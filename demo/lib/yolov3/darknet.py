from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import sys

from lib.yolov3.util import convert2cpu as cpu
from lib.yolov3.util import predict_transform


class test_net(nn.Module):
    def __init__(self, num_layers, input_size):
        super(test_net, self).__init__()
        self.num_layers= num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList([nn.Linear(5,5) for x in range(num_layers)])
        self.output = nn.Linear(5,2)

    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416, 416))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :]/255.0
    img_ = torch.from_numpy(img_).float()
    return img_


def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """
    # cfgfile = os.path.join(sys.path[-1], cfgfile)
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')     # store the lines in a list
    lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":               # This marks the start of a new block
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


class MaxPoolStride1(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

    def forward(self, x):
        padded_x = F.pad(x, (0, self.pad, 0, self.pad), mode="replicate")
        pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
        return pooled_x


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

    def forward(self, x, inp_dim, num_classes, confidence):
        x = x.data
        global CUDA
        prediction = x
        prediction = predict_transform(prediction, inp_dim, self.anchors, num_classes, confidence, CUDA)
        return prediction


class Upsample(nn.Module):
    def __init__(self, stride=2):
        super(Upsample, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        ws = stride
        hs = stride
        x = x.view(B, C, H, 1, W, 1).expand(B, C, H, stride, W, stride).contiguous().view(B, C, H*stride, W*stride)
        return x


class ReOrgLayer(nn.Module):
    def __init__(self, stride=2):
        super(ReOrgLayer, self).__init__()
        self.stride= stride

    def forward(self, x):
        assert(x.data.dim() == 4)
        B, C, H, W = x.data.shape
        hs = self.stride
        ws = self.stride
        assert(H % hs == 0),  "The stride " + str(self.stride) + " is not a proper divisor of height " + str(H)
        assert(W % ws == 0),  "The stride " + str(self.stride) + " is not a proper divisor of height " + str(W)
        x = x.view(B, C, H // hs, hs, W // ws, ws).transpose(-2, -3).contiguous()
        x = x.view(B, C, H // hs * W // ws, hs, ws)
        x = x.view(B, C, H // hs * W // ws, hs*ws).transpose(-1, -2).contiguous()
        x = x.view(B, C, ws*hs, H // ws, W // ws).transpose(1, 2).contiguous()
        x = x.view(B, C*ws*hs, H // ws, W // ws)
        return x


def create_modules(blocks):
    net_info = blocks[0]     # Captures the information about the input and pre-processing

    module_list = nn.ModuleList()

    index = 0    # indexing blocks helps with implementing route  layers (skip connections)
    prev_filters = 3
    output_filters = []

    for x in blocks:
        module = nn.Sequential()
        if x["type"] == "net":
            continue

        # If it's a convolutional layer
        if x["type"] == "convolutional":
            # Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)

            # Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # Check the activation.
            # It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)

        # If it's an upsampling layer
        # We use Bilinear2dUpsampling

        elif x["type"] == "upsample":
            stride = int(x["stride"])
#           upsample = Upsample(stride)
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            module.add_module("upsample_{}".format(index), upsample)

        # If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')

            # Start  of a route
            start = int(x["layers"][0])

            # end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0

            # Positive anotation
            if start > 0:
                start = start - index

            if end > 0:
                end = end - index

            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        # shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            from_ = int(x["from"])
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        elif x["type"] == "maxpool":
            stride = int(x["stride"])
            size = int(x["size"])
            if stride != 1:
                maxpool = nn.MaxPool2d(size, stride)
            else:
                maxpool = MaxPoolStride1(size)

            module.add_module("maxpool_{}".format(index), maxpool)

        # Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        else:
            print("Something I dunno")
            assert False

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        index += 1

    return (net_info, module_list)


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0

    def get_blocks(self):
        return self.blocks

    def get_module_list(self):
        return self.module_list

    def forward(self, x, CUDA):
        detections = []
        modules = self.blocks[1:]
        outputs = {}   # We cache the outputs for the route layer

        write = 0
        for i in range(len(modules)):

            module_type = (modules[i]["type"])
            if module_type == "convolutional" or module_type == "upsample" or module_type == "maxpool":

                x = self.module_list[i](x)
                outputs[i] = x

            elif module_type == "route":
                layers = modules[i]["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)
                outputs[i] = x

            elif module_type == "shortcut":
                from_ = int(modules[i]["from"])
                x = outputs[i-1] + outputs[i+from_]
                outputs[i] = x

            elif module_type == 'yolo':

                anchors = self.module_list[i][0].anchors
                # Get the input dimensions
                inp_dim = int(self.net_info["height"])

                # Get the number of classes
                num_classes = int(modules[i]["classes"])

                # Output the result
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)

                if type(x) == int:
                    continue

                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

                outputs[i] = outputs[i-1]

        try:
            return detections
        except:
            return 0
        
    def load_weights(self, weightfile):
        """
        Load weights with improved error handling and debugging
        """
        try:
            fp = open(weightfile, "rb")
        except FileNotFoundError:
            print(f"Weight file not found: {weightfile}")
            return False
        
        # Read header information
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        
        print(f"Loading weights from: {weightfile}")
        print(f"Header info - Major: {header[0]}, Minor: {header[1]}, Revision: {header[2]}")
        print(f"Images seen during training: {self.seen}")
        
        # Load all weights
        weights = np.fromfile(fp, dtype=np.float32)
        fp.close()
        
        print(f"Total weights available: {len(weights)}")
        
        ptr = 0
        weights_loaded = 0
        
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            
            if module_type == "convolutional":
                model = self.module_list[i]
                
                # Check if batch normalization is used
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                
                conv = model[0]
                
                print(f"Layer {i}: Conv layer - filters: {conv.out_channels}, "
                      f"kernel: {conv.kernel_size}, batch_norm: {batch_normalize}")
                
                if batch_normalize:
                    # Handle batch normalization layers
                    if len(model) < 2:
                        print(f"Warning: Expected batch norm layer but not found at index {i}")
                        continue
                        
                    bn = model[1]
                    num_bn_biases = bn.bias.numel()
                    
                    # Check if we have enough weights
                    if ptr + 4 * num_bn_biases > len(weights):
                        print(f"Error: Not enough weights for batch norm layer {i}")
                        print(f"Need: {4 * num_bn_biases}, Available: {len(weights) - ptr}")
                        break
                    
                    # Load batch norm parameters: bias, weight, running_mean, running_var
                    try:
                        bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                        ptr += num_bn_biases
                        
                        bn_weights = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                        ptr += num_bn_biases
                        
                        bn_running_mean = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                        ptr += num_bn_biases
                        
                        bn_running_var = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                        ptr += num_bn_biases
                        
                        # Reshape and copy to model
                        bn.bias.data.copy_(bn_biases.view_as(bn.bias.data))
                        bn.weight.data.copy_(bn_weights.view_as(bn.weight.data))
                        bn.running_mean.copy_(bn_running_mean.view_as(bn.running_mean))
                        bn.running_var.copy_(bn_running_var.view_as(bn.running_var))
                        
                        print(f"  ✓ Loaded batch norm parameters: {num_bn_biases} each")
                        
                    except Exception as e:
                        print(f"  ✗ Failed to load batch norm for layer {i}: {e}")
                        continue
                else:
                    # Handle bias for conv layers without batch norm
                    if conv.bias is not None:
                        num_biases = conv.bias.numel()
                        
                        if ptr + num_biases > len(weights):
                            print(f"Error: Not enough weights for conv bias layer {i}")
                            break
                        
                        try:
                            conv_biases = torch.from_numpy(weights[ptr:ptr + num_biases])
                            ptr += num_biases
                            conv.bias.data.copy_(conv_biases.view_as(conv.bias.data))
                            print(f"  ✓ Loaded conv bias: {num_biases}")
                        except Exception as e:
                            print(f"  ✗ Failed to load conv bias for layer {i}: {e}")
                            continue
                
                # Load convolutional weights
                num_weights = conv.weight.numel()
                
                if ptr + num_weights > len(weights):
                    print(f"Error: Not enough weights for conv layer {i}")
                    print(f"Need: {num_weights}, Available: {len(weights) - ptr}")
                    print(f"Conv shape: {conv.weight.data.shape}")
                    break
                
                try:
                    conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                    ptr += num_weights
                    
                    # Reshape weights to match conv layer
                    target_shape = conv.weight.data.shape
                    if conv_weights.numel() == conv.weight.data.numel():
                        conv_weights = conv_weights.view(target_shape)
                        conv.weight.data.copy_(conv_weights)
                        weights_loaded += 1
                        print(f"  ✓ Loaded conv weights: {num_weights} ({target_shape})")
                    else:
                        print(f"  ✗ Weight size mismatch for layer {i}")
                        print(f"    Expected: {conv.weight.data.numel()}, Got: {conv_weights.numel()}")
                        
                except Exception as e:
                    print(f"  ✗ Failed to load conv weights for layer {i}: {e}")
                    continue
        
        print(f"\nWeight loading summary:")
        print(f"Total conv layers processed: {weights_loaded}")
        print(f"Weights consumed: {ptr}/{len(weights)}")
        
        if ptr < len(weights):
            print(f"Warning: {len(weights) - ptr} weights remain unused")
        
        return weights_loaded > 0


    def create_modules(blocks):
        """
        Enhanced module creation with better error handling
        """
        net_info = blocks[0]
        module_list = nn.ModuleList()
        
        index = 0
        prev_filters = 3
        output_filters = []
        
        print("Creating YOLO model architecture:")
        
        for x in blocks:
            module = nn.Sequential()
            
            if x["type"] == "net":
                print(f"Network config: {x}")
                continue
            
            if x["type"] == "convolutional":
                # Get layer parameters
                activation = x["activation"]
                try:
                    batch_normalize = int(x["batch_normalize"])
                    bias = False
                except:
                    batch_normalize = 0
                    bias = True
                
                filters = int(x["filters"])
                padding = int(x["pad"])
                kernel_size = int(x["size"])
                stride = int(x["stride"])
                
                if padding:
                    pad = (kernel_size - 1) // 2
                else:
                    pad = 0
                
                # Create convolutional layer
                conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
                module.add_module("conv_{0}".format(index), conv)
                
                print(f"Layer {index}: Conv2d({prev_filters}, {filters}, {kernel_size}, stride={stride}, pad={pad})")
                
                # Add batch normalization if specified
                if batch_normalize:
                    bn = nn.BatchNorm2d(filters)
                    module.add_module("batch_norm_{0}".format(index), bn)
                    print(f"  + BatchNorm2d({filters})")
                
                # Add activation function
                if activation == "leaky":
                    activn = nn.LeakyReLU(0.1, inplace=True)
                    module.add_module("leaky_{0}".format(index), activn)
                    print(f"  + LeakyReLU(0.1)")
            
            elif x["type"] == "upsample":
                stride = int(x["stride"])
                upsample = nn.Upsample(scale_factor=stride, mode="nearest")
                module.add_module("upsample_{}".format(index), upsample)
                print(f"Layer {index}: Upsample(scale_factor={stride})")
            
            elif x["type"] == "route":
                x["layers"] = x["layers"].split(',')
                start = int(x["layers"][0])
                
                try:
                    end = int(x["layers"][1])
                except:
                    end = 0
                
                if start > 0:
                    start = start - index
                if end > 0:
                    end = end - index
                
                route = EmptyLayer()
                module.add_module("route_{0}".format(index), route)
                
                if end < 0:
                    filters = output_filters[index + start] + output_filters[index + end]
                    print(f"Layer {index}: Route (concatenate layers {index + start} and {index + end})")
                else:
                    filters = output_filters[index + start]
                    print(f"Layer {index}: Route (from layer {index + start})")
            
            elif x["type"] == "shortcut":
                from_ = int(x["from"])
                shortcut = EmptyLayer()
                module.add_module("shortcut_{}".format(index), shortcut)
                print(f"Layer {index}: Shortcut (from layer {index + from_})")
            
            elif x["type"] == "maxpool":
                stride = int(x["stride"])
                size = int(x["size"])
                if stride != 1:
                    maxpool = nn.MaxPool2d(size, stride)
                else:
                    maxpool = MaxPoolStride1(size)
                module.add_module("maxpool_{}".format(index), maxpool)
                print(f"Layer {index}: MaxPool2d({size}, stride={stride})")
            
            elif x["type"] == "yolo":
                mask = x["mask"].split(",")
                mask = [int(x) for x in mask]
                
                anchors = x["anchors"].split(",")
                anchors = [int(a) for a in anchors]
                anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in mask]
                
                detection = DetectionLayer(anchors)
                module.add_module("Detection_{}".format(index), detection)
                print(f"Layer {index}: YOLO Detection ({len(anchors)} anchors)")
            
            else:
                print(f"Unknown layer type: {x['type']}")
                assert False
            
            module_list.append(module)
            prev_filters = filters
            output_filters.append(filters)
            index += 1
        
        return (net_info, module_list)


# Alternative: Download correct weights function
    def download_yolo_weights():
        """
        Download the correct YOLO weights if they don't exist
        """
        import urllib.request
        import os
        
        weights_url = "https://pjreddie.com/media/files/yolov3.weights"
        weights_path = "yolov3.weights"
        
        if not os.path.exists(weights_path):
            print("Downloading YOLO v3 weights...")
            try:
                urllib.request.urlretrieve(weights_url, weights_path)
                print("Download completed!")
                return weights_path
            except Exception as e:
                print(f"Download failed: {e}")
                return None
        else:
            print("Weights file already exists.")
            return weights_path


    # Usage example with error handling
    def safe_load_yolo_model(cfg_path, weights_path=None):
        """
        Safely load YOLO model with proper error handling
        """
        try:
            # Create model
            model = Darknet(cfg_path)
            
            # Download weights if not provided
            if weights_path is None:
                weights_path = download_yolo_weights()
                if weights_path is None:
                    return None
            
            # Load weights with error handling
            if os.path.exists(weights_path):
                success = model.load_weights(weights_path)
                if success:
                    print("Model loaded successfully!")
                    return model
                else:
                    print("Failed to load weights properly")
                    return None
            else:
                print(f"Weight file not found: {weights_path}")
                return None
                
        except Exception as e:
            print(f"Error loading model: {e}")
            return None



