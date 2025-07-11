import sys
import argparse
import cv2
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
import os 
import gc
import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
import copy
import pandas as pd
sys.path.append(os.getcwd())
from demo.lib.utils import normalize_screen_coordinates, camera_to_world
from model.MotionAGFormer import MotionAGFormer

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import cv2
import os
import numpy as np
import csv 

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

print(torch.cuda.is_available())

def show2Dpose(kps, img):
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    # for j,c in enumerate(connections):
    #     start = map(int, kps[c[0]])
    #     end = map(int, kps[c[1]])
    #     start = list(start)
    #     end = list(end)
    #     cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
    #     cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
    #     cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)


    # # Draw a marker at each joint
    # for i in range(kps.shape[0]):
    #     x, y = int(kps[i][0]), int(kps[i][1])
    #     cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

        # Vẽ xương giữa các khớp
    for j, c in enumerate(connections):
        start = list(map(int, kps[c[0]]))
        end = list(map(int, kps[c[1]]))
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)

    # Vẽ marker và số chỉ khớp
    for idx, kp in enumerate(kps):
        x, y = int(kp[0]), int(kp[1])  # Lấy chỉ x, y từ mỗi khớp
        cv2.circle(img, (x, y), radius=3, color=(0, 255, 0), thickness=-1)  # chấm xanh
        cv2.putText(img, str(idx), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)  # số khớp

    return img


def show3Dpose(vals, ax):
    ax.view_init(elev=15., azim=70)

    lcolor=(0,0,1)
    rcolor=(1,0,0)

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7
     # 🟢 Draw a marker at each joint
    ax.scatter(vals[:, 0], vals[:, 1], vals[:, 2], c='k', s=25, marker='o')  # 'k' = black

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    ax.set_aspect('auto') # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)

# def get_pose2D(video_path, output_dir):
#     cap = cv2.VideoCapture(video_path)
#     width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#     height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

#     print('\nGenerating 2D pose...')
#     keypoints, scores = hrnet_pose(video_path, det_dim=416, num_peroson=1, gen_output=True)
#     keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
#     print(f"Number of valid frames: {len(valid_frames)}")
    
#     # Add conf score to the last dim
#     keypoints = np.concatenate((keypoints, scores[..., None]), axis=-1)
#     print(f"Shape of keypoints: {keypoints.shape}")

#     # Xử lý shape: (1, frames, joints, 3) → (frames, joints, 3)
#     if keypoints.shape[0] == 1:
#         keypoints = keypoints[0]


#     # Chỉ giữ các joints theo new_order (loại bỏ joint số 9)
#     new_order = [0, 7, 8, 10, 11, 12, 13, 14, 15, 16, 4, 5, 6, 1, 2, 3]
#     keypoints = keypoints[:, new_order, :]  # Chọn đúng joints cần giữ

#     # Tạo thư mục output nếu chưa tồn tại
#     # output_dir += 'input_2D/'
#     output_dir = 'D:/PBL4/MotionAGFormer1/demo/Dataset0/2D/'
#     os.makedirs(output_dir, exist_ok=True)

#     # Lưu file CSV
#     existing_files = [f for f in os.listdir(output_dir) if f.startswith('VIDEO_RGB_') and f.endswith('.csv')]
#     existing_indices = [int(f.split('_')[-1].split('.')[0]) for f in existing_files]
#     next_index = max(existing_indices + [-1]) + 1
#     print(f"Next index for CSV file: {next_index}")

#     output_csv = output_dir + f'VIDEO_RGB_{next_index:03d}.csv'

#     header = ['frame']
#     num_joints = keypoints.shape[1]
#     for j in range(num_joints):
#         header += [f'kp{j}_x', f'kp{j}_y', f'kp{j}_conf']

#     with open(output_csv, mode='w', newline='') as csv_file:
#         writer = csv.writer(csv_file)
#         writer.writerow(header)

#         for frame_idx in range(keypoints.shape[0]):
#             row = [frame_idx]
#             for j in range(num_joints):
#                 x, y, conf = keypoints[frame_idx, j]
#                 row += [x, y, conf]
#             writer.writerow(row)

#     print(f"Keypoints đã được lưu vào CSV: {output_csv}")

def get_pose2D(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print('\nGenerating 2D pose...')
    keypoints, scores = hrnet_pose(video_path, det_dim=416, num_peroson=1, gen_output=True)
    #keypoints, scores = hrnet_pose(video_path, det_dim=608, num_peroson=2, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    print(f"Number of valid frames: {len(valid_frames)}")
    # Add conf score to the last dim
    keypoints = np.concatenate((keypoints, scores[..., None]), axis=-1)
    print(f"Shape of keypoints: {keypoints.shape}")

    # print(output_dir)
    output_dir += 'input_2D/'
    # print(f"output_dir: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    output_npz = output_dir + 'keypoints.npz'
    np.savez_compressed(output_npz, reconstruction=keypoints)


def img2video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 5

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])

    videoWrite = cv2.VideoWriter(output_dir + video_name + '.mp4', fourcc, fps, size) 

    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)

    videoWrite.release()


def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)


def resample(n_frames):
    even = np.linspace(0, n_frames, num=243, endpoint=False)
    result = np.floor(even)
    result = np.clip(result, a_min=0, a_max=n_frames - 1).astype(np.uint32)
    return result


def turn_into_clips(keypoints):
    clips = []
    n_frames = keypoints.shape[1]
    if n_frames <= 243:
        new_indices = resample(n_frames)
        clips.append(keypoints[:, new_indices, ...])
        downsample = np.unique(new_indices, return_index=True)[1]
    else:
        for start_idx in range(0, n_frames, 243):
            keypoints_clip = keypoints[:, start_idx:start_idx + 243, ...]
            clip_length = keypoints_clip.shape[1]
            if clip_length != 243:
                new_indices = resample(clip_length)
                clips.append(keypoints_clip[:, new_indices, ...])
                downsample = np.unique(new_indices, return_index=True)[1]
            else:
                clips.append(keypoints_clip)
    return clips, downsample

def turn_into_h36m(keypoints):
    new_keypoints = np.zeros_like(keypoints)
    new_keypoints[..., 0, :] = (keypoints[..., 11, :] + keypoints[..., 12, :]) * 0.5
    new_keypoints[..., 1, :] = keypoints[..., 11, :]
    new_keypoints[..., 2, :] = keypoints[..., 13, :]
    new_keypoints[..., 3, :] = keypoints[..., 15, :]
    new_keypoints[..., 4, :] = keypoints[..., 12, :]
    new_keypoints[..., 5, :] = keypoints[..., 14, :]
    new_keypoints[..., 6, :] = keypoints[..., 16, :]
    new_keypoints[..., 8, :] = (keypoints[..., 5, :] + keypoints[..., 6, :]) * 0.5
    new_keypoints[..., 7, :] = (new_keypoints[..., 0, :] + new_keypoints[..., 8, :]) * 0.5
    new_keypoints[..., 9, :] = keypoints[..., 0, :]
    new_keypoints[..., 10, :] = (keypoints[..., 1, :] + keypoints[..., 2, :]) * 0.5
    new_keypoints[..., 11, :] = keypoints[..., 6, :]
    new_keypoints[..., 12, :] = keypoints[..., 8, :]
    new_keypoints[..., 13, :] = keypoints[..., 10, :]
    new_keypoints[..., 14, :] = keypoints[..., 5, :]
    new_keypoints[..., 15, :] = keypoints[..., 7, :]
    new_keypoints[..., 16, :] = keypoints[..., 9, :]

    return new_keypoints


def flip_data(data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]):
    """
    data: [N, F, 17, D] or [F, 17, D]
    """
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1  # flip x of all joints
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]  # Change orders
    return flipped_data


def reorder_vector_list(vector_list, new_order, exclude_ids=None):
    """
    Sắp xếp lại danh sách vector (dạng list) theo thứ tự chỉ mục và loại bỏ các chỉ mục không mong muốn.

    :param vector_list: Danh sách các vector (list of list)
    :param new_order: Danh sách index mong muốn sắp xếp lại
    :param exclude_ids: Danh sách index cần loại bỏ (mặc định: None)
    :return: Danh sách vector đã sắp xếp lại
    """
    if exclude_ids is None:
        exclude_ids = []
    vector_list = [[x, z, y] for x, y, z in vector_list]

    return [vector_list[i] for i in new_order if i not in exclude_ids]


@torch.no_grad()
def get_pose3D(video_path, output_dir, output_csv,video_name):
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.n_layers, args.dim_in, args.dim_feat, args.dim_rep, args.dim_out = 16, 3, 128, 512, 3
    args.mlp_ratio, args.act_layer = 4, nn.GELU
    args.attn_drop, args.drop, args.drop_path = 0.0, 0.0, 0.0
    args.use_layer_scale, args.layer_scale_init_value, args.use_adaptive_fusion = True, 0.00001, True
    args.num_heads, args.qkv_bias, args.qkv_scale = 8, False, None
    args.hierarchical = False
    args.use_temporal_similarity, args.neighbour_num, args.temporal_connection_len = True, 2, 1
    args.use_tcn, args.graph_only = False, False
    args.n_frames = 243
    args = vars(args)

    ## Reload 
    model = nn.DataParallel(MotionAGFormer(**args)).cuda()

    # Put the pretrained model of MotionAGFormer in 'checkpoint/'
    model_path = sorted(glob.glob(os.path.join('checkpoint', 'motionagformer-b-h36m.pth.tr')))[0]

    pre_dict = torch.load(model_path, weights_only=False)
    model.load_state_dict(pre_dict['model'], strict=True)

    model.eval()

    ## input
    keypoints = np.load(output_dir + 'input_2D/keypoints.npz', allow_pickle=True)['reconstruction']
    # keypoints = np.load('demo/lakeside3.npy')
    # keypoints = keypoints[:240]
    # keypoints = keypoints[None, ...]
    # keypoints = turn_into_h36m(keypoints)
    

    clips, downsample = turn_into_clips(keypoints)
    


    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    ## 3D
    print('\nGenerating 2D pose image...')
    for i in tqdm(range(video_length)):
        ret, img = cap.read()
        if img is None:
            continue
        img_size = img.shape

                # === LƯU FRAME GỐC ===
        output_dir_raw = output_dir + 'frames_raw/'
        os.makedirs(output_dir_raw, exist_ok=True)
        cv2.imwrite(output_dir_raw + str(('%04d' % i)) + '.png', img)

        input_2D = keypoints[0][i]
        # print(f"Shape of input_2D: {input_2D.shape}")
        # print(input_2D)

        image = show2Dpose(input_2D, copy.deepcopy(img))

        output_dir_2D = output_dir +'pose2D/'
        os.makedirs(output_dir_2D, exist_ok=True)
        cv2.imwrite(output_dir_2D + str(('%04d'% i)) + '_2D.png', image)

    
    print('\nGenerating 3D pose...')
    all_rows = []  # List to store CSV rows
    pose3d_all_frames = []
    for idx, clip in enumerate(clips):
        input_2D = normalize_screen_coordinates(clip, w=img_size[1], h=img_size[0]) 
        input_2D_aug = flip_data(input_2D)
        
        input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()
        input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32')).cuda()

        output_3D_non_flip = model(input_2D) 
        output_3D_flip = flip_data(model(input_2D_aug))
        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        if idx == len(clips) - 1:
            output_3D = output_3D[:, downsample]

        output_3D[:, :, 0, :] = 0
        post_out_all = output_3D[0].cpu().detach().numpy()
        
        # Add this print statement
        # print(f"Shape of post_out_all: {post_out_all.shape}")
        # print(f"post_out_all min: {np.min(post_out_all)}, max: {np.max(post_out_all)}")
        # print(f"Sample values (first joint): {post_out_all[0]}")

        for j, post_out in enumerate(post_out_all):
            rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
            rot = np.array(rot, dtype='float32')
            post_out = camera_to_world(post_out, R=rot, t=0)
            post_out[:, 2] -= np.min(post_out[:, 2])
            max_value = np.max(post_out)
            post_out /= max_value

            fig = plt.figure(figsize=(9.6, 5.4))
            gs = gridspec.GridSpec(1, 1)
            gs.update(wspace=-0.00, hspace=0.05) 
            ax = plt.subplot(gs[0], projection='3d')
            # print(f"Shape of post_out: {post_out.shape}")
            # print(post_out)
            show3Dpose(post_out, ax)

            output_dir_3D = output_dir +'pose3D/'
            os.makedirs(output_dir_3D, exist_ok=True)
            str(('%04d'% (idx * 243 + j)))
            plt.savefig(output_dir_3D + str(('%04d'% (idx * 243 + j))) + '_3D.png', dpi=200, format='png', bbox_inches='tight')
            plt.close(fig)

            new_order = [0, 7, 8, 10, 11, 12, 13, 14, 15, 16, 4, 5, 6, 1, 2, 3]
            exclude = [9]
            reordered_vectors = reorder_vector_list(post_out, new_order, exclude)
            reordered_vectors = np.array(reordered_vectors)
            flattened =  reordered_vectors.flatten() 
            # print(f"Shape of flattened: {flattened.shape}") 
            pose3d_all_frames.append(flattened.tolist())    

        
    print('Generating 3D pose successful!')
    # 🔵 Save CSV
    df = pd.DataFrame(pose3d_all_frames)
    csv_path = os.path.join(output_csv, f'{video_name}.csv')
    # df.to_csv(csv_path)
    df.to_csv(csv_path, index=False, header=False)
    print(f"✅ 3D data saved to {csv_path}")
    ## all
    image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

# import cv2
# import os
# import gc
# from tqdm import tqdm

    print('\nGenerating demo...')
    for i in tqdm(range(len(image_2d_dir))):
        # Đọc ảnh bằng cv2 (OpenCV)
        image_2d = cv2.imread(image_2d_dir[i])
        image_3d = cv2.imread(image_3d_dir[i])

        # Convert ảnh từ BGR sang RGB (cv2 mặc định đọc ảnh ở định dạng BGR)
        image_2d = cv2.cvtColor(image_2d, cv2.COLOR_BGR2RGB)
        image_3d = cv2.cvtColor(image_3d, cv2.COLOR_BGR2RGB)

        ## crop ảnh
        edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
        image_2d = image_2d[:, edge:image_2d.shape[1] - edge]

        edge = 130
        image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

          # Kiểm tra và thay đổi kích thước ảnh sao cho cùng chiều cao và chiều rộng
        height_2d, width_2d = image_2d.shape[:2]
        height_3d, width_3d = image_3d.shape[:2]

        if height_2d != height_3d or width_2d != width_3d:
            # Thay đổi kích thước của một trong các ảnh sao cho giống nhau
            image_3d = cv2.resize(image_3d, (width_2d, height_2d))


        ## hiển thị ảnh
        font_size = 12
        # Tạo một canvas để vẽ (hình ảnh output sẽ được lưu vào đây)
        output_image = cv2.hconcat([image_2d, image_3d])

        # Tạo cửa sổ với OpenCV để hiển thị
        # Mặc dù `imshow` có thể hiển thị nhưng sẽ chậm nếu bạn muốn sử dụng toàn bộ quy trình hiệu quả
        cv2.putText(output_image, 'Input', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(output_image, 'Reconstruction', (image_2d.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        ## lưu hình ảnh
        output_dir_pose = output_dir + 'pose/'
        os.makedirs(output_dir_pose, exist_ok=True)

        # Lưu ảnh output
        cv2.imwrite(f"{output_dir_pose}{i:04d}_pose.png", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

        # Dọn dẹp bộ nhớ
        gc.collect()


    print('\nGenerating demo...')
    for i in tqdm(range(len(image_2d_dir))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])

        ## crop
        edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
        image_2d = image_2d[:, edge:image_2d.shape[1] - edge]

        edge = 130
        image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

        ## show
        font_size = 12
        fig = plt.figure(figsize=(15.0, 5.4))
        ax = plt.subplot(121)
        showimage(ax, image_2d)
        ax.set_title("Input", fontsize = font_size)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("Reconstruction", fontsize = font_size)

        ## save
        output_dir_pose = output_dir +'pose/'
        os.makedirs(output_dir_pose, exist_ok=True)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(output_dir_pose + str(('%04d'% i)) + '_pose.png', dpi=200, bbox_inches = 'tight')
        plt.close(fig)
        plt.close()
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='sample_video.mp4', help='input video')
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    input_folder = 'D:/PBL4/MotionAGFormer/demo/Video' 
    output_folder = 'D:/PBL4/MotionAGFormer/demo/Dataset' 
    output_csv = 'D:/PBL4/MotionAGFormer/demo/Dataset/Data'
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_csv, exist_ok=True)
    # Duyệt từng file trong thư mục
    i = 0
    for filename in os.listdir(input_folder):
        print(f'{i} Video : {filename}')
        i += 1

        if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Lọc file video
            # video_path = os.path.join(output_folder, filename)
            # print(f"Processing video: {video_path}")
            # output_dir = os.path.join(output_folder, f"{filename}")
            # output_3d_folder = os.path.join(output_3d_folder, f"{filename}")

            video_path = './demo/Video/' + filename
            video_name = video_path.split('/')[-1].split('.')[0]
            output_dir = './demo/Dataset/' + video_name + '/'
            get_pose2D(video_path, output_dir)
            get_pose3D(video_path, output_dir, output_csv,video_name)
            # print('Generating 3D Video pose start!!!')
            # img2video(video_path, output_dir)
            print('Generating demo successful!')
    
    print('All done!')


