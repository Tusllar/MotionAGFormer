# MotionAGFormer: 3D Human Pose Estimation & Feature Extraction on Ewalk Dataset

[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/) [![arXiv](https://img.shields.io/badge/arXiv-2210.06551-b31b1b.svg)](https://arxiv.org/abs/2310.16288) [![Paper Explanation](https://img.shields.io/badge/-Paper%20Explanation%20in%209%20Minutes-ea3323?logo=youtube)](https://youtu.be/iyLhxPjwBuQ?si=yoG-wlz7N1fq-PmY)

[![PapersWithCode: Human3.6M](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/motionagformer-enhancing-3d-human-pose/monocular-3d-human-pose-estimation-on-human3)](https://paperswithcode.com/sota/monocular-3d-human-pose-estimation-on-human3?p=motionagformer-enhancing-3d-human-pose)
[![PapersWithCode: MPI-INF-3DHP](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/motionagformer-enhancing-3d-human-pose/3d-human-pose-estimation-on-mpi-inf-3dhp?metric=MPJPE)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-mpi-inf-3dhp?metric=MPJPE&p=motionagformer-enhancing-3d-human-pose)

---

## Giới thiệu

Đây là mã nguồn chính thức của bài báo [MotionAGFormer: Enhancing 3D Human Pose Estimation With a Transformer-GCNFormer Network (WACV 2024)](https://openaccess.thecvf.com/content/WACV2024/html/Mehraban_MotionAGFormer_Enhancing_3D_Human_Pose_Estimation_With_a_Transformer-GCNFormer_Network_WACV_2024_paper.html).

Repo này đã được chỉnh sửa để hỗ trợ trích xuất đặc trưng từ bộ dữ liệu **Ewalk** gồm 84 video.

---

## Môi trường cài đặt

- Python 3.8.10
- PyTorch 2.0.0
- CUDA 12.2

Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

---

## Hỗ trợ dữ liệu

### 1. Bộ dữ liệu Ewalk (84 video)

Repo đã được chỉnh sửa để hỗ trợ trích xuất đặc trưng từ 84 video của bộ dữ liệu Ewalk. Sau khi thực hiện bạn sẽ có cấu trúc dữ liệu như dưới đây:

```
demo/Dataset/
  └── <Tên_video>/
      ├── input_2D/
      │   └── keypoints.npz
      ├── pose/
      ├── pose2D/
      ├── pose3D/
      └── <Tên_video>.mp4
```

### 2. Các bộ dữ liệu khác (Human3.6M, MPI-INF-3DHP)

Repo vẫn hỗ trợ đầy đủ các bộ dữ liệu chuẩn như Human3.6M, MPI-INF-3DHP. Xem hướng dẫn chi tiết trong thư mục `configs/` hoặc phần bên dưới.

---

## Hướng dẫn trích xuất đặc trưng từ video Ewalk

1. **Đặt video Ewalk vào thư mục** `demo/Video/`.
2. **Chạy script trích xuất đặc trưng** (ví dụ):
   ```bash
   python demo/vis.py
   ```
   - Kết quả sẽ được lưu trong các thư mục `pose2D`, `pose3D`, `input_2D` dưới mỗi video.
   - Đặc trưng 2D/3D và keypoints sẽ được lưu dưới dạng `.npz` hoặc `.png` (tùy script).

3. **Tùy chỉnh script**: Bạn có thể chỉnh sửa các file trong `demo/vis.py`, `demo/lib/preprocess.py` hoặc các script khác để phù hợp với mục đích trích xuất đặc trưng riêng.

---

## Huấn luyện & Đánh giá

- **Huấn luyện trên Human3.6M**:
  ```bash
  python train.py --config configs/h36m/MotionAGFormer-base.yaml
  ```
- **Huấn luyện trên MPI-INF-3DHP**:
  ```bash
  python train_3dhp.py --config configs/mpi/MotionAGFormer-base.yaml
  ```
- **Đánh giá mô hình**:
  ```bash
  python train.py --eval-only --checkpoint <CHECKPOINT-DIR> --checkpoint-file <FILE> --config <CONFIG>
  ```

---

## Demo trực quan

- Bạn cần chuẩn bị dữ liệu video Ewalk theo cấu trúc sau. Đặt video vào `demo/Video/`.
- Chạy:
  ```bash
  python demo/vis.py
  ```
- Kết quả sẽ được lưu trong `demo/output/` hoặc các thư mục tương ứng.

---

## Tài liệu tham khảo & Code gốc

- [MotionAGFormer (code gốc)](https://github.com/Mehraban-s/MotionAGFormer)
- [MotionBERT](https://github.com/Walter0807/MotionBERT)
- [P-STMO](https://github.com/paTRICK-swk/P-STMO)
- [MHFormer](https://github.com/Vegetebird/MHFormer)
- [Bài báo WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Mehraban_MotionAGFormer_Enhancing_3D_Human_Pose_Estimation_With_a_Transformer-GCNFormer_Network_WACV_2024_paper.html)

---

## Citation

Nếu bạn sử dụng repo này cho nghiên cứu, vui lòng trích dẫn:

```bibtex
@inproceedings{motionagformer2024,
  title     =   {MotionAGFormer: Enhancing 3D Human Pose Estimation with a Transformer-GCNFormer Network}, 
  author    =   {Soroush Mehraban, Vida Adeli, Babak Taati},
  booktitle =   {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year      =   {2024}
}
```

