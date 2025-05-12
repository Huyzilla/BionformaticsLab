# 🧫 Yeast Cell Detection and Analysis

## I. 📌 Giới thiệu bài toán

Bài toán đặt ra là phát hiện, phân loại và phân tích hình thái học của các tế bào nấm men được nuôi trong môi trường ethanol, chụp qua kính hiển vi. Dữ liệu được chụp trên buồng đếm gồm 16 ô vuông, mỗi ô chứa nhiều tế bào.

### ✔️ Mục tiêu:

- Dự đoán **mask** của tế bào nấm men trong ảnh.
- Vẽ **bounding box** cho từng tế bào nấm men trong ảnh.
- **Đếm số lượng** nấm men trong từng ô buồng đếm (16 ô).
- **Trích xuất thông số hình thái** của từng tế bào: diện tích, chu vi, circularity,...

---

## II. 🧹 Tiền xử lý dữ liệu

- Gán nhãn từng tế bào trên ảnh gốc bằng công cụ **Paint**.
- Tiền xử lý ảnh: lọc nhiễu, tăng tương phản, chuyển đổi kênh màu.
- Cắt ảnh thành từng **vùng chứa tế bào** từ ảnh gốc dựa trên mask.
- Chuẩn hóa dữ liệu để đưa vào mô hình: **Segmentation (UNet)**

---

## III. 🧠 Huấn luyện mô hình

### UNet – Segmentation (Mask Prediction)

- Mục tiêu: phân vùng vùng chứa tế bào trên ảnh.
- Mô hình: **UNet** (TensorFlow/Keras).
- Đầu ra: ảnh nhị phân (1 = vùng tế bào, 0 = nền).

<div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
  <img src="https://github.com/user-attachments/assets/2e1ea3a8-8257-4044-b422-67654dfcc82c" width="200"/>
  <span style="font-size: 40px;">➡️</span>
  <img src="https://github.com/user-attachments/assets/7a841f08-fdd9-4943-b289-2ee87ee908a9" width="200"/>
</div>

- Ảnh bounding box
<p align="center">
<img src="https://github.com/user-attachments/assets/acc72104-3802-441a-a513-65c79966bc1f" width="200"/>

## IV. 🧮 Phân tích buồng đếm & đo hình thái học

- **Tự động phát hiện lưới 4x4 (16 ô)** trong ảnh buồng đếm.
- **Đếm số lượng tế bào** trong từng ô (dựa trên contour hoặc blob detection).
- Với mỗi tế bào:
  - Tính **diện tích**, **chu vi**, **chiều dài trục lớn/nhỏ**.
  - Tính **độ tròn (circularity)**, **aspect ratio**,...

<div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
  <img src="https://github.com/user-attachments/assets/2e1ea3a8-8257-4044-b422-67654dfcc82c" width="200"/>
  <span style="font-size: 40px;">➡️</span>
  <img src="https://github.com/user-attachments/assets/42dd1df1-8270-4b2d-ba35-6cb5481df296" width="200"/>
</div>

## V. 🔗 Triển khai API với FastAPI

- Xây dựng API nhận **ảnh đầu vào** → trả về:
  - Số lượng nấm men trong từng ô
  - Ảnh mask kết quả
  - Danh sách tế bào: vị trí, loại, thông số hình thái học
- Tích hợp **Swagger UI** để test trực tiếp trên trình duyệt.

---


## Cài đặt Thư viện

Để chạy dự án, bạn cần cài đặt các thư viện cần thiết. Thực hiện các bước dưới đây:

1. **Tạo môi trường ảo** (khuyến khích):

```bash
python -m venv env
source env/bin/activate  # Với Linux/MacOS
env\Scripts\activate  # Với Windows
```
### Cài đặt các thư viện từ tệp requirements.txt:

```bash
pip install -r requirements.txt
```
### Training unet with keras
```bash
unet_keras.ipynb
```
### Transfer learning cnn with pytorch
```bash
cnn_training_pytorch.ipynb
```
### Chạy API với FASTAPI
```bash
uvicorn main:app --reload
```
### Truy cập để đọc hướng dẫn và thử nghiệm
```bash
http://localhost:8000/docs
```
Có thể import file json ở trên trong postman để test các api
