import torch
import os
from unet.unet_model import MyUnet  # Import model của bạn

# Định nghĩa đường dẫn model
model_path =  "C:\Users\huyzi\Downloads\yeast-api-main\yeast-api-main\models\unet_model_ethanol.py"

# Số lượng kênh đầu vào và số lớp đầu ra
num_classes = 1
in_channels = 3

# Xác định thiết bị (GPU nếu có, không thì CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pytorch_model(model_path):
    try:
        # Khởi tạo mô hình
        model = MyUnet(in_channels, num_classes).to(device)

        # Load trọng số từ file .pt
        model.load_state_dict(torch.load(model_path, map_location=device))

        # Đặt mô hình về chế độ đánh giá    
        model.eval()

        print(f"✅ Model loaded successfully from {model_path}!")
        return model
    except Exception as e:
        print(f"❌ Error loading model from {model_path}: {e}")
        return None

# Gọi hàm để load model
unet_model_ethanol = load_pytorch_model(model_path)
