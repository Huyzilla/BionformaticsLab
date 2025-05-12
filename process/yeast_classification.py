import torch
import torchvision.transforms as T
from PIL import Image
import base64
from io import BytesIO
import os
from process.model import WideResNet  # Đảm bảo bạn đã định nghĩa WideResNet trong file model.py

# Định nghĩa transform (giống với khi training)
transform = T.Compose([
    T.CenterCrop((32, 32)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Đường dẫn model
model_path = "test_WideResNet-28-10/best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Load model với error handling đúng cách"""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        print(f"Loading model from {model_path}...")

        # Khởi tạo mô hình với đúng kiến trúc
        model = WideResNet(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=2).to(device)

        # Load trọng số vào mô hình
        checkpoint = torch.load(model_path, map_location=device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        print("✅ Model loaded successfully")
        return model

    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None  # Trả về None thay vì raise Exception

# Khởi tạo model
print("Initializing model...")
model = load_model()
if model is None:
    print("⚠️ Failed to load model!")

def process_base64_image(base64_str):
    """Convert base64 string to PIL Image"""
    try:
        img_data = base64.b64decode(base64_str)
        img = Image.open(BytesIO(img_data)).convert('RGB')
        return img
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")

def classify_yeast_image(base64_image):
    """Classify a yeast cell image"""
    if model is None:
        raise RuntimeError("Model not properly loaded, classification unavailable.")

    try:
        # Chuyển base64 thành ảnh
        img = process_base64_image(base64_image)

        # Apply transforms
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred_class].item()

        # Mapping class ID thành tên
        class_labels = {0: "Normal", 1: "Abnormal"}
        class_name = class_labels.get(pred_class, "Unknown")

        return {
            "class": class_name,
            "confidence": round(confidence * 100, 2),
            "class_id": pred_class
        }

    except Exception as e:
        raise RuntimeError(f"Error processing image: {str(e)}")
