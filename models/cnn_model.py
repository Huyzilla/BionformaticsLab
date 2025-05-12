from keras.models import load_model
import os

model_path = "/root/FastAPIDemo/models/4_classes_new_copilot_model_time_2.keras"

try:
    cnn = load_model(model_path)
except Exception as e:
    print(f"Error loading CNN model from {model_path}: {e}")
    cnn = None
