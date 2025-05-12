import cv2
import numpy as np
from PIL import Image

def draw_bounding_box(mask, original, margin=3):
    if isinstance(original, Image.Image):
        original = np.array(original)

    image_copy = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

    # Tìm contours của mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Debug: Số lượng contours tìm thấy: {len(contours)}")

    bounding_boxes = []
    # Vẽ bounding box cho từng contour
    for i, contour in enumerate(contours, start=1):
        x, y, w, h = cv2.boundingRect(contour)

        # Mở rộng bounding box
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image_copy.shape[1] - x, w + 2 * margin)
        h = min(image_copy.shape[0] - y, h + 2 * margin)

        bounding_boxes.append((f"bbox_{i}", x, y, w, h))

        # Vẽ bounding box
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 0, 0), 2)

        # Vẽ số thứ tự vào bounding box
        text_x = x + w // 2
        text_y = y - 5
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(str(i), font, font_scale, font_thickness)[0]
        text_x -= text_size[0] // 2
        cv2.putText(image_copy, str(i), (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

    # Trả về ảnh dưới dạng PIL Image
    return Image.fromarray(image_copy), bounding_boxes
