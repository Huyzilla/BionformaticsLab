import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

def process_mask_with_bbox(mask_image, original_image, image_id, output_dir="bb_normal_images"):
    """
    Xử lý ảnh mask để tìm contours, vẽ bounding box và lưu ảnh kết quả.

    mask_image: Ảnh mask dạng PIL.Image (mode "L").
    original_image: Ảnh gốc dạng NumPy array.
    image_id: ID của ảnh để đặt tên file đầu ra.
    output_dir: Thư mục để lưu kết quả.
    """
    # Chuyển đổi ảnh mask từ PIL sang NumPy
    mask = np.array(mask_image)

    # Tạo bản sao của ảnh gốc để vẽ bounding box
    image_copy = original_image.copy()

    # Tìm contours trên ảnh mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Mức độ mở rộng bounding box
    margin = 3  

    # Vẽ bounding box và đánh số thứ tự
    for i, contour in enumerate(contours, start=1):
        x, y, w, h = cv2.boundingRect(contour)

        # Mở rộng bounding box
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image_copy.shape[1] - x, w + 2 * margin)
        h = min(image_copy.shape[0] - y, h + 2 * margin)

        # Vẽ bounding box màu đen
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 0, 0), 2)

        # Tính toán vị trí để đặt số thứ tự
        text_x = x + w // 2
        text_y = y - 5 

        # Cấu hình font chữ
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(str(i), font, font_scale, font_thickness)[0]

        # Điều chỉnh vị trí để căn giữa số thứ tự
        text_x -= text_size[0] // 2

        # Vẽ số thứ tự lên ảnh
        cv2.putText(image_copy, str(i), (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

    # Hiển thị ảnh kết quả
    plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Lưu ảnh với tên chứa image_id vào thư mục đích
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{image_id}.png"  # Đặt tên file theo image_id
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, image_copy)

    print(f"Ảnh đã được lưu tại: {output_path}")

    return image_copy  # Trả về ảnh có bounding box

    """
    Xử lý ảnh mask để tìm contours, vẽ bounding box và lưu ảnh kết quả.
    
    mask_image: Ảnh mask dạng PIL.Image (mode "L").
    original_image: Ảnh gốc dạng NumPy array.
    output_dir: Thư mục để lưu kết quả.
    """
    # Chuyển đổi ảnh mask từ PIL sang NumPy
    mask = np.array(mask_image)  # mask lúc này là ảnh grayscale uint8

    # Tạo bản sao của ảnh gốc để vẽ bounding box
    image_copy = original_image.copy()

    # Tìm contours trên ảnh mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Mức độ mở rộng bounding box
    margin = 3  

    # Vẽ bounding box và đánh số thứ tự
    for i, contour in enumerate(contours, start=1):
        x, y, w, h = cv2.boundingRect(contour)
        
        # Mở rộng bounding box
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image_copy.shape[1] - x, w + 2 * margin)
        h = min(image_copy.shape[0] - y, h + 2 * margin)

        # Vẽ bounding box màu đen
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 0, 0), 2)

        # Tính toán vị trí để đặt số thứ tự
        text_x = x + w // 2
        text_y = y - 5 

        # Cấu hình font chữ
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(str(i), font, font_scale, font_thickness)[0]

        # Điều chỉnh vị trí để căn giữa số thứ tự
        text_x -= text_size[0] // 2

        # Vẽ số thứ tự lên ảnh
        cv2.putText(image_copy, str(i), (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

    # Hiển thị ảnh kết quả
    plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Lưu ảnh kết quả
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "output_with_bbox.png")
    cv2.imwrite(output_path, image_copy)

    print(f"Ảnh đã được lưu tại: {output_path}")

    return image_copy  # Trả về ảnh có bounding box
