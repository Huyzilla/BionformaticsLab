import cv2
import numpy as np

# Process the original image
def cut_unecessary_img2(image):
    """
    Crop unnecessary parts of the image and keep only the main object.

    Parameters:
    image (array): The input image to process, in BGR format.

    Returns:
    array: The cropped image or the original image if no suitable contour is found.
    """
    # Check if the image is valid
    if image is None:
        print("Invalid image.")
        return image, (0, 0, image.shape[1], image.shape[0])  # ✅ Trả về đúng 2 giá trị

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold_value = 185
    ret, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    thresholded_image = cv2.bitwise_not(thresholded_image)

    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray_image)

    new_contours = []

    MIN_HEIGHT = image.shape[1] * 0.5

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= MIN_HEIGHT:
            new_contours.append(cnt)

    if not new_contours:
          return image, (0, 0, image.shape[1], image.shape[0])  # Không cắt, giữ nguyên ảnh gốc

    con = new_contours[0]
    x, y, w, h = cv2.boundingRect(con)
    if h < image.shape[0] and w < image.shape[1]:

        cv2.drawContours(mask, [con], -1, (255), thickness=cv2.FILLED)

        result = cv2.bitwise_and(image, image, mask=mask)
        result = result[y:y+h, x:x+w]

        result = result.astype(np.uint8)
        return result.astype(np.uint8), (x, y, w, h)  # Trả về ảnh đã cắt và tọa độ cắt
    else:
        return image, (0, 0, image.shape[1], image.shape[0])  # Nếu không cắt, giữ nguyên

# Process the cells
# Padding with the same color as the first pixel of the image
def resize_image2(image, value=0):
    """
    Resize the input image to target x target x3. If the image is smaller, pad it. If it is larger, crop it.

    Parameters:
    image (array): The input image in BGR format.

    Returns:
    array: The resized image of size target x target x3.
    """
    height, width, _ = image.shape
    target_height = ((height + 255) // 256) * 256
    target_width = ((width + 255) // 256) * 256
    pad_height = max(0, target_height - height)
    pad_width = max(0, target_width - width)
    # Pad the image
    padded_image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=value)
        
    # resized_image = padded_image[:target_height, :target_width, :]

    return padded_image, pad_height, pad_width 

def split_image2(image, patch_size=256):
    patches = []
    h, w, _ = image.shape
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)
    return patches

def new_resize_image2(image, target_size, value=0):
    """
    Resize the input image to target x target x3. If the image is smaller, pad it. If it is larger, crop it.

    Parameters:
    image (array): The input image in BGR format.

    Returns:
    array: The resized image of size target x target x3.
    """
    height, width, _ = image.shape

    if height < target_size or width < target_size:
        # Calculate padding
        pad_height = max(0, target_size - height)
        pad_width = max(0, target_size - width)
        # Pad the image
        padded_image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=value)
        # Crop to ensure the final size is exactly target_size x target_size
        resized_image = padded_image[:target_size, :target_size, :]
    else:
        # Crop the image
        start_x = (width - target_size) // 2
        start_y = (height - target_size) // 2
        resized_image = image[start_y:start_y + target_size, start_x:start_x + target_size, :]

    return resized_image

# Merge small images back into a large image
def merge_patches_into_image2(predicted_masks, resized_image):
    m, n = resized_image.shape[0] // 256, resized_image.shape[1] // 256
    print(f"✅ Expected patches: {m} x {n} = {m*n}")
    print(m, n)
    height, width = 256 * m, 256 * n
    merged_image = np.zeros((height, width), dtype=np.float32)

    # Xác định cách các patch sẽ được đặt vào ảnh lớn
    for i in range(m):
        for j in range(n):
            patch_index = i * n + j
            patch = predicted_masks[patch_index]

            start_x = i * 256
            start_y = j * 256
            
            merged_image[start_x:start_x + 256, start_y:start_y + 256] = patch

    return merged_image, resized_image


def restore_mask2(full_mask, original_size, crop_coords, pad_h, pad_w):
    """
    Khôi phục ảnh mask về đúng kích thước ảnh gốc.

    Parameters:
    - full_mask (numpy array): Ảnh mask có cùng kích thước với ảnh đã resize.
    - original_size (tuple): Kích thước ảnh gốc trước khi cắt (width, height).
    - crop_coords (tuple): Tọa độ (x, y, w, h) của vùng cắt trong ảnh gốc.
    - pad_h, pad_w (int): Số pixel padding đã thêm vào trước khi chia ảnh.

    Returns:
    - numpy array: Ảnh mask sau khi khôi phục và chèn vào ảnh gốc.
    """
    original_width, original_height = original_size
    x, y, w, h = crop_coords

    # 1️⃣ Cắt bỏ padding để mask khớp với ảnh đã cắt
    restored_mask = full_mask[:h, :w]

    # 2️⃣ Tạo ảnh mask có kích thước ảnh gốc và đặt mask đã cắt vào đúng vị trí
    final_mask = np.zeros((original_height, original_width), dtype=np.uint8)
    final_mask[y:y+h, x:x+w] = restored_mask

    return final_mask

