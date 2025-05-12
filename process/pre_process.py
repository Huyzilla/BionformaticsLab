import cv2
import numpy as np

# Process the original image
def cut_unecessary_img(image):
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
        return image

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Set the threshold value and threshold the image
    threshold_value = 185
    ret, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Invert the thresholded image
    thresholded_image = cv2.bitwise_not(thresholded_image)

    # Find all contours in the thresholded image
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Create a black mask with the same size as the grayscale image
    mask = np.zeros_like(gray_image)

    # Save contours that meet the condition into a list
    new_contours = []

    # Set the minimum height for contours (50% of the image height)
    MIN_HEIGHT = image.shape[1] * 0.5

    # Filter contours with height greater than or equal to MIN_HEIGHT
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= MIN_HEIGHT:
            new_contours.append(cnt)

    # If no suitable contour is found, return the original image
    if not new_contours:
        return image

    # Get the largest contour that meets the condition
    con = new_contours[0]
    x, y, w, h = cv2.boundingRect(con)
    if h < image.shape[0] and w < image.shape[1]:
        # Draw a white contour on the mask
        cv2.drawContours(mask, [con], -1, (255), thickness=cv2.FILLED)

        # Apply the mask to the original image to keep the white contour area
        result = cv2.bitwise_and(image, image, mask=mask)

        # Crop the image
        result = result[y:y+h, x:x+w]

    result = result.astype(np.uint8)
    return result

# Process the cells
# Padding with the same color as the first pixel of the image
def resize_image(image, value=0):
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
        
    resized_image = padded_image[:target_height, :target_width, :]

    return resized_image

def split_image(image, patch_size=256):
    patches = []
    h, w, _ = image.shape
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)
    return patches

def new_resize_image(image, target_size, value=0):
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
def merge_images(original_image, images, tile_size=256):
    height_orig, width_orig, _ = original_image.shape
    grid_height = (height_orig + tile_size - 1) // tile_size
    grid_width = (width_orig + tile_size - 1) // tile_size

    original_height = tile_size * grid_height
    original_width = tile_size * grid_width

    merged_image = np.zeros((original_height, original_width), dtype=np.float32)

    idx = 0
    for i in range(grid_height):
        for j in range(grid_width):
            if idx < len(images):
                tile = images[idx]
                if tile.ndim == 3:
                    tile = tile[:, :, 0]
                elif tile.ndim == 2:
                    tile = tile[:, :] 
                merged_image[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size] = tile
                idx += 1

    return merged_image
