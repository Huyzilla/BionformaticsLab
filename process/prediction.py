import cv2
import numpy as np
import math
from models.unet_model_ethanol import unet_model_ethanol
from models.unet_model_normal import unet_model_normal
from models.unet_model import unet
from PIL import Image
import io
import torch
from process.pre_process import split_image, merge_images, resize_image, cut_unecessary_img, new_resize_image
from process.pre_process_normal_images import split_image2, resize_image2, cut_unecessary_img2, merge_patches_into_image2, restore_mask2
from process.process_mask import get_watershed_mask
from process.calculator import black_percentage
import base64
    
def predict_cell(image,image_name,mask, model):
    """
    Parameters:
    image (numpy.ndarray): The input image containing cells, in BGR format.
    mask (numpy.ndarray): A binary mask to identify the regions containing cells.
    model (keras.Model): A pre-trained machine learning model for cell classification.
    
    Returns:
    normal (int): The number of cells classified as "normal".
    abnormal (int): The number of cells classified as "abnormal".
    normal_2x (int): The number of cells classified as "normal_2x".
    abnormal_2x (int): The number of cells classified as "abnormal_2x".
    image (numpy.ndarray): The input image with rectangles drawn around detected cells, including area and perimeter information.
    bounding_boxes (list): A list of objects containing information about the location, size, type, and contours of the cells.

    """
    with open("thangdo.txt") as f:
        pixel_micro_m = 10.0 / float(f.read())
    label_dict = {
        0: "abnormal",
        1: "abnormal_2x",
        2: "normal",
        3: "normal_2x"
    }
    
    ret, nguong1 = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    nguong2 = cv2.bitwise_not(nguong1)
    
    contours, _ = cv2.findContours(nguong2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    MIN_HEIGHT = 10
    MAX_HEIGHT = image.shape[0] * 0.25
    
    bounding_boxes = []
    contours_list = []
    normal = 0
    abnormal = 0
    normal_2x = 0
    abnormal_2x = 0
    id = 1
    for cnt in contours[:-1]:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > MIN_HEIGHT and w < MAX_HEIGHT and h > MIN_HEIGHT and h < MAX_HEIGHT:
            y1 = max(0, y - 4)
            y2 = min(mask.shape[0], y + h + 4)
            x1 = max(0, x - 4)
            x2 = min(mask.shape[1], x + w + 4)
            
            crop_number = image[y1:y2, x1:x2]
            crop_number = new_resize_image(crop_number, 64, value = image[0][0].tolist()) 
            # Dien tich
            area = cv2.contourArea(cnt) * pixel_micro_m * pixel_micro_m
            #   Chu vi
            perimeter = cv2.arcLength(cnt, True) * pixel_micro_m
            
            #   Circularity
            circularity = 4*math.pi * area / (perimeter*perimeter) if perimeter != 0 else 0

            # Convexity
            hull = cv2.convexHull(cnt) 
            convexity = cv2.arcLength(hull, True) / cv2.arcLength(cnt, True) 

            # CE Diameter
            CE_diameter = math.sqrt(4 * area / math.pi) 

            #  Major/minor axis length
            ellipse = cv2.fitEllipse(cnt) 
            major_axis_length = ellipse[1][1] 
            minor_axis_length = ellipse[1][0] 

            # Aspect Ratio
            aspect_ratio = minor_axis_length / major_axis_length if major_axis_length != 0 else 0

            # Max distance
            max_distance = 0
            for i in range(len(cnt)):
                for j in range(i + 1, len(cnt)):
                    distance = np.linalg.norm(cnt[i][0] - cnt[j][0]) * pixel_micro_m
                    max_distance = max(max_distance, distance) 
            # print("Max distance: ", max_distance)
            crop_number = crop_number.astype(np.float32) / 255.0  
            label = model.predict(np.expand_dims(crop_number, axis=0)) 
            predicted_class1 = np.argmax(label, axis =1)
            # predicted_class2 = np.argmax(predictions2, axis =1)
            #print(prediction)
            color = None
            if predicted_class1[0] == 2:
                # print("normal")
                # cv2.rectangle(image, (x-2, y-2), ( x + w + 4 , y + h + 4 ) , (0, 0, 255), 2)
                normal += 1
                color = "red"
            elif predicted_class1[0] == 0:
                #print("abnormal")
                # cv2.rectangle(image, (x-2, y-2), ( x + w + 4 , y + h + 4 ) , (128, 0, 128), 2)
                abnormal += 1
                color = "purple"
            elif predicted_class1[0] == 1:
                # print("abnormal_2x")
                # cv2.rectangle(image, (x-2, y-2), ( x + w + 4 , y + h + 4 ) , (255, 0, 0), 2)
                abnormal_2x += 1
                color = "blue"
            else:
                #normal 2x
                # cv2.rectangle(image, (x-2, y-2), ( x + w + 4 , y + h + 4 ) , (0, 255, 0), 2)
                normal_2x += 1
                color = "green"
            contour_points = [{"x": int(point[0][0]), "y": int(point[0][1])} for point in cnt]
            bounding_boxes.append({
                "cell_id" : f"{image_name}_" + str(id),
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "type": label_dict[int(predicted_class1[0])],
                "color": color,
            })
            contours_list.append({
                "cell_id": f"{image_name}_" + str(id),
                "area": area,
                "perimeter": perimeter,
                "circularity": circularity,
                "convexity": convexity,
                "CE_diameter": CE_diameter,
                "major_axis_length": major_axis_length,
                "minor_axis_length": minor_axis_length,
                "aspect_ratio": aspect_ratio,
                "max_distance": max_distance,
                "contour": contour_points
            })
            id += 1
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # base64_image = cv2.imencode('.jpg', image)[1].tobytes()
    # encoded_image = base64.b64encode(base64_image).decode('utf-8')
    
    return normal, abnormal, normal_2x, abnormal_2x, bounding_boxes, contours_list

def predict_mask(image_data):
    try:
        # Decode and preprocess image
        image = Image.open(io.BytesIO(image_data))
        image = np.array(image)
        
        # Perform image processing
        image = cut_unecessary_img(image)
        image = resize_image(image, image[0][0].tolist())
        image_normalize = image.astype(np.float32) / 255.0
        
        # Split image into patches
        image_array = split_image(image_normalize)
        
        # Predict using U-Net model
        predictions = []
        batch_size = 1  

        for i in range(0, len(image_array), batch_size):
            batch = np.array(image_array[i:i+batch_size])
            batch_predictions = unet.predict(batch)
            predictions.extend(batch_predictions)
        
        predictions = np.array(predictions)
        merge_mask = merge_images(image,predictions)
        merge_mask = (merge_mask > 0.5).astype(np.uint8) * 255
        
        return image, merge_mask
    except Exception as e:
        raise RuntimeError(f"Error in image prediction: {e}")

def predict_mask_v2(image_data):
    try:
        # Decode and preprocess image
        image = Image.open(io.BytesIO(image_data))
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Perform image processing
        image = cut_unecessary_img(image)
        image = resize_image(image, image[0][0].tolist())
        image_normalize = image.astype(np.float32) / 255.0
        
        # Split image into patches
        image_array = split_image(image_normalize)
        
        # Predict using U-Net model
        predictions = []
        batch_size = 1  

        for i in range(0, len(image_array), batch_size):
            batch = np.array(image_array[i:i+batch_size])
            batch_predictions = unet.predict(batch)
            predictions.extend(batch_predictions)
        
        predictions = np.array(predictions)
        merge_mask = merge_images(image,predictions)
        merge_mask = (merge_mask > 0.5).astype(np.uint8) * 255
        
        return image, merge_mask
    except Exception as e:
        raise RuntimeError(f"Error in image prediction: {e}")

def predict_mask_normal(image_data):
    try:
        print("📌 Debug: Bắt đầu predict_mask_v3")
        # Decode and preprocess image
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = np.array(image)
        print(f"✅ Image shape: {image.shape}")  # Kiểm tra kích thước ảnh

        image_size = (image.shape[1], image.shape[0])  # (width, height)
        print(f"✅ Image size: {image_size}")  # Debug kích thước ảnh

        # Perform image processing
        cropped_image, crop_coords = cut_unecessary_img2(image)
        print(f"✅ Cropped image shape: {cropped_image.shape}")

        # Resize ảnh thành bội số của 256 và lưu padding
        resized_image, pad_h, pad_w = resize_image2(cropped_image)
        print(f"✅ Resized image shape: {resized_image.shape}")

        test_image = resized_image.astype(np.float32)
        test_image = test_image/255.0
        image_patches = split_image2(test_image)
        print(f"✅ Number of patches: {len(image_patches)}")

        image_patches = np.array(image_patches)
        image_patches = torch.tensor(image_patches).permute(0,3,1,2).float()
        print(f"✅ Image patches shape: {image_patches.shape}")

        if unet_model_normal is None:
            raise RuntimeError("❌ Model is not loaded. Please check the model path.")

        unet_model_normal.eval()  # Đảm bảo model ở chế độ đánh giá
        print(image_patches.size())
        batch_size = 1 
        num_patches = image_patches.shape[0]
        predicted_masks = []

        unet_model_normal.eval()
        with torch.no_grad():
            for i in range(0, num_patches, batch_size):
                batch = image_patches[i:i+batch_size]
                batch_output = unet_model_normal(batch)
                batch_output = (torch.sigmoid(batch_output) > 0.5).float()
                predicted_masks.append(batch_output.cpu().numpy())

        predicted_masks = np.concatenate(predicted_masks, axis=0)
        print(f"✅ Predicted mask shape: {predicted_masks.shape}")
        
        
        merged_mask, resized_image = merge_patches_into_image2(predicted_masks, resized_image)
        print(f"✅ Merged mask shape: {merged_mask.shape}")

        final_mask = restore_mask2(merged_mask,image_size, crop_coords, pad_h, pad_w)
        print(f"✅ Final mask shape: {final_mask.shape}")

        final_mask = (final_mask * 255).astype(np.uint8)
        print(f"🛠 Mask Min: {final_mask.min()}, Max: {final_mask.max()}")
        return final_mask
    
    except Exception as e:
        raise RuntimeError(f"Error in image prediction: {e}")

def predict_mask_ethanol(image_data):
    try:
        print("📌 Debug: Bắt đầu predict_mask_v3")
        # Decode and preprocess image
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = np.array(image)
        print(f"✅ Image shape: {image.shape}")  # Kiểm tra kích thước ảnh

        image_size = (image.shape[1], image.shape[0])  # (width, height)
        print(f"✅ Image size: {image_size}")  # Debug kích thước ảnh

        # Giữ nguyên 
        cropped_image = image
        crop_coords = (0, 0, image.shape[1], image.shape[0]) 
        print(f"✅ Cropped image shape: {cropped_image.shape}")

        # Resize ảnh thành bội số của 256 và lưu padding
        resized_image, pad_h, pad_w = resize_image2(cropped_image)
        print(f"✅ Resized image shape: {resized_image.shape}")

        test_image = resized_image.astype(np.float32)
        test_image = test_image/255.0
        image_patches = split_image2(test_image)
        print(f"✅ Number of patches: {len(image_patches)}")

        image_patches = np.array(image_patches)
        image_patches = torch.tensor(image_patches).permute(0,3,1,2).float()
        print(f"✅ Image patches shape: {image_patches.shape}")

        if unet_model_ethanol is None:
            raise RuntimeError("❌ Model is not loaded. Please check the model path.")

        unet_model_ethanol.eval()  
        print(image_patches.size())
        batch_size = 1 
        num_patches = image_patches.shape[0]
        predicted_masks = []

        unet_model_ethanol.eval()
        with torch.no_grad():
            for i in range(0, num_patches, batch_size):
                batch = image_patches[i:i+batch_size]
                batch_output = unet_model_ethanol(batch)
                batch_output = (torch.sigmoid(batch_output) > 0.5).float()
                predicted_masks.append(batch_output.cpu().numpy())

        predicted_masks = np.concatenate(predicted_masks, axis=0)
        print(f"✅ Predicted mask shape: {predicted_masks.shape}")
        
        
        merged_mask, resized_image = merge_patches_into_image2(predicted_masks, resized_image)
        print(f"✅ Merged mask shape: {merged_mask.shape}")

        final_mask = restore_mask2(merged_mask,image_size, crop_coords, pad_h, pad_w)
        print(f"✅ Final mask shape: {final_mask.shape}")

        final_mask = (final_mask * 255).astype(np.uint8)
        print(f"🛠 Mask Min: {final_mask.min()}, Max: {final_mask.max()}")
        return final_mask
    
    except Exception as e:
        raise RuntimeError(f"Error in image prediction: {e}")
    


def dead_or_alive_black_percentage(img_goc1):
    id = 1
    image, mask_using_unet = predict_mask_v2(img_goc1)
    
    mask = get_watershed_mask(mask_using_unet)
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    black_white_img = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
    bounding_boxes = []
    contours_list = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > image.shape[0]*0.5:
            continue
        img_temp = black_white_img[y:y+h, x:x+w]
        
        type = None
        color = None
        if (black_percentage(img_temp) <= 35):
            #cv2.rectangle(image, (x-2, y-2), ( x + w + 4 , y + h + 4 ) , (0, 255, 0), 2)
            type = "alive"
            color = "green"
        else :
            #cv2.rectangle(image, (x-2, y-2), ( x + w + 4 , y + h + 4 ) , (255, 0, 0), 2)
            type = "dead"
            color = "red"
        contour_points = [{"x": int(point[0][0]), "y": int(point[0][1])} for point in cnt]
        bounding_boxes.append({
            "cell_id" : id,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "type": type,
            "color": color,
        })
        contours_list.append({
            "cell_id": id,
            "contour": contour_points
        })
        
        id += 1
    return bounding_boxes, contours_list, mask_using_unet
        
        