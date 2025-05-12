import base64
import json
import os
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from process.prediction import predict_mask, predict_cell, predict_mask_normal
from process.bounding_box import draw_bounding_box
from models.cnn_model import cnn
from concurrent.futures import ThreadPoolExecutor
from process.bounding_box import draw_bounding_box 

upload_normal_image_router = APIRouter()
executor = ThreadPoolExecutor(max_workers=4)

def analyze_image(image_data, image_id):
    try:
        # Giải mã ảnh gốc
        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Dự đoán mask
        image_array, mask = predict_mask(image_data)
        mask_image = Image.fromarray(mask)
        
        
        # Dự đoán bằng model mới
        try:
            print("🔍 Running predict_mask_v3...")
            mask_new = predict_mask_normal(image_data)
            mask_new_pil = Image.fromarray(mask_new.astype(np.uint8))
            bb_image, bboxes = draw_bounding_box(mask_new,image,3)
        except Exception as e:
            print(f"❌ Error in predict_mask_v3: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to run predict_mask_v3")

        

        # Chuyển mask sang base64
        buffer1 = BytesIO()
        mask_image.save(buffer1, format="PNG")
        mask_base64 = base64.b64encode(buffer1.getvalue()).decode('utf-8')

        
        buffer2 = BytesIO()
        mask_new_pil.save(buffer2, format="PNG")
        mask_new_base64 = base64.b64encode(buffer2.getvalue()).decode('utf-8')

        buffer3 = BytesIO()
        bb_image.save(buffer3, format="PNG")
        bb_base64 = base64.b64encode(buffer3.getvalue()).decode('utf-8')
        # Nhận diện cell bằng CNN
        normal, abnormal, normal_2x, abnormal_2x, bounding_boxes, contours_list = predict_cell(image_array, image_id, mask, cnn)

        # Lưu ảnh gốc và mask để xử lý bounding box
        # mask_path = f"temp/{image_id}_mask.png"
        # image_path = f"temp/{image_id}.png"
        # os.makedirs("temp", exist_ok=True)
        # mask_image.save(mask_path)
        # image.save(image_path)

        # Vẽ bounding box lên ảnh
        # output_json_path = f"temp/{image_id}.json"
        # output_image = draw_bounding_boxes(image, mask_image, image_id, output_json_path)

        # Chuyển ảnh kết quả sang base64
        # output_buffer = BytesIO()
        # output_image.save(output_buffer, format="PNG")
        # output_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')

        # # Đọc JSON bounding box
        # with open(output_json_path, "r", encoding="utf-8") as json_file:
        #     bounding_box_info = json.load(json_file)

        response_content = {
            "image_id": image_id,
            "cell_counts": {
                "abnormal": abnormal,
                "normal_2x": normal_2x,
                "abnormal_2x": abnormal_2x
            },
            "bounding_boxes": bboxes,
            "contours_list": contours_list,
            "bb_img": bb_base64,
            "mask_img": mask_new_base64,        # Ảnh mask
        }

        return response_content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@upload_normal_image_router.post("/upload_image/normal_image/")
async def upload_image(request: Request):
    try:
        body = await request.json()
        if "base64_image" not in body:
            raise HTTPException(status_code=400, detail="base64_image field is required")

        base64_image = body["base64_image"]
        image_id = body["image_id"]
        image_data = base64.b64decode(base64_image)

        future = executor.submit(analyze_image, image_data, image_id)
        content = future.result()
        mask_img = content.pop("mask_img", None)
        bb_img = content.pop("bb_img", None)

        response_content = {
            "json" : content,
            "mask_img" : mask_img,
            "bb_img": bb_img,
        }
        
        return JSONResponse(content = response_content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
