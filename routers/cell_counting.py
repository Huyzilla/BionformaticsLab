import base64
import json
import os
import numpy as np
from io import BytesIO
from PIL import Image
import cv2 
import tempfile

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from process.prediction import predict_mask_ethanol
from process.Bai_toan_buong_dem import Count_Yeast_in_16_Squares
from concurrent.futures import ThreadPoolExecutor

cell_counting_router = APIRouter()
executor = ThreadPoolExecutor(max_workers=4)

def save_temp_image(image: Image.Image, suffix=".png") -> str:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    image.save(temp_file.name)
    return temp_file.name

def analyze_image(image_data, image_id):
    try:
        image = Image.open(BytesIO(image_data)).convert("RGB")

        try:
            print("🔍 Running predict_mask_ethanol...")
            mask_array = predict_mask_ethanol(image_data)
            mask_image = Image.fromarray(mask_array.astype(np.uint8))

            # Save temp files
            origin_path = save_temp_image(image)
            mask_path = save_temp_image(mask_image)
        except Exception as e:
            print(f"❌ Error in predict_mask_ethanol: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to run predict_mask_ethanol")
        
        counted_img, squares_info = Count_Yeast_in_16_Squares(
            Origin_path=origin_path,
            Mask_path=mask_path,
            show_process=True
        )
        buffer = BytesIO()
        mask_image.save(buffer, format="PNG")
        mask_array_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        _, buffer = cv2.imencode(".png", counted_img)
        result_base64 = base64.b64encode(buffer).decode("utf-8")

        # Remove temp file
        os.remove(origin_path)
        os.remove(mask_path)

        response_content = {
            "image_id": image_id,
            "mask_img": mask_array_base64,  
            "result_img": result_base64,
            "squares": squares_info
        }

        return response_content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@cell_counting_router.post("/upload_image/yeast_count/")
async def cell_counting(request: Request):
    try:
        # Read JSON data from request body
        body = await request.json()
        
        if "image_id" not in body:
            raise HTTPException(status_code=400, detail="image_id field is required")
        
        base64_image = body["base64_image"]
        image_id = body["image_id"]
        image_data = base64.b64decode(base64_image)

        future = executor.submit(analyze_image, image_data, image_id)
        content = future.result()
        mask_img = content.pop("mask_img", None)
        result_img = content.pop("result_img", None)

        response_content = {
            "json" : content,
            "mask_img" : mask_img,
            "result_img" : result_img
        }
        
        return JSONResponse(content = response_content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
