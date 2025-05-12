import base64
import json
import os
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from process.prediction import predict_mask_ethanol, predict_mask, predict_cell
from process.bounding_box import draw_bounding_box
from models.cnn_model import cnn
from concurrent.futures import ThreadPoolExecutor
submit_router = APIRouter()
executor = ThreadPoolExecutor(max_workers=4)
RESULT_DIR = "results"

def analyze_image(image_data, image_id):
    try:
        image = Image.open(BytesIO(image_data)).convert("RGB")
        try:
            print("🔍 Running predict_mask_ethanol...")
            mask_new = predict_mask_ethanol(image_data)
            mask_new_pil = Image.fromarray(mask_new.astype(np.uint8))

            # === Lưu ảnh mask dưới dạng PNG và TIF ===
            mask_dir = os.path.join(RESULT_DIR, image_id)
            os.makedirs(mask_dir, exist_ok=True)

            png_path = os.path.join(mask_dir, "mask.png")
            tif_path = os.path.join(mask_dir, "mask.tif")

            mask_new_pil.save(png_path)
            mask_new_pil.save(tif_path)
            
            bb_image, bb_data = draw_bounding_box(mask_new,image,3)
        except Exception as e:
            print(f"❌ Error in predict_mask_ethanol: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to run predict_mask_ethanol")

        buffer2 = BytesIO()
        mask_new_pil.save(buffer2, format="PNG")
        mask_new_base64 = base64.b64encode(buffer2.getvalue()).decode('utf-8')

        buffer3 = BytesIO()
        bb_image.save(buffer3, format="PNG")
        bb_base64 = base64.b64encode(buffer3.getvalue()).decode('utf-8')
       
        response_content = {
            "image_id": image_id,
            "bounding_boxes": bb_data,
            "bb_img": bb_base64,
            "mask_img": mask_new_base64,      
        }

        return response_content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e) +" =_= Error in analyze_image function =)))))")

@submit_router.post("/submit_image/")
async def submit_image(request: Request):
    body = await request.json()
    base64_image = body.get("base64_image")
    image_id = body.get("image_id")

    if not image_id:
        raise HTTPException(status_code=400, detail="Missing image_id")

    status_path = f"{RESULT_DIR}/{image_id}/status.json"
    os.makedirs(os.path.dirname(status_path), exist_ok=True)

    # Kiểm tra nếu đã có status rồi thì trả lại trạng thái luôn
    if os.path.exists(status_path):
        with open(status_path, "r") as f:
            status_data = json.load(f)
        status = status_data.get("status")

        if status == "done":
            return {"status": "done", "result": status_data.get("result")}
        elif status == "error":
            return {"status": "error", "detail": status_data.get("detail", "Unknown error")}

    if not base64_image:
        return {"status": "not_found"}

    try:
        image_data = base64.b64decode(base64_image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64_image")

    # Xử lý trực tiếp và lưu trạng thái
    try:
        result = analyze_image(image_data, image_id)
        with open(status_path, "w") as f:
            json.dump({"status": "done", "result": result}, f)
        return {"status": "done", "result": result}
    except Exception as e:
        with open(status_path, "w") as f:
            json.dump({"status": "error", "detail": str(e)}, f)
        raise HTTPException(status_code=500, detail=str(e))
