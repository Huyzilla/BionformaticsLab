from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import json
import os

request_status_router = APIRouter()

RESULT_DIR = "results"

@request_status_router.post("/request_image_id/")
async def request_status(request: Request):
    try:
        body = await request.json()
        image_id = body.get("image_id")

        if not image_id:
            raise HTTPException(status_code=400, detail="Missing 'image_id' in request")

        image_dir = os.path.join(RESULT_DIR, image_id)
        status_path = os.path.join(image_dir, "status.json")

        if not os.path.exists(image_dir):
            raise HTTPException(status_code=404, detail=f"Image ID '{image_id}' does not exist")

        if not os.path.exists(status_path):
            raise HTTPException(status_code=404, detail=f"Status file for image_id '{image_id}' not found")

        with open(status_path, "r") as f:
            status_data = json.load(f)

        return JSONResponse(content=status_data)

    except HTTPException as http_err:
        raise http_err  # giữ nguyên các lỗi HTTP đã tạo

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
