import math
import os
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from skimage.measure import label, regionprops

analyze_router = APIRouter()
executor = ThreadPoolExecutor(max_workers=4)

class AnalyzeRequest(BaseModel):
    image_id: str
    cell_id: int

def extract_cell_mask(image_id: str, cell_id: int, margin: int = 3) -> np.ndarray:
    mask_path = f"results/{image_id}/mask.png"
    if not os.path.exists(mask_path):
        raise HTTPException(status_code=404, detail=f"Mask not found for image_id {image_id}")
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    labeled = label(mask)
    props = regionprops(labeled)

    if cell_id < 1 or cell_id > len(props):
        raise HTTPException(status_code=400, detail=f"Invalid cell_id {cell_id} for image {image_id}")

    region = props[cell_id - 1]
    minr, minc, maxr, maxc = region.bbox
    minr = max(minr - margin, 0)
    minc = max(minc - margin, 0)
    maxr = min(maxr + margin, mask.shape[0])
    maxc = min(maxc + margin, mask.shape[1])

    cell_mask = (labeled[minr:maxr, minc:maxc] == region.label).astype(np.uint8) * 255
    return cell_mask

def compute_from_mask(mask: np.ndarray, cell_id: int) -> dict:
    try:
        with open("thangdo.txt") as f:
            px2um = 10.0 / float(f.read().strip())
    except Exception:
        px2um = 1.0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise HTTPException(status_code=404, detail="No contour found in the mask.")

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt) * px2um**2
    perimeter = cv2.arcLength(cnt, True) * px2um
    circularity = 4 * math.pi * area / (perimeter**2) if perimeter else 0

    hull = cv2.convexHull(cnt)
    convexity = (cv2.arcLength(hull, True) / perimeter) if perimeter else 0

    CE_diameter = math.sqrt(4 * area / math.pi)

    try:
        ellipse = cv2.fitEllipse(cnt)
        major_axis_length = ellipse[1][1]
        minor_axis_length = ellipse[1][0]
        aspect_ratio = minor_axis_length / major_axis_length if major_axis_length else 0
    except:
        major_axis_length = minor_axis_length = aspect_ratio = 0

    pts = cnt.reshape(-1, 2)
    max_distance = max(
        np.linalg.norm(pts[i] - pts[j]) * px2um
        for i in range(len(pts)) for j in range(i+1, len(pts))
    )

    contour_points = [{"x": int(p[0]), "y": int(p[1])} for p in pts]

    return {
        "cell_id": cell_id,
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
    }

@analyze_router.post("/compute_features/")
def analyze_mask(request: AnalyzeRequest):
    cell_mask = extract_cell_mask(request.image_id, request.cell_id)
    cell_info = compute_from_mask(cell_mask, request.cell_id)
    return {"cell_info": cell_info}