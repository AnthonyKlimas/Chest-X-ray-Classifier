import os, glob
import numpy as np
from PIL import Image
from multiprocessing import Pool
import cv2
from dataset import CLIP_LIMIT, TILE_GRID_SIZE
from util import normalize_cxr_image

IMAGE_ROOT = "../chest_xray_dataset/CXR8/images"
CACHE_ROOT = "../chest_xray_dataset/CXR8/images_preprocessed"
os.makedirs(CACHE_ROOT, exist_ok=True)

clahe = cv2.createCLAHE(
    clipLimit=CLIP_LIMIT,
    tileGridSize=(TILE_GRID_SIZE, TILE_GRID_SIZE)
)

def process_one(src_path):
    rel_path = os.path.relpath(src_path, IMAGE_ROOT)
    dst_path = os.path.join(CACHE_ROOT, rel_path)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    if os.path.exists(dst_path):
        return "skip"
    try:
        img = Image.open(src_path)
        arr = normalize_cxr_image(img)
        if arr is None:
            return "lateral"
        arr = clahe.apply(arr)
        Image.fromarray(arr).save(dst_path)
        return "ok"
    except Exception as e:
        return f"error: {e}"

if __name__ == "__main__":
    all_png = glob.glob(os.path.join(IMAGE_ROOT, "**", "*.png"), recursive=True)
    print(f"Processing {len(all_png)} images...")

    with Pool(processes=8) as pool:
        results = pool.map(process_one, all_png)

    from collections import Counter
    print(Counter(results))