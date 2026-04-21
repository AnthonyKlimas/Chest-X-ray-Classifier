import cv2
import numpy as np

from dataset import (
    ROTATION_DEGREES, ROTATION_PROB, AFFINE_TRANSLATION, CLAHE_PROB,
    CLIP_LIMIT, IMAGENET_MEAN, IMAGENET_STD, JITTER_BRIGHTNESS,
    JITTER_CONTRAST, NIH_CXR8_CUSTOM_MEAN, NIH_CXR8_CUSTOM_STD,
    TILE_GRID_SIZE
)

from train import (
    BASE_LR, BATCH_SIZE_TRAIN, BATCH_SIZE_VAL,
    CLASSIFIER_DROPOUT, FEATURE_DROPOUT, HARDWARE_PITY, IMAGE_ROOT,
    LR_LAYER_DECAY, METADATA_CSV_PATH, MODEL_OUTPUT_FILE,
    NUM_EPOCHS, PATIENCE, PERSISTENT_WORKERS, PREFETECH_FACTOR,
    SAMPLER_POWER, SSL_CKPT, LOADER_WORKERS_TRAIN,
    UNFREEZE_SCHEDULE, LOADER_WORKERS_VALUE, WARMUP_END_FACTOR,
    WARMUP_EPOCHS, WARMUP_START_FACTOR, WEIGHT_DECAY
)



"""
Normalize orientation of a CXR image using:
- EXIF orientation (if present)
- Pixel-based heuristics (heart-side, diaphragm, ribs)
- Lateral detection (remove laterals)
Input: PIL Image (grayscale or RGB)
Output: PIL Image or None (if lateral)
"""
def normalize_cxr_image(img):
    # PIL operations first
    if img.mode != "L":
        img = img.convert("L")

    # Fix EXIF orientation (best-effort, rare on CXRs)
    try:
        exif = img._getexif()
        if exif is not None:
            orientation_key = next(k for k, v in ExifTags.TAGS.items() if v == "Orientation")
            orientation = exif.get(orientation_key)
            rotations = {3: 180, 6: 270, 8: 90}
            if orientation in rotations:
                img = img.rotate(rotations[orientation], expand=True)
    except Exception:
        pass

    # Convert to numpy once
    arr = np.array(img)
    h, w = arr.shape

    # Lateral detection
    left_edge  = arr[:, :int(w * 0.15)].mean()
    right_edge = arr[:, int(w * 0.85):].mean()
    if max(left_edge, right_edge) > arr.mean() * 1.35:
        return None  # handle None in __getitem__

    # Upside-down detection (diaphragm should be brighter at bottom)
    if arr[:h//3].mean() > arr[-h//3:].mean() * 1.10:
        arr = np.flipud(arr)

    # Rib gradient check (after upright correction)
    arr_f = arr.astype(np.float32)
    sobel_y    = cv2.Sobel(arr_f, cv2.CV_32F, 0, 1, ksize=5)
    top_grad   = np.abs(sobel_y[:h//2]).mean()
    bottom_grad = np.abs(sobel_y[h//2:]).mean()
    if top_grad < bottom_grad * 0.85:
        arr = np.flipud(arr)

    return arr.astype(np.uint8)
    # return Image.fromarray(arr.astype(np.uint8))

def fix_rotation(img):
    w, h = img.size
    if w > h:
        img = img.rotate(90, expand=True)
    return img


