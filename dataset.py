# cxr_data.py
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

# Constants
CLIP_LIMIT = 1.5
TILE_GRID_SIZE = 4
CLAHE_PROB = 1.0 #making consistant for now
AFFINE_DEGREES = 1.2
AFFINE_TRANSLATION = 0.0 # causes instability
AFFINE_PROB = 0.3

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

ALL_CLASSES = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema",
    "Effusion","Emphysema","Fibrosis","Hernia",
    "Infiltration","Mass","No Finding","Nodule",
    "Pleural_Thickening","Pneumonia","Pneumothorax",
]


# Removes global brightness variation
class PerImageStandardize(object):
    def __call__(self, x):
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + 1e-6)



def normalize_cxr_image(img):
    """
    Normalize orientation of a CXR image using:
    - EXIF orientation (if present)
    - Pixel-based heuristics (heart-side, diaphragm, ribs)
    - Lateral detection (remove laterals)
    Input: PIL Image (grayscale or RGB)
    Output: PIL Image or None (if lateral)
    """


    # Ensure grayscale
    if img.mode != "L":
        img = img.convert("L")

    # -----------------------------
    # 1. Fix EXIF orientation
    # -----------------------------
    try:
        exif = img._getexif()
        if exif is not None:
            for k, v in ExifTags.TAGS.items():
                if v == "Orientation":
                    orientation_key = k
                    break

            orientation = exif.get(orientation_key, None)

            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
    except:
        pass

    # Convert to numpy
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape

    # -----------------------------
    # 2. Lateral detection (pixel-based)
    # -----------------------------
    # Laterals have bright shoulder/arm on one side
    left_edge = arr[:, :int(w*0.15)].mean()
    right_edge = arr[:, int(w*0.85):].mean()
    global_mean = arr.mean()

    if max(left_edge, right_edge) > global_mean * 1.35:
        return None  # lateral → drop

    # -----------------------------
    # 3. Heart-side heuristic
    # -----------------------------
    left_mean = arr[:, :w//2].mean()
    right_mean = arr[:, w//2:].mean()

    # Heart is on the left → left side darker
    if right_mean < left_mean * 0.92:
        arr = np.fliplr(arr)

    # -----------------------------
    # 4. Upside-down detection (diaphragm brightness)
    # -----------------------------
    top_brightness = arr[:h//3].mean()
    bottom_brightness = arr[-h//3:].mean()

    # Diaphragm is brighter → bottom should be brighter
    if top_brightness > bottom_brightness * 1.10:
        arr = np.flipud(arr)

    # -----------------------------
    # 5. Rib orientation heuristic (vertical gradient)
    # -----------------------------
    sobel_y = cv2.Sobel(arr, cv2.CV_64F, 0, 1, ksize=5)
    top_grad = np.abs(sobel_y[:h//2]).mean()
    bottom_grad = np.abs(sobel_y[h//2:]).mean()

    # If ribs appear upside-down
    if top_grad < bottom_grad * 0.85:
        arr = np.flipud(arr)

    # Return final PIL image
    return Image.fromarray(arr.astype(np.uint8))


# Utility
def fix_rotation(img):
    w, h = img.size
    if w > h:
        img = img.rotate(90, expand=True)
    return img

# CLAHE Transform
class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )
        img_np = np.array(img)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced)

# Training Transform
def make_train_tf(size):
    # ops = [
    #     transforms.Resize((size, size)),
    #     transforms.RandomApply(
    #         [CLAHETransform(clip_limit=CLIP_LIMIT,
    #                         tile_grid_size=(TILE_GRID_SIZE, TILE_GRID_SIZE))],
    #         p=CLAHE_PROB,
    #     ),
    #     transforms.RandomApply(
    #         [transforms.RandomAffine(
    #             AFFINE_DEGREES,
    #             translate=(AFFINE_TRANSLATION, AFFINE_TRANSLATION)
    #         )],
    #         p=AFFINE_PROB,
    #     ),
    #     transforms.ToTensor(),
    #     transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    # ]

    ops = [
    transforms.Resize((size, size)),
    transforms.RandomApply(
        [CLAHETransform(clip_limit=CLIP_LIMIT,
                        tile_grid_size=(TILE_GRID_SIZE, TILE_GRID_SIZE))],
        p=CLAHE_PROB,
    ),
    transforms.RandomApply(
        [transforms.RandomAffine(
            AFFINE_DEGREES,
            translate=None  # remove translation
        )],
        p=AFFINE_PROB,
    ),
    transforms.ToTensor(),
    PerImageStandardize(),
]

    return transforms.Compose(ops)

# Dataset
class CXR8Dataset(Dataset):
    def __init__(self, df, labels, idx_array, transform, lookup):
        self.df        = df.iloc[idx_array].reset_index(drop=True)
        self.labels    = labels[idx_array]
        self.transform = transform
        self.lookup    = lookup

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        while True:
            fname = self.df.loc[i, "Image Index"]
            img = Image.open(self.lookup[fname]).convert("L")
            img = normalize_cxr_image(img)
            if img is None:
                i = (i + 1) % len(self.df)
                continue

            # now stack to 3-channel
            arr = np.array(img, dtype=np.uint8)
            arr = np.stack([arr, arr, arr], axis=-1)
            img = Image.fromarray(arr)

            img = self.transform(img)

            lbl = torch.tensor(self.labels[i], dtype=torch.float32)
            view_id = torch.tensor(self.df.loc[i, "view_id"], dtype=torch.long)
            return img, lbl, view_id



# Worker Init
def worker_init_fn(worker_id):
    cv2.setNumThreads(0)
