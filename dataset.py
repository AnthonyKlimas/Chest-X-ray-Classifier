"""
Author: Nicholas J. Calabro
Computing for Health and Medicine
Group 5
Coyyright 2026
"""
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from PIL import Image, ExifTags


### Preprocessing Constants

HORIZONTAL_FLIP_PROB = 0.5

ROTATION_DEGREES = 2.2
ROTATION_PROB = 0.5

JITTER_BRIGHTNESS = 0.08
JITTER_CONTRAST   = 0.08

# CLAHE constants are only used in preprocess.py
# It is handled prior to training
# To enable, uncomment in make_*_tf
CLIP_LIMIT = 1.5
TILE_GRID_SIZE = 4
CLAHE_PROB = 1.0 # making consistant for now to increase stability


### Calculated Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

NIH_CXR8_CUSTOM_MEAN = [0.5249, 0.5249, 0.5249]
NIH_CXR8_CUSTOM_STD  = [0.2622, 0.2622, 0.2622]

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

# Unused in train.py; done in another process
class CLAHETransform:
    def __init__(self, clip_limit, tile_grid_size):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self._clahe = None

    def _get_clahe(self):
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=self.tile_grid_size
            )
        return self._clahe

    def __getstate__(self):
        # Called when pickling — drop the unpicklable cv2 object
        state = self.__dict__.copy()
        state['_clahe'] = None
        return state

    def __setstate__(self, state):
        # Called when unpickling in each worker — restore without cv2 object
        self.__dict__.update(state)

    def __call__(self, img):
        clahe = self._get_clahe()
        img_np = (img.numpy() * 255).astype("uint8")
        img_np = np.transpose(img_np, (1, 2, 0))

        out = []
        for c in range(img_np.shape[2]):
            out.append(clahe.apply(img_np[:, :, c]))

        out = np.stack(out, axis=2)
        out = np.transpose(out, (2, 0, 1))
        out = out.astype("float32") / 255.0
        return torch.from_numpy(out)

### Transformer Creation ###

# Value Transform
def make_value_tf(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        # CLAHETransform(clip_limit=CLIP_LIMIT, tile_grid_size=(TILE_GRID_SIZE, TILE_GRID_SIZE)),
        transforms.Normalize(NIH_CXR8_CUSTOM_MEAN, NIH_CXR8_CUSTOM_STD)
    ])

# Training Transform
def make_train_tf(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=HORIZONTAL_FLIP_PROB),
        transforms.RandomApply([
            transforms.RandomRotation(
                ROTATION_DEGREES,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0,
                expand=False
            )
        ], p=ROTATION_PROB),
        transforms.ColorJitter(brightness=JITTER_BRIGHTNESS, contrast=JITTER_CONTRAST),
        transforms.ToTensor(),
        # transforms.RandomApply(
        #     [CLAHETransform(clip_limit=CLIP_LIMIT,
        #                     tile_grid_size=(TILE_GRID_SIZE, TILE_GRID_SIZE))],
        #     p=CLAHE_PROB,
        # ),
        transforms.Normalize(NIH_CXR8_CUSTOM_MEAN, NIH_CXR8_CUSTOM_STD)
    ])

### Dataset ###
class CXR8Dataset(Dataset):
    def __init__(self, df, labels, idx_array, transform, lookup):
        self.df        = df.iloc[idx_array].reset_index(drop=True)
        self.labels    = labels[idx_array]
        self.transform = transform
        self.lookup    = lookup

    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        fname = self.df.loc[i, "Image Index"]
        path = self.lookup[fname]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        lbl = torch.tensor(self.labels[i], dtype=torch.float32)
        view_id = torch.tensor(self.df.loc[i, "view_id"], dtype=torch.long)
        return img, lbl, view_id


### Helper functions ###
def print_dataset_parameters():
    print("CLIP_LIMIT", CLIP_LIMIT)
    print("TILE_GRID_SIZE", TILE_GRID_SIZE)
    print("CLAHE_PROB", CLAHE_PROB)
    print("HORIZONTAL_FLIP_PROB", HORIZONTAL_FLIP_PROB)
    print("ROTATION_DEGREES", ROTATION_DEGREES)
    print("ROTATION_PROB", ROTATION_PROB)
    print("JITTER_BRIGHTNESS", JITTER_BRIGHTNESS)
    print("JITTER_CONTRAST", JITTER_CONTRAST)
    print("IMAGENET_MEAN", IMAGENET_MEAN)
    print("IMAGENET_STD", IMAGENET_STD)
    print("NIH_CXR8_CUSTOM_MEAN", NIH_CXR8_CUSTOM_MEAN)
    print("NIH_CXR8_CUSTOM_STD", NIH_CXR8_CUSTOM_STD)
    

# Worker Init; keep for memory safety
def worker_init_fn(worker_id):
    cv2.setNumThreads(0)

