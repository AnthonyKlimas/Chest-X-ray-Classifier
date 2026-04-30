# Chest X-ray Multi-Classifier

### Project Description

This project focuses on building a deep learning model for multi-label classification of chest X-ray images using the NIH dataset.

The task is challenging because each image can contain multiple diseases, and the dataset is imbalanced.

To handle this, we chose different approaches, including preprocessing techniques, transformer-based architectures, and advanced training strategies, in order to improve performance and understand what actually works best. Key Approaches:

Image preprocessing techniques such as CLAHE
Transformer-based architectures (Swin Transformer and SwinV2)
Training strategies such as asymmetric loss, unfreeze schedules, and self-supervised pretraining (SimMIM)
Our goal was to improve AUC performance, focusing on our most advanced model, Model 3, which uses a SimMIM pretrained SwinV2 backbone and to push its performance above 81 percent.


### Install

Use Python 3.11
```python --version```

```pip install -r requirements.txt```


Beyond Python there are two other dependencies:

  [NIH Chest X-ray dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC) (42 GB)
  
  [Microsoft SimMIM Swin small checkpoint](https://huggingface.co/zdaxie/SimMIM/blob/main/simmim_swinv2_pretrain_models/swinv2_small_1k_500k.pth) (.2 GB)


### Run

Use the 3.11 interpreter to run train.py. No arguments are needed.

```python train.py```

This will output a best checkpoint and log file.


### Results
[Model 1](old_architecture/swin.ipynb)  ->  ~83%

[Model 2](old_architecture/swin_clahe.ipynb) -> 84%

The following resulting AUCs discard Hernia representation in the latest model because it is difficult to measure with minimal error.
The figures above are closer to 81% and 82% for comparison purposes because of the calculation error, but the following ones are more accurate:

Model 3 -> 81.5%

| Model 4 Version | Validation AUC |
|---|---:|
| ImageNet baseline | 74.43% |
| 30 epochs on NIH | 80.80% |
| 100 epochs on NIH | 81.9866% |
| 100 epochs + thresholding | 82.01% |


### Dataset
The NIH Chest X-ray Dataset contains 112,120 images of various resolutions
It includes 14 disease catagories with a large imbalance:
```
No Finding           | ██████████████████████████████████████████████ 60361
Infiltration         | ████████████████ 19894
Effusion             | ████████████ 13317
Atelectasis          | ███████████ 11559
Nodule               | ██████ 6331
Mass                 | █████ 5782
Pneumothorax         | █████ 5302
Consolidation        | ████ 4667
Pleural Thickening   | ███ 3385
Cardiomegaly         | ██ 2776
Emphysema            | ██ 2516
Edema                | ██ 2303
Fibrosis             | █ 1686
Pneumonia            | █ 1431
Hernia               | ▏ 227
```
Mean: ```5798.3```

Standard Deviation: ```5429.6```


## Models

### Model 1: Baseline Swin Transformer

The first model is the baseline version of the project. It used the NIH Chest X-ray dataset with images resized and cropped to 224 × 224. The model architecture is a Swin Transformer Tiny, initialized with ImageNet1K V1 weights.


Preprocessing includes:
- Resize to 256 x 256 for training
- RandomHorizontalFlip
- RandomRotation of 10 degrees
- CenterCrop to 224 x 224
- ImageNet normalization

Validation preprocessing included:
- Resize to 224 x 224
- ImageNet normalization

Training setup:
- Model: Swin Transformer Tiny
- Weights: ImageNet1K V1
- Loss function: BCEWithLogitsLoss
- Optimizer: AdamW
- Learning rate: 1e-4
- Weight decay: 1e-2
- Scheduler: CosineAnnealingLR
- Batch size: 32
- Epochs: 10
- Train/validation split: 85% / 15%
- Training images: 95,302
- Validation images: 16,818

Result:
- Original reported AUC: 83.62%
- Patient-level comparison AUC: 81.3%

[Model 1](old_architecture/swin.ipynb) -> original ~83.62%, patient-level comparison ~81.3%

### Model 2: Swin Transformer + CLAHE

The second model used the same Swin Transformer Tiny architecture as the baseline, initialized with ImageNetV1 weights. The main change was the addition of CLAHE preprocessing, which stands for Contrast Limited Adaptive Histogram Equalization.

CLAHE was applied to each image before resizing and normalization. The image was converted from RGB to LAB color space, CLAHE was applied to the lightness channel, and the image was converted back to RGB. This helped improve local contrast in the chest X-ray images.

Preprocessing included:
- CLAHE with clip limit = 2.0 and tile grid size = 8 x 8
- Resize to 256 x 256 for training
- RandomHorizontalFlip
- RandomRotation of 10 degrees
- CenterCrop to 224 x 224
- ImageNet normalization

Training setup:
- Model: Swin Transformer Tiny
- Weights: ImageNet1K V1
- Loss function: BCEWithLogitsLoss
- Optimizer: AdamW
- Learning rate: 1e-4
- Weight decay: 1e-2
- Scheduler: CosineAnnealingLR
- Batch size: 32
- Epochs: 10
- Train/validation split: 85% / 15%

Result:
- Original reported mean validation AUC: 0.840
- Patient-level comparison AUC: 82.0%
- Best validation loss: 0.1708
- Highest per-class AUC: Emphysema = 0.932

Class validation AUC:
Per-class validation AUC:

| Class | AUC |
|---|---:|
| Infiltration | 0.723 |
| Pneumonia | 0.765 |
| Nodule | 0.778 |
| No Finding | 0.789 |
| Pleural Thickening | 0.808 |
| Consolidation | 0.818 |
| Atelectasis | 0.822 |
| Fibrosis | 0.827 |
| Hernia | 0.872 |
| Mass | 0.874 |
| Effusion | 0.889 |
| Cardiomegaly | 0.898 |
| Pneumothorax | 0.903 |
| Edema | 0.904 |
| Emphysema | 0.932 |
| Mean AUC | 0.840 |

[Model 2](old_architecture/swin_clahe.ipynb) -> original ~84.0%, patient-level comparison ~82.0%

### Model 3: SimMIM SwinV2 + MLP Head

The third model introduces a more advanced architecture using a SimMIM-pretrained SwinV2 Small backbone.

The backbone is initialized using a learning checkpoint trained with SimMIM, which allows the model to learn better feature representations before fine-tuning on the chest X-ray dataset.

Key architectural changes:
- Backbone: SwinV2 Small (SimMIM pretrained)
- Custom model wrapper with feature aggregation
- MLP classification head with dropout
- View position embedding (PA/AP) incorporated into the model

Training strategy:
- Unfreeze schedule applied to the backbone to stabilize early training
- Layers are gradually unfrozen across epochs instead of training all at once
- Warmup applied during early epochs to avoid unstable updates
- Layer-wise learning rate decay used across the network

Optimization setup:
- Loss function: Asymmetric Loss (designed for class imbalance)
- Optimizer: AdamW
- Base learning rate: 7e-5
- Head learning rate multiplier: 6x
- Weight decay: 1e-2
- EMA applied during training

Data handling improvements:
- Stratified split based on patient ID to avoid data leakage
- Weighted sampling to address class imbalance
- Minimum positive samples required for valid AUC calculation
- Improved evaluation for rare classes such as Hernia

Regularization and tuning:
- Feature dropout: 0.2
- Classifier dropout: 0.1
- Early stopping with patience
- Checkpointing during training

Training setup:
- Batch size: 16
- Epochs: up to 48
- GPU training (CUDA enabled)

Result:
- Validation AUC: ~0.815 (81.5%)
- Evaluated using patient-level split

Model 3 Class table for best epoch:
| Class | AUC |
|------|-----:|
| Hernia | 0.684 |
| Infiltration | 0.702 |
| Nodule | 0.731 |
| Pneumonia | 0.770 |
| Fibrosis | 0.766 |
| No Finding | 0.779 |
| Pleural Thickening | 0.795 |
| Consolidation | 0.793 |
| Atelectasis | 0.807 |
| Mass | 0.829 |
| Effusion | 0.867 |
| Cardiomegaly | 0.868 |
| Emphysema | 0.904 |
| Pneumothorax | 0.877 |
| Edema | 0.882 |
| **Mean AUC** | **0.8148** |

Per-class behavior:
- Strong performance on frequent classes (Edema, Pneumothorax, Emphysema)
- Lower performance on Lower classes (Hernia, Infiltration)
- Class imbalance remains a significant challenge

Model 3 -> 81.5%

### Model 4: Extended SwinV2 with Improved Preprocessing

Model 4 builds on Model 3, maintaining the same architecture and training framework while introducing some modifications improving data quality and generalization.

Key changes include:
- Increased input resolution to 256 × 256
- Use of a SimMIM-pretrained SwinV2 backbone trained on NIH chest X-ray images at 384 × 384 resolution
- Replacement of standard preprocessing with Albumentations
- Addition of elastic transformations for stronger data augmentation
- Additional evaluation using both NIH-trained and ImageNet pretrained weights

Model 4 results:

| Model 4 Version | Validation AUC |
|---|---:|
| ImageNet baseline | 74.43% |
| 30 epochs on NIH | 80.80% |
| 100 epochs on NIH | 81.9866% |
| 100 epochs + thresholding | 82.01% |

### Authors:
Nicholas Calabro

Anthony Klimas

Luke MacVicar

Hilary Jaen Rodriguez

Instructed by Professor Wenjin Zhou

__Final Project for a Computer Science Special Topics Elective:__ _Computing for Health and Medicine_
