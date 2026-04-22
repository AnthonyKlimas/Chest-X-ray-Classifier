# Chest X-ray Multi-Classifier

### Project Description

In this project, we attempt to tune a preprocessing pipeline, a SimMIM SwinV2 backbone, and a MLP head to push AUC to 84%



### Results
[Model 1](old_architecture/swin.ipynb)  ->  ~83%

[Model 2](old_architecture/swin_clahe.ipynb) -> 84%

The following resulting AUCs discard Hernia representation in the latest model because it is difficult to measure with minimal error.
The figures above are closer to 81% and 82% for comparison purposes because of the calculation error, but the folowing ones are more accurate:

[Model 3](train_save.py) -> 81.5%


### Model

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

Standard Deviation: ```5429.592```


### Authors:
Nicholas Calabro

Anthony Klimas

Luke MacVicar

Hilary Jaen Rodriguez

Instructed by Professor Wenjin Zhou

__Final Project for a Computer Science Special Topics Elective:__ _Computing for Health and Medicine_
