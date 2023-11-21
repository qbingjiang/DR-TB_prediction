# DR-TB_prediction

The code will be coming soon!!

This is a Python implementation of the DLHC model. For the detailed architecture please refer to the original paper.

## Introduction&#x20;

A fully automated artificial intelligence framework (DLHC) was proposed and designed to extract comprehensive MDR-TB-related data from CT images for the purpose of predicting MDR-TB status in patients prior to initiating first-line treatment. DLHC utilizes the combined power of deep learning features and hand-crafted features for prediction of MDR-TB from CT data. 

## Environments and Requirements

### Requirements

- pandas

- pydicom

- pyradiomics

- scikit-image

- scikit-learn

- seaborn

- SimpleITK

- torch

- torchvision

### Installation

```bash
pip install -r requirements.txt
```

## Lung segmentation

We modified the code from the pre-trained [model](https://github.com/JoHof/lungmask) for convenience.

## Deep learning feature extraction

3D Residual network (3D Resnet) \[7] with a pre-trained [weight ](https://pytorch.org/vision/main/models/generated/torchvision.models.video.r3d_18.html) was employed to extract the deep learning features from the lung region. Finally, a total of 512 features were extracted from each lung region.&#x20;

## Hand-crafted feature extraction

## Model building (including dimension reduction)

## Visualization

## Reference

###
