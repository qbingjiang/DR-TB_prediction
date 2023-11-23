# DR-TB_prediction

The code will be coming soon!!

This is a Python implementation of the DLHC model. For the detailed architecture please refer to the original paper.

## Introduction&#x20;

A fully automated artificial intelligence framework (DLHC) was proposed and designed to extract comprehensive MDR-TB-related data from CT images for the purpose of predicting MDR-TB status in patients prior to initiating first-line treatment. DLHC utilizes combined power of deep learning features and hand-crafted features for prediction of MDR-TB from CT data. Diverging from traditional radiomic methods, which rely on lesion annotation, our proposed framework operates as an automatic pipeline, exclusively utilizing lung CT scans, eliminating the requirement for manual lesion annotation.&#x20;

## Environments and Requirements

#### Requirements

- Python >\= 3.6

- PyTorch >\= 1.6

- torchvision that matches the PyTorch installation.

- pandas

- pydicom >\= 1.0

- pyradiomics

- scikit-image

- scikit-learn

- seaborn

- SimpleITK

#### Installation

```Shell
pip install -r requirements.txt
```

## Lung segmentation

We modified the code from the pre-trained [model](https://github.com/JoHof/lungmask) for convenience.

## Deep learning feature extraction

3D Residual network (3D Resnet) with a pre-trained [weight ](https://pytorch.org/vision/main/models/generated/torchvision.models.video.r3d_18.html) was employed to extract the deep learning features from the lung region. Finally, a total of 512 features were extracted from each lung region.&#x20;

## Hand-crafted feature extraction

## Model building (including dimension reduction)

[11](https://latex.codecogs.com/png.image?\dpi{110}0.28566381*log\mbox{-}sigma\mbox{-}3\mbox{-}0\mbox{-}mm\mbox{-}3D\_firstorder\_Maximum&plus;0.27731318*log\mbox{-}sigma\mbox{-}3\mbox{-}0\mbox{-}mm\mbox{-}3D\_glszm\_ZoneEntropy-0.29851531*wavelet\mbox{-}HHH\_firstorder\_Minimum-0.17285832*wavelet\mbox{-}LLL\_firstorder\_Minimum-0.41540002*square\_gldm\_SmallDependenceHighGrayLevelEmphasis-0.1525067*lbp\mbox{-}3D\mbox{-}m1\_firstorder\_90Percentile&plus;0.07975887*lbp\mbox{-}3D\mbox{-}k\_firstorder\_Skewness-0.68838761&space;)

```math
0.28566381*log\mbox{-}sigma\mbox{-}3\mbox{-}0\mbox{-}mm\mbox{-}3D\_firstorder\_Maximum + 0.27731318*log\mbox{-}sigma\mbox{-}3\mbox{-}0\mbox{-}mm\mbox{-}3D\_glszm\_ZoneEntropy-0.29851531*wavelet\mbox{-}HHH\_firstorder\_Minimum-0.17285832*wavelet\mbox{-}LLL\_firstorder\_Minimum-0.41540002*square\_gldm\_SmallDependenceHighGrayLevelEmphasis-0.1525067*lbp\mbox{-}3D\mbox{-}m1\_firstorder\_90Percentile+0.07975887*lbp\mbox{-}3D\mbox{-}k\_firstorder\_Skewness-0.68838761
```

```math
0.15431351*DLfeat\_23 + 0.37772542 * DLfeat\_113 - 0.32864266 * DLfeat_114	- 0.16367939 * DLfeat_181 + 0.17699039 * DLfeat_317 - 0.0797936 * DLfeat_361 + 0.29851433 * DLfeat_383 -0.31098062 * DLfeat_479 - 0.70178702
```

| DLfeat_23  | 0.15431351  |
| :--------- | :---------- |
| DLfeat_113 | 0.37772542  |
| DLfeat_114 | -0.32864266 |
| DLfeat_181 | -0.16367939 |
| DLfeat_317 | 0.17699039  |
| DLfeat_361 | -0.0797936  |
| DLfeat_383 | 0.29851433  |
| DLfeat_479 | -0.31098062 |
| Intercept  | -0.70178702 |

## Visualization

## Reference

- [lungmask]()

-

###
