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
[PyRadiomics](http://PyRadiomics.readthedocs.io/en/latest/) package was applied to extract handcrafted features from the regions of interest (ROIs) (from the lung region). A total of 1906 quantitative features were calculated from each lung ROI, encompassing four categories: (1) 14 shape-based 3D features, (2) 18 first-order statistics features, (3) 68 texture-based features, and (4) 1806 transformation features (original, LoG, Wavelet, Square, SquareRoot, Logarithm, Exponential, Gradient, LBP3D). The PyRadiomics documentation provides detailed information about these features. 

## Model building (including dimension reduction)

To identify robust features across VOIs defined by different annotators, a consistency analysis was performed for radiomics features. Only features with a Pearson correlation coefficient > 0.9 with other features were excluded from feature selection. Feature selection was performed separately for DL and HC features. Least Absolute Shrinkage and Selection Operator (LASSO), with penalty parameter tuning conducted by 10-fold cross-validation, was used for feature selection. To determine the importance of features, the LASSO regression was iterated 10 times, with votes being accumulated. Features that received more than 20 votes were considered as the most valuable feature set for predicting MDR-TB. Subsequently, two unimodal models (DL and HC) for predicting MDR-TB were calculated with a linear combination of the final selection of features and multiplied by their normalized coefficients using the multivariable logistic regression model. Finally, the DLHC logistic model was built based on DL score and HC score for predicting MDR-TB, in which the scores were computed via a weighted linear combination of the selected texture features and their corresponding coefficients.

Finally, [HC_score](https://latex.codecogs.com/png.image?\dpi{110}HC\_score=0.28566381*log\mbox{-}sigma\mbox{-}3\mbox{-}0\mbox{-}mm\mbox{-}3D\_firstorder\_Maximum&plus;0.27731318*log\mbox{-}sigma\mbox{-}3\mbox{-}0\mbox{-}mm\mbox{-}3D\_glszm\_ZoneEntropy-0.29851531*wavelet\mbox{-}HHH\_firstorder\_Minimum-0.17285832*wavelet\mbox{-}LLL\_firstorder\_Minimum-0.41540002*square\_gldm\_SmallDependenceHighGrayLevelEmphasis-0.1525067*lbp\mbox{-}3D\mbox{-}m1\_firstorder\_90Percentile&plus;0.07975887*lbp\mbox{-}3D\mbox{-}k\_firstorder\_Skewness-0.68838761&space;) and [DL_score](https://latex.codecogs.com/png.image?\dpi{110}DL\_score=0.15431351*DLfeat\_23&plus;0.37772542*DLfeat\_113-0.32864266*DLfeat_114-0.16367939*DLfeat\_181&plus;0.17699039*DLfeat\_317-0.0797936*DLfeat\_361&plus;0.29851433*DLfeat\_383-0.31098062*DLfeat\_479-0.70178702) were calculated by the 

| **Feature** | **Coefficient** |
| :---------- | :-------------- |
| DLfeat_23   | 0.15431351      |
| DLfeat_113  | 0.37772542      |
| DLfeat_114  | -0.32864266     |
| DLfeat_181  | -0.16367939     |
| DLfeat_317  | 0.17699039      |
| DLfeat_361  | -0.0797936      |
| DLfeat_383  | 0.29851433      |
| DLfeat_479  | -0.31098062     |
| Intercept   | -0.70178702     |


| **Feature**                                      | **Coefficient** |
| :----------------------------------------------- | :-------------- |
| log-sigma-3-0-mm-3D_firstorder_Maximum           | 0.28566381      |
| log-sigma-3-0-mm-3D_glszm_ZoneEntropy            | 0.27731318      |
| wavelet-HHH_firstorder_Minimum                   | -0.29851531     |
| wavelet-LLL_firstorder_Minimum                   | -0.17285832     |
| square_gldm_SmallDependenceHighGrayLevelEmphasis | -0.41540002     |
| lbp-3D-m1_firstorder_90Percentile                | -0.1525067      |
| lbp-3D-k_firstorder_Skewness                     | 0.07975887      |
| Intercept                                        | -0.68838761     |

## Visualization



## Reference

- [lungmask]()

-

###
