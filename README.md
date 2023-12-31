# DR-TB_prediction

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

We modified the code from the pre-trained [model](https://github.com/JoHof/lungmask) [1] for convenience.

## Deep learning feature extraction

3D Residual network (3D Resnet) [2] with a pre-trained [weight ](https://pytorch.org/vision/main/models/generated/torchvision.models.video.r3d_18.html) was employed to extract the deep learning features from the lung region. Finally, a total of 512 features were extracted from each lung region.&#x20;

## Hand-crafted feature extraction
[PyRadiomics](http://PyRadiomics.readthedocs.io/en/latest/) package was applied to extract handcrafted features from the regions of interest (ROIs) (from the lung region). A total of 1906 quantitative features were calculated from each lung ROI, encompassing four categories: (1) 14 shape-based 3D features, (2) 18 first-order statistics features, (3) 68 texture-based features, and (4) 1806 transformation features (original, LoG, Wavelet, Square, SquareRoot, Logarithm, Exponential, Gradient, LBP3D). The PyRadiomics documentation provides detailed information about these features. 

## Model building (including dimension reduction)

To identify robust features across VOIs defined by different annotators, a consistency analysis was performed for radiomics features. Only features with a Pearson correlation coefficient > 0.9 with other features were excluded from feature selection. Feature selection was performed separately for DL and HC features. Least Absolute Shrinkage and Selection Operator (LASSO), with penalty parameter tuning conducted by 10-fold cross-validation, was used for feature selection. To determine the importance of features, the LASSO regression was iterated 10 times, with votes being accumulated. Features that received more than 20 votes were considered as the most valuable feature set for predicting MDR-TB. 
Subsequently, two unimodal models (DL and HC) for predicting MDR-TB were calculated with a linear combination of the final selection of features and multiplied by their normalized coefficients using the multivariable logistic regression model. [HC_score](https://latex.codecogs.com/png.image?\dpi{110}HC\_score=0.28566381*log\mbox{-}sigma\mbox{-}3\mbox{-}0\mbox{-}mm\mbox{-}3D\_firstorder\_Maximum&plus;0.27731318*log\mbox{-}sigma\mbox{-}3\mbox{-}0\mbox{-}mm\mbox{-}3D\_glszm\_ZoneEntropy-0.29851531*wavelet\mbox{-}HHH\_firstorder\_Minimum-0.17285832*wavelet\mbox{-}LLL\_firstorder\_Minimum-0.41540002*square\_gldm\_SmallDependenceHighGrayLevelEmphasis-0.1525067*lbp\mbox{-}3D\mbox{-}m1\_firstorder\_90Percentile&plus;0.07975887*lbp\mbox{-}3D\mbox{-}k\_firstorder\_Skewness-0.68838761&space;) and [DL_score](https://latex.codecogs.com/png.image?\dpi{110}DL\_score=0.15431351*DLfeat\_23&plus;0.37772542*DLfeat\_113-0.32864266*DLfeat_114-0.16367939*DLfeat\_181&plus;0.17699039*DLfeat\_317-0.0797936*DLfeat\_361&plus;0.29851433*DLfeat\_383-0.31098062*DLfeat\_479-0.70178702) were calculated by their Logistic regression. The coefficients are showed in Table 1 and 2. 
Finally, the DLHC logistic model was built based on DL score and HC score for predicting MDR-TB, in which the scores were computed via a weighted linear combination of the selected texture features and their corresponding coefficients (Table 3). 


<div align="center">

Table 1: The coefficient of the DL model.
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

</div>


<div align="center">

Table 2: The coefficient of the HC model.
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

</div>

<div align="center">

Table 3: The coefficient of the DLHC model.
| **Feature**                                      | **Coefficient** |
| :-------- | :---------- |
| DL_score | 0.70839935 |
| HC_score | 0.67204375 |
| Intercept | 0.23975794 |

</div>


## Visualization
This serves as a visualization tool for DL, HC and DLHC features. For the HC features, Pyradiomics was employed for feature extraction, and additional details regarding parameter settings can be found at https://pyradiomics.readthedocs.io/. For the DL features, the intermediate feature maps in the last layer of the 3D ResNet from the Axial plane are showed, which illustrates that the DL model actually learned the structural representation of TB-related region. The intermediate results demonstrate the feasibility of extracting image features by the proposed framework. 


<div align="center">

![](https://github.com/qbingjiang/DR-TB_prediction/blob/main/visualization/feature%20mapping.png)
Figure 1: . Insight of each feature of two MDR-TB and two DS-TB cases in axial plane. The numbers following the letters correspond to case 1 and case 2, respectively.
(a) The CT image. 
(b)-(h) feature maps of log-sigma-3-0-mm-3D_firstorder_Maximum, log-sigma-3-0-mm-3D_glszm_ZoneEntropy, wavelet-HHH_firstorder_Minimum, wavelet-LLL_firstorder_Minimum, square_gldm_SmallDependenceHighGrayLevelEmphasis, lbp-3D-m1_firstorder_90Percentile, and lbp-3D-k_firstorder_Skewness, respectively. 
(i) logistic combination of HC feature maps. 
(j-q) feature map of feature map of DLfeat_23, DLfeat_113, DLfeat_114, DLfeat_181, DLfeat_317, DLfeat_361, DLfeat_383, and DLfeat_479, respectively. 
(r) logistic combination of DL feature maps.
NOTE: DLHC Suspicious area is a logistic combination of the one from HC and DL feature maps. 
  
</div>


## Reference

- [1] Hofmanninger J, Prayer F, Pan J, Röhrich S, Prosch H, Langs G. Automatic lung segmentation in routine imaging is primarily a data diversity problem, not a methodology problem. Eur Radiol Exp 2020; 4: 50.
- [2] Tran D, Wang H, Torresani L, Ray J, Lecun Y, Paluri M. A Closer Look at Spatiotemporal Convolutions for Action Recognition. Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition 2018; 6450–6459.


###
