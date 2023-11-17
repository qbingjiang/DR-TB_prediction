import glob
import pydicom as dicom
import os
import re
import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn
import re
from src.get_data import * 
import argparse
from src.build_model_v2_3d_pretrained import generate_model
from src.train_model_v2_3d_pretrained import run_train, train, test


######################################################step 1:: data input######################################################
path=r'/home/bj/Documents/code_workspace/38 drug-resistant TB/data preparation'
##KASHI FeiKe hospital
feature_file_training_set1 = path + r'/path_DRstatus_DRtime_485_label_first_time.csv'
feature_file_training_set2 = path + r'/path_DRstatus_DRtime_104_label_first_time.csv'
##KASHI YeCheng hospital
feature_file_training_set3 = path + r'/path_DRstatus_DRtime_154_label_first_time.csv'
###KASHI Shufu hospital
feature_file_training_set4 = path + r'/path_DRstatus_DRtime_305_label_first_time.csv'
###TBPortal public dataset
feature_file_training_set5 = path + r'/path_DRstatus_TBPortals_190_label_first_time.csv' 

df_features1 = pd.read_csv(feature_file_training_set1 )
df_features2 = pd.read_csv(feature_file_training_set2 ) 
df_features3 = pd.read_csv(feature_file_training_set3 )  ##
df_features4 = pd.read_csv(feature_file_training_set4 )  ##  shufu
df_features5 = pd.read_csv(feature_file_training_set5 )  ##   TBPortals


df_features_D1 = pd.concat([df_features1, df_features2])
df_features_D1 = df_features_D1.reset_index(drop=True)

df_features_D2 = pd.concat([df_features3, df_features4])
# df_features_D3 = pd.concat([, ])
df_features_D4 = pd.concat([df_features5, ])

df_features_total = df_features_D1

image_files = df_features_total['patient_image'].to_list() 
mask_files  = df_features_total['patient_mask'].to_list() 

image_files_ = image_files
mask_files_ = mask_files

image_mask_files_times = []
for i in range(len(image_files_)): 
    a = [image_files_[i], mask_files_[i]] 
    image_mask_files_times.append(a)

label_DrugRest         = df_features_total['label'].tolist() 
# label_DrugRest_ptLever = [ label_DrugRest[i] for i in inds_start ] 
label_DrugRest_ptLever = label_DrugRest


######################################################step 1:: data input-- for dataloader ######################################################

from sklearn.model_selection import StratifiedKFold
def stratified_k_fold(X, y, n_splits=4, shuffle=False, random_state=None): 

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state) 
    skf.get_n_splits(X, y)
    train_ind_list, test_ind_list = [], []
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        train_ind_list.append(train_index) 
        test_ind_list.append(test_index) 
    train_X_list, test_X_list = [], [] 
    train_y_list, test_y_list = [], [] 
    for i in range(len(train_ind_list)): 
        train_X_list.append([X[i_ind] for i_ind in train_ind_list[i] ] ) 
        test_X_list.append([X[i_ind] for i_ind in test_ind_list[i] ] ) 

        train_y_list.append([y[i_ind] for i_ind in train_ind_list[i] ] ) 
        test_y_list.append([y[i_ind] for i_ind in test_ind_list[i] ] ) 
    
    return train_X_list, train_y_list, test_X_list, test_y_list, train_ind_list, test_ind_list 


if __name__ == '__main__': 

    image_mask_files_times_D1 = image_mask_files_times 
    label_DrugRest_ptLever_D1 = label_DrugRest_ptLever 


    model = generate_model(backbone='resnet3d')

    ifCUDA = True 
    if torch.cuda.is_available() and ifCUDA: 
        model.cuda() 
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor
    testData_list = [image_mask_files_times_D1, label_DrugRest_ptLever_D1]
                    
    from src.make_dataloader import make_dataloader
    testloader = make_dataloader(testData_list[0:1], testData_list[1:2], 
                                bs=2, ifshuffle=False, iftransform=False, 
                                ifbatchSampler=False, ifrandomCrop=False )
    
    model.eval()
    Tensor = Tensor
    x_feats_list = []
    with torch.no_grad():
        for data in testloader:
            x, y, labels = data
            labels = labels.unsqueeze_(1)
            labels = Variable(labels.type(Tensor))
            x_t0 = x[0]
            y_t0 = y[0] 
            x_1_patch_list = [ (x_t0*y_t0).type(Tensor), ]
            x_1_patch_list = [torch.cat([x_1_patch_list[0] for _ in range(3)], dim=1) ]
            x_feats = model(*x_1_patch_list) 
            x_feats = x_feats.cpu().numpy()
            x_feats = np.squeeze(x_feats, axis=(2,3,4)) 
            x_feats_list.append(x_feats)

    ###save the deep learning features
    x_feats_list = np.concatenate(x_feats_list, axis=0 ) 
    df = pd.DataFrame(x_feats_list, columns=['DLfeat_{}'.format(i_feat) for i_feat in range(x_feats_list.shape[1])])
    df.to_csv('DLfeats_{}.csv'.format(x_feats_list.shape[0]), index=False) 
    

print()