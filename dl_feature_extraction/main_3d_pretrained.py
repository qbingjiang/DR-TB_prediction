import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn
from build_model import generate_model
from make_dataloader import make_dataloader
import os

def find_all_nii(path_image):
    folder_1_list = []
    for subdir, dirs, files in os.walk(path_image):
        for file in files:
            #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file
            if filepath.endswith('.nii.gz'): 
                folder_1_list.append(filepath)
    folder_1_list = sorted(folder_1_list) 
    return folder_1_list

######################################################step 1:: data input######################################################

path_image = r'/results_data_nii'
path_mask = r'/results'
df_image_info={}
image_files = find_all_nii(path_image)
mask_files  = find_all_nii(path_mask)

image_files_ = image_files
mask_files_ = mask_files

image_mask_files_times = []
for i in range(len(image_files_)): 
    a = [image_files_[i], mask_files_[i]] 
    image_mask_files_times.append(a)

label_DrugRest         = None

if __name__ == '__main__': 

    model = generate_model(backbone='resnet3d')

    ifCUDA = True 
    if torch.cuda.is_available() and ifCUDA: 
        model.cuda() 
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor
    testData_list = [image_mask_files_times, label_DrugRest]
                    
    testloader = make_dataloader(testData_list[0:1], testData_list[1:2], 
                                bs=2, ifshuffle=False, iftransform=False, 
                                ifbatchSampler=False, ifrandomCrop=False )
    
    model.eval()
    Tensor = Tensor
    x_feats_list = []
    with torch.no_grad():
        for data in testloader:
            x, y = data
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