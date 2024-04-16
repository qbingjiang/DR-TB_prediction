import csv
import json
# import nrrd
import numpy as np
import os
import pandas as pd
from radiomics import featureextractor
import SimpleITK as sitk

import scipy

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

# File which contains the extraction parameters for Pyradiomics
params_file  = 'source/feature_extraction_parameters_example.yaml'

# # File where the extracted pyradiomics features will be stored
label=3


path_image = r'/results_data_nii'
path_mask = r'/results'

df_image_info={}
df_image_info['image'] = find_all_nii(path_image)
df_image_info['mask'] = find_all_nii(path_mask)



def read_nii(path):   
    input_image = sitk.ReadImage(path)
    inimg_raw = sitk.GetArrayFromImage(input_image)
    directions = np.asarray(input_image.GetDirection())
    if len(directions) == 9:
        inimg_raw = np.flip(inimg_raw, np.where(directions[[0,4,8]][::-1]<0)[0]) 
    return inimg_raw    

def save_results_to_pandas(data, save_path=None):

    df = pd.DataFrame(data=data)
    if save_path != None:
        df.to_csv(save_path, index=False)
    return df


def run(imageFilepath, maskFilepath, params_file, label=1): 

    imageData = sitk.ReadImage(imageFilepath)
    imageData_np = sitk.GetArrayFromImage(imageData)
    imageData_np[imageData_np < -1024] = -1024
    imageData_np[imageData_np > 300] = 300
    imageData_update = sitk.GetImageFromArray(imageData_np)
    imageData_update.CopyInformation(imageData) 

    maskData = sitk.ReadImage(maskFilepath) 
    maskData_np = sitk.GetArrayFromImage(maskData) 
    if label==3:   ## 3 means both left and right lung
        maskData_np = (maskData_np>0.5)*1
    elif label==1:   ## 1 means both left lung
        maskData_np = (maskData_np==1)*1 
    elif label==2:   ## 2 means both right lung
        maskData_np = (maskData_np==2)*1 

    ifKeepUpperLung = True
    if ifKeepUpperLung: 
        maskData_np = (maskData_np>0.5)*1 
        maskData_np = scipy.ndimage.morphology.binary_erosion(maskData_np, structure=np.ones((3,3,3)) ).astype(maskData_np.dtype) 
        # a = np.where(y_1_t>0) 
        # z1, z2, x1, x2, y1, y2 = [np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1]), np.min(a[2]), np.max(a[2])] 
        maskData_np[:-maskData_np.shape[0]*2//3, :, :] = 0
    ##show the image
    # imageData_np_ = (imageData_np+1024)/1324
    # import skimage
    # for i_img in range(imageData_np_.shape[0]): 
    #     skimage.io.imsave('img_show_{}.bmp'.format(i_img), skimage.img_as_ubyte(imageData_np_[i_img]) )

    maskData_update = sitk.GetImageFromArray(maskData_np)
    maskData_update.CopyInformation(imageData)     
    # sitk.WriteImage(result_out, args.output)
    
    # # Initialize feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
    try: 
        print('Extract features...')
        # print(idx, ': ', imageFilepath)
        # featureVector = extractor.execute(imageFilepath, maskFilepath)    ##important: need to check the mask. label=None means label=1 
        featureVector = extractor.execute(imageData_update, maskData_update, label=1)     ##important: need to check the mask. label=None means label=1
        print(imageFilepath.split('/')[-1])
        # featureVector.update({'image_path': imageFilepath, 'mask_path': maskFilepath })  ### to check if in order
        return featureVector

    except Exception:
        print('EXCEPTION: Feature extraction failed!')
        print(imageFilepath)
        print(maskFilepath)
        return None

# Run main and extract pyradiomics features
if __name__ == '__main__':
    n_workers = 15
    image_paths = df_image_info['image']
    mask_paths = df_image_info['mask']

    featureVectors = []
    for i in range(len(image_paths)): 
        featureVectors.append(run(image_paths[i], mask_paths[i], params_file, label) )

    # featureVectors = run(image_paths[0], mask_paths[0], params_file, label )

    output_path  = 'extracted_radiomics_features_first_2outof3/extracted_example_features_train_lung_whole.csv'

    # Overwrite feature file if it already exists?
    overwrite = True

    # Remove previously calculated feautures
    if os.path.exists(output_path) and overwrite:
            print('Remove {}'.format(output_path))
            os.remove(output_path)

    # save_results_csv_path = f'{save_dir}/{patient_id}_40x_patch_nuclear_feat.csv'
    save_results_csv_path = output_path 
    data = {}
    data['patient_image'] = []
    data['patient_mask'] = []
    for i in range(len(featureVectors)):
        if featureVectors[i] is None:
            continue
        data['patient_image'].append(df_image_info['image'][i])
        data['patient_mask'].append(df_image_info['mask'][i])
        for feature_name, feature_value in featureVectors[i].items():
            if data.get(feature_name) is None:
                data[feature_name] = [feature_value]
            else:
                data[feature_name].append(feature_value)

    df = save_results_to_pandas(data, save_results_csv_path)
