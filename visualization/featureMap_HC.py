# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd
from radiomics import featureextractor
import SimpleITK as sitk
import scipy
import six
import skimage


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
params_file  = './hc_feature_extraction/feature_extraction_parameters_example.yaml'
# # File where the extracted pyradiomics features will be stored
label=3

def read_nii(path):   ##bj
    input_image = sitk.ReadImage(path)
    inimg_raw = sitk.GetArrayFromImage(input_image)
    directions = np.asarray(input_image.GetDirection())
    if len(directions) == 9:
        inimg_raw = np.flip(inimg_raw, np.where(directions[[0,4,8]][::-1]<0)[0]) 
    return inimg_raw    ###bj ^1

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

def crop_image(imageData, maskData): 
    imageData_np = sitk.GetArrayFromImage(imageData)
    maskData_np = sitk.GetArrayFromImage(maskData)

    y_1_t = (maskData_np>0.5)*1 
    a = np.where(y_1_t>0) 
    z1, z2, x1, x2, y1, y2 = [np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1]), np.min(a[2]), np.max(a[2])] 

    x_1_patch = imageData_np[z1:z2+1, x1:x2+1, y1:y2+1] 
    y_1_patch = (maskData_np[z1:z2+1, x1:x2+1, y1:y2+1] >0.5)*1 

    imageData_crop_update = sitk.GetImageFromArray(x_1_patch)
    imageData_crop_update.SetSpacing(imageData.GetSpacing() )
    imageData_crop_update.SetOrigin(imageData.GetOrigin() )
    imageData_crop_update.SetDirection(imageData.GetDirection() )

    maskData_crop_update = sitk.GetImageFromArray(y_1_patch)
    maskData_crop_update.CopyInformation(imageData_crop_update) 

    ori_size = imageData_np.shape
    patch_size = x_1_patch.shape
    print('crop size, ', patch_size)
    return imageData_crop_update, maskData_crop_update, ori_size, [z1, z2+1, x1, x2+1, y1, y2+1]

def load_data(imageFilepath, maskFilepath, label=3): 


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
    maskData_np = scipy.ndimage.binary_erosion(maskData_np, structure=np.ones((3,3,3)) ).astype(maskData_np.dtype)   ##(7,7,7)
    ifKeepUpperLung = False
    if ifKeepUpperLung: 
        maskData_np = (maskData_np>0.5)*1 
        # maskData_np = scipy.ndimage.binary_erosion(maskData_np, structure=np.ones((3,3,3)) ).astype(maskData_np.dtype) 
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
    
    return imageData_update, maskData_update 

def run_featureMap(imageFilepath, maskFilepath, params_file, label=1, image_type=None, feature_class=None,feature_name=None, save_path='.' ): 
    imageData_update, maskData_update = load_data(imageFilepath, maskFilepath, label)
    imageData_crop_update, maskData_crop_update, ori_size, [z1, z2, x1, x2, y1, y2] = crop_image(imageData_update, maskData_update)

    # # Initialize feature extractor
    setting_dict = {
                    # 'label': 1, 
                    # 'interpolator': 'sitkBSpline', 
                    # 'resampledPixelSpacing': [1.0, 1.0, 3], 
                    # 'binWidth': 25.0,  
                    # 'voxelArrayShift': 1024, 
                    'voxelBatch': 10000
                    }
    extractor = featureextractor.RadiomicsFeatureExtractor(params_file, **setting_dict)
    # extractor = featureextractor.RadiomicsFeatureExtractor(**setting_dict)
    extractor.disableAllImageTypes()
    extractor.disableAllFeatures()

    if 'original' in image_type:
        extractor.enableImageTypeByName('Original')
    elif 'log' in image_type and 'logarithm' not in image_type:
        extractor.enableImageTypeByName('LoG')
        # sigma = int(image_type.split('-')[2])
        sigma = float(image_type.replace('log-sigma-', '').replace('-mm-3D','').replace('-','.') )
        if sigma is not None:
            extractor.settings['sigma'] = [sigma]
    elif 'wavelet' in image_type: 
        extractor.enableImageTypeByName('Wavelet')  # LLH, LHL, LHH are also available depending on your needs
    elif 'square' in image_type: 
        extractor.enableImageTypeByName('Square')
    elif 'squareRoot' in image_type: 
        extractor.enableImageTypeByName('SquareRoot')
    elif 'logarithm' in image_type: 
        extractor.enableImageTypeByName('Logarithm')
    elif 'exponential' in image_type: 
        extractor.enableImageTypeByName('Exponential')
    elif 'gradient' in image_type: 
        extractor.enableImageTypeByName('Gradient')
    elif 'lbp-3D' in image_type: 
        extractor.enableImageTypeByName('LBP3D')
    else:
        raise ValueError("Invalid image_type. Please choose from 'Original', 'LoG', 'Wavelet-LLL', 'Square', 'SquareRoot', 'Logarithm', 'Exponential', 'Gradient', or 'LBP3D'.")
    # extractor.enableImageTypeByName(image_type)
    extractor.enableFeaturesByName(**dict(zip([feature_class], [feature_name])))

    # try: 
    print('Extract features...')
    result = extractor.execute(imageData_crop_update, maskData_crop_update, voxelBased=True)     ##  important: need to check the mask. label=None means label=1
    # print(imageFilepath.split('/')[-1])
    # featureVector.update({'image_path': imageFilepath, 'mask_path': maskFilepath })  ### to check if in order
    # print(result)
    for key, val in six.iteritems(result):
        if isinstance(val, sitk.Image) and image_type in key:
            shape = (sitk.GetArrayFromImage(val)).shape
            # Feature map
            sitk.WriteImage(val, save_path + '/' + key + '.nii.gz', True)
            print()
        # return featureVector

    
def split_feature_name(feature_name):
    parts = feature_name.split('_')
    ##{'imageType': 'Original'}
    if len(parts) == 3:
        image_type = parts[0]
        feature_class = parts[1]
        feature_name = parts[2]
        return image_type, feature_class, [feature_name]
    else:
        raise ValueError("Invalid feature name format. It should be in 'image_type_feature_name' format.")

def run_featureMap_multi(image_path, mask_path, params_file, label, features_name):
    save_path = './featureVisulaization/' + image_path.split('/')[-1].rsplit('.', 2)[0] 
    if not os.path.exists( save_path): 
        os.makedirs(save_path) 

    if not isinstance(features_name, list): 
        if not os.path.exists( save_path + '/' + features_name+ '.nii.gz'): 
            image_type, feature_class, feat_name = split_feature_name(features_name)
            featureVectors = run_featureMap(image_path, mask_path, params_file, label, image_type, feature_class, feat_name, save_path) 
    else: 
        for i in range(len(features_name)): 
            if not os.path.exists( save_path + '/' + features_name[i] + '.nii.gz'): 
                image_type, feature_class, feat_name = split_feature_name(features_name[i])
                featureVectors = run_featureMap(image_path, mask_path, params_file, label, image_type, feature_class, feat_name, save_path)
    print('finished', image_path)

def Normalize01(data ):
    new_data = np.asarray(data, dtype=np.float32)
    new_data = new_data - np.min(new_data)
    new_data = new_data / np.max(new_data)
    return new_data 

def featsMap_weighted_with_coeffs(featsMap_sel_resize_list, coeffs):
    featsMap_by_rogistic_coeffs = np.zeros(featsMap_sel_resize_list[0].shape )
    for idx in range(len(coeffs)): 
        featsMap_by_rogistic_coeffs += featsMap_sel_resize_list[idx]*coeffs[idx]
    return featsMap_by_rogistic_coeffs

import matplotlib.pyplot as plt

def visualize_featsMap(img_lung_patch, featsMap_sel_resize_list, featsMap_by_rogistic_coeffs, save_path='./results_png'): 
    for i in range(len(img_lung_patch)): 
        plt.figure()  ##figsize=(8, 6)
        plt.subplot(181)
        output_dir = 'name' 
        plt.imshow(img_lung_patch[i], cmap='gray')
        plt.axis('off')

        plt.subplot(182) 
        plt.imshow(featsMap_sel_resize_list[0][i], cmap='jet')
        # cbar = plt.colorbar()
        # cbar.set_label('Title (Unit)')
        # fig.axes.get_xaxis().set_visible(False)
        # fig.axes.get_yaxis().set_visible(False)
        plt.axis('off')
        plt.tight_layout()
        # set the spacing between subplots
        plt.subplots_adjust(wspace=0.01, hspace=0)

        plt.subplot(183) 
        plt.imshow(featsMap_sel_resize_list[1][i], cmap='jet')
        plt.axis('off')
        plt.tight_layout()
        # set the spacing between subplots
        plt.subplots_adjust(wspace=0.01, hspace=0)

        plt.subplot(184) 
        plt.imshow(featsMap_sel_resize_list[2][i], cmap='jet')
        plt.axis('off')
        plt.tight_layout()
        # set the spacing between subplots
        plt.subplots_adjust(wspace=0.01, hspace=0)

        plt.subplot(185) 
        plt.imshow(featsMap_sel_resize_list[3][i], cmap='jet')
        plt.axis('off')
        plt.tight_layout()
        # set the spacing between subplots
        plt.subplots_adjust(wspace=0.01, hspace=0)

        plt.subplot(186) 
        plt.imshow(featsMap_sel_resize_list[4][i], cmap='jet')
        plt.axis('off')
        plt.tight_layout()
        # set the spacing between subplots
        plt.subplots_adjust(wspace=0.01, hspace=0)

        plt.subplot(187) 
        plt.imshow(featsMap_sel_resize_list[5][i], cmap='jet')
        plt.axis('off')
        plt.tight_layout()
        # set the spacing between subplots
        plt.subplots_adjust(wspace=0.01, hspace=0)

        plt.subplot(188) 
        plt.imshow(featsMap_by_rogistic_coeffs[i], cmap='jet')
        plt.axis('off')
        plt.tight_layout()
        # set the spacing between subplots
        plt.subplots_adjust(wspace=0.01, hspace=0)
        # plt.show()

        plt.axis('off')
        plt.savefig(os.path.join(save_path, f"slice_{i}.png"), bbox_inches='tight', pad_inches=0, dpi=600 )
        plt.close()



def run_save_featsMap(image_path, mask_path, coeffs = [0.20082066, 0.15587652, 0.17159753, 0.11601547, 0.07308697, -0.14703178, 0.31023642] ): 

    imageData_update, maskData_update = load_data(image_path, mask_path)
    imageData_crop_update, maskData_crop_update, ori_size, [z1, z2, x1, x2, y1, y2] = crop_image(imageData_update, maskData_update)
    x_1_patch, y_1_patch = sitk.GetArrayFromImage(imageData_crop_update), sitk.GetArrayFromImage(maskData_crop_update)
    
    x_1_patch = Normalize01(x_1_patch)
    x_1_patch_copy = x_1_patch.copy()
    x_1_patch = x_1_patch*y_1_patch

    featsMap_sel_resize_list = []
    outputPath = './featureVisulaization/' + image_path.split('/')[-1].rsplit('.', 2)[0] 
    radioFeatsMapPaths = [outputPath + '/' + fname +'.nii.gz' for fname in features_name_list]
    for i in range(len(radioFeatsMapPaths)): 
        featsMapData = sitk.ReadImage(radioFeatsMapPaths[i])
        featsMapData_np = sitk.GetArrayFromImage(featsMapData) 
        featsMapData_np = skimage.transform.resize(featsMapData_np, [z2-z1, x2-x1, y2-y1], order=1, preserve_range=True, anti_aliasing=False)
        featsMapData_np = Normalize01(featsMapData_np)
        featsMapData_np = featsMapData_np*y_1_patch
        featsMap_sel_resize_list.append(featsMapData_np) 

    featsMap_by_rogistic_coeffs = featsMap_weighted_with_coeffs(featsMap_sel_resize_list, coeffs) 
    featsMap_by_rogistic_coeffs = Normalize01(featsMap_by_rogistic_coeffs)
    featsMap_by_rogistic_coeffs = featsMap_by_rogistic_coeffs*y_1_patch

    if not os.path.exists( outputPath): 
        os.makedirs(outputPath)

    visualize_featsMap(x_1_patch, featsMap_sel_resize_list, featsMap_by_rogistic_coeffs, save_path=outputPath) 
    print(image_path.split('/')[-1].rsplit('.', 2)[0])

    outputPath_S = './featureVisulaization/' + image_path.split('/')[-1].rsplit('.', 2)[0] + '/S'
    if not os.path.exists( outputPath_S): 
        os.makedirs(outputPath_S) 
    ##resize
    x_1_patch = skimage.transform.resize( x_1_patch, [x_1_patch.shape[0]*6, ]+list(x_1_patch.shape[1:3]), order=1, preserve_range=True, anti_aliasing=False ) 
    featsMap_sel_resize_list = [skimage.transform.resize( featsMap_sel_resize_list[i_fm], [featsMap_sel_resize_list[i_fm].shape[0]*6, ]+list(featsMap_sel_resize_list[i_fm].shape[1:3]), order=1, preserve_range=True, anti_aliasing=False )\
                                for i_fm in range(len(featsMap_sel_resize_list)) ]
    featsMap_by_rogistic_coeffs = skimage.transform.resize( featsMap_by_rogistic_coeffs, [featsMap_by_rogistic_coeffs.shape[0]*6, ]+list(featsMap_by_rogistic_coeffs.shape[1:3]), order=1, preserve_range=True, anti_aliasing=False ) 
    ## flip
    x_1_patch = np.flip(x_1_patch, axis=0) 
    featsMap_sel_resize_list = [np.flip(featsMap_sel_resize_list[i_fm], axis=0) for i_fm in range(len(featsMap_sel_resize_list)) ]
    featsMap_by_rogistic_coeffs = np.flip(featsMap_by_rogistic_coeffs, axis=0) 
    
    ##save in Sagittal direction
    x_1_patch_S = np.transpose(x_1_patch, [1, 0, 2]) 
    featsMap_sel_resize_list_S = [np.transpose(featsMap_sel_resize_list[i_fm], [1, 0, 2]) for i_fm in range(len(featsMap_sel_resize_list)) ]
    featsMap_by_rogistic_coeffs_S = np.transpose(featsMap_by_rogistic_coeffs, [1, 0, 2]) 
    visualize_featsMap(x_1_patch_S, featsMap_sel_resize_list_S, featsMap_by_rogistic_coeffs_S, save_path=outputPath_S ) 

    ##save in Coronal direction
    outputPath_C = './featureVisulaization/' + image_path.split('/')[-1].rsplit('.', 2)[0] + '/C'
    if not os.path.exists( outputPath_C): 
        os.makedirs(outputPath_C) 
    img_lung_patch_C = np.transpose(x_1_patch, [2, 0, 1])
    featsMap_sel_resize_list_C = [np.transpose(featsMap_sel_resize_list[i_fm], [2, 0, 1]) for i_fm in range(len(featsMap_sel_resize_list)) ]
    featsMap_by_rogistic_coeffs_C = np.transpose(featsMap_by_rogistic_coeffs, [2, 0, 1]) 
    visualize_featsMap(img_lung_patch_C, featsMap_sel_resize_list_C, featsMap_by_rogistic_coeffs_C, save_path= outputPath_C ) 



# Run main and extract pyradiomics features
if __name__ == '__main__':
    n_workers = 15
    
    path_image = r'/results_data_nii'
    path_mask = r'/results'

    df_image_info={}
    df_image_info['image'] = find_all_nii(path_image)
    df_image_info['mask'] = find_all_nii(path_mask)
    image_paths = find_all_nii(path_image)
    mask_paths = find_all_nii(path_mask)

    features_name_list = ['log-sigma-3-0-mm-3D_firstorder_Maximum', 'log-sigma-3-0-mm-3D_glszm_ZoneEntropy', 'wavelet-HHH_firstorder_Minimum', 
                            'wavelet-LLL_firstorder_Minimum', 'square_gldm_SmallDependenceHighGrayLevelEmphasis', 'lbp-3D-m1_firstorder_90Percentile', 
                            'lbp-3D-k_firstorder_Skewness'] 
    coeffs = [ 0.28566381, 0.27731318, -0.29851531, -0.17285832, -0.41540002, -0.1525067, 0.07975887]

    for i in range(len(image_paths)): 
        run_featureMap_multi(image_paths[i], mask_paths[i], params_file, label, features_name_list[1] ) 
        run_save_featsMap(image_paths[i], mask_paths[i], coeffs=coeffs)
