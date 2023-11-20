import torch
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pandas as pd
import skimage
import scipy
from multiprocessing import Pool, Manager
from itertools import repeat

def read_itk_files(img_path, label_path): 
    image_sitk = sitk.ReadImage( img_path ) 
    x = sitk.GetArrayFromImage(image_sitk) 
    originalimg_spacing = image_sitk.GetSpacing() 
    label_sitk = sitk.ReadImage( label_path ) 
    y = sitk.GetArrayFromImage(label_sitk) 
    return x, y, originalimg_spacing

def mipAndMinip(images, labels, mode='mip', num_proj=3): 
    ###mip,minip 10 mm and 15 mm 
    labels_ = scipy.ndimage.binary_erosion(labels, structure=np.ones((3,3,3)) ).astype(labels.dtype) 
    images_ = images*labels_ 
    img_shape = images_.shape
    np_mip = np.zeros(img_shape) 
    if mode=='mip': 
        for i in range(img_shape[0]): 
            start = max(0, i-num_proj) 
            np_mip[i,:,:] = np.amax(images_[start:i+1],0) 
    if mode=='minip': 
        for i in range(img_shape[0]): 
            start = max(0, i-num_proj) 
            np_mip[i,:,:] = np.amin(images_[start:i+1],0) 
            
    return np_mip, labels_

def process_data(x_1_patch_, y_1_patch_): 
    # Forward pass with an input tensor
    windowCenterWidth=(-200, 1600)
    imgMinMax = [ windowCenterWidth[0] - windowCenterWidth[1]/2.0, windowCenterWidth[0] + windowCenterWidth[1]/2.0 ]

    x_1_patch_ = np.clip(x_1_patch_, a_min=imgMinMax[0], a_max=imgMinMax[1] ) 
    x_1_patch_ = (x_1_patch_ - imgMinMax[0] ) / (imgMinMax[1] - imgMinMax[0])
    # if i==0: 
    y_1_t = (y_1_patch_>0.5)*1 
    a = np.where(y_1_t>0) 
    z1, z2, x1, x2, y1, y2 = [np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1]), np.min(a[2]), np.max(a[2])] 

    x_1_patch = x_1_patch_[z1:z2+1, x1:x2+1, y1:y2+1] 
    y_1_patch = (y_1_patch_[z1:z2+1, x1:x2+1, y1:y2+1] >0.5)*1 

    # x_1_patch_ = skimage.transform.resize(x_1_patch, [32, 192, 256], order=1, preserve_range=True, anti_aliasing=False)  ##[64, 224, 224] 192, 256
    # y_1_patch_ = skimage.transform.resize(y_1_patch, [32, 192, 256], order=0, preserve_range=True, anti_aliasing=False)  ##[64, 224, 320]  [64, 192, 256]
    x_1_patch_ = skimage.transform.resize(x_1_patch, [64, 224, 320], order=1, preserve_range=True, anti_aliasing=False)  ##[64, 224, 224] 192, 256
    y_1_patch_ = skimage.transform.resize(y_1_patch, [64, 224, 320], order=0, preserve_range=True, anti_aliasing=False)  ##[64, 224, 320]  [64, 192, 256]

    x_1_patch_, y_1_patch_ =  mipAndMinip(x_1_patch_*y_1_patch_, y_1_patch_, mode='mip', num_proj=5)  ###mask use the

    x_1_patch_ = skimage.exposure.adjust_gamma(x_1_patch_, 0.5)  

    x_1_patch_stack = np.stack([x_1_patch_, x_1_patch_, x_1_patch_], axis=0)
    x_1_patch_stack = x_1_patch_stack[None,:]
    input_tensor = torch.from_numpy(x_1_patch_stack) 
    input_tensor = input_tensor.type(torch.FloatTensor) 
    return input_tensor, [z1, z2+1, x1, x2+1, y1, y2+1], x_1_patch, y_1_patch


def prepare_model_and_make_inter_outputs(input_tensor): 

    # Define a hook function to extract the output from each layer
    def hook_fn(layer_name):
        def hook(module, input, output):
            # Store the output of the intermediate layer
            intermediate_outputs[layer_name] = output.detach().cpu()
        return hook

    # Load the R3D-18 model
    model = models.video.r3d_18(pretrained=True)  ##weights=R3D_18_Weights.KINETICS400_V1

    # Create a dictionary to store intermediate outputs
    intermediate_outputs = {}

    # Register hooks to all layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d):
            hook_handle = module.register_forward_hook(hook_fn(name))

    # input_tensor = torch.randn(1, 3, 16, 112, 112)
    model(input_tensor)  
    return intermediate_outputs



def save_results(intermediate_outputs): 


    # Create a directory to save the intermediate outputs
    output_dir = "intermediate_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Save the intermediate outputs as NumPy arrays and PNG images
    for name, output in intermediate_outputs.items():
        # Save as NumPy array
        np.save(os.path.join(output_dir, f"{name}.npy"), output.numpy())

        # Visualize and save as PNG image
        if output.ndim == 5:
            # Take a central slice from the 3D volume
            slice_index = output.size(2) // 2
            slice_image = output[0, :, slice_index].numpy()
            for channel_idx in range(slice_image.shape[0]):
                plt.imshow(slice_image[channel_idx], cmap='gray')
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, f"{name}_slice_{channel_idx}.png"), bbox_inches='tight', pad_inches=0)
                plt.close()
        else:
            # Visualize and save as PNG image assuming 3D volume with one channel
            plt.imshow(output[0, 0].numpy(), cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f"{name}.png"), bbox_inches='tight', pad_inches=0)
            plt.close()

def Normalize01(data ):
    new_data = np.asarray(data, dtype=np.float32)
    new_data = new_data - np.min(new_data)
    new_data = new_data / np.max(new_data)
    return new_data 



def run_featureMap(img_path, label_path, save_path='.'): 


    x_1_patch_, y_1_patch_, originalimg_spacing = read_itk_files(img_path, label_path)
    input_tensor, [z1, z2, x1, x2, y1, y2], x_1_patch, y_1_patch = process_data(x_1_patch_, y_1_patch_)
    intermediate_outputs = prepare_model_and_make_inter_outputs(input_tensor) 

    featsMap = intermediate_outputs['layer4.1.conv2.0'].squeeze().cpu().numpy()
    print(featsMap.shape)
    # a = ['DLfeat_23','DLfeat_114','DLfeat_148','DLfeat_181','DLfeat_317','DLfeat_383']
    featsMap_sel = featsMap[featsName_DL_num,:,:,:]

    featsMap_sel_resize_list = [] 
    for i in range(featsMap_sel.shape[0]): 
        featsMap_sel_resize = skimage.transform.resize( featsMap_sel[i], [z2-z1, x2-x1, y2-y1], order=1, preserve_range=True, anti_aliasing=False ) 
        featsMap_sel_resize_list.append(featsMap_sel_resize)

    # y_1_patch = scipy.ndimage.binary_erosion(y_1_patch, structure=np.ones((7,7,7)) ).astype(y_1_patch.dtype) 

    lung_img_itk = sitk.GetImageFromArray(x_1_patch) ## * y_1_patch
    lung_img_itk.SetSpacing(originalimg_spacing) 
    sitk.WriteImage(lung_img_itk, save_path + '/' + 'lung_img' + '.nii.gz', True)

    lung_mask_itk = sitk.GetImageFromArray(y_1_patch) ## * y_1_patch
    lung_mask_itk.CopyInformation(lung_img_itk)
    sitk.WriteImage(lung_mask_itk, save_path + '/' + 'lung_mask' + '.nii.gz', True)

    for i in range(len(featsMap_sel_resize_list)): 
        # fMap = sitk.GetImageFromArray(featsMap_sel_resize_list[i] * y_1_patch)
        fMap = sitk.GetImageFromArray(featsMap_sel_resize_list[i])
        fMap.CopyInformation(lung_img_itk)
        sitk.WriteImage(fMap, save_path + '/' + f'DL_{featsName_DL_num[i]}.nii.gz', True)

def featsMap_weighted_with_coeffs(featsMap_sel_resize_list, coeffs):
    featsMap_by_rogistic_coeffs = np.zeros(featsMap_sel_resize_list[0].shape )
    for idx in range(len(coeffs)): 
        featsMap_by_rogistic_coeffs += featsMap_sel_resize_list[idx]*coeffs[idx]
    return featsMap_by_rogistic_coeffs

def visualize_featsMap(img_lung_patch, featsMap_sel_resize_list, featsMap_by_rogistic_coeffs, save_path='./results_png'): 
    for i in range(len(img_lung_patch)): 
        plt.figure()
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

def run_save_featsMap(image_path, mask_path, coeffs = [0.16433159, -0.36936321, -0.14956905, -0.24486382, 0.26456043, 0.39575924] ): 
    
    x_1_patch_, y_1_patch_, originalimg_spacing = read_itk_files(image_path, mask_path)
    input_tensor, [z1, z2, x1, x2, y1, y2], x_1_patch, y_1_patch = process_data(x_1_patch_, y_1_patch_)
    # y_1_patch = scipy.ndimage.binary_erosion(y_1_patch, structure=np.ones((7,7,7)) ).astype(y_1_patch.dtype) 

    outputPath = './featureVisulaization_DL/' + image_path.split('/')[-1].rsplit('.', 2)[0] 
    img_lung_patch_itk = sitk.ReadImage(outputPath + '/lung_img.nii.gz') 
    img_lung_patch = sitk.GetArrayFromImage(img_lung_patch_itk) 

    features_name_list = [f'DL_{featsName_DL_num[i]}' for i in range(len(featsName_DL_num))] 
    FeatsMapPaths = [outputPath + '/' + fname +'.nii.gz' for fname in features_name_list] 
    featsMap_sel_resize_list = []
    for i_fm in range(len(FeatsMapPaths)): 
        featsMapData = sitk.ReadImage(FeatsMapPaths[i_fm])
        featsMapData_np = sitk.GetArrayFromImage(featsMapData) 
        # featsMapData_np = skimage.transform.resize(featsMapData_np, [z2-z1, x2-x1, y2-y1], order=1, preserve_range=True, anti_aliasing=False)
        featsMapData_np = Normalize01(featsMapData_np)
        # featsMapData_np = featsMapData_np*y_1_patch
        featsMap_sel_resize_list.append(featsMapData_np) 

    
    featsMap_by_rogistic_coeffs = featsMap_weighted_with_coeffs(featsMap_sel_resize_list, coeffs) 
    featsMap_by_rogistic_coeffs = Normalize01(featsMap_by_rogistic_coeffs)
    # featsMap_by_rogistic_coeffs = featsMap_by_rogistic_coeffs*y_1_patch

    visualize_featsMap(img_lung_patch, featsMap_sel_resize_list, featsMap_by_rogistic_coeffs, save_path= outputPath ) 

    print(image_path.split('/')[-1].rsplit('.', 2)[0]) 

    outputPath_S = './featureVisulaization_DL/' + image_path.split('/')[-1].rsplit('.', 2)[0] + '/S'
    if not os.path.exists( outputPath_S): 
        os.makedirs(outputPath_S) 
    ##resize
    img_lung_patch = skimage.transform.resize( img_lung_patch, [img_lung_patch.shape[0]*6, ]+list(img_lung_patch.shape[1:3]), order=1, preserve_range=True, anti_aliasing=False ) 
    featsMap_sel_resize_list = [skimage.transform.resize( featsMap_sel_resize_list[i_fm], [featsMap_sel_resize_list[i_fm].shape[0]*6, ]+list(featsMap_sel_resize_list[i_fm].shape[1:3]), order=1, preserve_range=True, anti_aliasing=False )\
                                   for i_fm in range(len(featsMap_sel_resize_list)) ]
    featsMap_by_rogistic_coeffs = skimage.transform.resize( featsMap_by_rogistic_coeffs, [featsMap_by_rogistic_coeffs.shape[0]*6, ]+list(featsMap_by_rogistic_coeffs.shape[1:3]), order=1, preserve_range=True, anti_aliasing=False ) 
    ## flip
    img_lung_patch = np.flip(img_lung_patch, axis=0) 
    featsMap_sel_resize_list = [np.flip(featsMap_sel_resize_list[i_fm], axis=0) for i_fm in range(len(featsMap_sel_resize_list)) ]
    featsMap_by_rogistic_coeffs = np.flip(featsMap_by_rogistic_coeffs, axis=0) 
    
    ##save in Sagittal direction
    img_lung_patch_S = np.transpose(img_lung_patch, [1, 0, 2]) 
    featsMap_sel_resize_list_S = [np.transpose(featsMap_sel_resize_list[i_fm], [1, 0, 2]) for i_fm in range(len(featsMap_sel_resize_list)) ]
    featsMap_by_rogistic_coeffs_S = np.transpose(featsMap_by_rogistic_coeffs, [1, 0, 2]) 
    visualize_featsMap(img_lung_patch_S, featsMap_sel_resize_list_S, featsMap_by_rogistic_coeffs_S, save_path= outputPath_S ) 

    ##save in Coronal direction
    outputPath_C = './featureVisulaization_DL/' + image_path.split('/')[-1].rsplit('.', 2)[0] + '/C'
    if not os.path.exists( outputPath_C): 
        os.makedirs(outputPath_C) 
    img_lung_patch_C = np.transpose(img_lung_patch, [2, 0, 1])
    featsMap_sel_resize_list_C = [np.transpose(featsMap_sel_resize_list[i_fm], [2, 0, 1]) for i_fm in range(len(featsMap_sel_resize_list)) ]
    featsMap_by_rogistic_coeffs_C = np.transpose(featsMap_by_rogistic_coeffs, [2, 0, 1]) 
    visualize_featsMap(img_lung_patch_C, featsMap_sel_resize_list_C, featsMap_by_rogistic_coeffs_C, save_path= outputPath_C ) 

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


path_image = r'/results_data_nii'
path_mask = r'/results'

image_paths = find_all_nii(path_image)
mask_paths = find_all_nii(path_mask)

features_name_list = ['DLfeat_23', 'DLfeat_113', 'DLfeat_114', 'DLfeat_181', 'DLfeat_317',
                        'DLfeat_361', 'DLfeat_383', 'DLfeat_479']
featsName_DL_num = [int(features_name_list[i].split('_')[-1]) for i in range(len(features_name_list))]
coeffs = [ 0.15431351, 0.37772542, -0.32864266, -0.16367939, 0.17699039, -0.0797936, 0.29851433, -0.31098062] 

for i in range(len(image_paths)): 
    outputPath = './featureVisulaization_DL/' + image_paths[i].split('/')[-1].rsplit('.', 2)[0] 
    if not os.path.exists( outputPath): 
        os.makedirs(outputPath) 

    run_featureMap(image_paths[i], mask_paths[i], save_path=outputPath) 
    run_save_featsMap(image_paths[i], mask_paths[i])
print()



