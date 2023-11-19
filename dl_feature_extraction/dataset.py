import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import pickle
import skimage
from skimage.transform import resize
import SimpleITK as sitk
import json
import os
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import torchio as tio
import random
import scipy 


def load_json(dataset_name, json_name, root_path ):
    
    real_path = os.path.join(root_path, dataset_name, json_name)
    return json.load(open(real_path))

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image):
        return torch.from_numpy(image).type(torch.FloatTensor)

def np2nii(array, spacing=[1,1,1], outDir=None): 
    '''
        # image_np_crop_sitk = np2nii(image_np_crop, spacing=out_size, outDir=data_save_path )
    save numpy array as simple itk 
    inputs:: 
        array: np.array
        spacing: [1.5,1.5,2]
        outDir: save directory
    output:: 
        array_sitk
    '''
    array_sitk = sitk.GetImageFromArray(array ) 
    array_sitk.SetSpacing(spacing)  ##SetSize
    # image_np_crop_sitk.SetOutputDirection(image.GetDirection()) 
    sitk.WriteImage( array_sitk, outDir ) 
    return array_sitk



dicom_transform = transforms.Compose([ToTensor()])
label_transform = transforms.Compose([ToTensor()])

class data_set(Dataset): 
    def __init__(self, image_paths, label_trg=None, ifSaveDatasetTemp=False, ifReadDatasetTemp=False, 
                        windowCenterWidth=(-200, 1600), iftransform=False, ifrandomCrop=True):
        ##
        self.image_paths = image_paths
        self.label_trg = label_trg
        self.ifSaveDatasetTemp = ifSaveDatasetTemp
        self.ifReadDatasetTemp = ifReadDatasetTemp
        # self.windowCenterWidth = windowCenterWidth
        self.imgMinMax = [ windowCenterWidth[0] - windowCenterWidth[1]/2.0, windowCenterWidth[0] + windowCenterWidth[1]/2.0 ]
        self.iftransform = iftransform
        self.ifrandomCrop = ifrandomCrop

    def __len__(self):
        return len(self.image_paths) 

    def _read_itk_files(self, img_path, label_path): 
        image_sitk = sitk.ReadImage( img_path ) 
        x = sitk.GetArrayFromImage(image_sitk) 

        originalimg_spacing = image_sitk.GetSpacing() 
        label_sitk = sitk.ReadImage( label_path ) 
        y = sitk.GetArrayFromImage(label_sitk) 
        return x, y, originalimg_spacing

    def _random_crop(self, img, mask, width, height):
        assert img.shape[-2] >= width
        assert img.shape[-1] >= height
        x = random.randint(0, img.shape[-2] - width)
        y = random.randint(0, img.shape[-1] - height)
        img = img[:, x:x+width, y:y+height]
        mask = mask[:, x:x+width, y:y+height]
        return img, mask


    def __getitem__(self, index): 

        paths = self.image_paths[index]
        x_1_patch_list=[]
        y_1_patch_list=[] 

        img_path, clu_path = paths[0], paths[1] 

        x_1_patch_, y_1_patch_, originalimg_spacing = self._read_itk_files(img_path, clu_path) 

        x_1_patch_ = np.clip(x_1_patch_, a_min=self.imgMinMax[0], a_max=self.imgMinMax[1] ) 
        x_1_patch_ = (x_1_patch_ - self.imgMinMax[0] ) / (self.imgMinMax[1] - self.imgMinMax[0])


        # if i==0: 
        y_1_t = (y_1_patch_>0.5)*1 
        a = np.where(y_1_t>0) 
        z1, z2, x1, x2, y1, y2 = [np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1]), np.min(a[2]), np.max(a[2])] 

        x_1_patch = x_1_patch_[z1:z2+1, x1:x2+1, y1:y2+1] 
        y_1_patch = (y_1_patch_[z1:z2+1, x1:x2+1, y1:y2+1] >0.5)*1 


        if random.random()>0.5 and self.ifrandomCrop:  ##0.6
            x_1_patch, y_1_patch = self._random_crop(x_1_patch, y_1_patch, 
                                                    width=random.randint( x_1_patch.shape[-2]*2//3, x_1_patch.shape[-2] ), 
                                                    height=random.randint( x_1_patch.shape[-1]*2//3, x_1_patch.shape[-1] ) 
                                                    )

        x_1_patch_ = skimage.transform.resize(x_1_patch, [32, 192, 256], order=1, preserve_range=True, anti_aliasing=False)  ##[64, 224, 224] 192, 256
        y_1_patch_ = skimage.transform.resize(y_1_patch, [32, 192, 256], order=0, preserve_range=True, anti_aliasing=False)  ##[64, 224, 320]  [64, 192, 256]
        
        x_1_patch_ =  np.expand_dims(x_1_patch_, axis=0 )
        y_1_patch_ =  np.expand_dims(y_1_patch_, axis=0 )
        x_1_patch_list = [x_1_patch_]
        y_1_patch_list = [y_1_patch_]

        # # ifTransform = False
        # ## only transform the positive case
        # ifTransform = True if self.label_trg[index] and self.iftransform else False
        ifTransform = True if self.iftransform else False
        if ifTransform: 
            x_1_patch_list, y_1_patch_list = self._transforms(x_1_patch_list, y_1_patch_list) 
        else: 
            x_1_patch_list = [torch.from_numpy(x_1_patch_ ).type(torch.FloatTensor) for x_1_patch_ in x_1_patch_list ]
            y_1_patch_list = [torch.from_numpy(y_1_patch_ ).type(torch.FloatTensor) for y_1_patch_ in y_1_patch_list ]

        if self.label_trg==None: 
            return x_1_patch_list, y_1_patch_list
        else: 
            return x_1_patch_list, y_1_patch_list, self.label_trg[index] 

    def _normalization_with_CenterWidth(self, x_1_patch_, windowCenterWidth = [[-200, 1600], 
                                                                                [-500, 1750], 
                                                                                [-500, 1250], 
                                                                                [50, 300], 
                                                                                [-500, 600] ] ): 
    
        imgMinMax_list = []
        for cw in windowCenterWidth: 
            imgMinMax = [ cw[0] - cw[1]/2.0, cw[0] + cw[1]/2.0 ]
            imgMinMax_list.append(imgMinMax) 
        x_1_patch_list = []
        for mm in imgMinMax_list: 
            x_1_patch_c = np.clip(x_1_patch_, a_min=mm[0], a_max=mm[1] ) 
            x_1_patch_cn = (x_1_patch_c - mm[0] ) / (mm[1] - mm[0]) 
            x_1_patch_list.append(x_1_patch_cn) 
        return x_1_patch_list

    def _transforms(self, image_3d_list, mask_3d_list=None, ): 
        ifAffine     = random.random()>0.7
        deg          = random.randint(-20, 20) 
        ifFlip       = random.random()>0.7
        ifGamma      = random.random()>0.7
        ifNoise      = random.random()>0.7 
        ifBiasField  = random.random()>0.7 
        ifBlur       = random.random()>0.7 
        ifSwap       = random.random()>0.7 
        ifMotion     = random.random()>0.7 
        tran    = 0 # random.randint(-5, 5) 
        sca     = 0 # random.randint(6,9)/10.0 

        image_3d_list_transformed, mask_3d_list_transformed = [], []
        for i in range(len(image_3d_list)): 
            image_3d = image_3d_list[i]
            subject = tio.Subject( 
                                    image=tio.ScalarImage(tensor=image_3d),)
            if mask_3d_list is not None: 
                mask_3d = mask_3d_list[i]
                # mask = tio.Image(mask_3d, type=tio.LABEL)
                subject = tio.Subject(
                                        image=tio.ScalarImage(tensor=image_3d),
                                        mask=tio.LabelMap(tensor=mask_3d),
                                    )
            # if ifCrop: 
            #     transform = tio.Crop()

            if ifAffine: 
                transform = tio.RandomAffine(degrees=(deg, deg ),
                                            # translation=(tran, tran, tran),
                                            # scale=(sca, sca, sca),
                                            # shear=(10, 10, 10),
                                            isotropic=False,
                                            default_pad_value='otsu',
                                            image_interpolation='linear',
                                            label_interpolation='nearest',
                                            p=1, 
                                            ) 
                if mask_3d is not None: 
                    subject = transform(subject ) 
                    image = subject['image'] 
                    mask = subject['mask']
                else: 
                    subject = transform(subject ) 
                    image = subject['image'] 
            if ifFlip: 
                transform  = tio.RandomFlip(axes=2, 
                                            flip_probability=1
                                            ) 
                if mask_3d is not None: 
                    subject = transform(subject ) 
                    image = subject['image'] 
                    mask = subject['mask'] 
                else: 
                    subject = transform(subject ) 
                    image = subject['image'] 
            if ifMotion: 
                transform  = tio.RandomMotion() 
                if mask_3d is not None: 
                    subject = transform(subject ) 
                    image = subject['image'] 
                    mask = subject['mask']
                else: 
                    subject = transform(subject ) 
                    image = subject['image'] 
            if ifSwap: 
                transform  = tio.RandomSwap(patch_size=[15,15,15], 
                                            num_iterations=100, 
                                            ) 
                if mask_3d is not None: 
                    subject = transform(subject ) 
                    image = subject['image'] 
                    mask = subject['mask']
                else: 
                    subject = transform(subject ) 
                    image = subject['image'] 
            if ifGamma: 
                transform = tio.RandomGamma(log_gamma=(-0.2, 0.2), p=1, include='image')
                if mask_3d is not None: 
                    subject = transform(subject ) 
                    image = subject['image'] 
                    mask = subject['mask']
                else: 
                    subject = transform(subject ) 
                    image = subject['image'] 
            if ifNoise: 
                transform = tio.RandomNoise(include='image', p=1)
                if mask_3d is not None: 
                    subject = transform(subject) 
                    image = subject['image'] 
                    mask = subject['mask']
                else: 
                    subject = transform(subject) 
                    image = subject['image'] 
            if ifBiasField: 
                transform = tio.RandomBiasField(include='image', p=1)
                if mask_3d is not None: 
                    subject = transform(subject ) 
                    image = subject['image'] 
                    mask = subject['mask']
                else: 
                    subject = transform(subject ) 
                    image = subject['image']  
            if ifBlur: 
                transform = tio.RandomBlur(include='image', p=1)
                if mask_3d is not None: 
                    subject = transform(subject ) 
                    image = subject['image'] 
                    mask = subject['mask']
                else: 
                    subject = transform(subject ) 
                    image = subject['image'] 

            if mask_3d is not None: 
                image = subject['image'] 
                mask = subject['mask']
            else: 
                image = subject['image'] 
            image_3d_list_transformed.append(image[tio.DATA])
            if mask_3d_list is not None: 
                mask_3d_list_transformed.append(mask[tio.DATA]) 
        
        # return {'image': image_transformed, 'mask': mask_transformed}
        if mask_3d_list is not None: 
            return image_3d_list_transformed, mask_3d_list_transformed
        else: 
            return image_3d_list_transformed



def get_dataloader(image_paths_full, label_trg, bs=1, ifshuffle=False, ifSaveDatasetTemp=True, ifReadDatasetTemp=True, iftransform=False):

    dataset = data_set(image_paths_full, label_trg, ifSaveDatasetTemp=ifSaveDatasetTemp, ifReadDatasetTemp=ifReadDatasetTemp, iftransform=iftransform) 
    # dataset = data_set(image_paths_full, label_trg, ifSaveDatasetTemp, ifReadDatasetTemp) 
    
    dataloader = DataLoader(dataset=dataset,
                            batch_size=bs,
                            num_workers=0,
                            shuffle=ifshuffle)
    return dataloader
