import sys
import os
curPath = os.path.dirname(__file__)
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
import logging
import SimpleITK as sitk
import pkg_resources
import numpy as np
import skimage
from skimage import measure
import matplotlib.pyplot as plt

def path(string):
    if os.path.exists(string):
        return string
    else:
        sys.exit(f'File not found: {string}')


def main(inputpath, labelpath=None,outputputh=None, modeltype='unet', modelname='R231', modelpath='./lungmask/unet_r231-d5d2fc3d.pth', 
            classes=3, cpu=False, nopostprocess=False, noHU=False, batchsize=20, trainORtest='test'):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/tr_im.nii.gz',type=path, help='Path to the input image, can be a folder for dicoms')
    parser.add_argument('--output', default='results', type=str, help='Filepath for output lungmask')
    parser.add_argument('--modeltype', help='Default: unet', type=str, choices=['unet'], default='unet')
    parser.add_argument('--modelname', help="spcifies the trained model, Default: R231", type=str, choices=['R231','LTRCLobes','LTRCLobes_R231','R231CovidWeb'], default='R231')
    parser.add_argument('--modelpath', help="spcifies the path to the trained model", default=None)
    parser.add_argument('--classes', help="spcifies the number of output classes of the model", default=3)
    parser.add_argument('--cpu', help="Force using the CPU even when a GPU is available, will override batchsize to 1", action='store_true')
    parser.add_argument('--nopostprocess', help="Deactivates postprocessing (removal of unconnected components and hole filling", action='store_true')
    parser.add_argument('--noHU', help="For processing of images that are not encoded in hounsfield units (HU). E.g. png or jpg images from the web. Be aware, results may be substantially worse on these images", action='store_true')
    parser.add_argument('--batchsize', type=int, help="Number of slices processed simultaneously. Lower number requires less memory but may be slower.", default=20)
    
    argsin = sys.argv[1:]
    args = parser.parse_args(argsin) 
    
    args.input = inputpath
    args.input_label = labelpath
    args.modelpath = modelpath
    args.output = outputputh
    args.trainORtest = trainORtest

    batchsize = args.batchsize
    if args.cpu:
        batchsize = 1

    logging.info(f'Load model')
    
    if args.trainORtest=='test': 
        input_image = get_input_image(args.input, path_label=args.input_label, trainORtest=args.trainORtest)
    if args.trainORtest=='train': 
        input_image, input_label = get_input_image(args.input, path_label=args.input_label, trainORtest=args.trainORtest)
    # input_image = utils.get_input_image(args.input)

    # ## print the image maximum and minimum 
    # inimg_raw = sitk.GetArrayFromImage(input_image)
    # print(np.min(inimg_raw), np.max(inimg_raw))

    # ###writing the original image as nii
    # sitk.WriteImage(input_image, args.output)

    logging.info(f'Infer lungmask')
    if args.modelname == 'LTRCLobes_R231':
        assert args.modelpath is None, "Modelpath can not be specified for LTRCLobes_R231 mode"
        result = apply_fused(input_image, force_cpu=args.cpu, batch_size=batchsize, volume_postprocessing=not(args.nopostprocess), noHU=args.noHU)
        # result = mask.apply_fused(input_image, force_cpu=args.cpu, batch_size=batchsize, volume_postprocessing=not(args.nopostprocess), noHU=args.noHU)
    else:
        model = get_model(args.modeltype, args.modelname, args.modelpath, args.classes, trainORtest=args.trainORtest)
        if args.trainORtest=='test': 
            result, xnew_box, inimg_raw = apply(input_image, model=model, force_cpu=args.cpu, batch_size=batchsize, volume_postprocessing=not(args.nopostprocess), noHU=args.noHU)
        elif args.trainORtest=='train':
            result, xnew_box, inimg_raw = apply(input_image, input_label, model=model, force_cpu=args.cpu, batch_size=batchsize, volume_postprocessing=not(args.nopostprocess), noHU=args.noHU, trainORtest=args.trainORtest)

    if args.noHU:
        file_ending = args.output.split('.')[-1]
        print(file_ending)
        if file_ending in ['jpg','jpeg','png']:
            result = (result/(result.max())*255).astype(np.uint8)
        result = result[0]
    
    result_out= sitk.GetImageFromArray(result)
    result_out.CopyInformation(input_image)
    logging.info(f'Save result to: {args.output}')
    # sys.exit(sitk.WriteImage(result_out, args.output))
    sitk.WriteImage(result_out, args.output)


import numpy as np
import torch
# from lungmask import utils
import SimpleITK as sitk
from resunet import UNet
# from .resunet import UNet
import warnings
import sys
from tqdm import tqdm
import skimage
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning)

# stores urls and number of classes of the models
model_urls = {('unet', 'R231'): ('https://github.com/JoHof/lungmask/releases/download/v0.0/unet_r231-d5d2fc3d.pth', 3),
              ('unet', 'LTRCLobes'): (
                  'https://github.com/JoHof/lungmask/releases/download/v0.0/unet_ltrclobes-3a07043d.pth', 6),
              ('unet', 'R231CovidWeb'): (
                  'https://github.com/JoHof/lungmask/releases/download/v0.0/unet_r231covid-0de78a7e.pth', 3)}


def apply(image, label=None, model=None, force_cpu=False, batch_size=20, volume_postprocessing=True, noHU=False, trainORtest='test'):
    if model is None:
        model = get_model('unet', 'R231')
    
    numpy_mode = isinstance(image, np.ndarray)
    if numpy_mode:
        if trainORtest == 'test': 
            inimg_raw = image.copy()
        elif trainORtest=='train':
            inimg_raw = image.copy()
            inlabel_raw = label.copy()
    else:
        if trainORtest == 'test': 
            inimg_raw = sitk.GetArrayFromImage(image)
            directions = np.asarray(image.GetDirection())
            if len(directions) == 9:
                inimg_raw = np.flip(inimg_raw, np.where(directions[[0,4,8]][::-1]<0)[0])
        elif trainORtest=='train':
            inimg_raw = sitk.GetArrayFromImage(image)
            directions = np.asarray(image.GetDirection())
            if len(directions) == 9:
                inimg_raw = np.flip(inimg_raw, np.where(directions[[0,4,8]][::-1]<0)[0])

            inlabel_raw = sitk.GetArrayFromImage(label)
            directions = np.asarray(label.GetDirection())
            if len(directions) == 9:
                inlabel_raw = np.flip(inlabel_raw, np.where(directions[[0,4,8]][::-1]<0)[0])

    del image
    if trainORtest=='train': del label
    if force_cpu:
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            logging.info("No GPU support available, will use CPU. Note, that this is significantly slower!")
            batch_size = 1
            device = torch.device('cpu')
    model.to(device)

    
    if not noHU:
        if trainORtest=='test':
            tvolslices, xnew_box = preprocess(inimg_raw, resolution=[256, 256], trainORtest=trainORtest)
        elif trainORtest=='train':
            tvolslices, xnew_box,tvollabels = preprocess(inimg_raw, label=inlabel_raw, resolution=[256, 256], trainORtest=trainORtest)

        # tvolslices, xnew_box = utils.preprocess(inimg_raw, resolution=[256, 256])
        tvolslices[tvolslices > 600] = 600
        tvolslices = np.divide((tvolslices + 1024), 1624)
    else:
        # support for non HU images. This is just a hack. The models were not trained with this in mind
        tvolslices = skimage.color.rgb2gray(inimg_raw)
        tvolslices = skimage.transform.resize(tvolslices, [256, 256])
        tvolslices = np.asarray([tvolslices*x for x in np.linspace(0.3,2,20)])
        tvolslices[tvolslices>1] = 1
        sanity = [(tvolslices[x]>0.6).sum()>25000 for x in range(len(tvolslices))]
        tvolslices = tvolslices[sanity]

    if trainORtest=='test':
        torch_ds_val = LungLabelsDS_inf(tvolslices, trainORtest=trainORtest)
    elif trainORtest=='train':
        torch_ds_val = LungLabelsDS_inf(tvolslices, label=tvollabels, trainORtest=trainORtest)

    # torch_ds_val = utils.LungLabelsDS_inf(tvolslices)
    dataloader_val = torch.utils.data.DataLoader(torch_ds_val, batch_size=batch_size, shuffle=False, pin_memory=False)

    if trainORtest=='train':
        model.train()
        criterion = torch.nn.NLLLoss()  ##https://blog.csdn.net/u012505617/article/details/103851298
        optimizer = torch.optim.SGD(model.parameters(),lr=0.0003)
        with tqdm(total=len(dataloader_val), desc=f'training the model') as pbar:
            for idx, [X, Y] in enumerate(tqdm(dataloader_val)):
                X = X.float().to(device)
                Y = Y.float().to(device)
                prediction = model(X)
                # pls = torch.max(prediction, 1)[1].detach().cpu().numpy().astype(np.uint8)
                loss = criterion(prediction, Y.squeeze(1).type(torch.long) ) 
                loss.backward()
                optimizer.step()
                # logging.info('batch index: {} loss: {:.7f} '.format(idx, loss.item() ) )
                # print('batch index: {} loss: {:.7f} '.format(idx, loss.item() ) )
                pbar.update(X.shape[0])
                pbar.set_postfix(**{'loss (batch)':loss.item()})

    if trainORtest=='test':
        timage_res = np.empty((np.append(0, tvolslices[0].shape)), dtype=np.uint8)
        with torch.no_grad():
            for X in tqdm(dataloader_val):
                X = X.float().to(device)
                prediction = model(X)
                pls = torch.max(prediction, 1)[1].detach().cpu().numpy().astype(np.uint8)
                timage_res = np.vstack((timage_res, pls))

        # postprocessing includes removal of small connected components, hole filling and mapping of small components to
        # neighbors
        if volume_postprocessing:
            outmask = postrocessing(timage_res)
            # outmask = utils.postrocessing(timage_res)
        else:
            outmask = timage_res

        if noHU:
            outmask = skimage.transform.resize(outmask[np.argmax((outmask==1).sum(axis=(1,2)))], inimg_raw.shape[:2], order=0, anti_aliasing=False, preserve_range=True)[None,:,:]
        else:
            outmask = np.asarray(
                [reshape_mask(outmask[i], xnew_box[i], inimg_raw.shape[1:]) for i in range(outmask.shape[0])],
                dtype=np.uint8)
            #  outmask = np.asarray(
            #     [utils.reshape_mask(outmask[i], xnew_box[i], inimg_raw.shape[1:]) for i in range(outmask.shape[0])],
            #     dtype=np.uint8)
        if not numpy_mode:
            if len(directions) == 9:
                outmask = np.flip(outmask, np.where(directions[[0,4,8]][::-1]<0)[0])    
        
        return outmask.astype(np.uint8), xnew_box, inimg_raw


def get_model(modeltype, modelname, modelpath=None, n_classes=3, trainORtest='test'):
    if modelpath is None:
        model_url, n_classes = model_urls[(modeltype, modelname)]
        state_dict = torch.hub.load_state_dict_from_url(model_url, progress=True, map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(modelpath, map_location=torch.device('cpu'))

    if modeltype == 'unet':
        model = UNet(n_classes=n_classes, padding=True, depth=5, up_mode='upsample', batch_norm=True, residual=False)
    elif modeltype == 'resunet':
        model = UNet(n_classes=n_classes, padding=True, depth=5, up_mode='upsample', batch_norm=True, residual=True)
    else:
        logging.exception(f"Model {modelname} not known")
    model.load_state_dict(state_dict)
    if trainORtest=='test': 
        model.eval()
    elif trainORtest=='train': 
        model.train()
    
    return model


def apply_fused(image, basemodel = 'LTRCLobes', fillmodel = 'R231', force_cpu=False, batch_size=20, volume_postprocessing=True, noHU=False):
    '''Will apply basemodel and use fillmodel to mitiage false negatives'''
    mdl_r = get_model('unet',fillmodel)
    mdl_l = get_model('unet',basemodel)
    logging.info("Apply: %s" % basemodel)
    res_l, _, _ = apply(image, mdl_l, force_cpu=force_cpu, batch_size=batch_size,  volume_postprocessing=volume_postprocessing, noHU=noHU)
    logging.info("Apply: %s" % fillmodel)
    res_r, _, _ = apply(image, mdl_r, force_cpu=force_cpu, batch_size=batch_size,  volume_postprocessing=volume_postprocessing, noHU=noHU)
    spare_value = res_l.max()+1
    res_l[np.logical_and(res_l==0, res_r>0)] = spare_value
    res_l[res_r==0] = 0
    logging.info("Fusing results... this may take up to several minutes!")
    return postrocessing(res_l, spare=[spare_value])
    # return utils.postrocessing(res_l, spare=[spare_value])

import scipy.ndimage as ndimage
import skimage.measure
import numpy as np
from torch.utils.data import Dataset
import os
import sys
import SimpleITK as sitk
import pydicom as pyd
import logging
from tqdm import tqdm
import fill_voids
import skimage.morphology


def preprocess(img, label=None, resolution=[192, 192], trainORtest='test'):
    imgmtx = np.copy(img)
    lblsmtx = np.copy(label)

    imgmtx[imgmtx < -1024] = -1024
    imgmtx[imgmtx > 600] = 600
    cip_xnew = []
    cip_box = []
    cip_mask = []
    for i in range(imgmtx.shape[0]):
        if label is None:
            (im, m, box) = crop_and_resize(imgmtx[i, :, :], width=resolution[0], height=resolution[1])
        else:
            (im, m, box) = crop_and_resize(imgmtx[i, :, :], mask=lblsmtx[i, :, :], width=resolution[0],
                                           height=resolution[1])
            cip_mask.append(m)
        cip_xnew.append(im)
        cip_box.append(box)
    if label is None:
        return np.asarray(cip_xnew), cip_box
    else:
        return np.asarray(cip_xnew), cip_box, np.asarray(cip_mask)


def simple_bodymask(img):
    maskthreshold = -500
    oshape = img.shape
    img = ndimage.zoom(img, 128/np.asarray(img.shape), order=0)
    bodymask = img > maskthreshold
    bodymask = ndimage.binary_closing(bodymask)
    bodymask = ndimage.binary_fill_holes(bodymask, structure=np.ones((3, 3))).astype(int)
    bodymask = ndimage.binary_erosion(bodymask, iterations=2)
    bodymask = skimage.measure.label(bodymask.astype(int), connectivity=1)
    regions = skimage.measure.regionprops(bodymask.astype(int))
    if len(regions) > 0:
        max_region = np.argmax(list(map(lambda x: x.area, regions))) + 1
        bodymask = bodymask == max_region
        bodymask = ndimage.binary_dilation(bodymask, iterations=2)
    real_scaling = np.asarray(oshape)/128
    return ndimage.zoom(bodymask, real_scaling, order=0)


def crop_and_resize(img, mask=None, width=192, height=192):
    bmask = simple_bodymask(img)
    # img[bmask==0] = -1024 # this line removes background outside of the lung.
    # However, it has been shown problematic with narrow circular field of views that touch the lung.
    # Possibly doing more harm than help
    reg = skimage.measure.regionprops(skimage.measure.label(bmask))
    if len(reg) > 0:
        bbox = np.asarray(reg[0].bbox)
    else:
        bbox = (0, 0, bmask.shape[0], bmask.shape[1])
    img = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    img = ndimage.zoom(img, np.asarray([width, height]) / np.asarray(img.shape), order=1)
    if not mask is None:
        mask = mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        mask = ndimage.zoom(mask, np.asarray([width, height]) / np.asarray(mask.shape), order=0)
        # mask = ndimage.binary_closing(mask,iterations=5)
    return img, mask, bbox


## For some reasons skimage.transform leads to edgy mask borders compared to ndimage.zoom
# def reshape_mask(mask, tbox, origsize):
#     res = np.ones(origsize) * 0
#     resize = [tbox[2] - tbox[0], tbox[3] - tbox[1]]
#     imgres = skimage.transform.resize(mask, resize, order=0, mode='constant', cval=0, anti_aliasing=False, preserve_range=True)
#     res[tbox[0]:tbox[2], tbox[1]:tbox[3]] = imgres
#     return res


def reshape_mask(mask, tbox, origsize):
    res = np.ones(origsize) * 0
    resize = [tbox[2] - tbox[0], tbox[3] - tbox[1]]
    imgres = ndimage.zoom(mask, resize / np.asarray(mask.shape), order=0)
    res[tbox[0]:tbox[2], tbox[1]:tbox[3]] = imgres
    return res


class LungLabelsDS_inf(Dataset):
    def __init__(self, ds, label=None, trainORtest='test'):
        self.dataset = ds
        self.dataset_label = label
        self.trainORtest = trainORtest

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.trainORtest=='test': 
            return self.dataset[idx, None, :, :].astype(float)
        elif self.trainORtest=='train':
            return self.dataset[idx, None, :, :].astype(float), self.dataset_label[idx, None, :, :].astype(float)


def read_dicoms(path, primary=True, original=True):
    allfnames = []
    for dir, _, fnames in os.walk(path):
        [allfnames.append(os.path.join(dir, fname)) for fname in fnames]

    dcm_header_info = []
    dcm_parameters = []
    unique_set = []  # need this because too often there are duplicates of dicom files with different names
    i = 0
    for fname in tqdm(allfnames):
        filename_ = os.path.splitext(os.path.split(fname)[1])
        i += 1
        if filename_[0] != 'DICOMDIR':
            try:
                dicom_header = pyd.dcmread(fname, defer_size=100, stop_before_pixels=True, force=True)  
                # if dicom_header.RescaleIntercept != 0 or  dicom_header.RescaleSlope !=1: 
                #     print('InterceptSlope problems', fname)
                if dicom_header is not None:   ## for yecheng 155 cases   TBPortal 190
                # if dicom_header is not None and (dicom_header.SeriesDescription==' Lung 5.0  CE' or dicom_header.SeriesDescription=='Lung 5.0  CE'): ### for 485 and 100 or 104 
                    if 'ImageType' in dicom_header:
                        if primary:
                            is_primary = all([x in dicom_header.ImageType for x in ['PRIMARY']])
                        else:
                            is_primary = True

                        if original:
                            is_original = all([x in dicom_header.ImageType for x in ['ORIGINAL']])
                        else:
                            is_original = True

                        # if 'ConvolutionKernel' in dicom_header:
                        #     ck = dicom_header.ConvolutionKernel
                        # else:
                        #     ck = 'unknown'
                        if is_primary and is_original and 'LOCALIZER' not in dicom_header.ImageType:
                            # h_info_wo_name = [dicom_header.StudyInstanceUID, dicom_header.SeriesInstanceUID,
                            #                     dicom_header.ImagePositionPatient]
                            # h_info = [dicom_header.StudyInstanceUID, dicom_header.SeriesInstanceUID, fname,
                            #             dicom_header.ImagePositionPatient] 
                            try: 
                                h_info_wo_name = [dicom_header.StudyInstanceUID, dicom_header.SeriesInstanceUID,
                                                  dicom_header.ImagePositionPatient]
                                h_info = [dicom_header.StudyInstanceUID, dicom_header.SeriesInstanceUID, fname,
                                          dicom_header.ImagePositionPatient]
                                useSOPInstanceUID=False 
                            except: 
                                h_info_wo_name = [dicom_header.StudyInstanceUID, dicom_header.SeriesInstanceUID,
                                                  dicom_header.SOPInstanceUID]
                                h_info = [dicom_header.StudyInstanceUID, dicom_header.SeriesInstanceUID, fname,
                                          dicom_header.SOPInstanceUID] 
                                useSOPInstanceUID=True
                            if h_info_wo_name not in unique_set:
                                unique_set.append(h_info_wo_name)
                                dcm_header_info.append(h_info)
                                # kvp = None
                                # if 'KVP' in dicom_header:
                                #     kvp = dicom_header.KVP
                                # dcm_parameters.append([ck, kvp,dicom_header.SliceThickness])
            except:
                logging.error("Unexpected error:", sys.exc_info()[0])
                logging.warning("Doesn't seem to be DICOM, will be skipped: ", fname)

    conc = [x[1] for x in dcm_header_info]
    sidx = np.argsort(conc)
    conc = np.asarray(conc)[sidx]
    dcm_header_info = np.asarray(dcm_header_info, dtype=object)[sidx]
    # dcm_parameters = np.asarray(dcm_parameters)[sidx]
    vol_unique = np.unique(conc, return_index=1, return_inverse=1)  # unique volumes
    n_vol = len(vol_unique[1])
    logging.info('There are ' + str(n_vol) + ' volumes in the study')

    relevant_series = []
    relevant_volumes = []

    for i in range(len(vol_unique[1])):
        curr_vol = i
        info_idxs = np.where(vol_unique[2] == curr_vol)[0]
        vol_files = dcm_header_info[info_idxs, 2]
        # positions = np.asarray([np.asarray(x[2]) for x in dcm_header_info[info_idxs, 3]]) 
        # slicesort_idx = np.argsort(positions)
        # vol_files = vol_files[slicesort_idx]
        # relevant_series.append(vol_files)
        # reader = sitk.ImageSeriesReader()
        # reader.SetFileNames(vol_files)
        # vol = reader.Execute()
        # relevant_volumes.append(vol)  ###ori 1^ 

        if useSOPInstanceUID: 
            positions = np.asarray([np.asarray(x) for x in dcm_header_info[info_idxs, 3]])
            slicesort_idx = np.argsort(positions)
            vol_files = vol_files[slicesort_idx]
            vol_files = reversed(vol_files)
            relevant_series.append(vol_files)
            vol=[]
            for fname in vol_files: 
                ds = pyd.dcmread(fname, force=True)
                ds.file_meta.TransferSyntaxUID =pyd.uid.ImplicitVRLittleEndian
                PixelData = ds.PixelData
                pixel_array = np.frombuffer(PixelData, dtype=np.int16)
                pixel_array = np.reshape(pixel_array, (512,512))
                vol.append(pixel_array) 
            vol = np.stack(vol, axis=0) 
            vol_sitk = sitk.GetImageFromArray(vol) 
            voxel_spacing = [ float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]), float(ds.SliceThickness) ] 
            vol_sitk.SetSpacing(voxel_spacing ) 
            # vol_sitk.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1] ) 
            relevant_volumes.append(vol_sitk)

            # print()
        else: 
            positions = np.asarray([np.asarray(x[2]) for x in dcm_header_info[info_idxs, 3]]) 
            slicesort_idx = np.argsort(positions)
            vol_files = vol_files[slicesort_idx]
            relevant_series.append(vol_files)
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(vol_files)
            vol = reader.Execute()
            relevant_volumes.append(vol)


    return relevant_volumes


def get_input_image(path, path_label=None, trainORtest='test'):
    if os.path.isfile(path):
        logging.info(f'Read input: {path}')
        input_image = sitk.ReadImage(path)
    
    if trainORtest=='train': ## or os.path.isfile(path_label): 
        logging.info(f'Read input label : {path_label}')
        input_label=sitk.ReadImage(path_label)
    else:
        logging.info(f'Looking for dicoms in {path}')
        dicom_vols = read_dicoms(path, original=False, primary=False)
        if len(dicom_vols) < 1:
            sys.exit('No dicoms found!')
        if len(dicom_vols) > 1:
            logging.warning("There are more than one volume in the path, will take the largest one")
        input_image = dicom_vols[np.argmax([np.prod(v.GetSize()) for v in dicom_vols], axis=0)]  ##ori
        # try: 
        #     input_image = dicom_vols[np.argmax([np.prod(v.GetSize()) for v in dicom_vols], axis=0)]
        # except: 
        #     input_image = dicom_vols[0]
    
    if trainORtest=='train':
        return input_image, input_label
    elif trainORtest=='test': 
        return input_image


def postrocessing(label_image, spare=[]):
    '''some post-processing mapping small label patches to the neighbout whith which they share the
        largest border. All connected components smaller than min_area will be removed
    '''

    # merge small components to neighbours
    regionmask = skimage.measure.label(label_image)
    origlabels = np.unique(label_image)
    origlabels_maxsub = np.zeros((max(origlabels) + 1,), dtype=np.uint32)  # will hold the largest component for a label
    regions = skimage.measure.regionprops(regionmask, label_image)
    regions.sort(key=lambda x: x.area)
    regionlabels = [x.label for x in regions]

    # will hold mapping from regionlabels to original labels
    region_to_lobemap = np.zeros((len(regionlabels) + 1,), dtype=np.uint8)
    for r in regions:
        r_max_intensity = int(r.max_intensity)
        if r.area > origlabels_maxsub[r_max_intensity]:
            origlabels_maxsub[r_max_intensity] = r.area
            region_to_lobemap[r.label] = r_max_intensity

    for r in tqdm(regions):
        r_max_intensity = int(r.max_intensity)
        if (r.area < origlabels_maxsub[r_max_intensity] or r_max_intensity in spare) and r.area>2: # area>2 improves runtime because small areas 1 and 2 voxel will be ignored
            bb = bbox_3D(regionmask == r.label)
            sub = regionmask[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
            dil = ndimage.binary_dilation(sub == r.label)
            neighbours, counts = np.unique(sub[dil], return_counts=True)
            mapto = r.label
            maxmap = 0
            myarea = 0
            for ix, n in enumerate(neighbours):
                if n != 0 and n != r.label and counts[ix] > maxmap and n != spare:
                    maxmap = counts[ix]
                    mapto = n
                    myarea = r.area
            regionmask[regionmask == r.label] = mapto
            # print(str(region_to_lobemap[r.label]) + ' -> ' + str(region_to_lobemap[mapto])) # for debugging
            if regions[regionlabels.index(mapto)].area == origlabels_maxsub[
                int(regions[regionlabels.index(mapto)].max_intensity)]:
                origlabels_maxsub[int(regions[regionlabels.index(mapto)].max_intensity)] += myarea
            regions[regionlabels.index(mapto)].__dict__['_cache']['area'] += myarea

    outmask_mapped = region_to_lobemap[regionmask]
    outmask_mapped[outmask_mapped==spare] = 0 

    if outmask_mapped.shape[0] == 1:
        # holefiller = lambda x: ndimage.morphology.binary_fill_holes(x[0])[None, :, :] # This is bad for slices that show the liver
        holefiller = lambda x: skimage.morphology.area_closing(x[0].astype(int), area_threshold=64)[None, :, :] == 1
    else:
        holefiller = fill_voids.fill

    outmask = np.zeros(outmask_mapped.shape, dtype=np.uint8)
    for i in np.unique(outmask_mapped)[1:]:
        outmask[holefiller(keep_largest_connected_component(outmask_mapped == i))] = i

    return outmask


def bbox_3D(labelmap, margin=2):
    shape = labelmap.shape
    r = np.any(labelmap, axis=(1, 2))
    c = np.any(labelmap, axis=(0, 2))
    z = np.any(labelmap, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    rmin -= margin if rmin >= margin else rmin
    rmax += margin if rmax <= shape[0] - margin else rmax
    cmin, cmax = np.where(c)[0][[0, -1]]
    cmin -= margin if cmin >= margin else cmin
    cmax += margin if cmax <= shape[1] - margin else cmax
    zmin, zmax = np.where(z)[0][[0, -1]]
    zmin -= margin if zmin >= margin else zmin
    zmax += margin if zmax <= shape[2] - margin else zmax
    
    if rmax-rmin == 0:
        rmax = rmin+1

    return np.asarray([rmin, rmax, cmin, cmax, zmin, zmax])


def keep_largest_connected_component(mask):
    mask = skimage.measure.label(mask)
    regions = skimage.measure.regionprops(mask)
    resizes = np.asarray([x.area for x in regions])
    max_region = np.argsort(resizes)[-1] + 1
    mask = mask == max_region
    return mask


import glob
import pydicom as dicom
def traverse_dicom(rootdir = r'path/to/dicom', dirSaveCSV=None, whichCity=None):

    folder_list = []
    folder_1_list = []
    folder_1_list_bmp = []
    for subdir, dirs, files in os.walk(rootdir):
        if len(files) > 20: 
            # filepath = subdir + os.sep + file
            folder_list.append(subdir)
        for file in files:
            if len(files) > 20: 
                filepath = subdir + os.sep + file
                if filepath.endswith(".txt") or filepath.endswith("Thumbs.db") or filepath.endswith(".xml") \
                    or filepath.endswith(".ini") or filepath.endswith(".ct") or filepath.endswith(".slice") \
                    or filepath.endswith(".npj") or filepath.endswith(".bin") or filepath.endswith(".zip") \
                    or filepath.endswith(".sppc") :  
                    continue
                if filepath.endswith(".BMP") or filepath.endswith(".TIFF"):
                    folder_1_list_bmp.append(filepath)
                folder_1_list.append(filepath)
    folder_1_set_date = set([folder_1_list[i].rsplit('/',1)[0] for i in range(len(folder_1_list))])
    folder_1_set_pt = set([folder_1_list[i].rsplit('/',2)[0] for i in range(len(folder_1_list))])
    folder_1_set_positive = set([folder_1_list[i].rsplit('/',3)[0] for i in range(len(folder_1_list))])

    folder_1_set_date = sorted(folder_1_set_date)
    folder_1_set_pt = sorted(folder_1_set_pt)
    folder_1_set_positive = sorted(folder_1_set_positive)

    folder_1_set_bmp_date = set([folder_1_list_bmp[i].rsplit('/',1)[0] for i in range(len(folder_1_list_bmp))])
    folder_1_set_bmp_date = sorted(folder_1_set_bmp_date)

    folder_1_set_date_filter = []
    outputpath_list_filter = []
    folders=[]
    pt_info_list_l = []
    aaa=[]
    for i in range(len(folder_1_set_date)): 
        f_name = folder_1_set_date[i].split('/')

        ##check if the data included  'Lung 5.0  CE'
        data_addrs = glob.glob(folder_1_set_date[i] + r'/*.dic')  ##
        if data_addrs == []:
            data_addrs = glob.glob(folder_1_set_date[i]+ r'/*.DCM')
        if data_addrs == []:
            data_addrs = glob.glob(folder_1_set_date[i]+ r'/*.dcm')

        data_addrs = sorted(data_addrs, key=lambda s: s.lower())
        slices = [dicom.read_file(s, force=True) for s in data_addrs]
        SeriesDescription_list = [slices[s_i].SeriesDescription for s_i in range(len(slices))]

        SeriesDescription_list_set = list(set(SeriesDescription_list))

        ##for 104 cases and 485 cases
        if ' Lung 5.0  CE' in SeriesDescription_list_set or 'Lung 5.0  CE' in SeriesDescription_list_set:
            folders.append(f_name[-3:]) 
            outputpath_list_filter.append(f_name[-3] + '_' + f_name[-2] + '_' + f_name[-1] + '.nii.gz') 

        else:
            folders.append(f_name[-3:]+['dont have Lung 5.0  CE'])    ##for 104 cases
            print(folder_1_set_date[i])
            print(set(SeriesDescription_list))

    if dirSaveCSV is not None: 
        import pandas as pd
        max_num = max( [len(f) for f in folders] )
        if max_num==4: 
            df = pd.DataFrame(folders, columns=['folder_name','pt_name','serials', 'NOTE'])
        else: 
            df = pd.DataFrame(folders, columns=['folder_name','pt_name','serials', ])
        df.to_csv(dirSaveCSV, index=False) 

    return folder_1_set_date_filter, outputpath_list_filter

def draw_TSNE(input, n_clustes=4, finalDF=None):
    from sklearn.manifold import TSNE
    import pandas as pd
    tsne = TSNE()
    tsne.fit_transform(input)
    tsne = pd.DataFrame(tsne.embedding_)
    import matplotlib.pyplot as plt
    plt.rcParams['axes.unicode_minus'] = False
    colors = ['r.', 'go','b*', 'c+']
    for i in range(n_clustes):
        d = tsne[finalDF['Cluster category'] == i]
        plt.plot(d[0],d[1], colors[i])
    plt.show()
    
from PIL import Image as pil_image
def array_to_img(x, scale=True):
    # target PIL image has format (height, width, channel)    (512,512,1)
    x = np.asarray(x, dtype=float)
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image).'
                         'Got array with shape', x.shape)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number:', x.shape[2])

if __name__ == "__main__":
    datapath = r'/media/bj/DataFolder3/Kashi/1 data_100'
    folder_1_set_date_filter, outputpath_list = traverse_dicom(rootdir = datapath)  ##'list_190.csv'
    ## auto-segmentation  
    for i in range(len(folder_1_set_date_filter)): 
        inputpath = folder_1_set_date_filter[i]
        output = r'./results_100/' + outputpath_list[i]
        print('finished: ', i )
        main(inputpath, outputputh=output, modeltype='unet', modelname='R231', modelpath='./unet_r231-d5d2fc3d.pth', 
                classes=3, cpu=False, nopostprocess=False, noHU=False, batchsize=20, trainORtest='test')
    