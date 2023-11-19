import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def resnet3d_pretrained(num_classes=1):
    model = models.video.r3d_18(pretrained=True)
    model.fc = nn.Linear(512, out_features=num_classes)
    # model_feat = torch.nn.Sequential(*list(model.children())[:-2]) 
    model_feat = torch.nn.Sequential(*list(model.children())[:-1]) 
    return model_feat
    # return model 

def r2plus1d_18_pretrained(num_classes=1):
    model = models.video.r2plus1d_18(pretrained=True)
    model.fc = nn.Linear(512, out_features=num_classes)
    # model_feat = torch.nn.Sequential(*list(model.children())[:-2]) 
    model_feat = torch.nn.Sequential(*list(model.children())[:-1]) 
    return model_feat 

def mc3_18_pretrained(num_classes=1):
    model = models.video.mc3_18(pretrained=True)
    model.fc = nn.Linear(512, out_features=num_classes)
    model_feat = torch.nn.Sequential(*list(model.children())[:-1]) 
    # feat_classifier = torch.nn.Sequential(*list(model.children())[-2:]) 
    return model_feat

def mvit_v1_b_pretrained(num_classes=1):
    model = models.video.mvit_v1_b(pretrained=True)
    model.fc = nn.Linear(512, out_features=num_classes)
    return model 

def mvit_v2_s_pretrained(num_classes=1):
    model = models.video.mvit_v2_s(pretrained=True)
    model.fc = nn.Linear(512, out_features=num_classes)
    return model 

def s3d_pretrained(num_classes=1):
    model = models.video.s3d(pretrained=True)
    model.classifier = nn.Linear(512, out_features=num_classes)
    return model 

def swin3d_b_pretrained(num_classes=1):
    model = models.video.swin3d_b(pretrained=True)
    model.fc = nn.Linear(512, out_features=num_classes)
    return model 

def swin3d_s_pretrained(num_classes=1):
    model = models.video.swin3d_s(pretrained=True)
    model.fc = nn.Linear(512, out_features=num_classes)
    return model 

def swin3d_t_pretrained(num_classes=1):
    model = models.video.swin3d_t(pretrained=True)
    model.fc = nn.Linear(512, out_features=num_classes)
    return model 

def generate_model(backbone, num_classes=1, args=None): 

    if backbone == 'resnet3d':
        model = resnet3d_pretrained(num_classes=num_classes )
    elif backbone == 'r2plus1d_18':
        model = r2plus1d_18_pretrained(num_classes=num_classes )
    elif backbone == 'mc3_18':
        model = mc3_18_pretrained(num_classes=num_classes)
    elif backbone == 'mvit_v1_b':
        model = mvit_v1_b_pretrained(num_classes=num_classes)
    elif backbone == 'mvit_v2_s':
        model = mvit_v2_s_pretrained(num_classes=num_classes) 
    elif backbone == 's3d': 
        model = s3d_pretrained(num_classes)  ##att_type='CA'
    elif backbone == 'swin3d_b': 
        model = swin3d_b_pretrained(num_classes)  ##att_type='CA'
    elif backbone == 'swin3d_s': 
        model = swin3d_s_pretrained(num_classes)  ##att_type='CA'
    elif backbone == 'swin3d_t': 
        model = swin3d_t_pretrained(num_classes)  ##att_type='CA'
    else:
        raise ValueError('% model does not supported!'%backbone) 

    return model  

if __name__ == '__main__': 
    resnet3d_pretrained()
    r2plus1d_18_pretrained( )
    mc3_18_pretrained()
    mvit_v1_b_pretrained()
    mvit_v2_s_pretrained() 
    s3d_pretrained()  ##att_type='CA'
    swin3d_b_pretrained()  ##att_type='CA'
    swin3d_s_pretrained()  ##att_type='CA'
    swin3d_t_pretrained()  ##att_type='CA'