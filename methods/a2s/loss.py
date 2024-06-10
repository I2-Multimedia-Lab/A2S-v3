import torch
import torch.nn.functional as F
from util import *
from math import exp, pow
import numpy as np
from PIL import Image

def get_contour(label):
    lbl = label.gt(0.5).float()
    ero = 1 - F.max_pool2d(1 - lbl, kernel_size=5, stride=1, padding=2)  # erosion
    dil = F.max_pool2d(lbl, kernel_size=5, stride=1, padding=2)            # dilation
    
    edge = dil - ero
    return edge

# Boundary-aware Texture Matching Loss
def BTMLoss(pred, image, radius, config=None):
        alpha = config['rgb']
        modal = config['trset']
        num_modal = len(modal) if 'c' in modal else len(modal)+1
        if 'r' in modal: num_modal = num_modal-1
        slices = range(0, 3*num_modal+1, 3)
        sal_map =  F.interpolate(pred, scale_factor=0.25, mode='bilinear', align_corners=True)
        image_ = F.interpolate(image, size=sal_map.shape[-2:], mode='bilinear', align_corners=True)
        mask = get_contour(sal_map)
        features = torch.cat([image_, sal_map], dim=1) # mask is in the last column of features 
        N, C, H, W = features.shape
        diameter = 2 * radius + 1 # sample range. The values referred to by diameter and radius in Figure 5 of the original text are 3 and 1, respectively. 
        # For each prediction point in the mask and image, take (diameter^2-1) points around it as candidate points, and calculate the distance between the center point and the candidate points. 
        kernels = F.unfold(features, diameter, 1, radius).view(N, C, diameter, diameter, H, W) 
        kernels = kernels - kernels[:, :, radius, radius, :, :].view(N, C, 1, 1, H, W)
        dis_modal = 1
        for idx, slice in enumerate(slices):
            # Traverse each modality information. 
            if idx == len(slices) - 1:
                continue
            # When the center point in the image is close to the candidate point, The corresponding value in the dis_map is close to 1; On the contrary, the corresponding value in the dis_map is close to 0. 
            # Therefore, loss only takes effect when the center point in the mask deviates from the candidate point, and the center point in the corresponding position in the image tends to be consistent with the candidate point. 
            dis_map = (-alpha * kernels[:, slice:slices[idx+1]] ** 2).sum(dim=1, keepdim=True).exp()
            # Only RGB
            if config['only_rgb'] and idx > 0:
                dis_map = dis_map * 0 + 1
            dis_modal = dis_modal * dis_map
            
        dis_sal = torch.abs(kernels[:, slices[-1]:]) # Take the mask from the last column 
        distance = dis_modal * dis_sal

        loss = distance.view(N, 1, (radius * 2 + 1) ** 2, H, W).sum(dim=2)
        loss = torch.sum(loss * mask) / torch.sum(mask)
        return loss

# Confidenceaware Saliency Distilling Loss
def CSDloss(pred, feat, mask=False, epoch=1, config=None, name=None):
    mul = 2 ** (1 - (epoch - 1) / config['epoch'])
    loss_map = torch.abs(pred - 0.5)
    w = torch.ones_like(pred)
    if(mask):
        p = 0.2-0.6*(epoch/config['epoch'])
        mask = (loss_map<p).bool()
        w.masked_fill_(mask,0)
        #loss_map.masked_fill_(mask,0.5)
        
    loss_map = torch.pow(loss_map, mul)

    # The pow(0.5, mul) is used to keep the loss greater than 0. It has no impact on the training process.
    loss = pow(0.5, mul) - loss_map#.mean() 
    return (loss*w).mean()

def Loss(pre1, img1, pre2, img2, epoch, ws, config, name):
    sal1 = pre1['sal'][0]
    sal2 = pre2['sal'][0]
    
    p1 = torch.sigmoid(sal1)
    p2 = torch.sigmoid(sal2)
    
    adb_loss = CSDloss(p1, pre1['feat'][0], config['pcl'], epoch, config, name) + CSDloss(p2, pre2['feat'][0], config['pcl'], epoch, config, name)
    
    if ws[1] > 0:
        ac_loss = BTMLoss(p1, img1, 5, config) + BTMLoss(p2, img2, 5, config)
    else:
        ac_loss = 0
    
    p2 = F.interpolate(p2, size=p1.size()[2:], mode='bilinear', align_corners=True)
    mse_loss = torch.mean(torch.pow(p1 - p2, 2))
    
    return adb_loss * ws[0], ac_loss * ws[1], mse_loss * ws[2]
