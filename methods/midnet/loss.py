import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
from util import *

import numpy as np
from math import exp

def IOU(pred, target):
    inter = target * pred
    union = target + pred - target * pred
    iou_loss = 1 - torch.sum(inter, dim=(1, 2, 3)) / (torch.sum(union, dim=(1, 2, 3)) + 1e-7)
    return iou_loss.mean()

def iou_loss(pred, gt):
    target = gt.float()
    iou_out = IOU(pred, target)
    return iou_out

def Loss(preds1, target1, preds2, target2, preds_t, config):
    iou = 0
    bce = 0
    ws = [1, 0.2]
    
    for pred, w in zip(preds1['sal'], ws):
        pred = nn.functional.interpolate(pred, size=target1.size()[-2:], mode='bilinear')
        iou += iou_loss(torch.sigmoid(pred), target1) * w
        #bce += F.binary_cross_entropy_with_logits(pred,target1) * w
    
    for pred, w in zip(preds2['sal'], ws):
        pred = nn.functional.interpolate(pred, size=target2.size()[-2:], mode='bilinear')
        iou += iou_loss(torch.sigmoid(pred), target2) * (w/2)
       # #bce += F.binary_cross_entropy_with_logits(pred,target2) * w

    mse = torch.mean(torch.pow(torch.sigmoid(preds2['final']) - torch.sigmoid(preds_t), 2))
    return iou, bce, mse