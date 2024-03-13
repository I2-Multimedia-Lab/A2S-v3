import sys
import importlib
from data import Test_Dataset, get_test_list
#from data_esod import ESOD_Test
import torch
import time
from progress.bar import Bar
import os
from collections import OrderedDict
import cv2
from PIL import Image
from util import *
import numpy as np
from torch import nn
import torch.utils.data as data
from base.framework_factory import load_framework
from metric import *

def test_model(model, test_sets, config, epoch=None, saver=None, mode = 'test'):
    print(mode)
    stage = config['stage']
    model.eval()
    if epoch is not None:
        weight_path = os.path.join(config['weight_path'], '{}_{}_{}.pth'.format(config['model_name'], config['sub'], epoch))
        torch.save(model.state_dict(), weight_path)
    
    st = time.time()
    for set_name, test_set in test_sets.items():
        save_folder = os.path.join(config['save_path'], set_name)
        check_path(save_folder)
        
        titer = test_set.size
        MR = MetricRecorder(titer)
        ious = []
        dises = []
        if mode == 'train':
            test_set.mode = 'train'
            data_loader = data.DataLoader(dataset=test_set,
                                  batch_size=config['batch'],
                                  shuffle=False,
                                  num_workers=1,
                                  pin_memory=True,
                                  drop_last=False)
        else:
            test_set.mode = 'test'
            data_loader = data.DataLoader(dataset=test_set,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=1,
                                  pin_memory=True,
                                  drop_last=False)
        test_bar = Bar('Dataset {:10}:'.format(set_name), max=len(data_loader))
        for i, pack in enumerate(data_loader, start=1):
            
            images = pack['image'].float()
            modals = pack['modal'].float()
            gt = pack['gt']
            name = pack['name']

            images, modals = images.cuda(), modals.cuda()
            
            priors = [images]
    
            priors.append(modals.cuda())
            #print(priors[0].shape,priors[1].shape)
            out_shape = gt.shape[-2:]
            if config['net_name'] == 'a2s':
                Y = model(images)
            else:
                Y = model(priors)
            
            pred = Y['final'].sigmoid_().cpu().data.numpy()#[0, 0]
            #print(pred.shape,gt.shape)
            gt = np.asarray((gt > 0.5)).astype(np.float32)

            if mode == 'train':
                for i in range(0,pred.shape[0]):
                    p, g = pred[i][0],gt[i]
                    p = np.clip(np.round(cv2.resize(p, out_shape[::-1]) * 255) / 255., 0, 1)
                    #print(p.shape,g.shape)
                    MR.update(pre=p, gt=g)
                    g = (g > 0.5).astype(np.float32)
                    iou = cal_iou(p, g)
                    ious.append(iou)
                    dis = cal_dis(p, g)
                    dises.append(dis)
            else:
                p, g = pred[0][0],gt[0]
                last = config['stage'] == 1 and epoch == config['epoch']
                if config['crf'] or last:
                    p = (p * 255).astype(np.uint8)
                    thre, p = cv2.threshold(p, 0, 255, cv2.THRESH_OTSU)
                    p, g = normalize_pil(p, g)
                    
                    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
                    std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
                    orig_img = images[0].cpu().numpy().transpose(1, 2, 0)
                    orig_img = ((orig_img * std + mean) * 255.).astype(np.uint8)
                    
                    p = (p > 0.5).astype(np.uint8)
                    #print(orig_img.shape,p.shape)
                    p = crf_inference_label(orig_img, p)
                    p = cv2.medianBlur(p.astype(np.uint8), 7)
                            
                p = np.clip(np.round(cv2.resize(p, out_shape[::-1]) * 255) / 255., 0, 1)
                MR.update(pre=p, gt=g)
                g = (g > 0.5).astype(np.float32)
                iou = cal_iou(p, g)
                ious.append(iou)
                dis = cal_dis(p, g)
                dises.append(dis)

            pred = p
            
            
            # save predictions
            if config['save'] or epoch == config['epoch']:
                if config['crf']:
                    tag = 'crf'
                else:
                    tag = 'final'
                fnl_folder = os.path.join(save_folder, tag)
                check_path(fnl_folder)
                im_path = os.path.join(fnl_folder, name[0].split('.')[0] + '.png')
                
                tlist = im_path.split('/')
                shufix = tlist[:-1]
                img_name = tlist[-1]
                new_path = '/'.join(shufix)
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                
                Image.fromarray((pred * 255)).convert('L').save(im_path)
                
                if saver is not None:
                    saver(Y, gt, name, save_folder, config)
                    pass
                
            Bar.suffix = '{}/{}'.format(i, titer)
            test_bar.next()
        
        mae, (maxf, meanf, *_), sm, em, wfm = MR.show(bit_num=3)
        print(' {}: Mean-F: {}, EM: {}, MAE: {:.3f}, IOU: {:.3f}, dis: {:.3f}.'.format(set_name, meanf, em, round(mae, 3), np.mean(ious), np.mean(dises)))
        
    print('Test using time: {}.'.format(round(time.time() - st, 3)))

def main():
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        print('Need model name!')
        return
    
    config, model, _, _, _, saver = load_framework(net_name)
    config['net_name'] = net_name
    
    if config['crf']:
        config['orig_size'] = True
    
    if config['weight'] != '':
        model.load_state_dict(torch.load(config['weight'], map_location='cpu'))
    else:
        print('No weight file provide!')

    test_sets = get_test_list(config['vals'], config)
    model = model.cuda()
    test_model(model, test_sets, config, saver=saver)
        
if __name__ == "__main__":
    main()