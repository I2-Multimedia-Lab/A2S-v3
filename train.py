import sys
import os
import time
import random
import cv2
from math import exp
from par import PAR
from progress.bar import Bar
from collections import OrderedDict
from util import *
from PIL import Image
from data import Train_Dataset, Test_Dataset, get_loader, get_test_list
from test import test_model
import torch
from torch.nn import utils
from base.framework_factory import load_framework

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

torch.set_printoptions(precision=5)

def main():
    # Loading model
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        print('Need model name!')
        return
        
    # Loading model
    config, model, optim, sche, model_loss, saver = load_framework(net_name)
    par = PAR(dilations=[1,2,4,8],num_iter=6)
    par.cuda()
    config['net_name'] = net_name
    stage = config['stage']
    
    if config['weight'] != '':
        print('Load weights from: {}.'.format(config['weight']))
        model.load_state_dict(torch.load(config['weight'], map_location='cpu'),strict=False)
        
    train_loader = get_loader(config)
    
    test_sets = get_test_list(config['vals'], config)
    
    debug = config['debug']
    num_epoch = config['epoch']
    num_iter = len(train_loader)
    ave_batch = config['ave_batch']
    #batch = ave_batch * config['batch']
    trset = config['trset']
    batch_idx = 0
    model.zero_grad()
    for epoch in range(1, num_epoch + 1):
        model.train()
        torch.cuda.empty_cache()
        
        if debug:
            test_model(model, test_sets, config, epoch)
        
        bar = Bar('{:10}-{:8} | epoch {:2}:'.format(net_name, config['sub'], epoch), max=num_iter)
        
        st = time.time()
        loss_count, adb_count, ac_count, mse_count = 0, 0, 0, 0
        optim.zero_grad()
        
        fin_lr = 0.2
        for i, pack in enumerate(train_loader, start=1):
            cur_it = i + (epoch-1) * num_iter
            total_it = num_epoch * num_iter
            itr = (1 - cur_it / total_it) * (1 - fin_lr) + fin_lr
            mul = itr
            
            if stage == 1:
                tune = 2 if 'MSB-TR' in pack['name'][0] else 20
                if epoch > tune:
                    optim.param_groups[0]['lr'] = config['lr'] * mul * 0.01
                else:
                    optim.param_groups[0]['lr'] = 0
                
                optim.param_groups[1]['lr'] = config['lr'] * mul
            else:
                if epoch == 1:
                    optim.param_groups[0]['lr'] = 0
                else:
                    optim.param_groups[0]['lr'] = config['lr'] * mul * 0.1
                optim.param_groups[1]['lr'] = config['lr'] * mul
                
            if config['finetune']:
                optim.param_groups[0]['lr'] = 0
                optim.param_groups[1]['lr'] = 0
                optim.param_groups[2]['lr'] = config['lr'] * mul
                    
            images = pack['image'].float()
            gts = pack['gt'].float()
            modals = pack['modal'].float()
            gt_names = pack['name']
            flips = pack['flip']
            
            images, gts, modals = images.cuda(), gts.cuda(), modals.cuda()
            
            priors = [images]

            #priors.append(images)
            loss = 0
            if stage == 1:
                if 'dep' in pack.keys():
                    priors.append(pack['dep'].float().cuda())
                if 'of' in pack.keys():
                    priors.append(pack['of'].float().cuda())
                if 'th' in pack.keys():
                    priors.append(pack['th'].float().cuda())
                Y = model(images, 'train')

                config['param'] = tran_param(config)
                images_temp = transform(images, False, config)
                priors = torch.cat(priors, dim=1)
                priors_temp = transform(priors, False, config)
                
                Y_ref = model(images_temp, 'train')
                
                lr_weight = np.array(config['lrw'].split(',')).astype(np.float32)
                if lr_weight is None or len(lr_weight) != 3:
                    lr_weight = [0.5, 0.05, 1]
                
                loss0, loss1, loss2 = model_loss(Y, priors, Y_ref, priors_temp, epoch, lr_weight, config, gt_names)
                loss += loss0 + loss1 + loss2
                    
                ac_count += loss1
                mse_count += loss2
                
            elif stage > 1:
                priors.append(modals)
                Y = model(priors, 'train')

                config['param'] = tran_param(config)
                images_t = transform(images, False, config)
                modals_t = transform(modals, False, config)
                gts_t = transform(gts, True, config)
                Y_t = transform(Y['final'].detach(), False, config)
                
                priors_t = [images_t]
                priors_t.append(modals_t)
                
                Y_ref = model(priors_t, 'train')
                #print(images.shape,images_t.shape)
                #print(Y['final'].shape,Y_ref['final'].shape,Y_t.shape)
                loss0, loss1, loss2 = model_loss(Y, gts.gt(0.5).float(), Y_ref, gts_t.gt(0.5).float(), Y_t, config)
                loss = loss0+loss1+loss2
                ac_count += loss1
                mse_count += loss2
            loss_count += loss.data
            adb_count += loss0
            loss.backward()

            batch_idx += 1
            if batch_idx == ave_batch:
                if config['clip_gradient']:
                    utils.clip_grad_norm_(model.parameters(), config['clip_gradient'])
                optim.step()
                optim.zero_grad()
                batch_idx = 0
            
            lrs = ','.join([format(param['lr'], ".2e") for param in optim.param_groups])
            if stage == 1:
                Bar.suffix = '{:4}/{:4} | loss: {:1.3f}, csd: {:1.3f}, btm: {:1.3f}, mse: {:1.3f}, LRs: [{}], time: {:1.3f}.'.format(
                    i, num_iter, float(loss_count / i), float(adb_count / i), float(ac_count / i), float(mse_count / i), lrs, time.time() - st)
            else:
                Bar.suffix = '{:4}/{:4} | loss: {:1.3f}, iou: {:1.3f}, bce: {:1.3f}, mse: {:1.3f}, LRs: [{}], time: {:1.3f}.'.format(i, num_iter, float(loss_count / i), float(adb_count / i), float(ac_count / i), float(mse_count / i), lrs, time.time() - st)
            bar.next()
            
            if epoch > 1 and stage > 1 and config['olr']:
                lamda = [0.7,0.2,0.1]
                for gt_path, image, pred, gt, flip in zip(gt_names, images, torch.sigmoid(Y['final'].detach()), gts, flips):
                    pred = F.interpolate(pred.unsqueeze(0), size=gt.size()[1:], mode='bilinear', align_corners=True)[0]
                    #print(pred.shape,image.shape)
                    ref = par(image.unsqueeze(0),pred.unsqueeze(0)).squeeze(0)
                    ref = ref/ref.max()
                    #print(ref.shape,pred.shape,gt.shape)
                    if flip:
                        pred = pred.flip(2)
                        ref = ref.flip(2)
                        gt = gt.flip(2)
                    new_gt = (pred * lamda[0]).cpu().numpy().transpose(1, 2, 0)+(gt * lamda[1]).cpu().numpy().transpose(1, 2, 0)+(ref*lamda[2]).cpu().numpy().transpose(1, 2, 0)
                    new_gt = ((new_gt/new_gt.max())).astype(np.float32)
                    #print(new_gt.shape)
                    cv2.imwrite(gt_path, new_gt * 255)
                    #print(gt_path)
                    #print(gt_path.split('/'))
                    #print('./pseudo/spr_/ori/'+gt_path.split('/')[-1])
                    if epoch == 10:
                        cv2.imwrite('./pseudo/spr_/ori/'+gt_path.split('/')[-1], gt.cpu().numpy().transpose(1, 2, 0)*255)
                        cv2.imwrite('./pseudo/spr_/par/'+gt_path.split('/')[-1], ref.cpu().numpy().transpose(1, 2, 0)*255)
                        cv2.imwrite('./pseudo/spr_/sal/'+gt_path.split('/')[-1], pred.cpu().numpy().transpose(1, 2, 0)*255)
                        cv2.imwrite('./pseudo/spr_/ref/'+gt_path.split('/')[-1], new_gt * 255)
                
        sche.step()
        bar.finish()
        #torch.cuda.empty_cache()
        i = len(train_loader)
        print('| loss: {:1.3f}, dfs: {:1.3f}, bac: {:1.3f}, mse: {:1.3f}, LRs: [{}], time: {:1.3f}.'.format(
                    float(loss_count / i), float(adb_count / i), float(ac_count / i), float(mse_count / i), lrs, time.time() - st))
        
        #if num_epoch-epoch<5:
        if epoch != config['epoch']:
            test_model(model, test_sets, config, epoch, mode='train')
        else:
            test_model(model, test_sets, config, epoch, mode='test')

if __name__ == "__main__":
    main()