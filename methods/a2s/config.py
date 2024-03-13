import sys
import argparse  
import os
from base.config import base_config, cfg_convert


def get_config():
    # Default configure
    cfg_dict = {
        'optim': 'SGD',
        'schedule': 'StepLR',
        'lr': 0.1,
        'batch': 8,
        'ave_batch': 1,
        'epoch': 20,
        'step_size': '12,16',
        'gamma': 0.1,
        'clip_gradient': 0,
        'test_batch': 1,
    }
    
    parser = base_config(cfg_dict)
    # Add custom params here
    parser.add_argument('--ac', default=0.05, type=float)
    parser.add_argument('--rgb', default=200, type=float)
    parser.add_argument('--lrw', type=str, default='1,0.05,1')
    parser.add_argument('--temp', default='temp1', help='Job name')
    parser.add_argument('--refine', action='store_true')

    params = parser.parse_args()
    config = vars(params)
    cfg_convert(config)
    print('Training {} network with {} backbone using Gpu: {}'.format(config['model_name'], config['backbone'], config['gpus']))
    
    # Config post-process
    if config['finetune']:
        config['params'] = [['encoder', 0], ['decoder', 0], ['adapter', config['lr']]]
        if config['weight'] == '':
            config['weight'] = config['load'] + 'a2s/base.pth'
        config['pcl'] = False
    else:
        config['params'] = [['encoder', 0], ['decoder', config['lr']]]
    config['lr_decay'] = 0.9
    
    return config, None