import os, random
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torch
from shutil import copyfile, copy
from collections import OrderedDict
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])

sub = 'split' # 'joint'

class random_image_enhance:
    def __init__(self, methods=['contrast', 'brightness', 'sharpness']):
        self.enhance_method = []
        if 'contrast' in methods:
            self.enhance_method.append(ImageEnhance.Contrast)
        if 'brightness' in methods:
            self.enhance_method.append(ImageEnhance.Brightness)
        if 'sharpness' in methods:
            self.enhance_method.append(ImageEnhance.Sharpness)

    def __call__(self, sample):
        np.random.shuffle(self.enhance_method)

        for method in self.enhance_method:
            if np.random.random() > 0.5:
                enhancer = method(sample)
                factor = float(1 + np.random.random() / 10)
                sample = enhancer.enhance(factor)

        return sample
    
def copydirs(from_file, to_file):
    if not os.path.exists(to_file):
        os.makedirs(to_file)
    files = os.listdir(from_file)
    for f in files:
        if os.path.isdir(from_file + '/' + f):
            copydirs(from_file + '/' + f, to_file + '/' + f)
        else:
            copy(from_file + '/' + f, to_file + '/' + f)

def OLR(gt_root, name):
    gt_orig = gt_root
    gt_root = os.path.join('./pseudo', 'temp2', name)
    
    print("Copying labels to temp folder: {}.".format(gt_root))
    copydirs(gt_orig, gt_root)
    print('Using temp labels from {}'.format(gt_root))
    return gt_orig, gt_root
    

def get_color_list(name, config, phase):
    name_list = []
    image_root = os.path.join(config['data_path'], name, 'RGB')
    img_list = os.listdir(image_root)
    
    if config['stage'] > 1 and phase == 'train':
        gt_root = './pseudo/c/{}'.format(name)
        print('Using pseudo labels from {}'.format(gt_root))

        if config['olr']:
            gt_orig, gt_root = OLR(gt_root, name)
    else:
        gt_root = os.path.join(config['data_path'], name, 'GT')
    
    for img_name in img_list:
        img_tag = img_name.split('.')[0]
        
        tag_dict = {}
        tag_dict['rgb'] = os.path.join(image_root, img_name)
        tag_dict['gt'] = os.path.join(gt_root, img_tag + '.png')
        name_list.append(tag_dict)
    
    return name_list

def get_rgbd_list(name, config, phase):
    name_list = []
    image_root = os.path.join(config['data_path'], '{}/RGB'.format(name))
    dep_root = os.path.join(config['data_path'], '{}/depth'.format(name))
    
    if config['stage'] > 1 and phase == 'train':
        gt_root = './pseudo/d/{}'.format(name)
        print('Using pseudo labels from {}'.format(gt_root))
        
        if config['olr']:
            gt_orig, gt_root = OLR(gt_root, name)
    else:
        gt_root = os.path.join(config['data_path'], '{}/GT'.format(name))
    
    img_list = os.listdir(image_root)
    for img_name in img_list:
        img_tag = img_name.split('.')[0]
        
        tag_dict = {}
        tag_dict['rgb'] = os.path.join(image_root, img_name)
        tag_dict['gt'] = os.path.join(gt_root, img_tag + '.png')
        if os.path.exists(os.path.join(dep_root, img_tag + '.jpg')):
            tag_dict['dep'] = os.path.join(dep_root, img_tag + '.jpg')
        elif os.path.exists(os.path.join(dep_root, img_tag + '.png')):
            tag_dict['dep'] = os.path.join(dep_root, img_tag + '.png')
        elif os.path.exists(os.path.join(dep_root, img_tag + '.bmp')):
            tag_dict['dep'] = os.path.join(dep_root, img_tag + '.bmp')
        name_list.append(tag_dict)
    
    return name_list
    
def get_rgbt_list(name, config, phase):
    name_list = []
    image_root = os.path.join(config['data_path'], '{}/RGB'.format(name))
    th_root = os.path.join(config['data_path'], '{}/T'.format(name))
    
    if config['stage'] > 1 and phase == 'train':
        gt_root = './pseudo/t/{}'.format(name)
        print('Using pseudo labels from {}'.format(gt_root))
        
        if config['olr']:
            gt_orig, gt_root = OLR(gt_root, name)
    else:
        gt_root = os.path.join(config['data_path'], '{}/GT'.format(name))
    
    img_list = os.listdir(image_root)
    for img_name in img_list:
        img_tag = img_name.split('.')[0]
        
        tag_dict = {}
        tag_dict['rgb'] = os.path.join(image_root, img_name)
        if os.path.exists(os.path.join(gt_root, img_tag + '.jpg')):
            tag_dict['gt'] = os.path.join(gt_root, img_tag + '.jpg')
        elif os.path.exists(os.path.join(gt_root, img_tag + '.png')):
            tag_dict['gt'] = os.path.join(gt_root, img_tag + '.png')
        if os.path.exists(os.path.join(th_root, img_tag + '.jpg')):
            tag_dict['th'] = os.path.join(th_root, img_tag + '.jpg')
        elif os.path.exists(os.path.join(th_root, img_tag + '.png')):
            tag_dict['th'] = os.path.join(th_root, img_tag + '.png')
        elif os.path.exists(os.path.join(th_root, img_tag + '.bmp')):
            tag_dict['th'] = os.path.join(th_root, img_tag + '.bmp')

        name_list.append(tag_dict)
    
    return name_list
    

def get_frame_list(name, config, phase):
    name_list = []
    
    base_path = os.path.join(config['data_path'], 'VSOD', name)
    videos = os.listdir(os.path.join(base_path, 'JPEGImages'))
        
    if config['stage'] > 1 and phase == 'train':
        gt_base = './pseudo/o-joint/{}'.format(name)
        print('Using pseudo labels from {}'.format(gt_base))
    
        if config['olr']:
            gt_orig, gt_base = OLR(gt_base, name)
    else:
        gt_base = os.path.join(base_path, 'Annotations')
    
    
    for video in videos:
        image_root = os.path.join(base_path, 'JPEGImages', video)
        gt_root = os.path.join(gt_base, video)
        of_root = os.path.join(base_path, 'optical', video)
        
        img_list = os.listdir(image_root)
        img_list = sorted(img_list)
        if phase == 'train' and 'select' in video:
            img_list = img_list[::5]

        for img_name in img_list:
            img_tag = img_name.split('.')[0]
            
            tag_dict = {}
            tag_dict['rgb'] = os.path.join(image_root, img_name)
            tag_dict['gt'] = os.path.join(gt_root, img_tag + '.png')
            tag_dict['of'] = os.path.join(of_root, img_tag + '.jpg')
            name_list.append(tag_dict)
         
    return name_list

def get_remote_list(name, config, phase):
    name_list = []

    base_path = os.path.join(config['data_path'], 'ORSSD')
    image_root = os.path.join(base_path, name, 'RGB')
    img_list = os.listdir(image_root)
    
    if config['stage'] > 1 and phase == 'train':
        gt_root = './pseudo/r/{}'.format(name)
        print('Using pseudo labels from {}'.format(gt_root))

        if config['olr']:
            gt_orig, gt_root = OLR(gt_root, name)
    else:
        gt_root = os.path.join(base_path, name, 'GT')
    
    for img_name in img_list:
        img_tag = img_name.split('.')[0]
        
        tag_dict = {}
        tag_dict['rgb'] = os.path.join(image_root, img_name)
        tag_dict['gt'] = os.path.join(gt_root, img_tag + '.png')
        name_list.append(tag_dict)
    
    return name_list

def get_train_image_list(names, config):
    image_list = []
    phase = 'train'
    ccc, ddd, ooo, ttt, rrr = [], [], [], [], []
    if 'c' in names:
        ccc = get_color_list('DUTS-TR', config, phase)
        #ccc = get_color_list('MSB-TR', config, phase)
        if config['reduce_data']:
            l = int(len(ccc)*config['reduce_rate'])
            ccc = ccc[0:l]
        image_list += ccc
    if 'd' in names:
        ddd = get_rgbd_list('RGBD-TR', config, phase)
        if config['reduce_data']:
            l = int(len(ddd)*config['reduce_rate'])
            ddd = ddd[0:l]
        image_list += ddd
    if 'o' in names:
        ooo = get_frame_list('VSOD-TR', config, phase)
        if config['reduce_data']:
            l = int(len(ooo)*config['reduce_rate'])
            ooo = ooo[0:l]
        image_list += ooo
    if 't' in names:
        ttt = get_rgbt_list('VT5000-TR', config, phase)
        if config['reduce_data']:
            l = int(len(ttt)*config['reduce_rate'])
            ttt = ttt[0:l]
        image_list += ttt
    if 'r' in names:
        rrr = get_remote_list('RSSD-TR', config, phase)
        if config['reduce_data']:
            l = int(len(rrr)*config['reduce_rate'])
            rrr = rrr[0:l]
        image_list += rrr
    print('Loading {} images for {}: RGB({}), RGBD({}), VSOD({}), RGBT({}), ORSSD({}).'.format(len(image_list), phase, len(ccc), len(ddd), len(ooo), len(ttt), len(rrr)))
    return image_list

def get_test_list(modes='cr', config=None):
    test_dataset = OrderedDict()
    
    for mode in modes:
        modal, subset = mode
        if subset == 'e':
            if modal == 'c':
                test_list = ['DUT-O', 'DUTS-TE', 'ECSSD', 'HKU-IS', 'PASCAL-S']#, 'SOD']
            elif modal == 'd':
                #test_list = ['DUT', 'LFSD', 'NJUD', 'NLPR', 'RGBD135', 'SIP', 'SSD', 'STERE1000']
                test_list = ['NJUD-TE', 'NLPR-TE', 'RGBD135', 'SIP']
            elif modal == 'o':
                test_list = ['FBMS', 'SegV2', 'DAVIS-TE', 'DAVSOD-TE']
            elif modal == 't':
                test_list = ['VT5000-TE', 'VT1000', 'VT821']
            elif modal == 'r':
                test_list = ['EORSSD-TE', 'ORSSD-TE', 'ORS-TE']                    
            for test_set in test_list:
                set_name = '_'.join((modal, test_set))
                test_dataset[set_name] = Test_Dataset(test_set, modal, config)
        else:
            if modal == 'c':
                trset = 'DUTS-TR' # 'MSB-TR' #
            elif modal == 'd':
                trset = 'RGBD-TR'
            elif modal == 'o':
                trset = 'VSOD-TR'
            elif modal == 't':
                trset = 'VT5000-TR'
            elif modal == 'r':
                trset = 'RSSD-TR'
            set_name = '_'.join((modal, trset))
            test_dataset[set_name] = Test_Dataset(trset, modal, config)
        
    return test_dataset
    
def get_loader(config):
    dataset = Train_Dataset(config)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['batch'],
                                  shuffle=True,
                                  num_workers=1,
                                  pin_memory=True,
                                  drop_last=True)
    return data_loader

def read_modality(sub, sample_path, flip, img_size):
    if sub in sample_path.keys():
        m_name = sample_path[sub]
        modal = Image.open(m_name).convert('RGB')
        modal = modal.resize((img_size, img_size))
        modal = np.array(modal).astype(np.float32) / 255.
        if flip:
            modal = modal[:, ::-1].copy()
        modal = modal.transpose((2, 0, 1))
    else:
        modal = np.zeros((3, img_size, img_size)).astype(np.float32)
        
    return modal

class Train_Dataset(data.Dataset):
    def __init__(self, config):
        self.config = config
        self.modality = config['trset']
        self.image_list = get_train_image_list(self.modality, config)
        self.size = len(self.image_list)
        self.rie = random_image_enhance(methods=  ['contrast', 'sharpness', 'brightness'])

    def __getitem__(self, index):
        sample_path = self.image_list[index]
        img_name = sample_path['rgb']
        gt_name = sample_path['gt']
        
        image = Image.open(img_name).convert('RGB')
        gt = Image.open(gt_name).convert('L')
        
        img_size = self.config['size']
        image = image.resize((img_size, img_size))
        gt = gt.resize((img_size, img_size), Image.NEAREST)
    
        image_ = np.array(image).astype(np.float32)
        gt = np.array(gt)
        
        #image = self.rie(image)

        flip = random.random() > 0.5
        if flip:
            image_ = image_[:, ::-1].copy()
            gt = gt[:, ::-1].copy()
        
        image = ((image_ / 255.) - mean) / std
        image = image.transpose((2, 0, 1))
        gt = (gt - np.min(gt)) / (np.max(gt) - np.min(gt) + 1e-5)
        
        gt = np.expand_dims(gt, axis=0)
        
        out_dict = {'image': image, 'gt': gt, 'name': gt_name, 'flip': flip}
        
        for modality in self.modality:
            if modality == 'c' or modality == 'r':
                continue
            elif modality == 'd':
                sub = 'dep'
                out_dict[sub] = read_modality(sub, sample_path, flip, img_size)
                if sub in sample_path.keys():
                    out_dict['modal'] = read_modality(sub, sample_path, flip, img_size)
            elif modality == 'o':
                sub = 'of'
                out_dict[sub] = read_modality(sub, sample_path, flip, img_size)
                if sub in sample_path.keys():
                    out_dict['modal'] = read_modality(sub, sample_path, flip, img_size)
            elif modality == 't':
                sub = 'th'
                out_dict[sub] = read_modality(sub, sample_path, flip, img_size)
                if sub in sample_path.keys():
                    out_dict['modal'] = read_modality(sub, sample_path, flip, img_size)
        
        if 'modal' not in out_dict:
            out_dict['modal'] = (image_/255).transpose((2, 0, 1))

        
        return out_dict

    def __len__(self):
        return self.size

class Test_Dataset(data.Dataset):
    def __init__(self, name, mode, config=None):
        self.config = config
        
        read_list = None
        if mode == 'c':
            read_list = get_color_list
        elif mode == 'd':
            read_list = get_rgbd_list
        elif mode == 'o':
            read_list = get_frame_list
        elif mode == 't':
            read_list = get_rgbt_list
        elif mode == 'r':
            read_list = get_remote_list
        self.image_list = read_list(name, config, 'test')
        
        self.set_name = name
        self.size = len(self.image_list)
        self.modality = mode
        self.mode = 'test'

    def __getitem__(self, index):
        sample_path = self.image_list[index]
        img_name = sample_path['rgb']
        gt_name = sample_path['gt']
        
        image = Image.open(img_name).convert('RGB')
        image = image.resize((self.config['size'], self.config['size']))
        image_ = np.array(image).astype(np.float32)
        gt = Image.open(gt_name).convert('L')
        if self.mode != 'test':
            gt = gt.resize((self.config['size'], self.config['size']))
        img_size = self.config['size']
        
        img_pads = img_name.split('/')
        name = '/'.join(img_pads[img_pads.index(self.set_name) + 2:])
        
        image = ((image_ / 255.) - mean) / std
        image = image.transpose((2, 0, 1))
        image = image
        gt = (gt - np.min(gt)) / (np.max(gt) - np.min(gt) + 1e-5)
        
        out_dict = {'image': image, 'gt': gt, 'name': name}

        for modality in self.modality:
            if modality == 'c' or modality == 'r':
                continue
            elif modality == 'd':
                sub = 'dep'
                out_dict[sub] = read_modality(sub, sample_path, False, img_size)
                if sub in sample_path.keys():
                    out_dict['modal'] = read_modality(sub, sample_path, False, img_size)
            elif modality == 'o':
                sub = 'of'
                out_dict[sub] = read_modality(sub, sample_path, False, img_size)
                if sub in sample_path.keys():
                    out_dict['modal'] = read_modality(sub, sample_path, False, img_size)
            elif modality == 't':
                sub = 'th'
                out_dict[sub] = read_modality(sub, sample_path, False, img_size)
                if sub in sample_path.keys():
                    out_dict['modal'] = read_modality(sub, sample_path, False, img_size)
        
        if 'modal' not in out_dict:
            out_dict['modal'] = (image_/255).transpose((2, 0, 1))

        return out_dict
    def __len__(self):
        return self.size
def test_data():
    config = {'orig_size': True, 'size': 288, 'data_path': '../dataset'}
    dataset = 'SOD'
    
    '''
    data_loader = Test_Dataset(dataset, config)
    #data_loader = Train_Dataset(dataset, config)
    data_size = data_loader.size
    
    for i in range(data_size):
        img, gt, name = data_loader.load_data(i)
        #img, gt = data_loader.__getitem__(i)
        new_img = (img * std + mean) * 255.
        #new_img = gt * 255
        print(np.min(new_img), np.max(new_img))
        new_img = (new_img).astype(np.uint8)
        #print(new_img.shape).astype(np.)
        im = Image.fromarray(new_img)
        #im.save('temp/' + name + '.jpg')
        im.save('temp/' + str(i) + '.jpg')
    
    '''
    
    data_loader = Val_Dataset(dataset, config)
    imgs, gts, names = data_loader.load_all_data()
    print(imgs.shape, gts.shape, len(names))
    

if __name__ == "__main__":
    test_data()