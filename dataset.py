import os
import re
import cv2
import glob
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

def make_some_noise(img, sigma, seed=None):
    s = np.random.uniform(sigma[0], sigma[1]) if len(sigma)!=1 else sigma[0]
    if seed is not None:
        torch.random.manual_seed(seed)
    noise = torch.randn(*img.shape)*s/255
    img = (img + noise)
    return img

class DatasetRandomCrop(Dataset):
    def __init__(self, crop_size, model_time, patch_load, data_name=None, gray=False):
        super(DatasetRandomCrop, self).__init__()
        assert model_time is not None, "model_time should be assigned!"
        
        self.crop_size = crop_size
        self.model_time = model_time
        self.epoch_dict = dict() if patch_load else None
        self.gray = gray
    
    def __getitem__(self, idx):
        iidx = idx
        idx%=len(self.file_list_hq)
        if self.epoch_dict is not None:
            if len(self.epoch_dict[iidx])==1:
                crop_h, crop_w, random_flip, random_rotate = self.epoch_dict[iidx][0]
            else:
                aa = 0 if iidx%2==0 else 1
                crop_h, crop_w, random_flip, random_rotate = self.epoch_dict[iidx][aa]
            if self.scale not in [1,2]: # only for SR x3, x4
                crop_h = int(crop_h/self.scale*2)
                crop_w = int(crop_w/self.scale*2)
        file_hq = self.file_list_hq[idx]
        file_lq = self.file_list_lq[idx] if self.file_list_lq is not None else None
        
        if not self.gray:
            img_hq = torch.from_numpy(np.load(file_hq))/255
        else:
            img_hq = np.transpose(np.load(file_hq), (1,2,0))
            img_hq = cv2.cvtColor(img_hq, cv2.COLOR_RGB2GRAY)
            img_hq = torch.from_numpy(img_hq).unsqueeze(0)/255
        img_lq = torch.from_numpy(np.load(file_lq))/255 if file_lq is not None else torch.clone(img_hq)
        
        if img_hq.size(0) == 4:
            img_hq = TF.to_tensor(TF.to_pil_image(img_hq).convert('RGB'))
            img_lq = TF.to_tensor(TF.to_pil_image(img_hq).convert('RGB'))
        
        _, lq_h, lq_w = img_lq.size()
        
        if hasattr(self, 'sigma'):
            img_lq = make_some_noise(img_lq, self.sigma)
        
        # random crop patch
        if self.epoch_dict is None:
            crop_h = torch.randint(0, lq_h-self.crop_size, (1,)).item()
            crop_w = torch.randint(0, lq_w-self.crop_size, (1,)).item()
        crop_lq = TF.crop(img_lq, crop_h, crop_w, self.crop_size, self.crop_size)
        crop_hq = TF.crop(img_hq, crop_h*self.scale, crop_w*self.scale, self.crop_size*self.scale, self.crop_size*self.scale)
        
        # random horizontal flip, random rotation, image normalization
        if self.epoch_dict is None:
            random_flip = torch.randint(0,2,(1,)).item()
            random_rotate = torch.randint(0,4,(1,)).item()
        crop_hq, crop_lq = (TF.hflip(crop_hq), TF.hflip(crop_lq)) if random_flip else (crop_hq, crop_lq)
        crop_hq, crop_lq = TF.rotate(crop_hq, angle=90*random_rotate), TF.rotate(crop_lq, angle=90*random_rotate)
        return crop_hq, crop_lq
    
class DIV2KDatasetRandomCrop(DatasetRandomCrop):
    def __init__(self, root_hq, root_lq, crop_size, model_time=None, patch_load=False):
        super(DIV2KDatasetRandomCrop, self).__init__(crop_size, model_time, patch_load, 'DIV2K')
        self.scale = int(re.search('.+(\d)', root_lq).group(1))
        
        self.file_list_hq = sorted(glob.glob(os.path.join(root_hq, '*.npy')))
        self.file_list_lq = sorted(glob.glob(os.path.join(root_lq, '*.npy')))
        
    def __len__(self):
        return len(self.file_list_hq)*80
    
class DF2KDatasetRandomCrop(DatasetRandomCrop):
    def __init__(self, D_root_hq, D_root_lq, F_root_hq, F_root_lq, crop_size, model_time=None, patch_load=False):
        super(DF2KDatasetRandomCrop, self).__init__(crop_size, model_time, patch_load, 'DF2K')
        self.scale = int(re.search('.+(\d)', D_root_lq).group(1))
        
        D_file_list_hq = sorted(glob.glob(os.path.join(D_root_hq, '*.npy')))
        F_file_list_hq = sorted(glob.glob(os.path.join(F_root_hq, '*.npy')))
        D_file_list_lq = sorted(glob.glob(os.path.join(D_root_lq, '*.npy')))
        F_file_list_lq = sorted(glob.glob(os.path.join(F_root_lq, '*.npy')))
        
        self.file_list_hq = D_file_list_hq + F_file_list_hq
        self.file_list_lq = D_file_list_lq + F_file_list_lq
        
    def __len__(self):
        return int(len(self.file_list_hq)*18.551)
    
class DFBWDatasetRandomCrop(DatasetRandomCrop):
    def __init__(self, D_root_hq, F_root_hq, B_root_hq, W_root_hq, crop_size, sigma, gray=False, model_time=None, patch_load=False):
        super(DFBWDatasetRandomCrop, self).__init__(crop_size, model_time, patch_load, 'DFBW', gray)
        self.scale = 1
        
        D_file_list_hq = sorted(glob.glob(os.path.join(D_root_hq, '*.npy')))
        F_file_list_hq = sorted(glob.glob(os.path.join(F_root_hq, '*.npy')))
        B_file_list_hq = sorted(glob.glob(os.path.join(B_root_hq, '*.npy')))
        W_file_list_hq = sorted(glob.glob(os.path.join(W_root_hq, '*.npy')))
        
        self.file_list_hq = D_file_list_hq + F_file_list_hq + B_file_list_hq + W_file_list_hq
        self.file_list_lq = None
        
        self.sigma = sigma
        
    def __len__(self):
        return len(self.file_list_hq)*3
    
class LLEDatasetRandomCrop(DatasetRandomCrop):
    def __init__(self, L_root_hq, V_root_hq, L_root_lq, V_root_lq, crop_size, model_time=None, patch_load=False):
        super(LLEDatasetRandomCrop, self).__init__(crop_size, model_time, patch_load, 'LLE')
        self.scale = 1
        
        L_file_list_hq = sorted(glob.glob(os.path.join(L_root_hq, '*.npy')))
        V_file_list_hq = sorted(glob.glob(os.path.join(V_root_hq, '*.npy')))
        L_file_list_lq = sorted(glob.glob(os.path.join(L_root_lq, '*.npy')))
        V_file_list_lq = sorted(glob.glob(os.path.join(V_root_lq, '*.npy')))
        
        self.file_list_hq = L_file_list_hq + V_file_list_hq
        self.file_list_lq = L_file_list_lq + V_file_list_lq
        
    def __len__(self):
        return int(len(self.file_list_lq)*14.006)
    
class DRDatasetRandomCrop(DatasetRandomCrop):
    def __init__(self, root_hq, root_lq, crop_size, model_time=None, patch_load=False):
        super(DRDatasetRandomCrop, self).__init__(crop_size, model_time, patch_load, 'DR')
        self.scale = 1
        
        self.file_list_hq = sorted(glob.glob(os.path.join(root_hq, '*.npy')))
        self.file_list_lq = sorted(glob.glob(os.path.join(root_lq, '*.npy')))
        
    def __len__(self):
        return int(len(self.file_list_lq)*1.8234)