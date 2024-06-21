from io import BytesIO
import lmdb
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
##########################
#import util as Util
import utils as Util
import scipy
import scipy.io
import os.path
from torchvision.utils import save_image
import numpy as np

IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = 'B'
LIST_FOLDER_NAME = 'list'
ANNOT_FOLDER_NAME = "label"
label_suffix = ".png"

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=np.str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list

def get_img_post_path(root_dir,img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)


def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)


def get_label_path(root_dir, img_name):
    return os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name) #.replace('.jpg', label_suffix)

class ISICDataset(Dataset):
    def __init__(self, args, datapath, transform = None, split = 'train', plane=False):
    # def __init__(self, dataroot, resolution=256, split='train', data_len=-1):
        
        self.res = 256
        # self.data_len = data_len
        self.split = "train"

        self.root_dir = "/media/lscsc/nas/yihan/ddpm_1/MedSegDiff/WHU"
        # self.split = split  #train | val | test
        
        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split+'.txt')
        
        self.img_name_list = load_img_name_list(self.list_path)

        self.dataset_len = len(self.img_name_list)

        # if self.data_len <= 0:
        #     self.data_len = self.dataset_len
        # else:
        self.data_len = self.dataset_len

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.data_len])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.data_len])

        img_A   = Image.open(A_path).convert("RGB")
        img_B   = Image.open(B_path).convert("RGB")
        
        L_path  = get_label_path(self.root_dir, self.img_name_list[index % self.data_len])
        img_lbl = Image.open(L_path).convert("L")
        # print("img_lbl1:",img_lbl.size)
        img_A   = Util.transform_augment_cd(img_A, split=self.split, min_max=(-1, 1))
        img_B   = Util.transform_augment_cd(img_B, split=self.split, min_max=(-1, 1))
        img_lbl = Util.transform_augment_cd(img_lbl, split=self.split, min_max=(0, 1))
        
        # if img_lbl.dim() > 2:
        #     print("img_lbl[0]:",img_lbl[0].shape)
        #     print("img_lbl:",img_lbl.shape)
        #     img_lbl = img_lbl[0]
        #     print("3")
        # img_lbl = img_lbl.unsqueeze(0)
        # print("lbl:",img_lbl.shape)
      
        return (img_A, img_B, img_lbl)
        # return {'A': img_A, 'B': img_B, 'L': img_lbl, 'Index': index}
