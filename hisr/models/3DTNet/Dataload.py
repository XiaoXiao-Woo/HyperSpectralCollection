import h5py
from torch.nn import functional as F
from os.path import exists, join, basename
import torch.utils.data as data
import torch
from torch.utils.data import DataLoader
import random
# from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam
import os
from torch.autograd import Variable

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        dataset = h5py.File(file_path, 'r')
        print(dataset.keys())
        self.GT = dataset.get("GT")
        print(self.GT.shape)
        self.UP = dataset.get("HSI_up")
        self.LRHSI = dataset.get("LRHSI")
        self.RGB = dataset.get("RGB")

    #####必要函数
    def __getitem__(self, index):
        rgb = torch.from_numpy(self.RGB[index, :, :, :]).float()
        LR = torch.from_numpy(self.LRHSI[index, :, :, :]).float()
        gt = torch.from_numpy(self.GT[index, :, :, :]).float()


        return LR, rgb, gt
        #####必要函数

    def __len__(self):
        return self.GT.shape[0]

def get_training_set(dir):
    return DatasetFromHdf5(dir)

def get_val_set(dir):
    return DatasetFromHdf5(dir)

def get_eval_set(dir):
    return DatasetFromHdf5(dir)