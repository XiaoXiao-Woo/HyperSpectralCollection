# GPL License
# Copyright (C) UESTC
# All Rights Reserved 
#
# @Time    : 2023/5/21 18:43
# @Author  : Xiao Wu
# @reference: 
#
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class ResTFNet(nn.Module):
    def __init__(self,
                 is_bn,
                 n_select_bands,
                 n_bands,
                 scale_ratio=None,):
        """Load the pretrained ResNet and replace top fc layer."""
        super(ResTFNet, self).__init__()
        self.scale_ratio = scale_ratio
        self.n_bands = n_bands
        self.n_select_bands = n_select_bands


        self.lr_conv1 = nn.Sequential(
            nn.Conv2d(n_bands, 32, kernel_size=3, stride=1, padding=1),
            # nn.PReLU(),
            nn.BatchNorm2d(32) if is_bn else nn.Identity(),
        )
        self.lr_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(32) if is_bn else nn.Identity(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.lr_down_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),
            nn.PReLU(),
        )
        self.hr_conv1 = nn.Sequential(
            nn.Conv2d(n_select_bands, 32, kernel_size=3, stride=1, padding=1),
            # nn.PReLU(),
            nn.BatchNorm2d(32) if is_bn else nn.Identity(),
        )
        self.hr_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.hr_down_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),
            nn.PReLU(),
        )

        self.fusion_conv1 = nn.Sequential(
            nn.BatchNorm2d(128) if is_bn else nn.Identity(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(128) if is_bn else nn.Identity(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.fusion_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0),
            nn.PReLU(),
        )
        self.fusion_conv3 = nn.Sequential(
            nn.BatchNorm2d(256) if is_bn else nn.Identity(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(256) if is_bn else nn.Identity(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.fusion_conv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
            nn.PReLU(),
        )

        self.recons_conv1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
        )
        self.recons_conv2 = nn.Sequential(
            nn.BatchNorm2d(128) if is_bn else nn.Identity(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(128) if is_bn else nn.Identity(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.recons_conv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
            nn.PReLU(),
        )
        self.recons_conv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
        )
        self.recons_conv5 = nn.Sequential(
            nn.BatchNorm2d(64) if is_bn else nn.Identity(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64) if is_bn else nn.Identity(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.recons_conv6 = nn.Sequential(
            nn.Conv2d(64, n_bands, kernel_size=3, stride=1, padding=1),
            # nn.PReLU(),
            nn.Tanh(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x_lr, x_hr):
        # feature extraction
        # x_lr = F.interpolate(x_lr, scale_factor=self.scale_ratio, mode='bilinear')
        x_lr = self.lr_conv1(x_lr)
        x_lr_cat = self.lr_conv2(x_lr)
        x_lr = x_lr + x_lr_cat
        x_lr = self.lr_down_conv(x_lr)

        x_hr = self.hr_conv1(x_hr)
        x_hr_cat = self.hr_conv2(x_hr)
        x_hr = x_hr + x_hr_cat
        x_hr = self.hr_down_conv(x_hr)
        x = torch.cat((x_hr, x_lr), dim=1)

        # feature fusion
        x = x + self.fusion_conv1(x)
        x_fus_cat = x
        x = self.fusion_conv2(x)
        x = x + self.fusion_conv3(x)
        x = self.fusion_conv4(x)
        x = torch.cat((x_fus_cat, x), dim=1)

        # image reconstruction
        x = self.recons_conv1(x)
        x = x + self.recons_conv2(x)
        x = self.recons_conv3(x)
        x = torch.cat((x_lr_cat, x_hr_cat, x), dim=1)
        x = self.recons_conv4(x)

        x = x + self.recons_conv5(x)
        x = self.recons_conv6(x)

        return x # , 0, 0, 0, 0, 0
    def train_step(self, data, *args, **kwargs):
        # log_vars = {}
        gt, up_hs, ms, rgb = data['gt'].cuda(), data['up'].cuda(), \
                           data['lrhsi'].cuda(), data['rgb'].cuda()
        sr = self(up_hs, rgb)
        loss = self.criterion(sr, gt, *args, **kwargs)['loss']
        # outputs = loss
        # return loss
        # log_vars.update(loss=loss.item())
        # metrics = {'loss': loss, 'log_vars': log_vars}
        return sr, loss

    def val_step(self, data, *args, **kwargs):
        # gt, lms, ms, pan = data
        gt, up_hs, ms, rgb = data['gt'].cuda(), data['up'].cuda(), \
                           data['lrhsi'].cuda(), data['rgb'].cuda()
        sr = self(up_hs, rgb)

        return sr#, gt

class TFNet(nn.Module):
    def __init__(self,
                 n_select_bands,
                 n_bands,
                 scale_ratio=None):
        """Load the pretrained ResNet and replace top fc layer."""
        super(TFNet, self).__init__()
        self.scale_ratio = scale_ratio
        self.n_bands = n_bands
        self.n_select_bands = n_select_bands

        self.lr_conv1 = nn.Sequential(
            nn.Conv2d(n_bands, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.lr_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  #
            nn.PReLU(),
        )
        self.lr_down_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),
            # nn.PReLU(),
        )
        self.hr_conv1 = nn.Sequential(
            nn.Conv2d(n_select_bands, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.hr_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  #
            nn.PReLU(),
        )
        self.hr_down_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),
            # nn.PReLU(),
        )

        self.fusion_conv1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.fusion_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0),
            nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
            nn.PReLU(),
        )

        self.recons_conv1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  #
            nn.PReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
            nn.PReLU(),
        )
        self.recons_conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  #
            nn.PReLU(),
            nn.Conv2d(64, n_bands, kernel_size=3, stride=1, padding=1),
            # nn.PReLU(),
            nn.Tanh(),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x_lr, x_hr):
        # feature extraction
        # x_lr = F.interpolate(x_lr, scale_factor=self.scale_ratio, mode='bilinear')
        x_lr = self.lr_conv1(x_lr)
        x_lr_cat = self.lr_conv2(x_lr)
        x_lr = self.lr_down_conv(x_lr_cat)  #

        x_hr = self.hr_conv1(x_hr)
        x_hr_cat = self.hr_conv2(x_hr)
        x_hr = self.hr_down_conv(x_hr_cat)
        x = torch.cat((x_hr, x_lr), dim=1)

        # feature fusion
        x = self.fusion_conv1(x)  #
        x = torch.cat((x, self.fusion_conv2(x)), dim=1)

        # image reconstruction
        x = self.recons_conv1(x)
        x = torch.cat((x_lr_cat, x_hr_cat, x), dim=1)
        x = self.recons_conv2(x)  #

        return x#, 0, 0, 0, 0, 0

    def train_step(self, data, *args, **kwargs):
        # log_vars = {}
        gt, up_hs, ms, rgb = data['gt'].cuda(), data['up'].cuda(), \
                           data['lrhsi'].cuda(), data['rgb'].cuda()
        sr = self(up_hs, rgb)
        loss = self.criterion(sr, gt, *args, **kwargs)['loss']
        # outputs = loss
        # return loss
        # log_vars.update(loss=loss.item())
        # metrics = {'loss': loss, 'log_vars': log_vars}
        return sr, loss

    def val_step(self, data, *args, **kwargs):
        # gt, lms, ms, pan = data
        gt, up_hs, ms, rgb = data['gt'].cuda(), data['up'].cuda(), \
                           data['lrhsi'].cuda(), data['rgb'].cuda()
        sr = self(up_hs, rgb)

        return sr#, gt


from hisr.models.base_model import HISRModel
from hisr.common.metrics import SetCriterion
from torch import optim

class build_TFNet(HISRModel, name='TFNet'):
    def __call__(self, cfg):
        spectral_num = 31
        loss = nn.L1Loss(size_average=True).cuda()  ## Define the Loss function
        weight_dict = {'loss': 1}
        losses = {'loss': loss}
        criterion = SetCriterion(losses, weight_dict)
        model = TFNet(3, spectral_num).cuda()
        model.criterion = criterion
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0)   ## optimizer 1: Adam
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=200,
                                                       gamma=0.1)

        return model, criterion, optimizer, scheduler


class build_ResTFNet(HISRModel, name='ResTFNet'):
    def __call__(self, cfg):
        # spectral_num = 150
        loss = nn.L1Loss(size_average=True).cuda()  ## Define the Loss function
        weight_dict = {'loss': 1}
        losses = {'loss': loss}
        criterion = SetCriterion(losses, weight_dict)
        model = ResTFNet(True, cfg.rgb_channel, cfg.spectral_num).cuda()
        model.criterion = criterion
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0)   ## optimizer 1: Adam
        # scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=550, #250,
        #                                                gamma=0.5)
        scheduler = None
        return model, criterion, optimizer, scheduler


if __name__ == '__main__':
    from udl_vis.Basis.auxiliary.torchstat import stat

    model = ResTFNet(True, 4, 150).cuda()
    # stat(model, [[1, 31, 64, 64], [1, 3, 64, 64]])
    stat(model, [[1, 150, 80, 80], [1, 4, 80, 80]])
    '''
    64
    Total params: 2.26M 0.0023G (2,264,688)
    Total memory: 27.47MB
    Total MAdd: 3.32GMAdd 0.0033TMAdd
    Total Flops: 1.87GFlops 0.0019TFlops
    Total MemR+W: 61.28MB 0.0598GB
    
    128
    Total Flops: 7.47GFlops 0.0075TFlops
    '''