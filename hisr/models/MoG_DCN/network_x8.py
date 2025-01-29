import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
# from clean_util import H_z, HT_y, para_setting
import torch.nn as nn
import scipy.io as sio
from hisr.models.MoG_DCN.Unet import Unet

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

"""
modify fast DVD(vedio denoising) 
"""


class VSR_CAS(torch.nn.Module):
    """
    network of 'Burst Denoising with Kernel Prediction Networks'
    """

    def __init__(self, channel0, factor, P, patch_size):
        super(VSR_CAS, self).__init__()

        self.channel0 = channel0
        self.up_factor = factor
        self.patch_size = patch_size

        self.P = torch.nn.Parameter(P)
        self.P.requires_grad = False
        self.acti = torch.nn.PReLU()

        self.delta_0 = torch.nn.Parameter(torch.tensor(0.1))
        self.eta_0 = torch.nn.Parameter(torch.tensor(0.9))
        self.delta_1 = torch.nn.Parameter(torch.tensor(0.1))
        self.eta_1 = torch.nn.Parameter(torch.tensor(0.9))
        self.delta_2 = torch.nn.Parameter(torch.tensor(0.1))
        self.eta_2 = torch.nn.Parameter(torch.tensor(0.9))
        self.delta_3 = torch.nn.Parameter(torch.tensor(0.1))
        self.eta_3 = torch.nn.Parameter(torch.tensor(0.9))
        self.delta_4 = torch.nn.Parameter(torch.tensor(0.1))
        self.eta_4 = torch.nn.Parameter(torch.tensor(0.9))
        self.delta_5 = torch.nn.Parameter(torch.tensor(0.1))
        self.eta_5 = torch.nn.Parameter(torch.tensor(0.9))
        self.delta_6 = torch.nn.Parameter(torch.tensor(0.1))
        self.eta_6 = torch.nn.Parameter(torch.tensor(0.9))
        self.spatial = Unet(31)
        # self.spatial1 = Unet_Spatial(3)  # if no use then comment it out
        self.fe_conv1 = torch.nn.Conv2d(in_channels=31, out_channels=64, kernel_size=3, padding=3 // 2)
        self.fe_conv2 = torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=3 // 2)
        self.fe_conv3 = torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=3 // 2)
        self.fe_conv4 = torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=3 // 2)

        self.fe_conv5 = torch.nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=3 // 2)
        self.fe_conv6 = torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=3 // 2)
        self.fe_conv7 = torch.nn.Conv2d(in_channels=448, out_channels=64, kernel_size=3, padding=3 // 2)
        self.fe_conv8 = torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=3 // 2)
        self.conv_downsample = torch.nn.Conv2d(in_channels=31, out_channels=31, kernel_size=13, stride=8,
                                               padding=13 // 2)
        self.conv_upsample = torch.nn.ConvTranspose2d(in_channels=31, out_channels=31, kernel_size=13, stride=8,
                                                      padding=13 // 2)
        self.conv_torgb = torch.nn.Conv2d(in_channels=31, out_channels=3, kernel_size=3, stride=1, padding=3 // 2)
        self.conv_tohsi = torch.nn.Conv2d(in_channels=3, out_channels=31, kernel_size=3, stride=1, padding=3 // 2)
        # self.spatial2 = Unet_Spatial(3)  # if no use then comment it out

        # self.spectral = Unet_Spectral(31)
        self.reset_parameters()

    def recon_noisy(self, z, noisy, v, RGB, id_layer):
        if id_layer == 0:
            DELTA = self.delta_0
            ETA = self.eta_0
        elif id_layer == 1:
            DELTA = self.delta_1
            ETA = self.eta_1
        elif id_layer == 2:
            DELTA = self.delta_2
            ETA = self.eta_2
        elif id_layer == 3:
            DELTA = self.delta_3
            ETA = self.eta_3
        elif id_layer == 4:
            DELTA = self.delta_4
            ETA = self.eta_4
        elif id_layer == 5:
            DELTA = self.delta_5
            ETA = self.eta_5

        sz = z.shape
        # err1 = RGB.reshape(sz[0], 3, sz[2] * sz[3]) - torch.matmul( self.P.transpose(0,1).unsqueeze(0) ,z.reshape(sz[0], sz[1], sz[2] * sz[3]))
        err1 = RGB - self.conv_torgb(z)
        # err1 = torch.matmul(self.P.unsqueeze(0), err1)
        err1 = self.conv_tohsi(err1)
        # err1 = err1.reshape(sz)
        err2 = noisy - ETA * v
        err2 = err2.reshape(sz)

        out = (1 - DELTA - DELTA * ETA) * z + DELTA * err1 + DELTA * err2
        return out

    def recon(self, features, recon, LR, RGB, id_layer):
        if id_layer == 0:
            DELTA = self.delta_0
            ETA = self.eta_0
        elif id_layer == 1:
            DELTA = self.delta_1
            ETA = self.eta_1
        elif id_layer == 2:
            DELTA = self.delta_2
            ETA = self.eta_2
        elif id_layer == 3:
            DELTA = self.delta_3
            ETA = self.eta_3
        elif id_layer == 4:
            DELTA = self.delta_4
            ETA = self.eta_4
        elif id_layer == 5:
            DELTA = self.delta_5
            ETA = self.eta_5

        # fft_B, fft_BT = para_setting('gaussian_blur', self.up_factor, [self.patch_size, self.patch_size])
        # fft_B = torch.cat((torch.Tensor(np.real(fft_B)).unsqueeze(2),
        #                         torch.Tensor(np.imag(fft_B)).unsqueeze(2)), 2).cuda()
        # fft_BT = torch.cat(
        #     (torch.Tensor(np.real(fft_BT)).unsqueeze(2), torch.Tensor(np.imag(fft_BT)).unsqueeze(2)), 2) .cuda()
        #
        # recon_h1 = int(recon.shape[2])
        # recon_h2 = int(recon.shape[3])

        # down = self.Down(recon, self.up_factor , fft_B )
        sz = recon.shape
        down = self.conv_downsample(recon)
        # err1 = self.UP(down - LR , self.up_factor ,fft_BT)
        err1 = self.conv_upsample(down - LR, output_size=sz)

        # to_rgb = torch.matmul(self.P.transpose(0,1).unsqueeze(0)   , recon.reshape(sz[0], sz[1], sz[2] * sz[3]))
        to_rgb = self.conv_torgb(recon)
        # err_rgb = RGB - to_rgb.reshape(sz[0], 3, sz[2] , sz[3])
        err_rgb = RGB - to_rgb
        # err3    = torch.matmul(self.P.unsqueeze(0),err_rgb.reshape(sz[0], 3, sz[2] * sz[3]))
        err3 = self.conv_tohsi(err_rgb)
        err3 = err3.reshape(sz)
        ################################################################

        out = (1 - DELTA * ETA) * recon + DELTA * err3 + DELTA * err1 + DELTA * ETA * features
        ################################################################
        return out

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def _forward_implem(self, lms, ms, pan):  # [batch_size ,3 ,7 ,270 ,480] ;
        ## LR [1 31 6 6]
        ## RGB [1 31 48 48]

        # label_h1 = int(LR.shape[2]) * self.up_factor
        # label_h2 = int(LR.shape[3]) * self.up_factor

        # x = bicubic_interp_2d(input, [label_h1, label_h2])
        # x = torch.nn.functional.interpolate(LR, scale_factor=self.up_factor, mode='bicubic', align_corners=False)
        # y = LR

        # for i in range(0, 3):
        x_now = self.spatial(self.fe_conv1(lms))[0]
        x_now = x_now + lms

        for _ in range(3):
            # z = self.recon_noisy(z, lms, v, pan, 0)
            x = self.recon(x_now, lms, ms, pan)
            x_prox = self.spatial(self.fe_conv2(x))[0]
            x_now = x_prox + x_now

        return conv_out

    def train_step(self, data, **kwargs):
        gt = data.pop("gt")
        data["lms"] = data.pop("up")
        data["ms"] = data.pop("lrhsi")
        data["pan"] = data.pop("rgb")
        sr = self._forward_implem(**data)
        loss_dicts = self.criterion(sr, gt, **kwargs)
        return loss_dicts

    def val_step(self, data, **kwargs):
        data["lms"] = data.pop("up")
        data["ms"] = data.pop("lrhsi")
        data["pan"] = data.pop("rgb")
        sr1 = self._forward_implem(**data)

        return sr1

    def test_step(self, data, **kwargs):
        data["lms"] = data.pop("up")
        data["ms"] = data.pop("lrhsi")
        data["pan"] = data.pop("rgb")
        sr1 = self._forward_implem(**data)

        return sr1


from hisr.models.base_model import HISRModel
from hisr.common.metrics import SetCriterion
from torch import optim
from torch import nn
from udl_vis.Basis.optim import LRScheduler, get_optimizer

class build_MoGDCN(HISRModel, name="MoGDCN_ori"):
    def __call__(self, cfg):
        loss = nn.L1Loss(size_average=True).cuda()  ## Define the Loss function
        weight_dict = {"loss": 1}
        losses = {"loss": loss}
        criterion = SetCriterion(losses, weight_dict)
        model = VSR_CAS(criterion, 31, cfg.scale).cuda()
        model.criterion = criterion
        optimizer = get_optimizer(model, model.parameters(), **cfg.optimizer_cfg)
        scheduler = LRScheduler(optimizer, **cfg.scheduler_cfg)

        return model, criterion, optimizer, scheduler


if __name__ == '__main__':
    
    hs_channel = 31
    ms_channel = 3
    factor = 8

    if factor == 8:
        LR = torch.randn(1, 31, 16, 16).cuda()
        UP = torch.randn(1, 31, 128, 128).cuda()
        RGB = torch.randn(1, 3, 128, 128).cuda()
        gt = torch.randn(1, 31, 128, 128).cuda()
    elif factor == 4:
        LR = torch.randn(1, 31, 16, 16).cuda()
        UP = torch.randn(1, 31, 64, 64).cuda()
        RGB = torch.randn(1, 3, 64, 64).cuda()
        gt = torch.randn(1, 31, 64, 64).cuda()
    else:
        raise ValueError("factor must be 4 or 8")

    model = VSR_CAS(
        n_stages=3,
        hs_channel=hs_channel,
        ms_channel=ms_channel,
        factor=factor,
        featEmbed=True,
        num_channel=64,
    ).cuda()

    model.criterion = nn.L1Loss().cuda()

    sr = model({"up": UP, "lrhsi": LR, "rgb": RGB, "gt": gt}, mode="train")
    print(sr.shape)
