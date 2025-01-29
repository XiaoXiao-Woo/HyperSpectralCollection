# coding=UTF-8

import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import scipy.io as sio
from torch import optim
from udl_vis.Basis.module import PatchMergeModule

"""
modify fast DVD(vedio denoising) 
"""


class Encoding_Block(torch.nn.Module):
    def __init__(self, c_in):
        super(Encoding_Block, self).__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels=c_in, out_channels=64, kernel_size=3, padding=3 // 2
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2
        )
        self.conv4 = torch.nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=3 // 2
        )
        self.conv5 = torch.nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=3 // 2
        )

        self.act = torch.nn.PReLU()

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, input):

        out1 = self.act(self.conv1(input))
        out2 = self.act(self.conv2(out1))
        out3 = self.act(self.conv3(out2))
        f_e = self.conv4(out3)
        down = self.act(self.conv5(f_e))
        return f_e, down


class Encoding_Block_End(torch.nn.Module):
    def __init__(self, c_in=64):
        super(Encoding_Block_End, self).__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels=c_in, out_channels=64, kernel_size=3, padding=3 // 2
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2
        )
        self.conv4 = torch.nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=3 // 2
        )
        self.act = torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, input):
        out1 = self.act(self.conv1(input))
        out2 = self.act(self.conv2(out1))
        out3 = self.act(self.conv3(out2))
        f_e = self.conv4(out3)
        return f_e


class Decoding_Block(torch.nn.Module):
    def __init__(self, c_in):
        super(Decoding_Block, self).__init__()
        self.conv0 = torch.nn.Conv2d(
            in_channels=256, out_channels=128, kernel_size=3, padding=3 // 2
        )
        self.conv1 = torch.nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=3 // 2
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=3 // 2
        )

        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1)
        # self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=3 // 2)
        self.batch = 1
        # self.up = torch.nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.up = torch.nn.ConvTranspose2d(
            c_in, 128, kernel_size=3, stride=2, padding=3 // 2
        )

        self.act = torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def up_sampling(self, input, label, kernel_size=3, out_features=64):

        batch_size = self.batch

        label_h1 = int(label.shape[2])
        label_h2 = int(label.shape[3])

        in_features = int(input.shape[1])
        Deconv = self.up(input)

        return Deconv

    def forward(self, input, map):

        up = self.up(
            input,
            output_size=[input.shape[0], input.shape[1], map.shape[2], map.shape[3]],
        )
        cat = torch.cat((up, map), 1)
        cat = self.act(self.conv0(cat))
        out1 = self.act(self.conv1(cat))
        out2 = self.act(self.conv2(out1))

        out3 = self.conv3(out2)

        return out3


class Feature_Decoding_End(torch.nn.Module):
    def __init__(self, c_out):
        super(Feature_Decoding_End, self).__init__()
        self.conv0 = torch.nn.Conv2d(
            in_channels=256, out_channels=128, kernel_size=3, padding=3 // 2
        )
        self.conv1 = torch.nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=3 // 2
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=3 // 2
        )

        self.conv3 = torch.nn.Conv2d(
            in_channels=128, out_channels=c_out, kernel_size=3, padding=3 // 2
        )
        self.batch = 1
        self.up = torch.nn.ConvTranspose2d(
            512, 128, kernel_size=3, stride=2, padding=3 // 2
        )
        self.act = torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def up_sampling(self, input, label, kernel_size=3, out_features=64):

        batch_size = self.batch

        label_h1 = int(label.shape[2])
        label_h2 = int(label.shape[3])

        in_features = int(input.shape[1])

        Deconv = self.up(input)

        return Deconv

    def forward(self, input, map):

        up = self.up(
            input,
            output_size=[input.shape[0], input.shape[1], map.shape[2], map.shape[3]],
        )
        cat = torch.cat((up, map), 1)
        cat = self.act(self.conv0(cat))
        out1 = self.act(self.conv1(cat))
        out2 = self.act(self.conv2(out1))

        out3 = self.conv3(out2)

        return out3


class Unet_Spatial(torch.nn.Module):
    def __init__(self, cin):
        super(Unet_Spatial, self).__init__()

        self.Encoding_block1 = Encoding_Block(64)
        self.Encoding_block2 = Encoding_Block(64)
        self.Encoding_block3 = Encoding_Block(64)
        self.Encoding_block4 = Encoding_Block(64)
        self.Encoding_block_end = Encoding_Block_End(64)

        self.Decoding_block1 = Decoding_Block(128)
        self.Decoding_block2 = Decoding_Block(512)
        self.Decoding_block3 = Decoding_Block(512)
        self.Decoding_block_End = Feature_Decoding_End(cin)

        self.acti = torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        sz = x.shape
        # x = x.view(-1,1,sz[2],sz[3])

        encode0, down0 = self.Encoding_block1(x)
        encode1, down1 = self.Encoding_block2(down0)
        encode2, down2 = self.Encoding_block3(down1)
        encode3, down3 = self.Encoding_block4(down2)

        media_end = self.Encoding_block_end(down3)

        decode3 = self.Decoding_block1(media_end, encode3)
        decode2 = self.Decoding_block2(decode3, encode2)
        decode1 = self.Decoding_block3(decode2, encode1)
        decode0 = self.Decoding_block_End(decode1, encode0)
        # y = x[:,1:2,:,:] + decode0
        # y = x + decode0

        return decode0, encode0


class Unet_Spectral(torch.nn.Module):
    def __init__(self, cin):
        super(Unet_Spectral, self).__init__()

        self.Encoding_block1 = Encoding_Block(cin)
        self.Encoding_block2 = Encoding_Block(64)
        self.Encoding_block3 = Encoding_Block(64)
        self.Encoding_block4 = Encoding_Block(64)
        self.Encoding_block_end = Encoding_Block_End(64)

        self.Decoding_block1 = Decoding_Block(128)
        self.Decoding_block2 = Decoding_Block(512)
        self.Decoding_block3 = Decoding_Block(512)
        self.Decoding_block_End = Feature_Decoding_End(cin)

        self.acti = torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        sz = x.shape
        x = x.view(-1, 31, sz[2], sz[3])

        encode0, down0 = self.Encoding_block1(x)
        encode1, down1 = self.Encoding_block2(down0)
        encode2, down2 = self.Encoding_block3(down1)
        encode3, down3 = self.Encoding_block4(down2)

        media_end = self.Encoding_block_end(down3)

        decode3 = self.Decoding_block1(media_end, encode3)
        decode2 = self.Decoding_block2(decode3, encode2)
        decode1 = self.Decoding_block3(decode2, encode1)
        decode0 = self.Decoding_block_End(decode1, encode0)
        y = x + decode0
        return y


class VSR_CAS(PatchMergeModule):
    """
    network of 'Burst Denoising with Kernel Prediction Networks'
    """

    def __init__(self, factor):
        super(VSR_CAS, self).__init__()

        self.up_factor = factor
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
        self.spatial = Unet_Spatial(31)
        # self.spatial1 = Unet_Spatial(3)  # if no use then comment it out
        self.fe_conv1 = torch.nn.Conv2d(
            in_channels=31, out_channels=64, kernel_size=3, padding=1
        )
        self.fe_conv2 = torch.nn.Conv2d(
            in_channels=192, out_channels=64, kernel_size=3, padding=1
        )
        self.fe_conv3 = torch.nn.Conv2d(
            in_channels=192, out_channels=64, kernel_size=3, padding=1
        )
        self.fe_conv4 = torch.nn.Conv2d(
            in_channels=192, out_channels=64, kernel_size=3, padding=1
        )

        self.fe_conv5 = torch.nn.Conv2d(
            in_channels=320, out_channels=64, kernel_size=3, padding=1
        )
        self.fe_conv6 = torch.nn.Conv2d(
            in_channels=192, out_channels=64, kernel_size=3, padding=1
        )
        self.fe_conv7 = torch.nn.Conv2d(
            in_channels=448, out_channels=64, kernel_size=3, padding=1
        )
        self.fe_conv8 = torch.nn.Conv2d(
            in_channels=192, out_channels=64, kernel_size=3, padding=1
        )
        self.conv_downsample = torch.nn.Conv2d(
            in_channels=31, out_channels=31, kernel_size=7, stride=4, padding=7 // 2
        )
        self.conv_upsample = torch.nn.ConvTranspose2d(
            in_channels=31, out_channels=31, kernel_size=7, stride=4, padding=7 // 2
        )
        self.conv_torgb = torch.nn.Conv2d(
            in_channels=31, out_channels=3, kernel_size=3, stride=1, padding=1
        )
        self.conv_tohsi = torch.nn.Conv2d(
            in_channels=3, out_channels=31, kernel_size=3, stride=1, padding=1
        )
        # self.spatial2 = Unet_Spatial(3)  # if no use then comment it out

        # self.spectral = Unet_Spectral(31)
        self.reset_parameters()

    def recon_noisy(self, z, up, v, rgb, id_layer):
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

        err1 = rgb - self.conv_torgb(z)
        err1 = self.conv_tohsi(err1)

        # out = (1 - DELTA - DELTA * ETA) * z + DELTA * err1 + DELTA * err2
        return DELTA * (err1 + up) + (1 - DELTA) * z - (DELTA * ETA) * (v + z)

        # if ema:
        #     if z is not None:
        #         return DELTA * (err1 + up) + (1 - DELTA) * z - (DELTA * ETA) * (v + z)

        #     else:
        #         return z + DELTA * (err1 + err2)

    def recon(self, features, recon, LR, RGB, id_layer, prox=True):
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

        sz = recon.shape
        down = self.conv_downsample(recon)
        err1 = self.conv_upsample(down - LR, output_size=sz)

        to_rgb = self.conv_torgb(recon)
        err_rgb = RGB - to_rgb
        err3 = self.conv_tohsi(err_rgb)
        err3 = err3.reshape(sz)

        # out = (1-DELTA*ETA)*recon +DELTA*err3 + DELTA*err1 + DELTA*ETA*features

        if prox:
            if features is not None:
                return recon - DELTA * (err3 + err1) - ETA * (recon - features)
            else:
                return recon - DELTA * (err3 + err1)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def _forward_implem(self, up, hs, rgb):  # [batch_size ,3 ,7 ,270 ,480] ;
        # x: HSI_UP [1, 31, 48, 48]  [1, 31, 64, 64]
        # y: LR [1 31 6 6] [1, 31, 16, 16]
        # RGB [1 31 48 48] [1, 31, 64, 64]

        x = up
        # for i in range(0, 3):
        z = x
        v, fe = self.spatial(self.fe_conv1(z))
        v = v + z
        z = self.recon_noisy(z, up, v, rgb, 0)
        conv_out, fe1 = self.spatial(
            self.fe_conv2(torch.cat((self.fe_conv1(z), fe), 1))
        )
        conv_out = conv_out + z

        x = self.recon(conv_out, x, hs, rgb, id_layer=3)

        z = x
        v, fe2 = self.spatial(self.fe_conv3(torch.cat((self.fe_conv1(z), fe), 1)))
        v = v + z
        z = self.recon_noisy(z, x, v, rgb, 0)
        conv_out, fe3 = self.spatial(
            self.fe_conv4(torch.cat((self.fe_conv1(z), fe2), 1))
        )
        conv_out = conv_out + z
        x = self.recon(conv_out, x, hs, rgb, id_layer=3)

        z = x
        v, fe4 = self.spatial(self.fe_conv5(torch.cat((self.fe_conv1(z), fe, fe2), 1)))
        v = v + z
        z = self.recon_noisy(z, x, v, rgb, 0)
        conv_out, fe5 = self.spatial(
            self.fe_conv6(torch.cat((self.fe_conv1(z), fe4), 1))
        )
        conv_out = conv_out + z
        x = self.recon(conv_out, x, hs, rgb, id_layer=3)

        z = x
        v, fe6 = self.spatial(
            self.fe_conv7(torch.cat((self.fe_conv1(z), fe, fe2, fe4), 1))
        )
        v = v + z
        z = self.recon_noisy(z, x, v, rgb, 0)
        conv_out, _ = self.spatial(self.fe_conv8(torch.cat((self.fe_conv1(z), fe6), 1)))
        conv_out = conv_out + z

        return conv_out

    def train_step(self, data, *args, **kwargs):
        gt = data.pop("gt")
        data["hs"] = data.pop("lrhsi")
        sr = self._forward_implem(**data)
        loss_dicts = self.criterion(sr, gt, **kwargs)

        return loss_dicts

    def val_step(self, data, **kwargs):
        data["hs"] = data.pop("lrhsi")
        sr1 = self._forward_implem(**data)
        return sr1

    def test_step(self, data, **kwargs):

        return self.val_step(data, **kwargs)


from hisr.models.base_model import HISRModel
from hisr.common.metrics import SetCriterion
from udl_vis.Basis.metrics.cal_ssim import ssim
from torch import optim
from torch import nn
from udl_vis.Basis.optim import LRScheduler, get_optimizer


class build_MoGDCN(HISRModel, name="MoGDCN"):
    def __call__(self, cfg, logger):
        loss = nn.L1Loss(size_average=True).cuda()  ## Define the Loss function
        loss = nn.L1Loss(size_average=True)  ## Define the Loss function
        weight_dict = {"loss": 1, "ssim_loss": 0.1}
        losses = {"loss": loss, "ssim_loss": lambda x, y: 1 - ssim(x, y)}
        criterion = SetCriterion(losses, weight_dict)
        model = VSR_CAS(
            cfg.factor
        ).cuda()
        # optimizer = optim.Adam(
        #     model.parameters(), lr=2e-4, weight_decay=1e-8
        # )  ## optimizer 1: Adam
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500)
        model.criterion = criterion
        optimizer = get_optimizer(model, model.parameters(), **cfg.optimizer_cfg)
        scheduler = LRScheduler(optimizer, **cfg.scheduler_cfg)

        return model, criterion, optimizer, scheduler


if __name__ == "__main__":
    hs_channel = 31
    ms_channel = 3
    factor = 4

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
        factor=factor,
    ).cuda()

    model.criterion = nn.L1Loss().cuda()

    sr = model({"gt": gt, "up": UP, "lrhsi": LR, "rgb": RGB}, mode="train")
    print(sr.shape)
