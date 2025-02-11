# coding=UTF-8

import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from clean_util import H_z, HT_y, para_setting

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
from UDL.pansharpening.common.evaluate import analysis_accu
from UDL.Basis.pytorch_msssim.cal_ssim import SSIM
import torch.nn as nn
from UDL.Basis.criterion_metrics import *
import scipy.io as sio
from torchsummary import summary

from torch import optim

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

    def Down(self, z, factor, fft_B):
        LR = H_z(z, factor, fft_B)
        return LR

    def UP(self, LR, factor, fft_BT):
        HR = HT_y(LR, factor, fft_BT)
        return HR

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

        err1 = RGB - self.conv_torgb(z)
        err1 = self.conv_tohsi(err1)

        # out = (1 - DELTA - DELTA * ETA) * z + DELTA * err1 + DELTA * err2

        if ema:
            if z is not None:
                return DELTA * (err1 + up) + (1 - DELTA) * z - (DELTA * ETA) * (v + z)

            else:
                return z + DELTA * (err1 + err2)

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

    def forward(self, up, hs, rgb):  # [batch_size ,3 ,7 ,270 ,480] ;
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
        
        
        x = self.recon(conv_out, x, y, RGB, id_layer=3)

        
        
        z = x
        v, fe2 = self.spatial(self.fe_conv3(torch.cat((self.fe_conv1(z), fe), 1)))
        v = v + z
        z = self.recon_noisy(z, x, v, RGB, 0)
        conv_out, fe3 = self.spatial(
            self.fe_conv4(torch.cat((self.fe_conv1(z), fe2), 1))
        )
        conv_out = conv_out + z
        x = self.recon(conv_out, x, y, RGB, id_layer=3)

        z = x
        v, fe4 = self.spatial(self.fe_conv5(torch.cat((self.fe_conv1(z), fe, fe2), 1)))
        v = v + z
        z = self.recon_noisy(z, x, v, RGB, 0)
        conv_out, fe5 = self.spatial(
            self.fe_conv6(torch.cat((self.fe_conv1(z), fe4), 1))
        )
        conv_out = conv_out + z
        x = self.recon(conv_out, x, y, RGB, id_layer=3)

        z = x
        v, fe6 = self.spatial(
            self.fe_conv7(torch.cat((self.fe_conv1(z), fe, fe2, fe4), 1))
        )
        v = v + z
        z = self.recon_noisy(z, x, v, RGB, 0)
        conv_out, _ = self.spatial(self.fe_conv8(torch.cat((self.fe_conv1(z), fe6), 1)))
        conv_out = conv_out + z

        return conv_out

    def train_step(self, batch, *args, **kwargs):
        gt, up, hsi, msi = (
            batch["gt"].cuda(),
            batch["up"].cuda(),
            batch["lrhsi"].cuda(),
            batch["rgb"].cuda(),
        )
        sr = self(hsi, msi)
        loss = self.criterion(sr, gt, *args, **kwargs)
        log_vars = {}
        with torch.no_grad():
            metrics = analysis_accu(gt, sr, 4, choices=4)
            log_vars.update(metrics)

        return {"loss": loss, "log_vars": log_vars}

    def eval_step(self, batch, *args, **kwargs):
        gt, up, hsi, msi = (
            batch["gt"].cuda(),
            batch["up"].cuda(),
            batch["lrhsi"].cuda(),
            batch["rgb"].cuda(),
        )
        # batch['lrhsi'].cuda(), \
        # print(msi.shape)
        sr1 = self.forward(hsi, msi)
        # x = torch.cat((up, msi), 1)
        # sr1 = self.forward_chop(x)
        with torch.no_grad():
            metrics = analysis_accu(gt[0].permute(1, 2, 0), sr1[0].permute(1, 2, 0), 4)
            metrics.update(metrics)

        return sr1, metrics

    def set_metrics(self, criterion, rgb_range=1.0):
        self.rgb_range = rgb_range
        self.criterion = criterion


def build(args):
    scheduler = None
    mode = "one"
    loss1 = nn.L1Loss().cuda()
    weight_dict = {"Loss": 1}
    losses = {"Loss": loss1}
    criterion = SetCriterion(losses, weight_dict)
    data = sio.loadmat("./HISR/MoGDCN/P.mat")
    P = data["P"]
    P = torch.FloatTensor(P)
    up_factor = 4
    channel = 31
    patch_size = 64
    WEIGHT_DECAY = 1e-8  # params of ADAM
    model = VSR_CAS(
        channel0=channel, factor=up_factor, P=P, patch_size=patch_size
    ).cuda()

    num_params = 0
    for param in VSR_CAS(
        channel0=channel, factor=up_factor, P=P, patch_size=patch_size
    ).parameters():
        num_params += param.numel()
    print(
        "[Network %s] Total number of parameters : %.3f M"
        % ("MoGnet", num_params / 1e6)
    )
    model.set_metrics(criterion)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY
    )

    return model, criterion, optimizer, scheduler


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    LR = torch.randn(1, 31, 16, 16).cuda()
    RGB = torch.randn(1, 3, 64, 64).cuda()
    data = sio.loadmat("./P.mat")
    P = data["P"]
    P = torch.FloatTensor(P)
    up_factor = 4
    channel = 31
    patch_size = 15
    WEIGHT_DECAY = 1e-8  # params of ADAM
    model = VSR_CAS(
        channel0=channel, factor=up_factor, P=P, patch_size=patch_size
    ).cuda()
    pred = model(LR, RGB)
    print(pred.shape)
    # summary(model, input_size=[(31, 16, 16), (3, 64, 64)], batch_size=1)
