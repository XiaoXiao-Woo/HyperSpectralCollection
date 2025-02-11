# coding=UTF-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from network_swinir import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

"""
modify fast DVD(vedio denoising) 
"""


class swin_Transformer(nn.Module):
    def __init__(self, n_feats, img_size=64, patch_size=4, depths=[6], num_heads=[6],
                 window_size=8, mlp_ratio=2, qkv_bias=True, qk_scale=None, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False, resi_connection='1conv',
                 **kwargs):
        super(swin_Transformer, self).__init__()

        self.patch_norm = patch_norm

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=n_feats, embed_dim=n_feats,
            norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=n_feats, embed_dim=n_feats,
            norm_layer=norm_layer if self.patch_norm else None)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        self.mlp_ratio = mlp_ratio
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = []  # nn.Sequential()
        for i_layer in range(1):
            layer = RSTB(dim=n_feats,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection

                         )
            self.layers.append(layer)
        self.layers = nn.Sequential(*self.layers)
        self.norm = norm_layer(n_feats)

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer((x, x_size))
        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        x = self.conv_after_body(self.forward_features(x)) + x

        return x


# a single branch of proposed SSPSR
class Prior(nn.Module):
    def __init__(self, n_colors, n_feats):
        super(Prior, self).__init__()
        kernel_size = 3
        self.head = nn.Conv2d(64, 128, kernel_size, padding=3 // 2)
        self.feture = nn.Conv2d(128, n_feats, kernel_size, padding=3 // 2)
        self.body = swin_Transformer(n_feats)
        # self.upsample = Upsampler(conv, up_scale, n_feats)

        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.conv3d1 = wn(nn.Conv3d(1, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)))
        self.conv3d2 = wn(nn.Conv3d(64, 1, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)))
        # self.conv3d3 = wn(nn.Conv3d(64, 1, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)))

        self.tail = nn.Conv2d(n_feats, n_colors, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.head(x)
        y = self.feture(x)
        y = self.body(y)
        y = self.relu(self.tail(y))
        y = y.unsqueeze(1)
        y = self.relu(self.conv3d1(y))
        y = self.conv3d2(y)
        y = y.squeeze(1)

        return y, x


class _3DT_Net(torch.nn.Module):

    def __init__(self, spectral_num, rgb_channel, factor, patch_size):
        super(_3DT_Net, self).__init__()

        # self.channel0 = channel0
        self.up_factor = factor
        self.patch_size = patch_size
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
        self.spatial = Prior(spectral_num, 180)
        self.fe_conv1 = torch.nn.Conv2d(in_channels=spectral_num, out_channels=64, kernel_size=3, padding=3 // 2)
        self.fe_conv2 = torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=3 // 2)
        self.fe_conv3 = torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=3 // 2)
        self.fe_conv4 = torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=3 // 2)

        self.fe_conv5 = torch.nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=3 // 2)
        self.fe_conv6 = torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=3 // 2)
        # self.fe_conv7 = torch.nn.Conv2d(in_channels=448, out_channels=64, kernel_size=3, padding=3 // 2)
        # self.fe_conv8 = torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=3 // 2)
        self.conv_downsample = torch.nn.Conv2d(in_channels=31, out_channels=31, kernel_size=13, stride=4,
                                               padding=13 // 2)
        self.conv_upsample = torch.nn.ConvTranspose2d(in_channels=31, out_channels=31, kernel_size=13, stride=4,
                                                      padding=13 // 2)
        self.conv_torgb = torch.nn.Conv2d(in_channels=31, out_channels=3, kernel_size=3, stride=1, padding=3 // 2)
        self.conv_tohsi = torch.nn.Conv2d(in_channels=rgb_channel, out_channels=31, kernel_size=3, stride=1, padding=3 // 2)
        # self.spatial2 = Unet_Spatial(3)  # if no use then comment it out
        self.bicubicsample = torch.nn.Upsample(scale_factor=factor, mode='bicubic', align_corners=False)

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
        err1 = RGB - self.conv_torgb(z)
        err1 = self.conv_tohsi(err1)
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

        sz = recon.shape
        down = self.conv_downsample(recon)
        err1 = self.conv_upsample(down - LR, output_size=sz)
        to_rgb = self.conv_torgb(recon)
        err_rgb = RGB - to_rgb
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

    def forward(self, LR, RGB):  # [batch_size ,3 ,7 ,270 ,480] ;
        ## LR [1 31 6 6]
        ## RGB [1 31 48 48]

        x = self.bicubicsample(LR)
        y = LR
        z = x
        v, fe = self.spatial(self.fe_conv1(z))
        v = v + z
        z = self.recon_noisy(z, x, v, RGB, 0)
        conv_out, fe1 = self.spatial(self.fe_conv2(torch.cat((self.fe_conv1(z), fe), 1)))
        conv_out = conv_out + z

        x = self.recon(conv_out, x, y, RGB, id_layer=3)
        z = x
        v, fe2 = self.spatial(self.fe_conv3(torch.cat((self.fe_conv1(z), fe), 1)))
        v = v + z
        z = self.recon_noisy(z, x, v, RGB, 0)
        conv_out, fe3 = self.spatial(self.fe_conv4(torch.cat((self.fe_conv1(z), fe2), 1)))
        conv_out = conv_out + z

        x = self.recon(conv_out, x, y, RGB, id_layer=3)
        z = x
        v, fe4 = self.spatial(self.fe_conv5(torch.cat((self.fe_conv1(z), fe, fe2), 1)))
        v = v + z
        z = self.recon_noisy(z, x, v, RGB, 0)
        conv_out, fe5 = self.spatial(self.fe_conv6(torch.cat((self.fe_conv1(z), fe4), 1)))
        conv_out = conv_out + z

        # x = self.recon(conv_out, x, y, RGB,  id_layer=3)
        # z = x
        # v,fe6 = self.spatial(self.fe_conv7(torch.cat((self.fe_conv1(z),fe,fe2,fe4),1)))
        # v=v+z
        # z = self.recon_noisy(z, x, v, RGB, 0)
        # conv_out,_ = self.spatial(self.fe_conv8(torch.cat((self.fe_conv1(z), fe6),1)))
        # conv_out=conv_out+z

        return conv_out

    def flops(self, inp, out):
        print([s.size() for s in inp])
        flops = 0
        B, C_lr, H, W = inp[0].size()
        _, C, H_lr, W_lr = inp[1].size()
        # v = v + z and conv_out = conv_out + z
        # recon_noisy
        flops += B * C * H * W * (1 + 5 + 1) + B * C_lr * H * W
        # recon
        flops += B * C * H_lr * W_lr + B * C * H * W * 4 + B * C_lr * H * W
        flops = flops * 3
        return flops

    def train_step(self, data, *args, **kwargs):

        gt, up_hs, hs, rgb = data['gt'].cuda(), data['up'].cuda(), \
                           data['lrhsi'].cuda(), data['rgb'].cuda()

        return model(hs, rgb)


    def val_step(self, data, *args, **kwargs):
        # gt, lms, ms, pan = data
        gt, up_hs, hs, rgb = data['gt'].cuda(), data['up'].cuda(), \
                           data['lrhsi'].cuda(), data['rgb'].cuda()
        sr = self(hs, rgb)

        return sr#, gt


from hisr.models.base_model import HISRModel
from hisr.common.metrics import SetCriterion
from torch import optim
from torch.optim import lr_scheduler


class build_3DTNet(HISRModel, name='3DTNet'):

    def __call__(self, cfg):
        model = _3DT_Net(cfg.spectral_num, rgb_channel=cfg.rgb_channel, factor=cfg.scale, patch_size=cfg.patch_size).cuda()
        criterion = nn.L1Loss(size_average=True)
        model.criterion = criterion
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[80], gamma=0.5)
        scheduler.by_epoch = True

        return model, criterion, optimizer, scheduler


if __name__ == '__main__':
    from udl_vis.Basis.auxiliary import torchstat

    # import sys
    # sys.path.append('../')
    # import argsParser
    # args = argsParser.argsParser()
    model = _3DT_Net(spectral_num=31, rgb_channel=3, factor=4, patch_size=64).cuda()
    # model = swin_Transformer(180).cuda()
    # model = SwinTransformerBlock(180, (64, 64), 6, 8, shift_size=0).cuda()
    # torchstat.stat(model, [[1, 64*64, 180]])

    # torchstat.stat(model, [[1, 31, 16, 16], [1, 3, 64, 64]])
    # torchstat.stat(model, [[1, 180, 64, 64]])
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    LR = torch.randn(1, 31, 16, 16).float().cuda()
    RGB = torch.randn(1, 3, 64, 64).float().cuda()
    print(flop_count_table(FlopCountAnalysis(model, (LR, RGB))))  # (LR, RGB)
    # print(flop_count_table(FlopCountAnalysis(model, torch.rand(1, 180, 64, 64).cuda()))) # (LR, RGB)

    '''
    (first four is divided by 1000)
    Total params: 3.46M 0.0035G (3,463,535)
    ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Total memory: 252.51MB
    Total MAdd: 13.0GMAdd 0.013TMAdd
    Total Flops: 14.06GFlops 0.0141TFlops
    Total MemR+W: 440.85MB 0.4305GB
    '''
