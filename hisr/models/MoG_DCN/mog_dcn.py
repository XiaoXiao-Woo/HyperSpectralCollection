import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import scipy.io as sio
from hisr.models.MoG_DCN.Unet import Unet
from udl_vis.Basis.module import PatchMergeModule
if __name__ == "__main__":
    from rich.traceback import install
    install()

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


class VSR_CAS(PatchMergeModule):
    """
    network of 'Burst Denoising with Kernel Prediction Networks'
    """

    def __init__(
        self,
        n_stages,
        hs_channel,
        ms_channel,
        num_channel=64,
        factor=4,
        prox=True,
        use_dense=False,
        featEmbed=False,
        args=None,
    ):
        super(VSR_CAS, self).__init__()

        self.n_stages = n_stages
        self.use_dense = use_dense

        self.hs_channel = hs_channel
        self.ms_channel = ms_channel
        self.factor = factor
        self.device = torch.cuda.current_device()
        self.prox = prox
        self.num_channel = num_channel
        self.act = torch.nn.PReLU()
        in_channel = num_channel if featEmbed else hs_channel
        self.spatial = Unet(in_channel, hs_channel, ms_channel)
        self.init_stages()
        self.init_recon()

        # self.fe_conv1 = torch.nn.Conv2d(in_channels=31, out_channels=64, kernel_size=3, padding=3 // 2)
        # self.fe_conv2 = torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=3 // 2)
        # self.fe_conv3 = torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=3 // 2)
        # self.fe_conv4 = torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=3 // 2)

        self.reset_parameters()

    def init_recon(self):

        if self.factor == 8:
            KERNEL_SIZE = 13
        else:
            KERNEL_SIZE = 3

        self.conv_downsample = torch.nn.Conv2d(
            in_channels=self.hs_channel,
            out_channels=self.hs_channel,
            kernel_size=KERNEL_SIZE,
            stride=self.factor,
            padding=KERNEL_SIZE // 2,
        )
        self.conv_upsample = torch.nn.ConvTranspose2d(
            in_channels=self.hs_channel,
            out_channels=self.hs_channel,
            kernel_size=KERNEL_SIZE,
            stride=self.factor,
            padding=KERNEL_SIZE // 2,
        )

        self.conv_topan = torch.nn.Conv2d(
            in_channels=self.hs_channel,
            out_channels=self.ms_channel,
            kernel_size=3,
            stride=1,
            padding=3 // 2,
        )
        self.conv_tolms = torch.nn.Conv2d(
            in_channels=self.ms_channel,
            out_channels=self.hs_channel,
            kernel_size=3,
            stride=1,
            padding=3 // 2,
        )

    def init_stages(self):
        if self.use_dense:
            # TODO: add in_channels, out_channels with list
            self.fe_conv = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=31,
                    out_channels=self.num_channel,
                    kernel_size=3,
                    padding=3 // 2,
                )
            for _ in range(self.n_stages + 1)
            ])
        else:
            self.fe_conv = nn.Conv2d(
                in_channels=self.hs_channel,
                out_channels=self.num_channel,
                kernel_size=3,
                padding=3 // 2,
            )

        # self.delta = nn.Parameter(torch.tensor(0.1))
        self.delta = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1)) for _ in range(self.n_stages)]
        )
        if self.prox:
            self.eta = nn.ParameterList(
                [nn.Parameter(torch.tensor(0.9)) for _ in range(self.n_stages)]
            )
            # self.eta = nn.Parameter(torch.tensor(0.9))

    def recon(self, features, up, lrhsi, rgb, DELTA, ETA=None):

        sz = up.shape
        down = self.conv_downsample(up)
        err1 = self.conv_upsample(down - lrhsi, output_size=sz)

        to_rgb = self.conv_topan(up)
        err_rgb = rgb - to_rgb
        err3 = self.conv_tolms(err_rgb)
        err3 = err3.reshape(sz)

        # out = (1 - DELTA * ETA) * recon + DELTA * err3 + DELTA * err1 + DELTA * ETA * features
        if self.prox:
            return up - DELTA * (err1 + err3) + DELTA * ETA * (features - up)
        else:
            return up - DELTA * (err1 + err3)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def _forward_implem(self, lms, ms, pan):

        if self.use_dense:
            x_now, fe = self.spatial(self.fe_conv[0](lms), pan)
            x_now = x_now + lms
        else:
            tmp = self.fe_conv(lms)
            x_now = self.spatial(tmp, pan)
            x_now = x_now + lms

        for i in range(self.n_stages):
            if self.use_dense:
                sr = self.recon(x_now, lms, ms, pan, self.delta[i], self.eta[i])
                x_now, fe = self.spatial(
                    self.fe_conv[i+1](torch.cat((self.fe_conv[0](sr), fe), 1)), pan
                )
                x_now = x_now + lms

            else:
                if self.prox:
                    sr = self.recon(x_now, lms, ms, pan, self.delta[i], self.eta[i])
                else:
                    sr = self.recon(x_now, lms, ms, pan, self.delta[i])
                x_now = self.spatial(self.fe_conv(sr), pan) + lms

        return x_now

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

    def set_metrics(self, criterion, rgb_range=1.0):
        self.rgb_range = rgb_range
        self.criterion = criterion


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
        model = VSR_CAS(cfg.stages, cfg.hs_channel, cfg.ms_channel, cfg.num_channel, cfg.factor, cfg.prox, cfg.use_dense, cfg.featEmbed).cuda()
        # optimizer = optim.Adam(
        #     model.parameters(), lr=cfg.lr, weight_decay=1e-8
        # )  ## optimizer 1: Adam
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500)
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
        n_stages=3, hs_channel=hs_channel, ms_channel=ms_channel, factor=factor,
        featEmbed=True
    ).cuda()
    
    model.criterion = nn.L1Loss().cuda()
    
    sr = model({"up": UP, "lrhsi": LR, "rgb": RGB, "gt": gt}, mode="train")
    print(sr.shape)
