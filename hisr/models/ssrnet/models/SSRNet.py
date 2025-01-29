import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from udl_vis.Basis.module import PatchMergeModule

def spatial_edge(x):
    edge1 = x[:, :, 0 : x.size(2) - 1, :] - x[:, :, 1 : x.size(2), :]
    edge2 = x[:, :, :, 0 : x.size(3) - 1] - x[:, :, :, 1 : x.size(3)]

    return edge1, edge2


def spectral_edge(x):
    edge = x[:, 0 : x.size(1) - 1, :, :] - x[:, 1 : x.size(1), :, :]

    return edge


class SSRNET(PatchMergeModule):
    def __init__(
        self, arch, n_bands, rgb_channel, scale_ratio=None, n_select_bands=None
    ):
        """Load the pretrained ResNet and replace top fc layer."""
        super(SSRNET, self).__init__()
        self.scale_ratio = scale_ratio
        self.n_bands = n_bands
        self.arch = arch
        self.n_select_bands = n_select_bands
        self.weight = nn.Parameter(torch.tensor([0.5]))

        self.conv_fus = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv_spat = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv_spec = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def lrhr_interpolate(self, x_lr, x_hr):
        x_lr = F.interpolate(x_lr, scale_factor=self.scale_ratio, mode="bilinear")
        gap_bands = self.n_bands / (self.n_select_bands - 1.0)
        for i in range(0, self.n_select_bands - 1):
            x_lr[:, int(gap_bands * i), ::] = x_hr[:, i, ::]
        x_lr[:, int(self.n_bands - 1), ::] = x_hr[:, self.n_select_bands - 1, ::]

        return x_lr

    def spatial_edge(self, x):
        edge1 = x[:, :, 0 : x.size(2) - 1, :] - x[:, :, 1 : x.size(2), :]
        edge2 = x[:, :, :, 0 : x.size(3) - 1] - x[:, :, :, 1 : x.size(3)]

        return edge1, edge2

    def spectral_edge(self, x):
        edge = x[:, 0 : x.size(1) - 1, :, :] - x[:, 1 : x.size(1), :, :]

        return edge

    def _forward_implem(self, x, x_lr, x_hr):
        x = self.lrhr_interpolate(x_lr, x_hr)
        # x = x.cuda()
        # x = torch.cat((x, x_hr), 1)
        x = self.conv_fus(x)

        if self.arch == "SSRNET":
            x_spat = x + self.conv_spat(x)
            spat_edge1, spat_edge2 = self.spatial_edge(x_spat)

            x_spec = x_spat + self.conv_spec(x_spat)
            spec_edge = self.spectral_edge(x_spec)

            x = x_spec

        elif self.arch == "SpatRNET":
            x_spat = x + self.conv_spat(x)

            spat_edge1, spat_edge2 = self.spatial_edge(x_spat)
            x_spec = x
            spec_edge = self.spectral_edge(x_spec)

        elif self.arch == "SpecRNET":
            x_spat = x
            spat_edge1, spat_edge2 = self.spatial_edge(x_spat)

            x_spec = x + self.conv_spec(x)
            spec_edge = self.spectral_edge(x_spec)

            x = x_spec

        return x, x_spat, x_spec, spat_edge1, spat_edge2, spec_edge

    def train_step(self, data, **kwargs):

        gt, up_hs, hs, rgb = (
            data["gt"].cuda(),
            data["up"].cuda(),
            data["lrhsi"].cuda(),
            data["rgb"].cuda(),
        )
        out, out_spat, out_spec, edge_spat1, edge_spat2, edge_spec = (
            self._forward_implem(up_hs, hs, rgb)
        )
        ref_edge_spat1, ref_edge_spat2 = spatial_edge(gt)
        ref_edge_spec = spectral_edge(gt)
        loss_fus = self.criterion(out, gt)
        loss_spat = self.criterion(out_spat, gt)
        loss_spec = self.criterion(out_spec, gt)
        loss_spec_edge = self.criterion(edge_spec, ref_edge_spec)
        loss_spat_edge = 0.5 * self.criterion(
            edge_spat1, ref_edge_spat1
        ) + 0.5 * self.criterion(edge_spat2, ref_edge_spat2)
        if self.arch == "SpatRNET":
            loss = loss_spat + loss_spat_edge
        elif self.arch == "SpecRNET":
            loss = loss_spec + loss_spec_edge
        elif self.arch == "SSRNET":
            loss = loss_fus + loss_spat_edge + loss_spec_edge

        return {"loss": loss}

    def val_step(self, data, **kwargs):
        # gt, lms, ms, pan = data
        up_hs, hs, rgb = (
            data["up"].cuda(),
            data["lrhsi"].cuda(),
            data["rgb"].cuda(),
        )
        sr = self._forward_implem(up_hs, hs, rgb)[0]

        return sr 

    def test_step(self, data, **kwargs):
        return self.val_step(data, **kwargs)


from hisr.models.base_model import HISRModel
from hisr.common.metrics import SetCriterion
from torch import optim, nn
from udl_vis.Basis.optim import LRScheduler, get_optimizer


class build_SSRNet(HISRModel, name="SSRNet"):

    def __call__(self, cfg, logger=None):
        # spectral_num = 31
        # if cfg.dataset['train'] == 'PaviaU':
        #     spectral_num = 103
        # elif cfg.dataset['train'] == 'Pavia':
        #     spectral_num = 102
        # elif cfg.dataset['train'] == 'Botswana':
        #     spectral_num = 145
        # elif cfg.dataset['train'] == 'KSC':
        #     spectral_num = 176
        # elif cfg.dataset['train'] == 'Urban':
        #     spectral_num = 162
        # elif cfg.dataset['train'] == 'IndianP':
        #     spectral_num = 200
        # elif cfg.dataset['train'] == 'Washington':
        #     spectral_num = 191
        # elif cfg.dataset['train'] == 'GF5-GF1':
        #     spectral_num = 150

        model = SSRNET(
            "SSRNET",
            n_bands=cfg.hs_channel,
            rgb_channel=cfg.ms_channel,
            scale_ratio=cfg.scale,
            n_select_bands=cfg.n_select_bands,
        ).cuda()
        criterion = nn.MSELoss(size_average=True)
        model.criterion = criterion
        optimizer = get_optimizer(model, model.parameters(), **cfg.optimizer_cfg)
        scheduler = None
        
        return model, criterion, optimizer, scheduler


if __name__ == "__main__":
    data = {
        "gt": torch.randn(1, 31, 64, 64).cuda(),
        "up": torch.randn(1, 31, 64, 64).cuda(),
        "lrhsi": torch.randn(1, 31, 16, 16).cuda(),
        "rgb": torch.randn(1, 3, 64, 64).cuda(),
    }

    model = SSRNET("SSRNET", 31, 3, scale_ratio=4, n_select_bands=3).cuda()
    model.criterion = nn.MSELoss(size_average=True)
    # stat(model, [[1, 31, 64, 64], [1, 3, 64, 64]])
    print(model.train_step(data)[0].shape)
    """
    Total params: 26.88K 0.0269M (26,877)
    Total memory: 2.91MB
    Total MAdd: 219.8MMAdd 0.2198GMAdd
    Total Flops: 110.47MFlops 0.1105GFlops
    Total MemR+W: 5.96MB 0.0058GB
    
    Total Flops: 441.88MFlops 0.4419GFlops
    """
