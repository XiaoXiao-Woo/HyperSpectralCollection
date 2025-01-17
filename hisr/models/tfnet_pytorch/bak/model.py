import torch
import torch.nn as nn
import math

class TFNet(nn.Module):
    def __init__(self, spectral_num, rgb_channel):
        super(TFNet, self).__init__()
        self.encoder1_pan=nn.Sequential(
            nn.Conv2d(in_channels=rgb_channel,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )
        self.encoder2_pan = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.encoder1_lr=nn.Sequential(
            nn.Conv2d(in_channels=spectral_num,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU())
        self.encoder2_lr = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.fusion1=nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )
        self.fusion2=nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=2,
                      stride=2),
            nn.PReLU(),
        )
        self.restore1=nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      ),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=2,
                               stride=2),
            nn.PReLU())
        self.restore2=nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=2,
                               stride=2),
            nn.PReLU()
        )
        self.restore3=nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=spectral_num,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Tanh()
        )
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         if m.bias is not None:
        #             m.bias.data.zero_()

    def forward(self, x_pan, x_lr):
        encoder1_pan = self.encoder1_pan(x_pan)
        encoder1_lr = self.encoder1_lr(x_lr)

        encoder2_pan = self.encoder2_pan(encoder1_pan)
        encoder2_lr = self.encoder2_lr(encoder1_lr)

        fusion1 = self.fusion1(torch.cat((encoder2_pan, encoder2_lr), dim=1))
        fusion2 = self.fusion2(fusion1)

        restore1 = self.restore1(fusion2)
        restore2 = self.restore2(torch.cat((restore1, fusion1),dim=1))
        restore3 = self.restore3(torch.cat((restore2, encoder1_lr, encoder1_pan), dim=1))

        return restore3

    def train_step(self, data, *args, **kwargs):
        # log_vars = {}
        gt, up_hs, ms, rgb = data['gt'].cuda(), data['up'].cuda(), \
                           data['lrhsi'].cuda(), data['rgb'].cuda()
        sr = self(rgb, up_hs)
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
        sr = self(rgb, up_hs)

        return sr, gt

from models.base_model import HISRModel
from common.metrics import SetCriterion
from torch import optim
class build_TFNet(HISRModel, name='TFNet'):
    def __call__(self, cfg):
        spectral_num = 31
        loss = nn.L1Loss(size_average=True).cuda()  ## Define the Loss function
        weight_dict = {'loss': 1}
        losses = {'loss': loss}
        criterion = SetCriterion(losses, weight_dict)
        model = TFNet(spectral_num, 3).cuda()
        model.criterion = criterion
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0)   ## optimizer 1: Adam
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=200,
                                                       gamma=0.1)

        return model, criterion, optimizer, scheduler