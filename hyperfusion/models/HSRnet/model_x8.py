# GPL License
# Copyright (C) UESTC
# All Rights Reserved 
#
# @Time    : 2023/5/21 22:51
# @Author  : Xiao Wu
# @reference: 
#
import torch
from torch import nn


def _phase_shift(I, r):
    bsize, c, h, w = I.shape
    bsize = I.shape[0]
    X = torch.reshape(I, (bsize, r, r, h, w))
    X = torch.chunk(X, chunks=w, dim=3)  # 在w通道上分成了w份， 将每一维分成了1
    # tf.squeeze删除axis上的1，然后在第三通道 即r通道上 将w个小x重新级联变成r * w
    X = torch.concat([torch.squeeze(x, dim=3) for x in X], 1)  # 最终变成 bsize, h, r * w, r
    X = torch.chunk(X, chunks=h, dim=3)
    X = torch.cat([torch.squeeze(x, dim=3) for x in X], 2)

    return torch.reshape(X, (bsize, 1, h * r, w * r))  # 最后变成这个shape


class HSRNet(nn.Module):
    def __init__(self, spectral_num, num_res=6, num_feature=64):
        super().__init__()

        self.ms_ch_attn = nn.Sequential(
            nn.Conv2d(31, 1, kernel_size=1, stride=1),
            nn.Conv2d(1, 31, kernel_size=1, stride=1)
        )

        self.rgb_spa_attn = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2)
        # (64 - k + 2*p)/4 + 1 = 16
        # 58/4 + p/2 = 15
        # (128-k+2*p)/8 + 1 = 16
        # 128-k+2*p = 120
        self.rgb_conv = nn.Conv2d(3, 3, kernel_size=6, stride=8, padding=1)  # kernel_size=6, stride=4

        self.rs_conv1 = nn.Conv2d(spectral_num+3, spectral_num * 2 * 2, kernel_size=3, stride=1, padding=1)
        self.rs_conv2 = nn.Conv2d(spectral_num, spectral_num * 2 * 2, kernel_size=3, stride=1, padding=1)
        self.rs_conv3 = nn.Conv2d(spectral_num, spectral_num * 2 * 2, kernel_size=3, stride=1, padding=1)

        self.rs_conv41 = nn.Conv2d(spectral_num + 3, num_feature, kernel_size=3, stride=1, padding=1)
        self.rs_blocks = nn.ModuleList()
        for i in range(num_res):
            self.rs_blocks.append(nn.Sequential(
                nn.Conv2d(num_feature, num_feature, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(num_feature, num_feature, kernel_size=3, stride=1, padding=1)
            ))
        self.rs_conv42 = nn.Conv2d(num_feature, spectral_num, kernel_size=3, stride=1, padding=1)

    def forward(self, ms, RGB):
        gap_ms_c = torch.mean(ms, dim=[2, 3], keepdim=True)
        CA = self.ms_ch_attn(gap_ms_c)
        gap_RGB_s = torch.mean(RGB, dim=1, keepdim=True)
        SA = self.rgb_spa_attn(gap_RGB_s)
        rgb = self.rgb_conv(RGB)
        rslice, gslice, bslice = torch.split(rgb, [1, 1, 1], dim=1)
        msp1, msp2 = torch.split(ms, [15, 16], dim=1)
        ms = torch.cat([rslice, msp1, gslice, msp2, bslice], dim=1)

        rs = self.rs_conv1(ms)
        Xc = torch.chunk(rs, chunks=31, dim=1)
        rs = torch.cat([_phase_shift(x, 2) for x in Xc], 1)  # each of x in Xc is r * r channel 分别每一个通道变为r*r

        rs = self.rs_conv2(rs)
        Xc = torch.chunk(rs, chunks=31, dim=1)
        rs = torch.cat([_phase_shift(x, 2) for x in Xc], 1)  # each of x in Xc is r * r channel 分别每一个通道变为r*r

        rs = self.rs_conv3(rs)
        Xc = torch.chunk(rs, chunks=31, dim=1)
        rs = torch.cat([_phase_shift(x, 2) for x in Xc], 1)  # each of x in Xc is r * r channel 分别每一个通道变为r*r


        Rslice, Gslice, Bslice = torch.split(RGB, [1, 1, 1], dim=1)
        Msp1, Msp2 = torch.split(rs, [15, 16], dim=1)
        rs = torch.concat([Rslice, Msp1, Gslice, Msp2, Bslice], dim=1)
        rs = self.rs_conv41(rs)
        for rs_block in self.rs_blocks:
            rs = rs + rs_block(rs)
        rs = SA * rs
        rs = self.rs_conv42(rs)
        rs = CA * rs
        return rs, CA, SA

    def flops(self, inp, out):
        flops = 0
        B, C, H, W = inp[1].size()
        flops += B * C * H * W * (2+len(self.rs_blocks))

        return flops

if __name__ == '__main__':
    from udl_vis.Basis.auxiliary import torchstat
    # ms = torch.randn([1, 31, 16, 16])
    # RGB = torch.randn([1, 3, 64, 64])
    model = HSRNet(31).cuda()
    torchstat.stat(model, [[1, 31, 16, 16], [1, 3, 128, 128]])
    '''
    (first four is divided by 1000)
    Total params: 588.63K 0.5886M (588,626)
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------
    Total memory: 56.55MB
    Total MAdd: 16.1GMAdd 0.0161TMAdd
    Total Flops: 8.06GFlops 0.0081TFlops
    Total MemR+W: 113.81MB 0.1111GB
    '''