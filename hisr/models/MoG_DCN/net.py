# coding=UTF-8
import torch


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
        self.up = torch.nn.ConvTranspose2d(
            c_in, 128, kernel_size=3, stride=2, padding=3 // 2
        )

        self.act = torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

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


class ResConv(torch.nn.Module):
    def __init__(self, cin, mid, cout, kernel_size=3):
        super(ResConv, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=cin, out_channels=mid, kernel_size=kernel_size, padding="same"
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=mid, out_channels=cout, kernel_size=1, padding="same"
        )
        self.relu = torch.nn.PReLU()

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        out = x + res
        return out


class Unet_Spatial(torch.nn.Module):
    def __init__(self, cin):
        super(Unet_Spatial, self).__init__()

        self.Encoding_block1 = Encoding_Block(31)
        self.Encoding_block2 = Encoding_Block(64)
        self.Encoding_block3 = Encoding_Block(64)
        self.Encoding_block4 = Encoding_Block(64)
        self.Encoding_block_end = Encoding_Block_End(64)

        self.Decoding_block1 = Decoding_Block(128)
        self.Decoding_block2 = Decoding_Block(512)
        self.Decoding_block3 = Decoding_Block(512)
        self.Decoding_block_End = Feature_Decoding_End(31)

        self.acti = torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        encode0, down0 = self.Encoding_block1(x)
        encode1, down1 = self.Encoding_block2(down0)
        encode2, down2 = self.Encoding_block3(down1)
        encode3, down3 = self.Encoding_block4(down2)

        media_end = self.Encoding_block_end(down3)

        decode3 = self.Decoding_block1(media_end, encode3)
        decode2 = self.Decoding_block2(decode3, encode2)
        decode1 = self.Decoding_block3(decode2, encode1)
        decode0 = self.Decoding_block_End(decode1, encode0)

        return decode0, encode0


class VSR_CAS(torch.nn.Module):
    """
    network of 'Burst Denoising with Kernel Prediction Networks'
    """

    def __init__(self, criterion, channel0, factor, P=None):
        super(VSR_CAS, self).__init__()

        self.criterion = criterion
        self.channel0 = channel0

        self.acti = torch.nn.PReLU()

        self.delta_0 = torch.nn.Parameter(torch.tensor(0.1))
        self.eta_0 = torch.nn.Parameter(torch.tensor(0.9))

        # self.spatial = ResConv(31, 64, 31) #
        self.spatial = Unet_Spatial(31)
        self.fe_conv1 = torch.nn.Conv2d(
            in_channels=31, out_channels=64, kernel_size=3, padding=3 // 2
        )
        self.fe_conv2 = torch.nn.Conv2d(
            in_channels=192, out_channels=64, kernel_size=3, padding=3 // 2
        )
        self.fe_conv3 = torch.nn.Conv2d(
            in_channels=192, out_channels=64, kernel_size=3, padding=3 // 2
        )
        self.fe_conv4 = torch.nn.Conv2d(
            in_channels=192, out_channels=64, kernel_size=3, padding=3 // 2
        )

        self.conv_downsample = torch.nn.Conv2d(
            in_channels=31,
            out_channels=31,
            kernel_size=13,
            stride=factor,
            padding=13 // 2,
        )
        self.conv_upsample = torch.nn.ConvTranspose2d(
            in_channels=31,
            out_channels=31,
            kernel_size=13,
            stride=factor,
            padding=13 // 2,
        )
        self.conv_torgb = torch.nn.Conv2d(
            in_channels=31, out_channels=3, kernel_size=3, stride=1, padding=3 // 2
        )
        self.conv_tohsi = torch.nn.Conv2d(
            in_channels=3, out_channels=31, kernel_size=3, stride=1, padding=3 // 2
        )

        self.reset_parameters()

    def recon_noisy(self, z, noisy, v, RGB, id_layer):

        DELTA = self.delta_0
        ETA = self.eta_0

        # sz = z.shape
        # err1 = RGB - self.conv_torgb(z)
        # err1 = self.conv_tohsi(err1)
        # err2 = noisy - ETA * (v + z)
        # err2 = err2.reshape(sz)

        # out = - DELTA * ETA * (v + 2*z) + ((1-DELTA) * z + DELTA * noisy)# + DELTA * err1
        # out = - DELTA * ETA * (v + z) + ((1 - DELTA) * z + DELTA * noisy)  # + DELTA * err1

        # return out

        return (1 - DELTA) * z + DELTA * noisy

    def recon(self, features, recon, LR, RGB):
        DELTA = self.delta_0
        ETA = self.eta_0

        sz = recon.shape
        down = self.conv_downsample(recon)
        err1 = self.conv_upsample(down - LR, output_size=sz)

        to_rgb = self.conv_torgb(recon)
        err_rgb = RGB - to_rgb
        err3 = self.conv_tohsi(err_rgb)
        err3 = err3.reshape(sz)
        ################################################################
        # features - recon: 误差？ F范数？ x
        out = recon - DELTA * (err3 + err1) + DELTA * ETA * (features - recon)
        ################################################################
        # out = (err3+err1)
        #
        return out

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x, y, RGB):  # [batch_size ,3 ,7 ,270 ,480] ;
        # x: HSI_UP [1, 31, 48, 48]  [1, 31, 64, 64]
        # y: LR [1 31 6 6] [1, 31, 16, 16]
        # RGB [1 31 48 48] [1, 31, 64, 64]
        # z = x
        # prox: refinement, proxnet, v=conv_out
        conv_out = self.spatial(x)[0]
        conv_out = conv_out + x

        for _ in range(3):
            # initialize and ema
            # V = lambda(Z - gradient) + lambda*eta(V-Z)
            x = self.recon(conv_out, x, y, RGB)
            z = x
            # prox: refinement, proxnet, v=conv_out
            conv_out = self.spatial(z)[0]
            conv_out = conv_out + z

        return conv_out

    def train_step(self, data, *args, **kwargs):
        gt, up_hs, hs, rgb = (
            data["gt"].cuda(),
            data["up"].cuda(),
            data["lrhsi"].cuda(),
            data["rgb"].cuda(),
        )
        sr = self(up_hs, hs, rgb)
        loss = self.criterion(sr, gt, *args, **kwargs)["loss"]

        return sr, loss

    def val_step(self, data, *args, **kwargs):
        # gt, lms, ms, pan = data
        gt, up_hs, hs, rgb = (
            data["gt"].cuda(),
            data["up"].cuda(),
            data["lrhsi"].cuda(),
            data["rgb"].cuda(),
        )
        sr = self(up_hs, hs, rgb)

        return sr


from hisr.models.base_model import HISRModel
from hisr.common.metrics import SetCriterion
from torch import optim
from torch import nn


class build_MoG_DCN(HISRModel, name="MoG_DCN"):
    def __call__(self, cfg):
        loss = nn.L1Loss(size_average=True).cuda()  ## Define the Loss function
        weight_dict = {"loss": 1}
        losses = {"loss": loss}
        criterion = SetCriterion(losses, weight_dict)
        model = VSR_CAS(criterion, 31, cfg.up_factor).cuda()
        optimizer = optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=1e-8
        )  ## optimizer 1: Adam
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500)
        scheduler = None

        return model, criterion, optimizer, scheduler


if __name__ == "__main__":
    ratio = 4
    model = VSR_CAS(None, 31, ratio).cuda()
    x = [
        torch.randn(shape).cuda()
        for shape in [[1, 31, 64, 64], [1, 31, 16, 16], [1, 3, 64, 64]]
    ]
    print(model(*x).shape)
