import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ResNet, TFNet


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

class LIIF(nn.Module):
    def __init__(self, encoder_spec=None, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        # self.encoder = models.make(encoder_spec)
        # self.encoder = nn.Conv2d(in_channels=31, out_channels=31, kernel_size=3, padding=1, stride=1)
        self.net = TFNet(input_channel=35, output_channel=31)

        # output = self.net(input_pan, input_lr_u)

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                imnet_in_dim *= 9
            imnet_in_dim += 2 # attach coord
            if self.cell_decode:
                imnet_in_dim += 2
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None


    def gen_feat(self, inp):
        # self.feat = self.encoder(inp)
        self.feat = inp
        return self.feat

    def query_rgb(self, rgb, coord, cell=None):

        feat = self.feat  # generate feature map

        if self.feat_unfold:  # unfold to stack the feature map
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:  # use the information (up, down, left, and right)
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        # feat_coord (h, w, 2)
        # permute feat_coord (2, h, w)
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda().permute(2, 0, 1).unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
        # expand feat_coord (B, 2, h, w)

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                c = coord_ - coord
                # q_feat = F.grid_sample(
                #     feat, coord_.flip(-1).unsqueeze(1),
                #     mode='nearest', align_corners=False)[:, :, 0, :] \
                #     .permute(0, 2, 1)
                # BxCxhxw --> BxCxHxW --> BxCxW --> BxWxC

                q_feat = F.grid_sample(feat, coord_, mode='nearest', align_corners=False)
                q_rgb = F.grid_sample(rgb, coord_, mode='nearest', align_corners=False)
                # BxCxHxW

                # q_coord = F.grid_sample(
                #     feat_coord, coord_.flip(-1).unsqueeze(1),
                #     mode='nearest', align_corners=False)[:, :, 0, :] \
                #     .permute(0, 2, 1)
                # Bx2xhxw --> Bx2xHxW --> Bx2xW --> BxWx2
                q_coord = F.grid_sample(feat_coord, coord_, mode='nearest', align_corners=False).permute(0, 2, 3, 1)
                # Bx2xHxW

                rel_coord = coord - q_coord
                rel_coord[:, :, :, 0] *= feat.shape[-2]
                rel_coord[:, :, :, 1] *= feat.shape[-1]
                rel_coord = rel_coord.permute(0, 3, 1, 2)
                inp = torch.cat([q_feat, rel_coord], dim=1)  # C_fold + 2

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, :, 1] *= feat.shape[-1]
                    rel_cell = rel_cell.permute(0, 3, 1, 2)
                    inp = torch.cat([inp, rel_cell], dim=1)  # C_fold + 2 + 2

                # bs, q = coord.shape[:2]
                # pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                inp = self.net(q_rgb, inp)
                preds.append(inp)

                rel_coord = rel_coord.permute(0, 2, 3, 1)
                area = torch.abs(rel_coord[:, :, :, 0] * rel_coord[:, :, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(1)

        return ret

    def forward(self, rgb, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(rgb, coord, cell)


if __name__ == '__main__':
    import torch

    b, c, h, w = 7, 31, 64, 64
    rgb = torch.randn([b, 3, h, w]).cuda()
    input = torch.randn([b, c, h//4, w//4]).cuda()

    # generate coordinates and cells
    coord = make_coord((h, w), flatten=False).expand(input.shape[0], h, w, 2).cuda()  # BxHxWx2
    cell = torch.ones_like(coord).cuda()  # BxHxWx2
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w

    # print(input.shape)
    # print(coord.shape)
    # print(cell.shape)

    liif = LIIF(encoder_spec=None, imnet_spec=None,
                 local_ensemble=True, feat_unfold=False, cell_decode=True).cuda()
    output = liif(rgb, input, coord, cell)
    print(output.shape)
    # print(output.shape)



    # model = TFNet(input_channel=c).cuda()
    # input_pan = torch.randn([b, 3, h, w]).cuda()
    # input_lr_u = torch.randn([b, c, h, w]).cuda()
    # output = model(input_pan, input_lr_u)
    # print(output.shape)


