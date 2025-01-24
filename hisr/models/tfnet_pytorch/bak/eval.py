import numpy as np

import argparse
import torch
from data import get_test
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import scipy.io as sio

from PIL import Image

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--dataset', type=str, default='F:\Data\HSI\cave_x4')
parser.add_argument('--checkpoint', type=str, default='2000')
parser.add_argument("--net", type=str, default='resnet', choices={'resnet','tfnet'})
parser.add_argument('--testBatchSize', type=int, default=96, help='testing batch size')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
opt = parser.parse_args()

test_set = get_test(opt.dataset)
test_data_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=1)
model = torch.load('model/resnet_cave_x4_202212282036/model_epoch_%s.pth'%opt.checkpoint)

image_path = 'images/%s'%opt.checkpoint
print(image_path)

def test(test_data_loader, model):
    model.eval()
    num_params = 0
    output = np.zeros((11, 512, 512, 31))
    for param in model.parameters():
        num_params += param.numel()
    print('[Network %s] Total number of parameters : %.3f M' % ('restf', num_params / 1e6))

    for index,batch in enumerate(test_data_loader):
        input_pan, input_lr, input_lr_u, target = Variable(batch[0]), Variable(batch[1]).cuda(), Variable(batch[2]).cuda(),Variable(batch[3], requires_grad=False).cuda()
        # input_pan = input_pan.numpy()
        # input_pan = input_pan[:, ::-1, :, :]
        # input_pan = torch.tensor(input_pan.copy()).cuda()
        input_pan = input_pan.cuda()
        input_lr_u = input_lr_u.cuda()
        out = model(input_pan, input_lr_u)
        output[index, :, :, :] = out.permute(0, 2, 3, 1).cpu().detach().numpy()

    sio.savemat('cave11-restf.mat', {'output': output})




if not os.path.exists(image_path):
    os.makedirs(image_path)


test(test_data_loader, model['model'])











