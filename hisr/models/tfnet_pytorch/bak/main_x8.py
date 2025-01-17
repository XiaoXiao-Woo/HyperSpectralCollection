import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import ResNet, TFNet
from data import get_training_set, get_test_set
import random
import os
import time
from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=96, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--dataset', type=str, default='F:\Data\HSI\harvard_x8')
parser.add_argument('--dataname', type=str, default='harvard_x8')
parser.add_argument("--resume", default='model/tfnet_202210261623/', type=str ,help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--step", type=int, default=250, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--net", type=str, default='tfnet', choices={'resnet','tfnet'})
parser.add_argument("--log", type=str, default="log/")
opt = parser.parse_args()

def main():
    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1,10000)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print('===> Loading datasets')
    train_set = get_training_set(opt.dataset)
    test_set = get_test_set(opt.dataset)
    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True)
    test_data_loader = DataLoader(dataset=test_set, batch_size=opt.testBatchSize, shuffle=False)

    print("===> Building model")
    if (opt.net=='resnet'):
        model = ResNet().cuda()
    else:
        model = TFNet().cuda()
    criterion = nn.L1Loss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()


    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    t = time.strftime("%Y%m%d%H%M")
    train_log_path = os.path.join(opt.log, "%s_%s_train.log")%(opt.net, t)
    test_log_path = os.path.join(opt.log, "%s_%s_test.log")%(opt.net, t)
    # if os.path.exists(train_log_path):
    #     pass
    # else:
    #     os.makedirs(train_log_path)
    #
    # if os.path.exists(test_log_path):
    #     pass
    # else:
    #     os.makedirs(test_log_path)

    train_log = open(train_log_path, "w")
    test_log = open(test_log_path, "w")

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch, train_log)
        if epoch%50==0:
            # test(test_data_loader, model, criterion, epoch, test_log)
            save_checkpoint(model, epoch, t, opt.dataname)
    train_log.close()
    test_log.close()

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.5 ** (epoch // opt.step))
    return lr

def train(training_data_loader, optimizer, model, criterion, epoch, train_log):
    lr = adjust_learning_rate(optimizer, epoch - 1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print ("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        input_pan, input_lr, input_lr_u, target = Variable(batch[0]).cuda(), Variable(batch[1]).cuda(), Variable(batch[2]).cuda(),Variable(batch[3], requires_grad=False).cuda()
        if(opt.net=="resnet"):
            output = model(input_pan, input_lr_u)
        else:
            output = model(input_pan, input_lr_u)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_log.write("{} {:.10f}\n".format((epoch-1)*len(training_data_loader)+iteration, loss.item()))
        if iteration%10 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader),
                                                                loss.item()))

def test(test_data_loader, model, criterion, epoch, test_log):
    avg_l1 = 0
    model.eval()
    for index,batch in enumerate(test_data_loader):
        input_pan, input_lr, input_lr_u, target = Variable(batch[0],volatile=True).cuda(), Variable(batch[1],volatile=True).cuda(), Variable(batch[2].cuda(),volatile=True).cuda(),Variable(batch[3], requires_grad=False,volatile=True).cuda()
        if opt.cuda:
            input_pan = input_pan.cuda()
#            input_lr = input_lr.cuda()
            input_lr_u = input_lr_u.cuda()
            target = target.cuda()
        if (opt.net == "resnet"):
            output = model(input_pan, input_lr_u)
        else:
            output = model(input_pan, input_lr_u)
        loss = criterion(output, target)
        avg_l1 += loss.item()
        # ref = target.detach().cpu().numpy()
        # out = output.detach().cpu().numpy()
        # psnr = calc_psnr(ref, out)
        # rmse = calc_rmse(ref, out)
        # ergas = calc_ergas(ref, out)
        # sam = calc_sam(ref, out)
        # print('RMSE:   {:.4f};'.format(rmse))
        # print('PSNR:   {:.4f};'.format(psnr))
        # print('ERGAS:   {:.4f};'.format(ergas))
        # print('SAM:   {:.4f}.'.format(sam))

    del (input_pan, input_lr, input_lr_u, target, output)
    test_log.write("{} {:.10f}\n".format((epoch-1), avg_l1 / len(test_data_loader)))
    print("===>Epoch{} Avg. L1: {:.4f} ".format(epoch, avg_l1 / len(test_data_loader)))

def save_checkpoint(model, epoch, t, data):
    model_out_path = "model/{}_{}_{}/model_epoch_{}.pth".format(opt.net,data,t,epoch)
    state = {"epoch": epoch, "model": model}

    if not os.path.exists("model/{}_{}_{}".format(opt.net, data, t)):
        os.makedirs("model/{}_{}_{}".format(opt.net, data, t))

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    import os
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DIVICES"] = "0"
    print(torch.cuda.is_available())
    main()
