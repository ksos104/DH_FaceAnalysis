'''
    Written by msson
    2021.08.10
'''
from functools import total_ordering
import torch
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import FaceDataset
from utils.transforms import Transpose
import argparse
import time
import os
from skimage.measure import compare_ssim
import numpy as np
from tensorboardX import SummaryWriter
from models import create_model
from models.bisenet_model import BiSeNet
from utils_bisenet.loss import OhemCELoss, WCELoss

import math
from glob import glob

CELoss = CrossEntropyLoss()
l2Loss = torch.nn.MSELoss()

NUM_CLASSES = 8

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='Root directory path that consists of train and test directories.', default=r'D:\DH_dataset\face_seg\HELENstar\helenstar_release', dest='root')
    parser.add_argument('--is_test', help='True / False (bool)', default=False, dest='is_test')
    parser.add_argument('--batch_size', help='Batch size (int)', default=8, dest='batch_size')
    parser.add_argument('--epoch', help='Number of epoch (int)', default=100, dest='n_epoch')
    parser.add_argument('--lr_seg', help='Learning rate for segementation network', default=1e-3, dest='lr_seg')
    parser.add_argument('--lr_pix', help='Learning rate for Pix2Pix network', default=1e-4, dest='lr_pix')
    parser.add_argument('--load', help='checkpoint directory name (ex. 2021-09-27_22-06)', default=None, dest='load')

    root = parser.parse_args().root
    is_test = parser.parse_args().is_test
    batch_size = parser.parse_args().batch_size
    n_epoch = parser.parse_args().n_epoch
    lr_seg = parser.parse_args().lr_seg
    lr_pix = parser.parse_args().lr_pix
    load = parser.parse_args().load

    return root, is_test, batch_size, n_epoch, lr_seg, lr_pix, load


def cal_miou(result, gt):                ## resutl.shpae == gt.shape == [batch_size, 512, 512]
    miou = 0
    result = torch.where(gt==0, torch.tensor(0).to(gt.device), result)
    tensor1 = torch.Tensor([1]).to(gt.device)
    tensor0 = torch.Tensor([0]).to(gt.device)
    for idx in range(1, NUM_CLASSES):              ## background 제외
        u = torch.sum(torch.where((result==idx) + (gt==idx), tensor1, tensor0)).item()
        o = torch.sum(torch.where((result==idx) * (gt==idx), tensor1, tensor0)).item()
        try:
            iou = o / u
        except:
            pass
        miou += iou
    
    return miou / (NUM_CLASSES-1)


def get_lr(epoch, max_epoch, iter, max_iter):
    lr0 = 1e-2
    it = (epoch * max_iter) + iter
    power = 0.9

    factor = (1-(it/(max_epoch*max_iter)))**power
    lr = lr0 * factor

    return lr


def train(model_Dparse, model_RGBparse, model_RGBD, dataloader, optimizer_Dparse, optimizer_RGBparse, epoch, n_epoch, writer, Losses, max_iter):
    model_Dparse.train()
    model_RGBparse.train()
    LossP, Loss2, Loss3 = Losses[0], Losses[1], Losses[2]
    
    totalLosses = 0
    avg_miou = 0
    for n_iter, (images, infos) in enumerate(dataloader):
        if torch.cuda.is_available():
            images = images.cuda()
            segments = infos[0].long().cuda()
            depths = infos[1].cuda()

        inputs = {'A': images, 'B': depths}
        model_RGBD.set_input(inputs)
        output_RGBD = model_RGBD.forward()
        outputs_rgbdp, outputs16_rgbdp, outputs32_rgbdp = model_Dparse(output_RGBD)

        outputs_dp, outputs16_dp, outputs32_dp = model_Dparse(depths)
        outputs_rgbp, outputs16_rgbp, outputs32_rgbp = model_RGBparse(images)
        segments = segments.squeeze(dim=1)

        MLloss = l2Loss(outputs_rgbp, outputs_dp)
        E2Eloss = LossP(outputs_rgbdp, segments) + Loss2(outputs16_rgbdp, segments) + Loss3(outputs32_rgbdp, segments)
        RGBparseLoss = LossP(outputs_rgbp, segments) + Loss2(outputs16_rgbp, segments) + Loss3(outputs32_rgbp, segments)
        DparseLoss = LossP(outputs_dp, segments) + Loss2(outputs16_dp, segments) + Loss3(outputs32_dp, segments)
        _, DTloss = model_RGBD.cal_losses()

        totalLoss = MLloss + E2Eloss + 0.5 * (RGBparseLoss + DparseLoss) + DTloss
        totalLosses = totalLosses + totalLoss

        optimizer_Dparse.zero_grad()
        optimizer_RGBparse.zero_grad()

        # model_RGBD.optimize_D_parameters()
        model_RGBD.set_requires_grad(model_RGBD.netD, False)
        model_RGBD.optimizer_G.zero_grad()
    
        totalLoss.backward()

        lr_seg = get_lr(epoch, n_epoch, n_iter, max_iter)

        for pg in optimizer_Dparse.param_groups:
            if pg.get('lr_mul', False):
                pg['lr'] = lr_seg * 10
            else:
                pg['lr'] = lr_seg
        if optimizer_Dparse.defaults.get('lr_mul', False):
            optimizer_Dparse.defaults['lr'] = lr_seg * 10
        else:
            optimizer_Dparse.defaults['lr'] = lr_seg

        for pg in optimizer_RGBparse.param_groups:
            if pg.get('lr_mul', False):
                pg['lr'] = lr_seg * 10
            else:
                pg['lr'] = lr_seg
        if optimizer_RGBparse.defaults.get('lr_mul', False):
            optimizer_RGBparse.defaults['lr'] = lr_seg * 10
        else:
            optimizer_RGBparse.defaults['lr'] = lr_seg

        optimizer_Dparse.step()
        optimizer_RGBparse.step()
        model_RGBD.optimizer_G.step()

        result_parse = torch.argmax(outputs_rgbp, dim=1)     ## result_parse.shape: [batch_size, 512, 512]
        miou = cal_miou(result_parse, segments)
        avg_miou += miou

        print('[Train] Epoch: {}/{}, Iter: {}/{}, Loss: {:.4f}, mIoU: {:.4f}'.format(epoch, n_epoch, n_iter, max_iter, totalLoss, miou))

    avg_loss = totalLosses / n_iter
    avg_miou = avg_miou / n_iter
    writer.add_scalar("Train/loss", avg_loss, epoch)
    writer.add_scalar("Train/mIoU", avg_miou, epoch)


def val(model_Dparse, model_RGBparse, model_RGBD, dataloader, epoch, n_epoch, writer, Losses, max_iter):
    model_Dparse.eval()
    model_RGBparse.eval()
    model_RGBD.eval()
    LossP, Loss2, Loss3 = Losses[0], Losses[1], Losses[2]

    with torch.no_grad():
        totalLosses = 0
        avg_miou = 0
        for n_iter, (images, infos) in enumerate(dataloader):
            if torch.cuda.is_available():
                images = images.cuda()
                segments = infos[0].long().cuda()
                depths = infos[1].cuda()

            inputs = {'A': images, 'B': depths}
            model_RGBD.set_input(inputs)
            output_RGBD = model_RGBD.forward()
            outputs_rgbdp, outputs16_rgbdp, outputs32_rgbdp = model_Dparse(output_RGBD)

            outputs_dp, outputs16_dp, outputs32_dp = model_Dparse(depths)
            outputs_rgbp, outputs16_rgbp, outputs32_rgbp = model_RGBparse(images)
            segments = segments.squeeze(dim=1)

            MLloss = l2Loss(outputs_rgbp, outputs_dp)
            E2Eloss = LossP(outputs_rgbdp, segments) + Loss2(outputs16_rgbdp, segments) + Loss3(outputs32_rgbdp, segments)
            RGBparseLoss = LossP(outputs_rgbp, segments) + Loss2(outputs16_rgbp, segments) + Loss3(outputs32_rgbp, segments)
            DparseLoss = LossP(outputs_dp, segments) + Loss2(outputs16_dp, segments) + Loss3(outputs32_dp, segments)
            _, DTloss = model_RGBD.cal_losses()

            totalLoss = MLloss + E2Eloss + 0.5 * (RGBparseLoss + DparseLoss) + DTloss
            totalLosses = totalLosses + totalLoss

            result_parse = torch.argmax(outputs_rgbp, dim=1)     ## result_parse.shape: [batch_size, 512, 512]
            miou = cal_miou(result_parse, segments)
            avg_miou += miou

            print('[Valid] Epoch: {}/{}, Iter: {}/{}, Loss: {:.4f}, mIoU: {:.4f}'.format(epoch, n_epoch, n_iter, max_iter, totalLoss, miou))

        avg_loss = totalLosses / n_iter
        avg_miou = avg_miou / n_iter
        writer.add_scalar("Valid/loss", avg_loss, epoch)
        writer.add_scalar("Valid/mIoU", avg_miou, epoch)

    return avg_loss, avg_miou


def main(root, is_test, batch_size, n_epoch, lr_seg, lr_pix, load):
    load_Dparse = '2021-11-01_15-34'
    load_RGBparse = '2021-10-29_17-19'
    load_RGBD = '2021-11-01_14-55'

    now = time.strftime(r'%Y-%m-%d_%H-%M',time.localtime(time.time()))
    model_save_pth = os.path.join('trained', now)
    os.makedirs(model_save_pth, exist_ok=True)

    train_dataset = FaceDataset(root, 'train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=4)
    test_dataset = FaceDataset(root, 'val')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=4)

    train_num_imgs = train_dataset.__len__()
    val_num_images = test_dataset.__len__()
    train_max_iter = math.ceil(train_num_imgs / batch_size)
    val_max_iter = math.ceil(val_num_images / batch_size)

    model_save_pth = os.path.join('pretrained', now)
    os.makedirs(model_save_pth, exist_ok=True)
    
    writer = SummaryWriter(os.path.join('logs',now))

    model_Dparse = BiSeNet(n_classes=NUM_CLASSES)
    model_RGBparse = BiSeNet(n_classes=NUM_CLASSES)
    model_RGBD = create_model(model_save_pth, lr_pix)

    continue_train = False
    load_iter = 0
    epoch = 'latest'
    verbose = False

    lr_policy = 'linear'
    epoch_count = 1
    n_epochs = n_epoch
    n_epochs_decay = n_epoch
    lr_decay_iters = 50

    model_RGBD.setup(continue_train=continue_train, load_iter=load_iter, epoch=epoch, verbose=verbose,
                lr_policy=lr_policy, epoch_count=epoch_count, n_epochs=n_epochs, n_epochs_decay=n_epochs_decay,
                lr_decay_iters=lr_decay_iters)

    if torch.cuda.is_available():
        model_Dparse = model_Dparse.cuda()
        model_RGBparse = model_RGBparse.cuda()
        # model_RGBD = model_RGBD.cuda()
    print("Model Structure: ", model_Dparse, "\n\n")
    print("Model Structure: ", model_RGBparse, "\n\n")
    print("Model Structure: ", model_RGBD, "\n\n")

    lr_seg0 = lr_seg
    optimizer_Dparse = torch.optim.SGD(model_Dparse.parameters(), lr=lr_seg0, momentum=0.9, weight_decay=5e-4)
    optimizer_RGBparse = torch.optim.SGD(model_RGBparse.parameters(), lr=lr_seg0, momentum=0.9, weight_decay=5e-4)
    

    score_thres = 0.7
    n_min = batch_size * 512 * 512//16      ## batch_size * crop_size[0] * crop_size[1]//16
    ignore_idx = -100

    LossP = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss2 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss3 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    epoch = 0
    best_loss = float('inf')
    best_miou = float(0)

    ## model load
    if load: 
        # model_Dparse = torch.nn.DataParallel(model_Dparse)
        # model_RGBparse = torch.nn.DataParallel(model_RGBparse)
        
        last_checkpoint_path = glob(os.path.join('pretrained', now, '*'))[-1]
        checkpoint = torch.load(last_checkpoint_path)

        epoch = checkpoint['epoch'] + 1
        model_Dparse.load_state_dict(checkpoint['model_Dparse_state_dict'])
        optimizer_Dparse.load_state_dict(checkpoint['opt_Dparse_state_dict'])
        for state in optimizer_Dparse.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        model_RGBparse.load_state_dict(checkpoint['model_RGBparse_state_dict'])
        optimizer_RGBparse.load_state_dict(checkpoint['opt_RGBparse_state_dict'])
        for state in optimizer_RGBparse.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        model_RGBD.netG.load_state_dict(checkpoint['modelG_state_dict'])
        model_RGBD.netD.load_state_dict(checkpoint['modelD_state_dict'])
        model_RGBD.optimizers[0].load_state_dict(checkpoint['optG_state_dict'])
        model_RGBD.optimizers[1].load_state_dict(checkpoint['optD_state_dict'])
        for state in model_RGBD.optimizers[0].state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda() 
        for state in model_RGBD.optimizers[1].state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
    else:
        model_Dparse_path = glob(os.path.join('pretrained', load_Dparse, '*'))[-1]
        checkpoint = torch.load(model_Dparse_path)
        model_Dparse.load_state_dict(checkpoint['model_state_dict'])
        optimizer_Dparse.load_state_dict(checkpoint['opt_state_dict'])
        
        model_RGBparse_path = glob(os.path.join('pretrained', load_RGBparse, '*'))[-1]
        checkpoint = torch.load(model_RGBparse_path)
        model_RGBparse.load_state_dict(checkpoint['model_state_dict'])
        optimizer_RGBparse.load_state_dict(checkpoint['opt_state_dict'])
        
        model_RGBD_path = glob(os.path.join('pretrained', load_RGBD, '*'))[-1]
        checkpoint = torch.load(model_RGBD_path)
        model_RGBD.netG.load_state_dict(checkpoint['modelG_state_dict'])
        model_RGBD.netD.load_state_dict(checkpoint['modelD_state_dict'])
        model_RGBD.optimizers[0].load_state_dict(checkpoint['optG_state_dict'])
        model_RGBD.optimizers[1].load_state_dict(checkpoint['optD_state_dict'])
        
        # model_Dparse = torch.nn.DataParallel(model_Dparse)
        # model_RGBparse = torch.nn.DataParallel(model_RGBparse)

    ## start train and validation
    for epoch in range(epoch, n_epoch):
        model_RGBD.update_learning_rate()
        if not is_test:
            train(model_Dparse, model_RGBparse, model_RGBD, train_dataloader, optimizer_Dparse, optimizer_RGBparse, epoch, n_epoch, writer=writer, Losses=(LossP, Loss2, Loss3), max_iter=train_max_iter)
        loss, miou = val(model_Dparse, model_RGBparse, model_RGBD, test_dataloader, epoch, n_epoch, writer=writer, Losses=(LossP, Loss2, Loss3), max_iter=val_max_iter)

        if miou > best_miou:
            best_miou = miou
            print("Best mIoU: {:.4f}".format(best_miou))
            file_name = '{:03d}_{:.4f}.ckpt'.format(epoch, best_miou)
            torch.save(
                {
                    'epoch': epoch,
                    'model_Dparse_state_dict': model_Dparse.state_dict(),
                    'model_RGBparse_state_dict': model_RGBparse.state_dict(),
                    'model_RGBD_G_state_dict': model_RGBD.netG.state_dict(),
                    'model_RGBD_D_state_dict': model_RGBD.netD.state_dict(),
                    'opt_Dparse_state_dict': optimizer_Dparse.state_dict(),
                    'opt_RGBparse_state_dict': optimizer_RGBparse.state_dict(),
                    'opt_RGBD_G_state_dict': model_RGBD.optimizers[0].state_dict(),
                    'opt_RGBD_D_state_dict': model_RGBD.optimizers[1].state_dict()
                },
                os.path.join(model_save_pth, file_name)
            )
    writer.close()


if __name__ == '__main__':
    root, is_test, batch_size, n_epoch, lr_seg, lr_pix, load = get_arguments()

    main(root, is_test, batch_size, n_epoch, lr_seg, lr_pix, load)