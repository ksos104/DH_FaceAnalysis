'''
    Written by msson
    2021.08.10

    Modified by msson
    2021.08.30

    Modified by msson
    2021.09.27
'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datasets import FaceDataset
# from models.bisenet_model import BiSeNet
from models.bisenet_dec_model import BiSeNet
from utils_bisenet.loss import OhemCELoss

import argparse
import time
import os
import numpy as np
from glob import glob
import math
from torch.nn import functional as F

NUM_CLASSES = 8

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='Root directory path that consists of train and test directories.', default=r'D:\DH_dataset\CelebA-HQ', dest='root')
    parser.add_argument('--batch_size', help='Batch size (int)', default=8, dest='batch_size')
    parser.add_argument('--epoch', help='Number of epoch (int)', default=100, dest='n_epoch')
    parser.add_argument('--lr', help='Learning rate', default=1e-2, dest='learning_rate')
    parser.add_argument('--load', help='checkpoint directory name (ex. 2021-09-27_22-06)', default=None, dest='load')

    root = parser.parse_args().root
    batch_size = int(parser.parse_args().batch_size)
    n_epoch = parser.parse_args().n_epoch
    learning_rate = parser.parse_args().learning_rate
    load = parser.parse_args().load

    return root, batch_size, n_epoch, learning_rate, load


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
            continue
        miou += iou

    return miou / (NUM_CLASSES-1)

def get_lr(epoch, max_epoch, iter, max_iter):
    lr0 = 1e-2
    it = (epoch * max_iter) + iter
    power = 0.9

    factor = (1-(it/(max_epoch*max_iter)))**power
    lr = lr0 * factor

    return lr


def train(model, dataloader, optimizer, epoch, n_epoch, writer, Losses, max_iter):
    model.train()
    LossSeg, LossDepth, LossNormal = Losses[0], Losses[1], Losses[2]

    losses= 0
    avg_miou = 0
    for n_iter, (images, infos) in enumerate(dataloader):
        if torch.cuda.is_available():
            images = images.cuda()
            segment = infos[0].long().cuda()
            depth = infos[1].cuda()
            normal = infos[2].cuda()

        outputs_seg, outputs_depth, outputs_normal = model(images)

        segment = segment.squeeze(dim=1)
        loss_seg = LossSeg(outputs_seg, segment)
        depth = depth.squeeze(dim=1)
        loss_depth = LossDepth(outputs_depth, depth)
        loss_normal = LossNormal(outputs_normal, normal)

        loss = loss_seg + loss_depth + loss_normal
        losses += loss

        optimizer.zero_grad()
        loss.backward()

        lr = get_lr(epoch, n_epoch, n_iter, max_iter)
        for pg in optimizer.param_groups:
            if pg.get('lr_mul', False):
                pg['lr'] = lr * 10
            else:
                pg['lr'] = lr
        if optimizer.defaults.get('lr_mul', False):
            optimizer.defaults['lr'] = lr * 10
        else:
            optimizer.defaults['lr'] = lr
        
        optimizer.step()

        result_parse = torch.argmax(outputs_seg, dim=1)     ## result_parse.shape: [batch_size, 512, 512]
        miou = cal_miou(result_parse, segment)
        avg_miou += miou

        print('[Train] Epoch: {}/{}, Iter: {}/{}, Loss: {:.4f}[{:.4f}], mIoU: {:.4f}[{:.4f}]'.format(epoch, n_epoch, n_iter, max_iter, loss, (losses/(n_iter+1)), miou, (avg_miou/(n_iter+1))))

    avg_loss = losses / n_iter
    avg_miou = avg_miou / n_iter
    writer.add_scalar("Train/loss", avg_loss, epoch)
    writer.add_scalar("Train/mIoU", avg_miou, epoch)


def val(model, dataloader, epoch, n_epoch, writer, Losses, max_iter):
    model.eval()
    LossSeg, LossDepth, LossNormal = Losses[0], Losses[1], Losses[2]

    with torch.no_grad():
        losses = 0
        avg_miou = 0
        for n_iter, (images, infos) in enumerate(dataloader):
            if torch.cuda.is_available():
                images = images.cuda()
                segment = infos[0].long().cuda()
                depth = infos[1].cuda()
                normal = infos[2].cuda()

            outputs_seg, outputs_depth, outputs_normal = model(images)

            segment = segment.squeeze(dim=1)
            loss_seg = LossSeg(outputs_seg, segment)
            depth = depth.squeeze(dim=1)
            loss_depth = LossDepth(outputs_depth, depth)
            loss_normal = LossNormal(outputs_normal, normal)

            loss = loss_seg + loss_depth + loss_normal
            losses += loss
            
            result_parse = torch.argmax(outputs_seg, dim=1)     ## result_parse.shape: [batch_size, 512, 512]
            miou = cal_miou(result_parse, segment)
            avg_miou += miou

            print('[Valid] Epoch: {}/{}, Iter: {}/{}, Loss: {:.4f}[{:.4f}], mIoU: {:.4f}[{:.4f}]'.format(epoch, n_epoch, n_iter, max_iter, loss, (losses/(n_iter+1)), miou, (avg_miou/(n_iter+1))))

        avg_loss = losses / n_iter
        avg_miou = avg_miou / n_iter
        writer.add_scalar("Valid/loss", avg_loss, epoch)
        writer.add_scalar("Valid/mIoU", avg_miou, epoch)

    return avg_loss, avg_miou


def pretrain(root, batch_size, n_epoch, learning_rate, load):
    if load:
        now = load
    else:
        now = time.strftime(r'%Y-%m-%d_%H-%M',time.localtime(time.time()))

    train_dataset = FaceDataset(root, 'train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=4)
    test_dataset = FaceDataset(root, 'val')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=4)

    model_save_pth = os.path.join('pretrained', now)
    os.makedirs(model_save_pth, exist_ok=True)

    writer = SummaryWriter(os.path.join('logs',now))
    model = BiSeNet(n_classes=NUM_CLASSES)

    if torch.cuda.is_available():
        model = model.cuda()
    print("Model Structure: ", model, "\n\n")

    lr0 = learning_rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr0, momentum=0.9, weight_decay=5e-4)
    train_num_imgs = train_dataset.__len__()
    val_num_images = test_dataset.__len__()
    train_max_iter = math.ceil(train_num_imgs / batch_size)
    val_max_iter = math.ceil(val_num_images / batch_size)
    
    score_thres = 0.7
    n_min = batch_size * 512 * 512//16      ## batch_size * crop_size[0] * crop_size[1]//16
    ignore_idx = -100

    LossSeg = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    LossDepth = nn.L1Loss()
    LossNormal = nn.L1Loss()

    epoch = 0
    best_loss = float('inf')
    best_miou = float(0)

    if load: 
        last_checkpoint_path = glob(os.path.join('pretrained', now, '*'))[-1]
        checkpoint = torch.load(last_checkpoint_path)

        epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['opt_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda() 

    for epoch in range(epoch, n_epoch):

        train(model, train_dataloader, optimizer, epoch, n_epoch, writer=writer, Losses=(LossSeg, LossDepth, LossNormal), max_iter=train_max_iter)
        loss, miou = val(model, test_dataloader, epoch, n_epoch, writer=writer, Losses=(LossSeg, LossDepth, LossNormal), max_iter=val_max_iter)

        if miou > best_miou:
            best_miou = miou
            print("Best mIoU: {:.4f}".format(best_miou))
            file_name = '{:03d}_{:.4f}.ckpt'.format(epoch, best_miou)
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'opt_state_dict': optimizer.state_dict()
                },
                os.path.join(model_save_pth, file_name)
            )
    writer.close()


if __name__ == '__main__':
    root, batch_size, n_epoch, learning_rate, load = get_arguments()

    # root = '/mnt/server7_hard0/msson/CelebA-HQ'

    pretrain(root, batch_size, n_epoch, learning_rate, load)