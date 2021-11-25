'''
    Written by msson
    2021.08.10

    Modified by msson
    2021.08.30
'''
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datasets import FaceDataset
from models.EAGR import *
from utils.eagr_criterion import CriterionCrossEntropyEdgeParsing_boundary_attention_loss

import argparse
import time
import os
from skimage.measure import compare_ssim
import numpy as np
from glob import glob

NUM_CLASSES = 11

criterion = CriterionCrossEntropyEdgeParsing_boundary_attention_loss()
if torch.cuda.is_available():
    criterion = criterion.cuda()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='Root directory path that consists of train and test directories.', default=r'D:\DH_dataset\face_seg\HELENstar\helenstar_release', dest='root')
    parser.add_argument('--batch_size', help='Batch size (int)', default=4, dest='batch_size')
    parser.add_argument('--epoch', help='Number of epoch (int)', default=100, dest='n_epoch')
    parser.add_argument('--lr', help='Learning rate', default=1e-3, dest='learning_rate')

    root = parser.parse_args().root
    batch_size = parser.parse_args().batch_size
    n_epoch = parser.parse_args().n_epoch
    learning_rate = parser.parse_args().learning_rate

    return root, batch_size, n_epoch, learning_rate


# def cal_miou(result, gt):                ## resutl.shpae == gt.shape == [512, 512]
#     # miou = np.zeros((10))
#     miou = 0
#     for idx in range(1,11):              ## background 제외
#         u = torch.sum(torch.where((result==idx) + (gt==idx), torch.Tensor([1]), torch.Tensor([0]))).item()
#         o = torch.sum(torch.where((result==idx) * (gt==idx), torch.Tensor([1]), torch.Tensor([0]))).item()
#         iou = o / u
#         # miou[idx-1] += iou
#         miou += iou

#     return miou / 10


def train(model, dataloader, optimizer, epoch, n_epoch, input, writer):
    model.train()
    losses= 0
    
    for n_iter, (images, infos) in enumerate(dataloader):
        if torch.cuda.is_available():
            images = images.cuda()
            segments = infos[0].long().cuda()
            edges = infos[1].long().cuda()
            depths = infos[2].cuda()

        if input == 'depth':
            outputs = model(depths)
        elif input == 'rgb':
            outputs = model(images)
        loss = criterion(outputs, [segments, edges])
        losses += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('[Train] Epoch: {}/{}, Iter: {}/{}, Loss: {:.4f}'.format(epoch, n_epoch, n_iter, 0, loss))

    avg_loss = losses / n_iter
    writer.add_scalar("Loss/train", avg_loss, epoch)


def val(model, dataloader, epoch, n_epoch, input, writer):
    model.eval()
    with torch.no_grad():
        losses = 0
        for n_iter, (images, infos) in enumerate(dataloader):
            if torch.cuda.is_available():
                images = images.cuda()
                segments = infos[0].long().cuda()
                edges = infos[1].long().cuda()
                depths = infos[2].cuda()

            if input == 'depth':
                outputs = model(depths)
            elif input == 'rgb':
                outputs = model(images)
            loss = criterion(outputs, [segments, edges])
            losses += loss

            print('[Valid] Epoch: {}/{}, Iter: {}/{}, Loss: {:.4f}'.format(epoch, n_epoch, n_iter, 0, loss))

        avg_loss = losses / n_iter
        writer.add_scalar("Loss/valid", avg_loss, epoch)

    return avg_loss


def pretrain(root, batch_size, n_epoch, learning_rate):
    # load = '2021-09-05_10-49'
    load = None
    if load:
        now = load
    else:
        now = time.strftime(r'%Y-%m-%d_%H-%M',time.localtime(time.time()))

    train_dataset = FaceDataset(root, 'train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    test_dataset = FaceDataset(root, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())

    # Depth-to-parsing Network
    # model_save_pth = os.path.join('pretrained', now, 'Dparse')
    # os.makedirs(model_save_pth, exist_ok=True)

    # Dwriter = SummaryWriter(os.path.join('logs',now,'Dparse'))
    # model = EAGRNet(num_classes=NUM_CLASSES)

    # if torch.cuda.is_available():
    #     model = model.cuda()
    # print("Model Structure: ", model, "\n\n")

    # lr = learning_rate
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.9975)     # polynomial learning rate decay

    # epoch = 0
    # best_loss = 100
    # for epoch in range(epoch, n_epoch):
    #     train(model, train_dataloader, optimizer, epoch, n_epoch, input='depth', writer=Dwriter)
    #     loss = val(model, test_dataloader, epoch, n_epoch, input='depth', writer=Dwriter)

    #     scheduler.step()

    #     if loss < best_loss:
    #         best_loss = loss
    #         print("Best Loss: {:.4f}".format(loss))
    #         file_name = '{:03d}_{:.4f}.ckpt'.format(epoch, loss)
    #         torch.save(
    #             {
    #                 'epoch': epoch,
    #                 'model_state_dict': model.state_dict(),
    #                 'opt_state_dict': optimizer.state_dict()
    #             },
    #             os.path.join(model_save_pth, file_name)
    #         )
    # Dwriter.close()

    # RGB-to-parsing Network
    model_save_pth = os.path.join('pretrained', now, 'RGBparse')
    os.makedirs(model_save_pth, exist_ok=True)

    RGBwriter = SummaryWriter(os.path.join('logs',now,'RGBparse'))
    model = EAGRNet(num_classes=NUM_CLASSES)

    if torch.cuda.is_available():
        model = model.cuda()
    print("Model Structure: ", model, "\n\n")

    lr = learning_rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - n_epoch) / float(n_epoch + 1)
            return lr_l
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    
    epoch = 0
    best_loss = float('inf')

    if load: 
        last_checkpoint_path = glob(os.path.join('pretrained', now, 'RGBparse', '*'))[-1]
        checkpoint = torch.load(last_checkpoint_path)

        epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['opt_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda() 

    for epoch in range(epoch, n_epoch):
        scheduler.step()

        train(model, train_dataloader, optimizer, epoch, n_epoch, input='rgb', writer=RGBwriter)
        loss = val(model, test_dataloader, epoch, n_epoch, input='rgb', writer=RGBwriter)

        if loss < best_loss:
            best_loss = loss
            print("Best Loss: {:.4f}".format(loss))
            file_name = '{:03d}_{:.4f}.ckpt'.format(epoch, loss)
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'opt_state_dict': optimizer.state_dict()
                },
                os.path.join(model_save_pth, file_name)
            )
    RGBwriter.close()


if __name__ == '__main__':
    root, batch_size, n_epoch, learning_rate = get_arguments()

    pretrain(root, batch_size, n_epoch, learning_rate)