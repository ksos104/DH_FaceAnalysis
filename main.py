'''
    Written by msson
    2021.08.10
'''
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
from models import EAGR, create_model
from torch.optim import lr_scheduler

CELoss = CrossEntropyLoss()
l2Loss = torch.nn.MSELoss()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='Root directory path that consists of train and test directories.', default=r'D:\DH_dataset\face_seg\HELENstar\helenstar_release', dest='root')
    parser.add_argument('--is_test', help='True / False (bool)', default=False, dest='is_test')
    parser.add_argument('--batch_size', help='Batch size (int)', default=1, dest='batch_size')
    parser.add_argument('--epoch', help='Number of epoch (int)', default=500, dest='n_epoch')
    parser.add_argument('--lr', help='Learning rate', default=1e-3, dest='learning_rate')

    root = parser.parse_args().root
    is_test = parser.parse_args().is_test
    batch_size = parser.parse_args().batch_size
    n_epoch = parser.parse_args().n_epoch
    learning_rate = parser.parse_args().learning_rate

    return root, is_test, batch_size, n_epoch, learning_rate


def train(model_Dparse, model_RGBparse, model_RGBD, dataloader, optimizer_Dparse, optimizer_RGBparse, epoch, n_epoch):
    model_Dparse.train()
    model_RGBparse.train()
    
    totalLosses = 0
    for n_iter, (images, (segments, edges, depths)) in enumerate(dataloader):
        if torch.cuda.is_available():
            images = images.cuda()
            segments = segments.cuda()
            edges = edges.cuda()
            depths = depths.cuda()

        inputs = {'A': images, 'B': depths}
        model_RGBD.set_input(inputs)
        output_RGBD = model_RGBD.forward()
        output_RGBD_Dparse = model_Dparse(output_RGBD)
        output_Dparse = model_Dparse(depths)
        output_RGBparse = model_RGBparse(images)

        MLloss = l2Loss(output_RGBparse, output_Dparse)
        E2Eloss = CELoss(output_RGBD_Dparse, segments)
        RGBparseLoss = CELoss(output_RGBparse, segments)
        DparseLoss = CELoss(output_Dparse, segments)
        _, DTloss = model_RGBD.cal_losses()

        totalLoss = MLloss + E2Eloss + 0.5 * (RGBparseLoss + DparseLoss) + DTloss

        optimizer_Dparse.zero_grad()
        optimizer_RGBparse.zero_grad()
        model_RGBD.optimizer_G.zero_grad()
        totalLoss.backward()
        optimizer_Dparse.step()
        optimizer_RGBparse.step()
        model_RGBD.optimizer_G.step()


        print('Epoch: {:03d}/{:03d}, Iter: {:03d}/{:03d}, Loss: {:.4f}'.format(epoch, n_epoch, n_iter, 0, totalLoss))


def val(model_Dparse, model_RGBparse, model_RGBD, dataloader, epoch, n_epoch):
    model_Dparse.eval()
    model_RGBparse.eval()
    model_RGBD.eval()
    with torch.no_grad():
        totalLosses = 0
        for n_iter, (images, segments, edges, depths) in enumerate(dataloader):
            if torch.cuda.is_available():
                images = images.cuda()
                segments = segments.cuda()
                edges = edges.cuda()
                depths = depths.cuda()

            inputs = {'A': images, 'B': depths}
            model_RGBD.set_input(inputs)
            output_RGBD = model_RGBD.forward()
            output_RGBD_Dparse = model_Dparse(output_RGBD)
            output_Dparse = model_Dparse(depths)
            output_RGBparse = model_RGBparse(images)

            MLloss = l2Loss(output_RGBparse, output_Dparse)
            E2Eloss = CELoss(output_RGBD_Dparse, segments)
            RGBparseLoss = CELoss(output_RGBparse, segments)
            DparseLoss = CELoss(output_Dparse, segments)
            _, DTloss = model_RGBD.cal_losses()

            totalLoss = MLloss + E2Eloss + 0.5 * (RGBparseLoss + DparseLoss) + DTloss
            totalLosses += totalLoss

            print('Epoch: {:03d}/{:03d}, Iter: {:03d}/{:03d}, Loss: {:.4f}'.format(epoch, n_epoch, n_iter, 0, totalLoss))

        avg_loss = totalLosses / n_iter

    return avg_loss


def main(root, is_test, batch_size, n_epoch, learning_rate):
    load_Dparse = '2021-09-02_19-34'
    load_RGBparse = '2021-09-05_13-33'
    load_RGBD = '2021-09-16_13-32'

    now = time.strftime(r'%Y-%m-%d_%H-%M',time.localtime(time.time()))
    model_save_pth = os.path.join('trained', now)
    os.makedirs(model_save_pth, exist_ok=True)

    train_dataset = FaceDataset(root, 'train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    test_dataset = FaceDataset(root, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())

    model_Dparse = EAGR()
    model_RGBparse = EAGR()
    model_RGBD = create_model()

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

    lr = learning_rate
    optimizer_Dparse = torch.optim.Adam(model_Dparse.parameters(), lr=lr)
    optimizer_RGBparse = torch.optim.Adam(model_RGBparse.parameters(), lr=lr)
    
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)
        return lr_l
    scheduler_Dparse = lr_scheduler.LambdaLR(optimizer_Dparse, lr_lambda=lambda_rule)
    scheduler_RGBparse = lr_scheduler.LambdaLR(optimizer_RGBparse, lr_lambda=lambda_rule)

    model_Dparse_root = os.path.join(r'C:\Users\Minseok\Desktop\DH_FaceAnalysis\pretrained', load_Dparse, 'Dparse')
    model_Dparse_path = os.path.join(model_Dparse_root, os.listdir(model_Dparse_root)[-1])
    checkpoint = torch.load(model_Dparse_path)
    model_Dparse.load_state_dict(checkpoint['model_state_dict'])
    optimizer_Dparse.load_state_dict(checkpoint['opt_state_dict'])
    
    model_RGBparse_root = os.path.join(r'C:\Users\Minseok\Desktop\DH_FaceAnalysis\pretrained', load_RGBparse, 'Dparse')
    model_RGBparse_path = os.path.join(model_RGBparse_root, os.listdir(model_RGBparse_root)[-1])
    checkpoint = torch.load(model_RGBparse_path)
    model_RGBparse.load_state_dict(checkpoint['model_state_dict'])
    optimizer_RGBparse.load_state_dict(checkpoint['opt_state_dict'])
    
    model_RGBD_root = os.path.join(r'C:\Users\Minseok\Desktop\DH_FaceAnalysis\pretrained', load_RGBD, 'Dparse')
    model_RGBD_path = os.path.join(model_RGBD_root, os.listdir(model_RGBD_root)[-1])
    checkpoint = torch.load(model_RGBD_path)
    model_RGBD.netG.load_state_dict(checkpoint['modelG_state_dict'])
    model_RGBD.netD.load_state_dict(checkpoint['modelD_state_dict'])
    model_RGBD.optimizers[0].load_state_dict(checkpoint['optG_state_dict'])
    model_RGBD.optimizers[1].load_state_dict(checkpoint['optD_state_dict'])

    epoch = 0
    best_loss = float('inf')
    for epoch in range(epoch, n_epoch):
        model_RGBD.update_learning_rate()
        scheduler_Dparse.step()
        scheduler_RGBparse.step()
        if not is_test:
            train(model_Dparse, model_RGBparse, model_RGBD, train_dataloader, optimizer_Dparse, optimizer_RGBparse, epoch, n_epoch)
        loss = val(model_Dparse, model_RGBparse, model_RGBD, test_dataloader, epoch, n_epoch)

        if loss < best_loss:
            print("Best Loss: {:.4f}".format(loss))
            dir_name = '{:03d}_{:.4f}'.format(epoch, loss)
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
                os.path.join(model_save_pth, dir_name, 'checkpoint.ckpt')
            )


if __name__ == '__main__':
    root, is_test, batch_size, n_epoch, learning_rate = get_arguments()

    main(root, is_test, batch_size, n_epoch, learning_rate)