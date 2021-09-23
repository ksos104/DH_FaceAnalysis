'''
    Written by msson
    2021.08.10
'''
import torch
from torch.utils.data import DataLoader
from datasets import FaceDataset
from utils.transforms import Transpose
import argparse
import time
import os
from skimage.measure import compare_ssim
import numpy as np
from models import EAGR, create_model


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='Root directory path that consists of train and test directories.', default=r'D:\DH_dataset\face_seg\HELENstar\helenstar_release', dest='root')
    parser.add_argument('--is_test', help='True / False (bool)', default=False, dest='is_test')
    parser.add_argument('--batch_size', help='Batch size (int)', default=1, dest='batch_size')
    parser.add_argument('--epoch', help='Number of epoch (int)', default=500, dest='n_epoch')
    parser.add_argument('--lr', help='Learning rate', default=1e-4, dest='learning_rate')

    root = parser.parse_args().root
    is_test = parser.parse_args().is_test
    batch_size = parser.parse_args().batch_size
    n_epoch = parser.parse_args().n_epoch
    learning_rate = parser.parse_args().learning_rate

    return root, is_test, batch_size, n_epoch, learning_rate


def train(model, dataloader, optimizer, epoch, n_epoch):
    model.train()
    
    for n_iter, (images, targets) in enumerate(dataloader):
        if torch.cuda.is_available():
            images = images.cuda()
            targets_seg = targets[0].cuda()
            targets_dep = targets[1].cuda()

        outputs = model(images)
        loss = cal_depth_loss(outputs[0], targets_dep)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch: {:03d}/{:03d}, Iter: {:03d}/{:03d}, Loss: {:.4f}'.format(epoch, n_epoch, n_iter, 0, loss))


def val(model, dataloader, epoch, n_epoch):
    model.eval()
    with torch.no_grad():
        losses = 0
        for n_iter, (images, targets) in enumerate(dataloader):
            if torch.cuda.is_available():
                images = images.cuda()
                targets_seg = targets[0].cuda()
                targets_dep = targets[1].cuda()

            outputs = model(images)
            loss = cal_depth_loss(outputs[0], targets_dep)
            losses += loss

            print('Epoch: {:03d}/{:03d}, Iter: {:03d}/{:03d}, Loss: {:.4f}'.format(epoch, n_epoch, n_iter, 0, loss))

        avg_loss = losses / n_iter

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

    if torch.cuda.is_available():
        model_Dparse = model_Dparse.cuda()
        model_RGBparse = model_RGBparse.cuda()
        model_RGBD = model_RGBD.cuda()
    print("Model Structure: ", model_Dparse, "\n\n")
    print("Model Structure: ", model_RGBparse, "\n\n")
    print("Model Structure: ", model_RGBD, "\n\n")

    lr = learning_rate
    optimizer_Dparse = torch.optim.Adam(model_Dparse.parameters(), lr=lr)
    optimizer_RGBparse = torch.optim.Adam(model_RGBparse.parameters(), lr=lr)

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
        if not is_test:
            train(model, train_dataloader, optimizer, epoch, n_epoch)
        loss = val(model, test_dataloader, epoch, n_epoch)

        if loss < best_loss:
            print("Best Loss: {:.4f}".format(loss))
            dir_name = '{:03d}_{:.4f}'.format(epoch, loss)
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'opt_state_dict': optimizer.state_dict()
                },
                os.path.join(model_save_pth, dir_name, 'checkpoint.ckpt')
            )


if __name__ == '__main__':
    root, is_test, batch_size, n_epoch, learning_rate = get_arguments()

    main(root, is_test, batch_size, n_epoch, learning_rate)