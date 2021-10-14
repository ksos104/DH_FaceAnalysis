'''
    Written by msson
    2021.08.10

    Modified by msson
    2021.08.30
    2021.09.03
'''
import torch
from torch.serialization import load
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datasets import FaceDataset
from models import create_model

import argparse
import time
import os
from skimage.measure import compare_ssim
import numpy as np


# criterion = ()
# if torch.cuda.is_available():
#     criterion = criterion.cuda()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='Root directory path that consists of train and test directories.', default=r'D:\DH_dataset\CelebA-HQ', dest='root')
    parser.add_argument('--batch_size', help='Batch size (int)', default=1, dest='batch_size')
    parser.add_argument('--epoch', help='Number of epoch (int)', default=100, dest='n_epoch')
    parser.add_argument('--lr', help='Learning rate', default=1e-3, dest='learning_rate')

    root = parser.parse_args().root
    batch_size = parser.parse_args().batch_size
    n_epoch = parser.parse_args().n_epoch
    learning_rate = parser.parse_args().learning_rate

    return root, batch_size, n_epoch, learning_rate


def train(model, dataloader, epoch, n_epoch, writer, max_iter):
    # model.train()
    Ggan_losses = 0
    Gl1_losses = 0
    Dreal_losses = 0
    Dfake_losses = 0
    
    for n_iter, (images, infos) in enumerate(dataloader):
        if torch.cuda.is_available():
            images = images.cuda()
            segments = infos[0].long().cuda()
            depths = infos[1].cuda()

        inputs = {'A': images, 'B': depths}

        model.set_input(inputs)
        model.optimize_parameters()
        loss = model.get_current_losses()

        Ggan_losses += loss['G_GAN']
        Gl1_losses += loss['G_L1']
        Dreal_losses += loss['D_real']
        Dfake_losses += loss['D_fake']

        print('[Train] Epoch: {}/{}, Iter: {}/{}, Loss: {:.4f}, G_GAN Loss: {:.4f}, G_L1 Loss: {:.4f}, D_real Loss: {:.4f}, D_fake Loss: {:.4f}'.format(epoch, n_epoch, n_iter, max_iter, loss['G_GAN']+loss['G_L1'], loss['G_GAN'], loss['G_L1'], loss['D_real'], loss['D_fake']))

    avg_loss = (Ggan_losses + Gl1_losses) / n_iter
    Ggan_avg_loss = Ggan_losses / n_iter
    Gl1_avg_loss = Gl1_losses / n_iter
    Dreal_avg_loss = Dreal_losses / n_iter
    Dfake_avg_loss = Dfake_losses / n_iter
    
    writer.add_scalar("Loss/train/Loss", avg_loss, epoch)
    writer.add_scalar("Loss/train/G_GAN", Ggan_avg_loss, epoch)
    writer.add_scalar("Loss/train/G_L1", Gl1_avg_loss, epoch)
    writer.add_scalar("Loss/train/D_real", Dreal_avg_loss, epoch)
    writer.add_scalar("Loss/train/D_fake", Dfake_avg_loss, epoch)



def val(model, dataloader, epoch, n_epoch, writer, max_iter):
    model.eval()
    with torch.no_grad():
        Ggan_losses = 0
        Gl1_losses = 0
        Dreal_losses = 0
        Dfake_losses = 0
        for n_iter, (images, infos) in enumerate(dataloader):
            if torch.cuda.is_available():
                images = images.cuda()
                segments = infos[0].long().cuda()
                depths = infos[1].cuda()

            inputs = {'A': images, 'B': depths}

            model.set_input(inputs)
            model.forward()
            model.cal_losses()
            loss = model.get_current_losses()

            Ggan_losses += loss['G_GAN']
            Gl1_losses += loss['G_L1']
            Dreal_losses += loss['D_real']
            Dfake_losses += loss['D_fake']

            print('[Valid] Epoch: {}/{}, Iter: {}/{}, Loss: {:.4f}, G_GAN Loss: {:.4f}, G_L1 Loss: {:.4f}, D_real Loss: {:.4f}, D_fake Loss: {:.4f}'.format(epoch, n_epoch, n_iter, max_iter, loss['G_GAN']+loss['G_L1'], loss['G_GAN'], loss['G_L1'], loss['D_real'], loss['D_fake']))

        avg_loss = (Ggan_losses + Gl1_losses) / n_iter
        Ggan_avg_loss = Ggan_losses / n_iter
        Gl1_avg_loss = Gl1_losses / n_iter
        Dreal_avg_loss = Dreal_losses / n_iter
        Dfake_avg_loss = Dfake_losses / n_iter
        
        writer.add_scalar("Loss/valid/Loss", avg_loss, epoch)
        writer.add_scalar("Loss/valid/G_GAN", Ggan_avg_loss, epoch)
        writer.add_scalar("Loss/valid/G_L1", Gl1_avg_loss, epoch)
        writer.add_scalar("Loss/valid/D_real", Dreal_avg_loss, epoch)
        writer.add_scalar("Loss/valid/D_fake", Dfake_avg_loss, epoch)

    return Ggan_avg_loss + Gl1_avg_loss


def pretrain(root, batch_size, n_epoch, learning_rate):
    now = time.strftime(r'%Y-%m-%d_%H-%M',time.localtime(time.time()))

    train_dataset = FaceDataset(root, 'train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    test_dataset = FaceDataset(root, 'val')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())

    # Depth-to-parsing Network
    model_save_pth = os.path.join('pretrained', now)
    os.makedirs(model_save_pth, exist_ok=True)

    writer = SummaryWriter(os.path.join('logs',now))
    model = create_model(model_save_pth)

    train_max_iter = train_dataset.__len__()
    val_max_iter = test_dataset.__len__()

    continue_train = False
    load_iter = 0
    epoch = 'latest'
    verbose = False

    lr_policy = 'linear'
    epoch_count = 1
    n_epochs = n_epoch
    n_epochs_decay = n_epoch
    lr_decay_iters = 50

    model.setup(continue_train=continue_train, load_iter=load_iter, epoch=epoch, verbose=verbose,
                lr_policy=lr_policy, epoch_count=epoch_count, n_epochs=n_epochs, n_epochs_decay=n_epochs_decay,
                lr_decay_iters=lr_decay_iters)

    print("Model Structure: ", model, "\n\n")

    epoch = 0
    best_loss = float('inf')
    for epoch in range(epoch, n_epoch):
        model.update_learning_rate()
        train(model, train_dataloader, epoch, n_epoch, writer=writer, max_iter=train_max_iter)
        loss = val(model, test_dataloader, epoch, n_epoch, writer=writer, max_iter=val_max_iter)

        if loss < best_loss:
            best_loss = loss
            print("Best Loss: {:.4f}".format(loss))
            file_name = '{:03d}_{:.4f}.ckpt'.format(epoch, loss)
            torch.save(
                {
                    'epoch': epoch,
                    'modelG_state_dict': model.netG.state_dict(),
                    'modelD_state_dict': model.netD.state_dict(),
                    'optG_state_dict': model.optimizers[0].state_dict(),
                    'optD_state_dict': model.optimizers[1].state_dict()
                },
                os.path.join(model_save_pth, file_name)
            )
    writer.close()


if __name__ == '__main__':
    root, batch_size, n_epoch, learning_rate = get_arguments()

    pretrain(root, batch_size, n_epoch, learning_rate)