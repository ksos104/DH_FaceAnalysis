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
import cv2


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='Root directory path that consists of train and test directories.', default=r'D:\DH_dataset\CelebA-HQ', dest='root')
    parser.add_argument('--batch_size', help='Batch size (int)', default=1, dest='batch_size')
    parser.add_argument('--output', help='depth / normal', default='depth', dest='output')
    parser.add_argument('--load', help='checkpoint directory name (ex. 2021-09-27_22-06)', default=None, dest='load_cp')

    root = parser.parse_args().root
    batch_size = parser.parse_args().batch_size
    output = parser.parse_args().output
    load_cp = parser.parse_args().load_cp

    return root, batch_size, output, load_cp


def val(model, dataloader, epoch):
    model.eval()
    with torch.no_grad():
        Ggan_losses = 0
        Gl1_losses = 0
        Dreal_losses = 0
        Dfake_losses = 0
        
        avg_time = 0
        avg_l1 = 0
        for n_iter, (images, infos) in enumerate(dataloader):
            if torch.cuda.is_available():
                images = images.cuda()
                depth = infos[1].cuda()
                normal = infos[2].cuda()

            if output == 'depth':
                inputs = {'A': images, 'B': depth}
            elif output == 'normal':
                inputs = {'A': images, 'B': normal}

            model.set_input(inputs)

            start = time.time()
            result = model.forward()
            end = time.time()
            avg_time += end - start

            result = torch.where(depth==0., depth, result)
            avg_l1 += torch.nn.L1Loss()(depth, result).item()

            model.cal_losses()
            loss = model.get_current_losses()

            Ggan_losses += loss['G_GAN']
            Gl1_losses += loss['G_L1']
            Dreal_losses += loss['D_real']
            Dfake_losses += loss['D_fake']

            print('[Valid] Epoch: {}/{}, Iter: {}/{}, Loss: {:.4f}, G_GAN Loss: {:.4f}, G_L1 Loss: {:.4f}, D_real Loss: {:.4f}, D_fake Loss: {:.4f}'.format(epoch, 1, n_iter, 0, loss['G_GAN']+loss['G_L1'], loss['G_GAN'], loss['G_L1'], loss['D_real'], loss['D_fake']))

            '''
                Visualization
            '''
            alpha = 0.15

            image = images.squeeze().permute(1,2,0).cpu()
            result = result.squeeze().permute(1,2,0).cpu()

            blended = (image * alpha) + (result * (1 - alpha))
            blended = blended.type(torch.uint8).numpy()

            cv2.imshow('blended', blended)
            cv2.waitKey()
            cv2.destroyAllWindows()

        print("avg_time: ", avg_time / n_iter)
        print("avg_l1: ", avg_l1 / n_iter)

        avg_loss = (Ggan_losses + Gl1_losses) / n_iter
        Ggan_avg_loss = Ggan_losses / n_iter
        Gl1_avg_loss = Gl1_losses / n_iter
        Dreal_avg_loss = Dreal_losses / n_iter
        Dfake_avg_loss = Dfake_losses / n_iter

    return Ggan_avg_loss + Gl1_avg_loss


def inference(root, batch_size, output, load_cp):
    test_dataset = FaceDataset(root, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    model = create_model('.')

    continue_train = False
    load_iter = 0
    epoch = 'latest'
    verbose = False

    lr_policy = 'linear'
    epoch_count = 1
    n_epochs = 1
    n_epochs_decay = 1
    lr_decay_iters = 50

    model.setup(continue_train=continue_train, load_iter=load_iter, epoch=epoch, verbose=verbose,
                lr_policy=lr_policy, epoch_count=epoch_count, n_epochs=n_epochs, n_epochs_decay=n_epochs_decay,
                lr_decay_iters=lr_decay_iters)

    print("Model Structure: ", model, "\n\n")

    model_dir = load_cp
    model_root = os.path.join('./pretrained', model_dir)
    model_path = os.path.join(model_root, os.listdir(model_root)[-1])
    checkpoint = torch.load(model_path)
    model.netG.load_state_dict(checkpoint['modelG_state_dict'])
    model.netD.load_state_dict(checkpoint['modelD_state_dict'])
    model.optimizers[0].load_state_dict(checkpoint['optG_state_dict'])
    model.optimizers[1].load_state_dict(checkpoint['optD_state_dict'])

    val(model, test_dataloader, epoch, output)


if __name__ == '__main__':
    root, batch_size, output, load_cp = get_arguments()

    inference(root, batch_size, output, load_cp)