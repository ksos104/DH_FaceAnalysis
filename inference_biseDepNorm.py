'''
    Written by msson
    2021.08.10

    Modified by msson
    2021.08.30

    Modified by msson
    2021.09.28
'''
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datasets import FaceDataset
# from models.bisenet_model import BiSeNet
from models.bisenet_dec_model import BiSeNet
from utils_bisenet.loss import OhemCELoss

import argparse
import time
import os
from skimage.measure import compare_ssim
import numpy as np
from torch.nn import functional as F
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt

criteria = nn.L1Loss()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='Root directory path that consists of train and test directories.', default=r'D:\DH_dataset\CelebA-HQ', dest='root')
    parser.add_argument('--batch_size', help='Batch size (int)', default=8, dest='batch_size')
    parser.add_argument('--epoch', help='Number of epoch (int)', default=100, dest='n_epoch')
    parser.add_argument('--lr', help='Learning rate', default=1e-2, dest='learning_rate')
    parser.add_argument('--load', help='checkpoint directory name (ex. 2021-09-27_22-06)', default=None, dest='load')

    root = parser.parse_args().root
    batch_size = parser.parse_args().batch_size
    n_epoch = parser.parse_args().n_epoch
    learning_rate = parser.parse_args().learning_rate
    load = parser.parse_args().load

    return root, batch_size, n_epoch, learning_rate, load


def inference(root, load_root, load, NUM_CLASSES):
    test_dataset = FaceDataset(root, 'test')
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=4)

    model = BiSeNet(n_classes=NUM_CLASSES)

    if torch.cuda.is_available():
        model = model.cuda()
    print("Model Structure: ", model, "\n\n")

    model_dir = load
    model_root = os.path.join(load_root, model_dir)
    model_path = os.path.join(model_root, os.listdir(model_root)[-1])
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    with torch.no_grad():
        avg_time = 0
        for n_iter, (images, infos) in enumerate(dataloader):
            if torch.cuda.is_available():
                images = images.cuda()
                segment = infos[0].long().cuda()
                depth = infos[1].cuda()
                normal = infos[2].cuda()

            import time
            start = time.time()
            outputs, _, _ = model(images)
            end = time.time()
            avg_time += end - start
            loss = criteria(outputs, depth[:,0,...])

            '''
                Run these lines except using random 2k images.
            '''
            # outputs = torch.stack([outputs, outputs, outputs], dim=1).type(torch.uint8)
            outputs = outputs.type(torch.uint8)

            images = images.cpu().squeeze().permute(1,2,0)
            outputs = outputs.cpu().squeeze(dim=0).permute(1,2,0)


            '''
                Visualization
            '''
            alpha = 0

            images = images.numpy()[...,::-1]
            outputs = outputs.numpy()[...,::-1]

            blended = (images * alpha) + (outputs * (1 - alpha))
            blended = torch.from_numpy(blended).type(torch.uint8)

            plt.imshow(blended)
            plt.show()

            print('{} Iterations / Loss: {:.4f}'.format(n_iter, loss))
        print("avg_time: ", avg_time / n_iter)

if __name__ == '__main__':
    root, _, _, _, load = get_arguments()

    load_root = './trained_biseDepth'
    load = '2021-11-13_00-14'
    NUM_CLASSES = 1
    # load_root = './trained_biseNormal'
    # load = '2021-11-14_16-01'
    # NUM_CLASSES = 3

    inference(root, load_root, load, NUM_CLASSES)