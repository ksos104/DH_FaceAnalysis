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
from models.bisenet_model import BiSeNet
from utils_bisenet.loss import OhemCELoss

import argparse
import time
import os
from skimage.measure import compare_ssim
import numpy as np
from torch.nn import functional as F
import cv2
import matplotlib.pyplot as plt
import math
import time

NUM_CLASSES = 8

label_to_color = {
    0: [128, 64,128],
    1: [244, 35,232],
    2: [ 70, 70, 70],
    3: [102,102,156],
    4: [190,153,153],
    5: [153,153,153],
    6: [250,170, 30],
    7: [220,220,  0],
    8: [107,142, 35],
    9: [152,251,152],
    10: [ 70,130,180]
    }


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='Root directory path that consists of train and test directories.', default=r'D:\DH_dataset\CelebA-HQ', dest='root')
    parser.add_argument('--batch_size', help='Batch size (int)', default=8, dest='batch_size')
    parser.add_argument('--epoch', help='Number of epoch (int)', default=100, dest='n_epoch')
    parser.add_argument('--lr', help='Learning rate', default=1e-3, dest='learning_rate')
    parser.add_argument('--input', help='depth / rgb', default='rgb', dest='input')
    parser.add_argument('--load', help='checkpoint directory name (ex. 2021-09-27_22-06)', default=None, dest='load')

    root = parser.parse_args().root
    batch_size = parser.parse_args().batch_size
    n_epoch = parser.parse_args().n_epoch
    learning_rate = parser.parse_args().learning_rate
    input = parser.parse_args().input
    load = parser.parse_args().load

    return root, batch_size, n_epoch, learning_rate, input, load


def cal_miou(result, gt):                ## resutl.shpae == gt.shape == [512, 512]
    # miou = np.zeros((10))
    miou = 0
    result = torch.where(gt==0, torch.tensor(0).to(gt.device), result)
    tensor1 = torch.Tensor([1]).to(gt.device)
    tensor0 = torch.Tensor([0]).to(gt.device)

    for idx in range(1, NUM_CLASSES):              ## background ??????
        u = torch.sum(torch.where((result==idx) + (gt==idx), tensor1, tensor0)).item()
        o = torch.sum(torch.where((result==idx) * (gt==idx), tensor1, tensor0)).item()
        try:
            iou = o / u
        except:
            pass
        miou += iou

    return miou / (NUM_CLASSES-1)


def get_lr(epoch, max_epoch, iter, max_iter):
    lr0 = 1e-3
    it = (epoch * max_iter) + iter
    power = 0.9

    factor = (1-(it/(max_epoch*max_iter)))**power
    lr = lr0 * factor

    return lr


def online(root, batch_size, learning_rate, input, load):
    now = time.strftime(r'%Y-%m-%d_%H-%M',time.localtime(time.time()))

    test_dataset = FaceDataset(root, 'test')
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=4)

    model_save_pth = os.path.join('online', now)
    os.makedirs(model_save_pth, exist_ok=True)

    model = BiSeNet(n_classes=NUM_CLASSES)

    if torch.cuda.is_available():
        model = model.cuda()
    print("Model Structure: ", model, "\n\n")

    lr0 = learning_rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr0, momentum=0.9, weight_decay=5e-4)
    num_imgs = test_dataset.__len__()
    max_iter = math.ceil(num_imgs / batch_size)

    model_dir = load
    model_root = os.path.join('./pretrained', model_dir)
    model_path = os.path.join(model_root, os.listdir(model_root)[-1])
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    score_thres = 0.7
    n_min = 1 * 512 * 512//16      ## batch_size * crop_size[0] * crop_size[1]//16
    # n_min = 1 * 1080 * 2048//16      ## batch_size * crop_size[0] * crop_size[1]//16
    ignore_idx = -100

    LossP = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss2 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss3 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    model.eval()
    avg_time = 0
    losses= 0
    avg_miou = 0
    best_miou = float(0)
    for n_iter, (images, infos) in enumerate(dataloader):
        if torch.cuda.is_available():
            images = images.cuda()
            segments = infos[0].long().cuda()
            depths = infos[1].cuda()

        start = time.time()
        if input == 'depth':
            outputs, outputs16, outputs32 = model(depths)
        elif input == 'rgb':
            outputs, outputs16, outputs32 = model(images)
        # end = time.time()
        # avg_time += end - start
        segments = segments.squeeze(dim=1)
        loss = LossP(outputs, segments) + Loss2(outputs16, segments) + Loss3(outputs32, segments)
        losses = losses + loss

        optimizer.zero_grad()
        loss.backward()

        lr = get_lr(epoch=0, max_epoch=1, iter=n_iter, max_iter=max_iter)
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
        end = time.time()
        avg_time += end - start

        '''
            Run these lines except using random 2k images.
        '''
        # scale_parse = F.upsample(input=outputs.unsqueeze(dim=1)[0], size=(512, 512), mode='bilinear') # parsing
        # result_parse = torch.argmax(scale_parse, dim=1).squeeze()
        # result_parse = torch.stack([result_parse, result_parse, result_parse], dim=0).type(torch.uint8)

        result_parse = torch.argmax(outputs, dim=1)     ## result_parse.shape: [batch_size, 512, 512]
        miou = cal_miou(result_parse, segments)
        avg_miou += miou

        '''
            Visualization
        '''
        # images = images.cpu().squeeze().permute(1,2,0)
        # alpha = 0.5

        # seg_color = np.zeros(result_parse.shape)
        # for key in label_to_color.keys():
        #     seg_color[result_parse[:,:,0] == key] = label_to_color[key]

        # # segments = segments.squeeze().cpu()
        # # segments = np.stack([segments, segments, segments], axis=-1)
        # # seg_color = np.zeros(segments.shape)
        # # for key in label_to_color.keys():
        # #     seg_color[segments[:,:,0] == key] = label_to_color[key]

        # blended = (images * alpha) + (seg_color * (1 - alpha))
        # blended = blended.type(torch.uint8)

        # plt.imshow(blended)
        # plt.show()

        print('{} Iterations / Loss: {:.4f}, mIoU: {:.4f}'.format(n_iter, loss, miou))

        if miou > best_miou:
            best_miou = miou
            print("Best mIoU: {:.4f}".format(best_miou))
            file_name = '{:03d}_{:.4f}.ckpt'.format(n_iter, best_miou)
            torch.save(
                {
                    'n_iter': n_iter,
                    'model_state_dict': model.state_dict(),
                    'opt_state_dict': optimizer.state_dict()
                },
                os.path.join(model_save_pth, file_name)
            )

    print("avg_time: ", avg_time / n_iter)
    print("avg_loss: ", losses / n_iter)
    print("avg_miou: ", avg_miou / n_iter)

if __name__ == '__main__':
    root, batch_size, _, learning_rate, input, load = get_arguments()

    # root = '/mnt/server7_hard0/msson/CelebA-HQ'
    # input = 'rgb'
    # load = '2021-10-29_17-19'

    online(root, batch_size, learning_rate, input, load)