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
from torch.nn import functional as F
import cv2
import matplotlib.pyplot as plt

NUM_CLASSES = 11

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


criterion = CriterionCrossEntropyEdgeParsing_boundary_attention_loss()
if torch.cuda.is_available():
    criterion = criterion.cuda()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='Root directory path that consists of train and test directories.', default=r'D:\DH_dataset\face_seg\HELENstar\helenstar_release', dest='root')
    parser.add_argument('--batch_size', help='Batch size (int)', default=4, dest='batch_size')
    parser.add_argument('--epoch', help='Number of epoch (int)', default=400, dest='n_epoch')
    parser.add_argument('--lr', help='Learning rate', default=1e-3, dest='learning_rate')

    root = parser.parse_args().root
    batch_size = parser.parse_args().batch_size
    n_epoch = parser.parse_args().n_epoch
    learning_rate = parser.parse_args().learning_rate

    return root, batch_size, n_epoch, learning_rate


def cal_miou(result, gt):                ## resutl.shpae == gt.shape == [512, 512]
    # miou = np.zeros((10))
    miou = 0
    for idx in range(1,11):              ## background 제외
        if idx == 3 or idx == 5 or idx == 9:
            continue
        elif idx == 2 or idx == 4:
            u = torch.sum(torch.where(((result==idx)+(result==idx+1)) + ((gt==idx)+(gt==idx+1)), torch.Tensor([1]), torch.Tensor([0]))).item()
            o = torch.sum(torch.where(((result==idx)+(result==idx+1)) * ((gt==idx)+(gt==idx+1)), torch.Tensor([1]), torch.Tensor([0]))).item()
        elif idx == 7:
            u = torch.sum(torch.where(((result==idx)+(result==idx+2)) + ((gt==idx)+(gt==idx+2)), torch.Tensor([1]), torch.Tensor([0]))).item()
            o = torch.sum(torch.where(((result==idx)+(result==idx+2)) * ((gt==idx)+(gt==idx+2)), torch.Tensor([1]), torch.Tensor([0]))).item()
        else:
            u = torch.sum(torch.where((result==idx) + (gt==idx), torch.Tensor([1]), torch.Tensor([0]))).item()
            o = torch.sum(torch.where((result==idx) * (gt==idx), torch.Tensor([1]), torch.Tensor([0]))).item()
        iou = o / u
        # miou[idx-1] += iou
        miou += iou

    return miou / 7


def inference(root):
    test_dataset = FaceDataset(root, 'test')
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=torch.cuda.is_available())

    model = EAGRNet(num_classes=NUM_CLASSES)

    if torch.cuda.is_available():
        model = model.cuda()
    print("Model Structure: ", model, "\n\n")

    model_dir = '2021-09-05_13-33'
    model_root = os.path.join(r'C:\Users\Minseok\Desktop\DH_FaceAnalysis\pretrained', model_dir, 'RGBparse')
    model_path = os.path.join(model_root, os.listdir(model_root)[-1])
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    with torch.no_grad():
        avg_time = 0
        avg_miou = 0
        for n_iter, (images, infos) in enumerate(dataloader):
            if torch.cuda.is_available():
                images = images.cuda()
                segments = infos[0].long().cuda()
                edges = infos[1].long().cuda()
                depths = infos[2].cuda()

            import time
            start = time.time()
            input = 'rgb'
            if input == 'depth':
                outputs = model(depths)
            elif input == 'rgb':
                outputs = model(images)
            end = time.time()
            avg_time += end - start
            loss = criterion(outputs, [segments, edges])

            scale_parse = F.upsample(input=outputs[0], size=(512, 512), mode='bilinear') # parsing
            result_parse = torch.argmax(scale_parse, dim=1).squeeze()
            result_parse = torch.stack([result_parse, result_parse, result_parse], dim=0).type(torch.uint8)
            # cv2.imshow('result', result_parse.cpu().numpy())
            # cv2.waitKey(0)

            alpha = 0.5

            images = images.cpu().squeeze().permute(1,2,0)
            result_parse = result_parse.cpu().squeeze().permute(1,2,0)

            miou = cal_miou(result_parse[...,0], segments.squeeze().cpu())
            avg_miou += miou

            seg_color = np.zeros(result_parse.shape)
            for key in label_to_color.keys():
                seg_color[result_parse[:,:,0] == key] = label_to_color[key]

            blended = (images * alpha) + (seg_color * (1 - alpha))
            blended = blended.type(torch.uint8)
            # blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

            # plt.imshow(blended)
            # plt.show()

            print('{} Iterations / Loss: {:.4f}'.format(n_iter, loss))
        print("avg_time: ", avg_time / 100)
        print("avg_miou\n", avg_miou / 100)

if __name__ == '__main__':
    root, _, _, _ = get_arguments()

    inference(root)