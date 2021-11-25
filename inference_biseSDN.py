'''
    Written by msson
    2021.08.10

    Modified by msson
    2021.08.30

    Modified by msson
    2021.09.28
'''
from ctypes import resize
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from BiSeSDN import NUM_CLASSES
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
    parser.add_argument('--lr', help='Learning rate', default=1e-2, dest='learning_rate')
    parser.add_argument('--load', help='checkpoint directory name (ex. 2021-09-27_22-06)', default=None, dest='load')

    root = parser.parse_args().root
    batch_size = parser.parse_args().batch_size
    n_epoch = parser.parse_args().n_epoch
    learning_rate = parser.parse_args().learning_rate
    load = parser.parse_args().load

    return root, batch_size, n_epoch, learning_rate, load


def get_freq(result):                   ## result.shpae == [batch_size, 512, 512]
    freq_list = torch.zeros((result.shape[0], NUM_CLASSES)).to(result.device)
    tensor1 = torch.Tensor([1]).to(result.device)
    tensor0 = torch.Tensor([0]).to(result.device)

    total_pixel = result.shape[1] * result.shape[2]
    for idx in range(NUM_CLASSES):
        freq_list[:, idx] = torch.sum(torch.sum(torch.where(result==idx, tensor1, tensor0), dim=-1), dim=-1)
    
    return freq_list, total_pixel       ## freq_list.shape == [batch_size, NUM_CLASSES]


def cal_miou(result, gt):                ## resutl.shpae == gt.shape == [batch_size, 512, 512]
    miou = 0
    batch_size = result.shape[0]
    
    tensor1 = torch.Tensor([1]).to(gt.device)
    tensor0 = torch.Tensor([0]).to(gt.device)

    '''
        WRONG CODE
    '''
    # for idx in range(NUM_CLASSES):
    #     u = torch.sum(torch.where((result==idx) + (gt==idx), tensor1, tensor0)).item()
    #     i = torch.sum(torch.where((result==idx) * (gt==idx), tensor1, tensor0)).item()
    #     try:
    #         iou = i / u
    #     except:
    #         continue
    #     miou += iou
    #     # iou_list[idx] += iou
    # miou = miou / NUM_CLASSES

    # return miou

    '''
        image별 계산 후 batch 평균
    '''
    iou_mat = torch.zeros((batch_size, NUM_CLASSES)).to(result.device)
    mask_mat = torch.zeros((batch_size, NUM_CLASSES)).to(result.device)

    for idx in range(NUM_CLASSES):
        mask_mat[:,idx] = torch.sum(torch.sum(torch.where(gt==idx, tensor1, tensor0), dim=-1), dim=-1)              ## mask_mat.shape == [batch_size, NUM_CLASSES]
    n_cls_tensor = torch.sum(torch.where(mask_mat>0, tensor1, tensor0), dim=-1)                                     ## n_cls_tensor.shape == [batch_size]

    for idx in range(NUM_CLASSES):              
        u_tensor = torch.sum(torch.sum(torch.where((result==idx) + (gt==idx), tensor1, tensor0),dim=-1),dim=-1)
        i_tensor = torch.sum(torch.sum(torch.where((result==idx) * (gt==idx), tensor1, tensor0),dim=-1),dim=-1)

        u_tensor = torch.where(u_tensor==0, tensor1, u_tensor)
        iou_tensor = i_tensor / u_tensor

        iou_list = iou_tensor
        iou_mat[:,idx] += iou_list              ## iou_mat.shape == [batch_size, NUM_CLASSES]

    batch_iou = torch.sum(iou_mat, dim=-1) / n_cls_tensor
    miou = (torch.sum(batch_iou) / batch_size).item()

    return miou

    '''
        image별 계산 후 batch 평균
        frequency weighted IoU
    '''
    # freq_list, total_pixel = get_freq(result)
    # iou_mat = torch.zeros((result.shape[0], NUM_CLASSES)).to(result.device)
    # for idx in range(NUM_CLASSES):              
    #     u_tensor = torch.sum(torch.sum(torch.where((result==idx) + (gt==idx), tensor1, tensor0),dim=-1),dim=-1)
    #     i_tensor = torch.sum(torch.sum(torch.where((result==idx) * (gt==idx), tensor1, tensor0),dim=-1),dim=-1)

    #     u_tensor = torch.where(u_tensor==0, tensor1, u_tensor)
    #     iou_tensor = i_tensor / u_tensor

    #     iou_list = iou_tensor
    #     iou_mat[:,idx] += iou_list * freq_list[:,idx]
    
    # batch_iou = torch.sum(iou_mat, dim=-1)
    # miou = (torch.sum(batch_iou) / result.shape[0]).item()

    # return miou / total_pixel


'''
    전체 dataset에 대해 i와 u 합산하여 mIoU 계산
'''
def cal_miou_total(result, gt):                ## resutl.shpae == gt.shape == [batch_size, 512, 512]    
    tensor1 = torch.Tensor([1]).to(gt.device)
    tensor0 = torch.Tensor([0]).to(gt.device)

    res_i = torch.zeros((NUM_CLASSES)).to(result.device)
    res_u = torch.zeros((NUM_CLASSES)).to(result.device)

    for idx in range(NUM_CLASSES):
        u = torch.sum(torch.where((result==idx) + (gt==idx), tensor1, tensor0)).item()
        i = torch.sum(torch.where((result==idx) * (gt==idx), tensor1, tensor0)).item()

        res_i[idx] += i
        res_u[idx] += u

    return res_i, res_u


def mul_weight(outputs):
    total_area = 11
    area_list = torch.tensor([1,1,1,1,1,1,1,1]).to(outputs.device) / total_area
    area_list = area_list.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
    
    outputs = outputs * area_list

    return outputs


def inference(root, batch_size, load_root, load, NUM_CLASSES):
    test_dataset = FaceDataset(root, 'val')
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=4)

    model = BiSeNet(n_classes=NUM_CLASSES)
    # model = nn.DataParallel(model)

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
        avg_miou = 0
        total_i = torch.zeros((NUM_CLASSES)).cuda()
        total_u = torch.zeros((NUM_CLASSES)).cuda()
        for n_iter, (images, infos) in enumerate(dataloader):
            if torch.cuda.is_available():
                images = images.cuda()
                segment = infos[0].long().cuda()
                depth = infos[1].cuda()
                normal = infos[2].cuda()

            import time
            start = time.time()
            outputs_seg, outputs_depth, outputs_normal = model(images)
            end = time.time()
            avg_time += end - start
            loss = criteria(outputs_depth, depth[:,0,...])

            '''
                weight 적용
            '''
            # outputs_seg = mul_weight(outputs_seg)

            '''
                Run these lines except using random 2k images.
            '''
            ## segmentation
            segment = segment.squeeze(dim=1)
            result_parse = torch.argmax(outputs_seg, dim=1)     ## result_parse.shape: [batch_size, 512, 512]
            # miou = cal_miou(result_parse, segment)
            # avg_miou += miou
            res_i, res_u = cal_miou_total(result_parse, segment)
            total_i += res_i
            total_u += res_u

            '''
                Visualization
            '''
            # images = images.cpu().squeeze().permute(1,2,0)
            # images = images.numpy()[...,::-1]

            # ## segmentation
            # alpha = 0.5

            # result_parse = result_parse.squeeze()
            # result_parse = torch.stack([result_parse, result_parse, result_parse], dim=0).type(torch.uint8)
            # result_parse = result_parse.cpu().squeeze().permute(1,2,0)

            # seg_color = np.zeros(result_parse.shape)
            # for key in label_to_color.keys():
            #     seg_color[result_parse[:,:,0] == key] = label_to_color[key]

            # blended = (images * alpha) + (seg_color * (1 - alpha))
            # blended = torch.from_numpy(blended).type(torch.uint8)

            # plt.imshow(blended)
            # plt.show()

            # ## depth
            # alpha = 0

            # outputs_depth = outputs_depth.type(torch.uint8)
            # outputs_depth = outputs_depth.cpu().squeeze(dim=0).permute(1,2,0)
            # outputs_depth = outputs_depth.numpy()[...,::-1]

            # blended = (images * alpha) + (outputs_depth * (1 - alpha))
            # blended = torch.from_numpy(blended).type(torch.uint8)

            # plt.imshow(blended)
            # plt.show()

            # ## normal
            # alpha = 0

            # outputs_normal = outputs_normal.type(torch.uint8)
            # outputs_normal = outputs_normal.cpu().squeeze(dim=0).permute(1,2,0)
            # outputs_normal = outputs_normal.numpy()[...,::-1]

            # blended = (images * alpha) + (outputs_normal * (1 - alpha))
            # blended = torch.from_numpy(blended).type(torch.uint8)

            # plt.imshow(blended)
            # plt.show()

            print('{} Iterations / Loss: {:.4f}'.format(n_iter, loss))
        print("avg_time: ", avg_time / (n_iter+1))

        avg_miou = torch.sum(total_i / total_u).item() / NUM_CLASSES
        print("avg_mIoU: ", avg_miou)

        # print("avg_mIoU: ", avg_miou / (n_iter+1))
        # print("avg_iou: ", iou_list / n_iter)

if __name__ == '__main__':
    root, batch_size, _, _, load = get_arguments()

    load_root = './pretrained'
    batch_size = 8
    load = '2021-11-18_10-21'
    # load = '2021-11-24_11-16'
    NUM_CLASSES = 8

    inference(root, batch_size, load_root, load, NUM_CLASSES)