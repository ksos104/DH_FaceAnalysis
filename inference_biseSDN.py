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
from datasets import FaceDataset
from models.bisenet_dec_model import BiSeNet

import argparse
import time
import os
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

criteria = nn.L1Loss()
NUM_CLASSES = 8

label_to_color = {
    # 0: [128, 64,128],
    # 1: [244, 35,232],
    # 2: [ 70, 70, 70],
    # 3: [102,102,156],
    # 4: [190,153,153],
    # 5: [153,153,153],
    # 6: [250,170, 30],
    # 7: [220,220,  0],
    # 8: [107,142, 35],
    # 9: [152,251,152],
    # 10: [ 70,130,180]
    0: [0, 0, 0], 
    1: [128, 0, 0],
    2: [0, 128, 0],
    3: [128, 128, 0],
    4: [0, 0, 128],
    5: [128, 0, 128],
    6: [0, 128, 128],
    7: [128, 128, 128]
    }

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='Root directory path that consists of train and test directories.', default=None, dest='root')
    parser.add_argument('--batch_size', help='Batch size (int)', default=1, dest='batch_size')
    parser.add_argument('--load', help='Checkpoint directory name (ex. 2021-09-27_22-06)', default='2021-12-01_20-18', dest='load')
    parser.add_argument('--save', help='Save result images (segmentation, depth, normal)', default=False, dest='save_res')
    parser.add_argument('--show', help='Show result images (segmentation, depth, normal)', default=False, dest='show_res')

    root = parser.parse_args().root
    batch_size = parser.parse_args().batch_size
    load = parser.parse_args().load
    save_res = parser.parse_args().save_res
    show_res = parser.parse_args().show_res

    return root, batch_size, load, save_res, show_res


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
        image??? ?????? ??? batch ??????
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
        image??? ?????? ??? batch ??????
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
    ?????? dataset??? ?????? i??? u ???????????? mIoU ??????
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


def inference(root, batch_size, load, save_res, show_res):
    test_dataset = FaceDataset(root, 'test')
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=4)

    model = BiSeNet(n_classes=NUM_CLASSES)
    # model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.cuda()
    print("Model Structure: ", model, "\n\n")

    load_root = './pretrained'
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

        avg_depth_loss = 0
        avg_normal_loss = 0
        for n_iter, (images, infos) in enumerate(dataloader):
            if torch.cuda.is_available():
                images = images.cuda()
                segment = infos[0].long().cuda()
                depth = infos[1].cuda()
                normal = infos[2].cuda()

            start = time.time()
            outputs_seg, outputs_depth, outputs_normal = model(images)
            end = time.time()
            avg_time += end - start
            avg_depth_loss += criteria(outputs_depth, depth[:,0,...])
            avg_normal_loss += criteria(outputs_normal, normal)

            ## segmentation
            segment = segment.squeeze(dim=1)
            result_parse = torch.argmax(outputs_seg, dim=1)     ## result_parse.shape: [batch_size, 512, 512]
            # result_parse = segment
            miou = cal_miou(result_parse, segment)
            avg_miou += miou
            res_i, res_u = cal_miou_total(result_parse, segment)
            total_i += res_i
            total_u += res_u

            '''
                Visualization
            '''
            images = images.cpu().squeeze().permute(1,2,0)
            images = images.numpy()[...,::-1]

            ## segmentation
            alpha = 0

            result_parse = result_parse.squeeze()
            result_parse = torch.stack([result_parse, result_parse, result_parse], dim=0).type(torch.uint8)
            result_parse = result_parse.cpu().squeeze().permute(1,2,0)

            seg_color = np.zeros(result_parse.shape)
            for key in label_to_color.keys():
                seg_color[result_parse[:,:,0] == key] = label_to_color[key]

            blended = (images * alpha) + (seg_color * (1 - alpha))

            '''
                SAVE RESULT IMAGE
            '''
            if save_res:
                save_path = os.path.join(root, 'test', 'result', load, 'seg')
                os.makedirs(save_path, exist_ok=True)
                save_name = 'res_seg' + str(n_iter) + '.png'
                plt.imsave(os.path.join(save_path, save_name), blended/255)

            '''
                SHOW RESULT IMAGE
            '''
            if show_res: 
                blended = torch.from_numpy(blended).type(torch.uint8)
                plt.imshow(blended)
                plt.show()


            ## depth
            alpha = 0

            outputs_depth = outputs_depth.type(torch.uint8)
            outputs_depth = outputs_depth.cpu().squeeze(dim=0).permute(1,2,0)
            outputs_depth = outputs_depth.numpy()[...,::-1]

            blended = (images * alpha) + (outputs_depth * (1 - alpha))

            '''
                SAVE RESULT IMAGE
            '''
            if save_res:
                save_path = os.path.join(root, 'test', 'result', load, 'depth')
                os.makedirs(save_path, exist_ok=True)
                save_name = 'res_depth' + str(n_iter) + '.png'
                plt.imsave(os.path.join(save_path, save_name), blended/255)

            '''
                SHOW RESULT IMAGE
            '''
            if show_res:
                blended = torch.from_numpy(blended).type(torch.uint8)
                plt.imshow(blended)
                plt.show()

            ## normal
            alpha = 0

            outputs_normal = outputs_normal.type(torch.uint8)
            outputs_normal = outputs_normal.cpu().squeeze(dim=0).permute(1,2,0)
            outputs_normal = outputs_normal.numpy()[...,::-1]

            blended = (images * alpha) + (outputs_normal * (1 - alpha))
            
            '''
                SAVE RESULT IMAGE
            '''
            if save_res:
                save_path = os.path.join(root, 'test', 'result', load, 'depth')
                os.makedirs(save_path, exist_ok=True)
                save_name = 'res_depth' + str(n_iter) + '.png'
                plt.imsave(os.path.join(save_path, save_name), blended/255)

            '''
                SHOW RESULT IMAGE
            '''
            if show_res:
                blended = torch.from_numpy(blended).type(torch.uint8)
                plt.imshow(blended)
                plt.show()

            print('{} Iterations'.format(n_iter))
        print("avg_time: ", avg_time / (n_iter+1))

        avg_miou = torch.sum(total_i / total_u).item() / NUM_CLASSES
        print("avg_mIoU: ", avg_miou)
        print("IoU: ", total_i / total_u)
        print("avg_depth_loss: ", avg_depth_loss / (n_iter+1))
        print("avg_normal_loss: ", avg_normal_loss / (n_iter+1))


if __name__ == '__main__':
    root, batch_size, load, save_res, show_res = get_arguments()

    root = r'D:\DH_dataset\FFHQ'
    batch_size = 1
    load = '2021-12-01_20-18'
    save_res = False
    show_res = True

    inference(root, batch_size, load, save_res, show_res)