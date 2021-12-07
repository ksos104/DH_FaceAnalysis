'''
    Written by msson
    2021.08.10
'''
import os
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import random

class FaceDataset(Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.aspect_ratio = 1
        self.crop_size = [512, 512]
        self.scale_factor = 0.25
        self.rotation_factor = 30

        self.onlyImg = True
        if len(os.listdir(os.path.join(self.root, self.mode))) >= 4:
            self.onlyImg = False

        self.imgs = os.listdir(os.path.join(self.root, self.mode, 'images'))
        if not self.onlyImg:
            self.seg_list = os.listdir(os.path.join(self.root, self.mode, 'segment8'))
            self.normal_list = os.listdir(os.path.join(self.root, self.mode, 'normal'))
            self.depth_list = os.listdir(os.path.join(self.root, self.mode, 'depth'))
        
            '''
                depth.npy 파일은 server11_hard0에 위치
                아래 depth 부분도 모두 수정 필요.
            '''
            # self.depth_root = '/mnt/server11_hard0/yoonji/DH_dataset/CelebAMask-HQ'
            # self.depth_list = os.listdir(os.path.join(self.depth_root, self.mode, 'depth'))

    def __len__(self):
        '''
            Only for 10,000 random 2k images
        '''
        # return 10000


        return len(self.imgs)

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)

        return center, scale

    def __getitem__(self, idx):
        '''
            Only for 10,000 random 2k images
        '''
        # img = torch.rand((3, 1080, 2048))
        # segment = torch.rand((1, 1080, 2048))
        # depth = torch.rand((3, 1080, 2048))
        # normal = torch.rand((3, 1080, 2048))
        # return img, (segment, depth, normal)

        img_path = os.path.join(self.root, self.mode, 'images', self.imgs[idx])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if self.onlyImg:
            segment = torch.rand((1, 512, 512))
            depth = torch.rand((3, 512, 512))
            normal = torch.rand((3, 512, 512))
        else:
            file_name = os.path.splitext(self.imgs[idx])[0] + '.png'
            normal_name = os.path.splitext(self.imgs[idx])[0] + '_normal.png'   
            segment_path = os.path.join(self.root, self.mode, 'segment', file_name)          ## segment 8 labels
            normal_path = os.path.join(self.root, self.mode, 'normal', normal_name)
            depth_name = os.path.splitext(self.imgs[idx])[0] + '_depth.png'
            depth_path = os.path.join(self.root, self.mode, 'depth', depth_name)

            segment = cv2.imread(segment_path, cv2.IMREAD_GRAYSCALE)
            normal = cv2.imread(normal_path, cv2.IMREAD_COLOR)
            depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            depth = np.stack([depth,depth,depth], axis=-1)

            # depth_name = os.path.splitext(self.imgs[idx])[0] + '_depth.npy'
            # depth_path = os.path.join(self.depth_root, self.mode, 'depth', depth_name)
            # depth = np.load(depth_path)

        '''
            CelebA dataset
                image (depth, normal) size : 1024 x 1024
                segment size : 512 x 512

            Resize to 512 x 512
        '''
        if self.onlyImg:
            seg_h, seg_w = 512, 512
            img = cv2.resize(img, dsize=(seg_w, seg_h))
            img = torch.Tensor(img).permute(2,0,1)
        else:
            seg_h, seg_w = segment.shape
            img = cv2.resize(img, dsize=(seg_w, seg_h))
            depth = cv2.resize(depth, dsize=(seg_w, seg_h))
            normal = cv2.resize(normal, dsize=(seg_w, seg_h))

            img = torch.Tensor(img).permute(2,0,1)
            segment = torch.Tensor(segment).unsqueeze(0)
            depth = torch.Tensor(depth).permute(2,0,1)
            normal = torch.Tensor(normal).permute(2,0,1)

        return img, (segment, depth, normal)