'''
    Written by msson
    2021.08.10
'''
import os
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
from utils.transforms import *
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
        
        self.imgs = os.listdir(os.path.join(self.root, self.mode, 'images'))
        self.seg_list = os.listdir(os.path.join(self.root, self.mode, 'segment'))
        self.depth_list = os.listdir(os.path.join(self.root, self.mode, 'depth'))
        self.normal_list = os.listdir(os.path.join(self.root, self.mode, 'normal'))

    def __len__(self):
        return len(self.imgs)
        # return 10000

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
        img_path = os.path.join(self.root, self.mode, 'images', self.imgs[idx])

        file_name = os.path.splitext(self.imgs[idx])[0] + '.png'
        depth_name = os.path.splitext(self.imgs[idx])[0] + '_depth.png'
        normal_name = os.path.splitext(self.imgs[idx])[0] + '_normal.png'
        segment_path = os.path.join(self.root, self.mode, 'segment', file_name)
        depth_path = os.path.join(self.root, self.mode, 'depth', depth_name)
        normal_path = os.path.join(self.root, self.mode, 'normal', normal_name)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        segment = cv2.imread(segment_path, cv2.IMREAD_GRAYSCALE)
        # depth = np.load(depth_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        depth = np.stack([depth,depth,depth], axis=-1)
        normal = cv2.imread(normal_path, cv2.IMREAD_COLOR)

        h, w, _ = img.shape
        center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        seg_h, seg_w = segment.shape
        seg_center, seg_s = self._box2cs([0, 0, seg_w - 1, seg_h - 1])
        seg_r = 0

        # if self.mode == 'train':
        #     sf = self.scale_factor
        #     rf = self.rotation_factor
        #     s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        #     r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
        #         if random.random() <= 0.6 else 0

        trans = get_affine_transform(center, s, r, self.crop_size)
        seg_trans = get_affine_transform(seg_center, seg_s, seg_r, self.crop_size)
        img = cv2.warpAffine(
            img,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        segment = cv2.warpAffine(
            segment,
            seg_trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        depth = cv2.warpAffine(
            depth,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        normal = cv2.warpAffine(
            normal,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        img = torch.Tensor(img).permute(2,0,1)
        segment = torch.Tensor(segment).unsqueeze(0)
        depth = torch.Tensor(depth).permute(2,0,1)
        normal = torch.Tensor(normal).permute(2,0,1)

        return img, (segment, depth, normal)

        
        # img = torch.rand((3, 1080, 2048))
        # segment = torch.rand((1, 1080, 2048))
        # depth = torch.rand((3, 1080, 2048))
        # normal = torch.rand((1, 1080, 2048))
        # return img, (segment, depth, normal)