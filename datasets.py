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

class FaceDataset(Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.aspect_ratio = 1
        self.crop_size = [512, 512]
        
        self.imgs = os.listdir(os.path.join(self.root, self.mode, 'images'))
        self.seg_list = os.listdir(os.path.join(self.root, self.mode, 'segments'))
        self.edge_list = os.listdir(os.path.join(self.root, self.mode, 'edges'))
        self.depth_list = os.listdir(os.path.join(self.root, self.mode, 'depth_npy'))

    def __len__(self):
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
        img_path = os.path.join(self.root, self.mode, 'images', self.imgs[idx])

        file_name = os.path.splitext(self.imgs[idx])[0] + '.png'
        depth_name = os.path.splitext(self.imgs[idx])[0] + '_depth.npy'
        segment_path = os.path.join(self.root, self.mode, 'segments', file_name)
        edge_path = os.path.join(self.root, self.mode, 'edges', file_name)
        depth_path = os.path.join(self.root, self.mode, 'depth_npy', depth_name)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        segment = cv2.imread(segment_path, cv2.IMREAD_GRAYSCALE)
        edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        depth = np.load(depth_path)
        depth = np.stack([depth,depth,depth], axis=-1)

        h, w, _ = img.shape
        center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        trans = get_affine_transform(center, s, r, self.crop_size)
        img = cv2.warpAffine(
            img,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        edge = cv2.warpAffine(
            edge,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        edge[np.where(edge != 0)] = 1
        segment = cv2.warpAffine(
            segment,
            trans,
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

        img = torch.Tensor(img).permute(2,0,1)
        segment = torch.Tensor(segment).unsqueeze(0)
        edge = torch.Tensor(edge).unsqueeze(0)
        depth = torch.Tensor(depth).permute(2,0,1)

        return img, (segment, edge, depth)
        # return img, (torch.Tensor([0]), torch.Tensor([0]), depth)