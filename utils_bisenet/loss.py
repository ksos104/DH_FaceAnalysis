#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# def seperate_labels(labels, n_classes=11, ignore_lb=255):
#     labels_sep = torch.Tensor(n_classes, labels.shape[0], labels.shape[1], labels.shape[2])
#     labels_sep = labels_sep.to(labels.device)

#     for cls_idx in range(n_classes):
#         labels_sep[cls_idx] = torch.where(labels==cls_idx, labels, torch.tensor([ignore_lb], dtype=torch.int64).to(labels.device))

#     labels_sep = labels_sep.permute(1,0,2,3)

#     return labels_sep
    

def get_freq_weights(labels, n_classes=11):
    n_clsPixel = torch.Tensor(labels.shape[0], n_classes).to(labels.device)     ## n_clsPixel.shape: [8, 11]
    freq = torch.Tensor(labels.shape[0], n_classes).to(labels.device)     ## freq.shape: [8, 11]
    freq_w = torch.Tensor(labels.shape[0], n_classes).to(labels.device)     ## freq_w.shape: [8, 11]
    
    tensor0 = torch.tensor([0], dtype=torch.int64).to(labels.device)
    tensor1 = torch.tensor([1], dtype=torch.int64).to(labels.device)
    for cls_idx in range(n_classes):    ## labels.shape: [8, 512, 512]
        oh_cls = torch.where(labels==cls_idx, tensor1, tensor0)
        oh_cls = oh_cls.reshape(labels.shape[0], -1)
        n_clsPixel[:,cls_idx] = torch.count_nonzero(oh_cls, dim=-1)

        n_clsSum = torch.sum(n_clsPixel, dim=0)
        freq[:,cls_idx] = n_clsPixel[:,cls_idx] / n_clsSum[cls_idx]

    median = torch.median(freq)
    freq = torch.where(freq==0, median, freq)
    freq_w = median / freq

    return freq_w


def get_area_weights(labels, n_classes=11):
    n_clsPixel = torch.Tensor(labels.shape[0], n_classes).to(labels.device)     ## n_clsPixel.shape: [8, 11]
    occurence = torch.Tensor(labels.shape[0], n_classes).to(labels.device)      ## occurence.shape: [8, 11]
    area = torch.Tensor(labels.shape[0], n_classes).to(labels.device)           ## area.shape: [8, 11]
    area_w = torch.Tensor(labels.shape[0], n_classes).to(labels.device)         ## area_w.shape: [8, 11]

    tensor0 = torch.tensor([0], dtype=torch.int64).to(labels.device)
    tensor1 = torch.tensor([1], dtype=torch.int64).to(labels.device)
    for cls_idx in range(n_classes):    ## labels.shape: [8, 512, 512]
        oh_cls = torch.where(labels==cls_idx, tensor1, tensor0)
        oh_cls = oh_cls.reshape(labels.shape[0], -1)
        n_clsPixel[:,cls_idx] = torch.count_nonzero(oh_cls, dim=-1)

        # n_clsSum = torch.sum(n_clsPixel, dim=0)
        # freq[:,cls_idx] = n_clsPixel[:,cls_idx] / n_clsSum

    return area_w

class WCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(WCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        '''
            MFB or MAB loss
        '''
        n_classes = logits.shape[1]
        # labels_sep = seperate_labels(labels, n_classes, ignore_lb=self.ignore_lb)
        freq_w = get_freq_weights(labels, n_classes)
        # area_w = get_area_weights(labels, n_classes)

        loss = self.criteria(logits, labels)

        cls_loss = torch.Tensor(labels.shape[0], n_classes, labels.shape[1], labels.shape[2]).to(labels.device)         ## cls_loss.shape = [8, 11, 512, 512]
        tensor0 = torch.tensor([0.]).to(labels.device)
        for cls_idx in range(n_classes):
            cls_loss[:,cls_idx] = torch.where(labels==cls_idx, loss, tensor0)

        result = cls_loss * (freq_w.unsqueeze(-1).unsqueeze(-1)).expand_as(cls_loss)
        return torch.mean(result)


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        # loss, _ = torch.sort(loss, descending=True)
        # if loss[self.n_min] > self.thresh:
        #     loss = loss[loss>self.thresh]
        # else:
        #     loss = loss[:self.n_min]
        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss


if __name__ == '__main__':
    torch.manual_seed(15)
    criteria1 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).cuda()
    criteria2 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).cuda()
    net1 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net1.cuda()
    net1.train()
    net2 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net2.cuda()
    net2.train()

    with torch.no_grad():
        inten = torch.randn(16, 3, 20, 20).cuda()
        lbs = torch.randint(0, 19, [16, 20, 20]).cuda()
        lbs[1, :, :] = 255

    logits1 = net1(inten)
    logits1 = F.interpolate(logits1, inten.size()[2:], mode='bilinear')
    logits2 = net2(inten)
    logits2 = F.interpolate(logits2, inten.size()[2:], mode='bilinear')

    loss1 = criteria1(logits1, lbs)
    loss2 = criteria2(logits2, lbs)
    loss = loss1 + loss2
    print(loss.detach().cpu())
    loss.backward()
