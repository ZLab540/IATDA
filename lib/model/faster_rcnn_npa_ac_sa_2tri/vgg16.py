from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn_npa_ac_sa_2tri.faster_rcnn import _fasterRCNN
from model.utils.config import cfg
import pdb


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


class netC(nn.Module):  # 用于分类黑或白
    def __init__(self):
        super(netC, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(256, 16, 1, 1, 0)
        self.excitation = nn.Conv2d(16, 256, 1, 1, 0)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        mask = F.sigmoid(out)
        out = self.fc2(self.fc1(out.view(b, c)))
        return mask, F.sigmoid(out)


class net_CSA(nn.Module):
    def __init__(self, kernel_size=7):
        super(net_CSA, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=4)
        self.fc1 = nn.Linear(37 * 46, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        b, _, h, w = x.size()
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        mask = self.sigmoid(x)

        x = F.relu(x)
        x = self.maxpool(x)
        x = F.relu(self.fc1(x.view(b, (h // 4) * (w // 4))))
        x = self.fc2(x)
        return mask, x


class netD_pixel(nn.Module):
    def __init__(self, context=False):
        super(netD_pixel, self).__init__()
        self.conv1 = conv1x1(256, 256)
        # self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = conv1x1(256, 128)
        # self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv1x1(128, 1)

        self.context = context

    def forward(self, x):
        x = F.relu(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.context:
            feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
            # feat = x
            x = F.sigmoid(self.conv3(x))
            return x.view(-1, 1), feat  # torch.cat((feat1,feat2),1)#F
        else:
            x = F.sigmoid(self.conv3(x))
            return x.view(-1, 1)  # F.sigmoid(x)


class netD(nn.Module):
    def __init__(self, context=False):
        super(netD, self).__init__()
        self.conv1 = conv3x3(512, 512, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = conv3x3(512, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128, 2)
        self.context = context

    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))), training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))), training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))), training=self.training)
        x = F.avg_pool2d(x, (x.size(2), x.size(3)))
        x = x.view(-1, 128)
        if self.context:
            feat = x
        x = self.fc(x)
        if self.context:
            return x, feat  # torch.cat((feat1,feat2),1)#F
        else:
            return x


class netD_dc(nn.Module):
    def __init__(self):
        super(netD_dc, self).__init__()
        self.fc1 = nn.Linear(2048, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.fc1(x))), training=self.training)
        x = F.dropout(F.relu(self.bn2(self.fc2(x))), training=self.training)
        x = self.fc3(x)
        return x


class vgg16(_fasterRCNN):
    def __init__(self, classes, pretrained=False, class_agnostic=False, lc=False, gc=False):
        self.model_path = cfg.VGG_PATH
        self.dout_base_model = 512
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        self.lc = lc
        self.gc = gc

        _fasterRCNN.__init__(self, classes, class_agnostic, self.lc, self.gc)

    def _init_modules(self):
        vgg = models.vgg16()
        if self.pretrained:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

        # not using the last maxpool layer
        # print(vgg.features)
        self.RCNN_base1 = nn.Sequential(*list(vgg.features._modules.values())[:14])

        self.RCNN_base2 = nn.Sequential(*list(vgg.features._modules.values())[14:-1])
        # print(self.RCNN_base1)
        # print(self.RCNN_base2)
        self.netD = netD(context=self.gc)
        self.netD_pixel = netD_pixel(context=self.lc)
        self.netCT = net_CSA()
        self.netCS = net_CSA()

        feat_d = 4096
        if self.lc:
            feat_d += 128
        if self.gc:
            feat_d += 128
        # Fix the layers before conv3:
        for layer in range(10):
            for p in self.RCNN_base1[layer].parameters(): p.requires_grad = False

        # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

        self.RCNN_top = vgg.classifier

        self.RCNN_cls_score = nn.Linear(feat_d, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(feat_d, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(feat_d, 4 * self.n_classes)

    def _head_to_tail(self, pool5):

        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5_flat)

        return fc7

