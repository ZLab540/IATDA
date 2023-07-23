import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np

from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer

import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        self.lower_margin = 0.5
        self.margin = 1

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

        self.vis_feature = None
        self.tgt_vis_feature = None
        self.mask = None

    def forward(self, im_data, im_info, gt_boxes, num_boxes, tgt_im_data, tgt_im_info, tgt_gt_boxes, tgt_num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        tgt_im_info = tgt_im_info.data
        tgt_gt_boxes = tgt_gt_boxes.data
        tgt_num_boxes = tgt_num_boxes.data
        if not self.training:
            self.netC_S = self.netC_T
        # feed image data to base model to obtain base feature map
        # base_feat = self.RCNN_base(im_data)
        # tgt_base_feat = self.RCNN_base(tgt_im_data)
        # feed image data to base model to obtain base feature map
        base_feat1 = self.RCNN_base1(im_data)  # l1 1 256 150 187
        mask, c = self.netC_S(base_feat1)
        base_feat1 = (1 - mask) * base_feat1
        # base_feat1 = mask * base_feat1
        base_feat = self.RCNN_base2(base_feat1)  # l2 512 75 94
        self.vis_feature = base_feat

        tgt_base_feat1 = self.RCNN_base1(tgt_im_data)
        tgt_mask, tgt_c = self.netC_T(tgt_base_feat1)
        # tgt_base_feat1 = tgt_mask * tgt_base_feat1
        tgt_base_feat1 = (1 - tgt_mask) * tgt_base_feat1
        tgt_base_feat = self.RCNN_base2(tgt_base_feat1)
        self.tgt_vis_feature = tgt_base_feat
        self.mask = tgt_mask

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            # feed base feature map tp RPN to obtain rois
            rois, rpn_loss_cls, rpn_loss_bbox, rpn_cls_prob = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
            tgt_rois, tgt_rpn_loss_cls, tgt_rpn_loss_bbox, tgt_rpn_cls_prob = self.RCNN_rpn(tgt_base_feat,
                                                                            tgt_im_info, tgt_gt_boxes, tgt_num_boxes)

            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))

            tgt_roi_data = self.RCNN_proposal_target(tgt_rois, tgt_gt_boxes, tgt_num_boxes)
            tgt_rois, tgt_rois_label, tgt_rois_target, tgt_rois_inside_ws, tgt_rois_outside_ws = tgt_roi_data

            tgt_rois_label = Variable(tgt_rois_label.view(-1).long())
            tgt_rois_target = Variable(tgt_rois_target.view(-1, tgt_rois_target.size(2)))
            tgt_rois_inside_ws = Variable(tgt_rois_inside_ws.view(-1, tgt_rois_inside_ws.size(2)))
            tgt_rois_outside_ws = Variable(tgt_rois_outside_ws.view(-1, tgt_rois_outside_ws.size(2)))
        else:

            # feed base feature map tp RPN to obtain rois
            rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
            tgt_rois, tgt_rpn_loss_cls, tgt_rpn_loss_bbox = self.RCNN_rpn(tgt_base_feat, tgt_im_info,
                                                                                            tgt_gt_boxes, tgt_num_boxes)

            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

            tgt_rois_label = None
            tgt_rois_target = None
            tgt_rois_inside_ws = None
            tgt_rois_outside_ws = None
            tgt_rpn_loss_cls = 0
            tgt_rpn_loss_bbox = 0

        rois = Variable(rois)
        tgt_rois = Variable(tgt_rois)

        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == 'crop':
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)

            tgt_grid_xy = _affine_grid_gen(tgt_rois.view(-1, 5), tgt_base_feat.size()[2:], self.grid_size)
            tgt_grid_yx = torch.stack([tgt_grid_xy.data[:, :, :, 1], tgt_grid_xy.data[:, :, :, 0]], 3).contiguous()
            tgt_pooled_feat = self.RCNN_roi_crop(tgt_base_feat, Variable(tgt_grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                tgt_pooled_feat = F.max_pool2d(tgt_pooled_feat, 2, 2)

        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
            tgt_pooled_feat = self.RCNN_roi_align(tgt_base_feat, tgt_rois.view(-1, 5))

        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))
            tgt_pooled_feat = self.RCNN_roi_pool(tgt_base_feat, tgt_rois.view(-1, 5))


        # get the adaptive feature for RPN
        if self.training:
            rpn_adapt_feat = self.rpn_adapt_feat(pooled_feat.view(pooled_feat.size(0), -1))
            tgt_rpn_adapt_feat = self.rpn_adapt_feat(tgt_pooled_feat.view(tgt_pooled_feat.size(0), -1))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)
        tgt_pooled_feat = self._head_to_tail(tgt_pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        tgt_bbox_pred = self.RCNN_bbox_pred(tgt_pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

            tgt_bbox_pred_view = tgt_bbox_pred.view(tgt_bbox_pred.size(0), int(tgt_bbox_pred.size(1) / 4), 4)
            tgt_bbox_pred_select = torch.gather(tgt_bbox_pred_view, 1,
                                                tgt_rois_label.view(tgt_rois_label.size(0), 1, 1).expand(
                                                    tgt_rois_label.size(0), 1, 4))
            tgt_bbox_pred = tgt_bbox_pred_select.squeeze(1)

        # compute object classification probability
        adapt_feat = self.RCNN_adapt_feat(pooled_feat)
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        tgt_adapt_feat = self.RCNN_adapt_feat(tgt_pooled_feat)
        tgt_cls_score = self.RCNN_cls_score(tgt_pooled_feat)
        tgt_cls_prob = F.softmax(tgt_cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        tgt_RCNN_loss_cls = 0
        tgt_RCNN_loss_bbox = 0
        RCNN_loss_tri = 0
        RPN_loss_tri = 0
        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            tgt_RCNN_loss_cls = F.cross_entropy(tgt_cls_score, tgt_rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
            tgt_RCNN_loss_bbox = _smooth_l1_loss(tgt_bbox_pred, tgt_rois_target, tgt_rois_inside_ws,
                                                 tgt_rois_outside_ws)

            RCNN_loss_tri = self.rpc_triplet_loss(adapt_feat, cls_prob, tgt_adapt_feat,
                                                                  tgt_cls_prob, batch_size)
            RPN_loss_tri = self.fb_adaptive_loss(rpn_adapt_feat, rpn_cls_prob, tgt_rpn_adapt_feat,
                                                                tgt_rpn_cls_prob, batch_size)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        tgt_cls_prob = tgt_cls_prob.view(batch_size, tgt_rois.size(1), -1)
        tgt_bbox_pred = tgt_bbox_pred.view(batch_size, tgt_rois.size(1), -1)

        if self.training:
        	return rois, tgt_rois, cls_prob, tgt_cls_prob, bbox_pred, tgt_bbox_pred, rpn_loss_cls.view(-1), tgt_rpn_loss_cls.view(-1), \
        		rpn_loss_bbox.view(-1), tgt_rpn_loss_bbox.view(-1), RCNN_loss_cls.view(-1), tgt_RCNN_loss_cls.view(-1), RCNN_loss_bbox.view(-1), \
        		tgt_RCNN_loss_bbox.view(-1), RCNN_loss_tri.view(-1), rois_label, tgt_rois_label, \
                RPN_loss_tri.view(-1), c, tgt_c
        else:
        	return rois, tgt_rois, cls_prob, tgt_cls_prob, bbox_pred, tgt_bbox_pred, rpn_loss_cls, tgt_rpn_loss_cls, rpn_loss_bbox, \
        		tgt_rpn_loss_bbox, RCNN_loss_cls, tgt_RCNN_loss_cls, RCNN_loss_bbox, tgt_RCNN_loss_bbox, \
        		0 , rois_label, tgt_rois_label, 0

    def triplet_loss(self, anchor, positive, negative, margin = 0.3):
        pos_dist = self.distance(anchor, positive)
        neg_dist = self.distance(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + margin)
        return loss.mean()

    def fb_adaptive_loss(self, pooled_feat, cls_prob, tgt_pooled_feat, tgt_cls_prob, batch_size, epsilon=1e-6):
        # get the feature embedding of every class for source and target domains
        pooled_feat = pooled_feat.view(batch_size, pooled_feat.size(0) // batch_size, pooled_feat.size(1))
        cls_prob = cls_prob.view(batch_size, cls_prob.size(0) // batch_size, cls_prob.size(1))
        tgt_pooled_feat = tgt_pooled_feat.view(batch_size, tgt_pooled_feat.size(0) // batch_size,
                                               tgt_pooled_feat.size(1))
        tgt_cls_prob = tgt_cls_prob.view(batch_size, tgt_cls_prob.size(0) // batch_size, tgt_cls_prob.size(1))

        num_classes = cls_prob.size(2)
        class_feat = list()
        tgt_class_feat = list()

        for i in range(num_classes):
            tmp_cls_prob = cls_prob[:, :, i].view(cls_prob.size(0), cls_prob.size(1), 1)
            tmp_class_feat = pooled_feat * tmp_cls_prob
            tmp_class_feat = torch.sum(torch.sum(tmp_class_feat, dim=1), dim=0) / (torch.sum(tmp_cls_prob) + epsilon)
            class_feat.append(tmp_class_feat)

            tmp_tgt_cls_prob = tgt_cls_prob[:, :, i].view(tgt_cls_prob.size(0), tgt_cls_prob.size(1), 1)
            tmp_tgt_class_feat = tgt_pooled_feat * tmp_tgt_cls_prob
            tmp_tgt_class_feat = torch.sum(torch.sum(tmp_tgt_class_feat, dim=1), dim=0) / (
                        torch.sum(tmp_tgt_cls_prob) + epsilon)
            tgt_class_feat.append(tmp_tgt_class_feat)

        class_feat = torch.stack(class_feat, dim=0)
        tgt_class_feat = torch.stack(tgt_class_feat, dim=0)

        tri_loss = 0
        tri_loss = self.triplet_loss(class_feat[0], tgt_class_feat[0], class_feat[1])
        tri_loss += self.triplet_loss(class_feat[1], tgt_class_feat[1], class_feat[0])
        tri_loss += self.triplet_loss(tgt_class_feat[0], class_feat[0], tgt_class_feat[1])
        tri_loss += self.triplet_loss(tgt_class_feat[1], class_feat[1], tgt_class_feat[0])

        # dist_loss = 0
        # for i in range(class_feat.size(0)):
        #     dist_loss += self.distance(class_feat[i], tgt_class_feat[i])
        #
        # return dist_loss / class_feat.size(0) + tri_loss / 4
        return tri_loss / 4

    def find_max_dist_index(self, i, list):
        max_dist_index = 0
        max_dist = 0
        for j in range(list.size(0)):
            if i == j:
                continue
            tmp_dist = self.distance(list[i], list[j])
            if tmp_dist > max_dist:
                max_dist_index = j
                max_dist = tmp_dist
        return max_dist_index

    def find_min_dist_index(self, i, list):
        min_dist_index = 0
        min_dist = 0
        for j in range(list.size(0)):
            if i == j:
                continue
            tmp_dist = self.distance(list[i], list[j])
            if j == 0:
                min_dist_index = j
                min_dist = tmp_dist
            elif tmp_dist < min_dist:
                min_dist_index = j
                min_dist = tmp_dist
        return min_dist_index

    def rpc_triplet_loss(self, pooled_feat, cls_prob, tgt_pooled_feat, tgt_cls_prob, batch_size, epsilon=1e-6):
        pooled_feat = pooled_feat.view(batch_size, pooled_feat.size(0) // batch_size, pooled_feat.size(1))
        cls_prob = cls_prob.view(batch_size, cls_prob.size(0) // batch_size, cls_prob.size(1))
        tgt_pooled_feat = tgt_pooled_feat.view(batch_size, tgt_pooled_feat.size(0) // batch_size,
                                               tgt_pooled_feat.size(1))
        tgt_cls_prob = tgt_cls_prob.view(batch_size, tgt_cls_prob.size(0) // batch_size, tgt_cls_prob.size(1))

        num_classes = cls_prob.size(2)
        class_feat = list()
        tgt_class_feat = list()

        for i in range(num_classes):
            tmp_cls_prob = cls_prob[:, :, i].view(cls_prob.size(0), cls_prob.size(1), 1)
            tmp_class_feat = pooled_feat * tmp_cls_prob
            tmp_class_feat = torch.sum(torch.sum(tmp_class_feat, dim=1), dim=0) / (torch.sum(tmp_cls_prob) + epsilon)
            class_feat.append(tmp_class_feat)

            tmp_tgt_cls_prob = tgt_cls_prob[:, :, i].view(tgt_cls_prob.size(0), tgt_cls_prob.size(1), 1)
            tmp_tgt_class_feat = tgt_pooled_feat * tmp_tgt_cls_prob
            tmp_tgt_class_feat = torch.sum(torch.sum(tmp_tgt_class_feat, dim=1), dim=0) / (
                    torch.sum(tmp_tgt_cls_prob) + epsilon)
            tgt_class_feat.append(tmp_tgt_class_feat)

        class_feat = torch.stack(class_feat, dim=0)
        tgt_class_feat = torch.stack(tgt_class_feat, dim=0)
        tri_loss = 0
        for i in range(1, class_feat.size(0)):  # 不需要背景类的对齐
            index = self.find_min_dist_index(i, class_feat)
            tgt_index = self.find_min_dist_index(i, tgt_class_feat)

            tri_loss += self.triplet_loss(class_feat[i], tgt_class_feat[i], class_feat[index])
            tri_loss += self.triplet_loss(tgt_class_feat[i], class_feat[i], tgt_class_feat[tgt_index])

        # dist_loss = 0
        # for i in class_feat.size(0):
        #     dist_loss += self.distance(class_feat[i], tgt_class_feat[i])
        #
        # return tri_loss / (2 * (class_feat.size(0) - 1)) + dist_loss / class_feat.size(0)

        return tri_loss / (2 * (class_feat.size(0) - 1))

    def distance(self, src_feat, tgt_feat):

        output = torch.pow(src_feat - tgt_feat, 2.0).mean()
        return output

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_adapt_feat, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.rpn_adapt_feat, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
