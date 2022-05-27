import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from mmdet.core import (delta2bbox, multiclass_nms, bbox_target, force_fp32,
                        auto_fp16)

from ..losses import accuracy
from .convfc_bbox_head import ConvFCBBoxHead
from ..registry import HEADS


@HEADS.register_module
class SharedMultiSelectiveFCBBoxHead(ConvFCBBoxHead):

    def __init__(self, num_fcs=2, fc_out_channels=1024, num_scales=4, *args, **kwargs):
        super(SharedMultiSelectiveFCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

        self.num_scales = num_scales
        self.scales = (0.25, 0.125, 0.0625, 0.03125)
        self.fc6_list = []
        self.fc7_list = []
        in_channels = kwargs['in_channels'] * kwargs['roi_feat_size'] ** 2

        if self.with_cls:
            self.fc_cls = nn.Linear(self.fc_out_channels, self.num_classes)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                                                             self.num_classes)
            self.fc_reg = nn.Linear(self.fc_out_channels, out_dim_reg)

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        self.alpha = Parameter(torch.FloatTensor(num_scales, 3))
        self.beta = Parameter(torch.FloatTensor(num_scales, 3))
        self.gamma = Parameter(torch.FloatTensor(num_scales, 3))

        self.init_weights()

    def init_weights(self):
        super(ConvFCBBoxHead, self).init_weights()
        nn.init.constant_(self.alpha, 1.)
        nn.init.constant_(self.beta, 10.)
        nn.init.constant_(self.gamma, 1.)
        for module_list in [self.shared_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    '''
    def forward(self, x, **kwargs):
        rois = kwargs.get('rois', None)
        scale_factor = kwargs.get('scale_factor', None)
        assert rois is not None
        assert scale_factor is not None
        bboxes = rois[:, 1:].clone()
        avg_width = 103.89 * scale_factor
        avg_height = 107.42 * scale_factor
        weights = []

        widths = bboxes[:, 2] - bboxes[:, 0] + 1
        heights = bboxes[:, 3] - bboxes[:, 1] + 1
        # print(self.alpha, self.beta)

        for i, scale in enumerate(self.scales):
            heights_diff = heights - avg_height
            widths_diff = widths - avg_width
            sign_heights = torch.sign(heights_diff)
            sign_widths = torch.sign(widths_diff)
            sign = torch.sign(sign_heights + sign_widths - 1)
            prod = torch.sqrt((heights_diff * widths_diff).abs())
            # weights_scale = -self.alpha[i] * sign * prod * scale / self.beta[i]
            weights_scale = (-self.alpha[i] * sign * prod * scale / self.beta[i]).exp()
            weights.append(weights_scale)
        # weights_cat = torch.cat([weight.unsqueeze(1) for weight in weights], dim=1)
        # weights = F.softmax(weights_cat, dim=1)
        sum_weight = sum(weights)
        weights[0] = weights[0] / sum_weight
        weights[1] = weights[1] / sum_weight
        weights[2] = weights[2] / sum_weight
        weights[3] = weights[3] / sum_weight

        # shared part
        result = []
        if self.num_shared_convs > 0:
            for roi_feat, conv in zip(x, self.shared_convs):
                result.append(conv(roi_feat))

        if self.num_shared_fcs > 0:
            #if self.with_avg_pool:
            #    x = self.avg_pool(x)
            #x = x.view(x.size(0), -1)
            for level, roi_feat in enumerate(x):
                result.append(roi_feat * weights[level].view(weights[level].size(0), 1, 1, 1))
            result = sum(result)
            result = result.view(result.size(0), -1)
            for fc in self.shared_fcs:
                result = F.relu(fc(result))

        # separate branches
        x_cls = result
        x_reg = result

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred

    '''

    def forward(self, x, **kwargs):
        rois = kwargs.get('rois', None)
        scale_factor = kwargs.get('scale_factor', None)
        assert rois is not None
        assert scale_factor is not None
        bboxes = rois[:, 1:].clone()
        avg_width = torch.tensor([64., 128., 256.]).to(rois.device)
        avg_height = torch.tensor([64., 128., 256.]).to(rois.device)

        # avg_width = torch.tensor([59.65, 121.20, 356.61]).to(rois.device)  # k-means together for coco
        # avg_height = torch.tensor([57.96, 364.36, 155.75]).to(rois.device)

        # avg_width = torch.tensor([43.17, 197.66, 481.94]).to(rois.device)  # k-means separate for coco
        # avg_height = torch.tensor([43.54, 177.23, 388.02]).to(rois.device)

        # avg_width = torch.tensor([41.72, 201.77, 576.98]).to(rois.device) # k-means together for AgriPest
        # avg_height = torch.tensor([41.71, 202.15, 576.92]).to(rois.device)
        # avg_width = 51.59 * scale_factor  # for agripest
        # avg_height = 61.63 * scale_factor  # for agripest
        # avg_width = 103.89 * scale_factor  # for coco
        # avg_height = 107.42 * scale_factor  # for coco
        weights = []

        widths = bboxes[:, 2] - bboxes[:, 0] + 1
        widths = widths.expand(3, -1).permute(1, 0)
        heights = bboxes[:, 3] - bboxes[:, 1] + 1
        heights = heights.expand(3, -1).permute(1, 0)
        # print(self.alpha, self.beta)

        for i, scale in enumerate(self.scales):
            heights_diff = heights - avg_height
            widths_diff = widths - avg_width
            sign_heights = torch.sign(heights_diff)
            sign_widths = torch.sign(widths_diff)
            sign = torch.sign(sign_heights + sign_widths - 1)
            # sign = torch.sign(sign_heights + sign_widths + 1)
            prod = torch.sqrt((heights_diff * widths_diff).abs())
            # weights_scale = -self.alpha[i] * sign * prod * scale / self.beta[i]
            weights_scale = (-self.alpha[i] * sign * prod * scale / self.beta[i])
            weights.append(weights_scale)
        weights_cat = torch.cat([weight.unsqueeze(1) for weight in weights], dim=1)
        weights = F.softmax(weights_cat, dim=1)
        # sum_weight = sum(weights)
        # weights[0] = weights[0] / sum_weight
        # weights[1] = weights[1] / sum_weight
        # weights[2] = weights[2] / sum_weight
        # weights[3] = weights[3] / sum_weight

        # shared part
        result = []
        if self.num_shared_convs > 0:
            for roi_feat, conv in zip(x, self.shared_convs):
                result.append(conv(roi_feat))

        if self.num_shared_fcs > 0:
            # if self.with_avg_pool:
            #    x = self.avg_pool(x)
            # x = x.view(x.size(0), -1)
            result1 = []
            result2 = []
            result3 = []
            for level, roi_feat in enumerate(x):
                feature1 = roi_feat.view(roi_feat.size(0), -1)
                for fc in self.shared_fcs:
                    feature1 = F.relu(fc(feature1))
                feature1 = feature1 * weights[:, level, 0].unsqueeze(1)
                # feature1 = feature1 * weights[level][:, 0].unsqueeze(1)
                result1.append(feature1)

                feature2 = roi_feat.view(roi_feat.size(0), -1)
                for fc in self.shared_fcs:
                    feature2 = F.relu(fc(feature2))
                feature2 = feature2 * weights[:, level, 1].unsqueeze(1)
                # feature2 = feature2 * weights[level][:, 1].unsqueeze(1)
                result2.append(feature2)

                feature3 = roi_feat.view(roi_feat.size(0), -1)
                for fc in self.shared_fcs:
                    feature3 = F.relu(fc(feature3))
                feature3 = feature3 * weights[:, level, 2].unsqueeze(1)
                # feature3 = feature3 * weights[level][:, 2].unsqueeze(1)
                result3.append(feature3)

        result = sum(result1) + sum(result2) + sum(result3)
        # result = torch.cat((sum(result1), sum(result2), sum(result3)))

        # separate branches
        x_cls = result
        x_reg = result

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        # cls_score = (self.fc_cls(x) if self.with_cls else None for x in x_cls)
        # bbox_pred = (self.fc_reg(x) if self.with_reg else None for x in x_reg)
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        # labels = labels.repeat(3)
        # label_weights = label_weights.repeat(3)
        # bbox_targets = bbox_targets.repeat(3, 1)
        # bbox_weights = bbox_weights.repeat(3, 1)
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            losses['loss_cls'] = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               4)[pos_inds, labels[pos_inds]]
            losses['loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       bbox_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        # rois = rois.repeat(3, 1)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        if rescale:
            bboxes /= scale_factor

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels
