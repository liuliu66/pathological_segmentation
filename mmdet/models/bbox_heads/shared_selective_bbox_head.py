import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from .bbox_head import BBoxHead
from .convfc_bbox_head import ConvFCBBoxHead
from ..registry import HEADS
from ..utils import ConvModule


@HEADS.register_module
class SharedSelectiveFCBBoxHead(ConvFCBBoxHead):

    def __init__(self, num_fcs=2, fc_out_channels=1024, num_scales=4, *args, **kwargs):
        super(SharedSelectiveFCBBoxHead, self).__init__(
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

        self.alpha = Parameter(torch.FloatTensor(num_scales))
        self.beta = Parameter(torch.FloatTensor(num_scales))

        self.init_weights()

    def init_weights(self):
        super(ConvFCBBoxHead, self).init_weights()
        nn.init.constant_(self.alpha, 1.)
        nn.init.constant_(self.beta, 10.)

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
        avg_width = 51.59 * scale_factor  # for agripest
        avg_height = 61.63 * scale_factor  # for agripest
        # avg_width = 103.89 * scale_factor  # for coco
        # avg_height = 107.42 * scale_factor  # for coco
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
                feature = roi_feat.view(roi_feat.size(0), -1)
                for fc in self.shared_fcs:
                    feature = F.relu(fc(feature))
                # feature = feature * weights[:, level].unsqueeze(1)
                feature = feature * weights[level].unsqueeze(1).exp()
                result.append(feature)
        result = sum(result)

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

