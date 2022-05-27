import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from mmcv.cnn import kaiming_init, normal_init

from .grid_head import GridHead
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule


@HEADS.register_module
class SelectiveGridHead(GridHead):

    def __init__(self,
                 grid_points=9,
                 num_convs=8,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 point_feat_channels=64,
                 deconv_kernel_size=4,
                 class_agnostic=False,
                 loss_grid=dict(
                     type='CrossEntropyLoss', use_sigmoid=True,
                     loss_weight=15),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=36)):
        super(SelectiveGridHead, self).__init__()
        self.num_scales = 4
        self.scales = (0.25, 0.125, 0.0625, 0.03125)
        self.alpha = Parameter(torch.FloatTensor(4))
        self.beta = Parameter(torch.FloatTensor(4))
        torch.nn.init.constant_(self.alpha, 1.)
        torch.nn.init.constant_(self.beta, 10.)

    def forward(self, x, **kwargs):
        assert x[0].shape[-1] == x[0].shape[-2] == self.roi_feat_size
        pos_rois = kwargs.get('pos_rois', None)
        scale_factor = kwargs.get('scale_factor', None)
        assert pos_rois is not None
        assert scale_factor is not None
        bboxes = pos_rois[:, 1:]
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
            weights.append((-self.alpha[i] * sign * prod * scale / self.beta[i]).exp())
        sum_weight = sum(weights)
        weights[0] = weights[0] / sum_weight
        weights[1] = weights[1] / sum_weight
        weights[2] = weights[2] / sum_weight
        weights[3] = weights[3] / sum_weight

        # RoI feature transformation, downsample 2x
        result = []
        for level, roi_feat in enumerate(x):
            feature = self.convs(roi_feat)
            # feature_shape = feature.shape
            # feature = feature.view(feature_shape[0], -1)
            feature = feature * weights[level].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            # feature = feature.view(feature_shape)
            result.append(feature)
        x = sum(result)

        c = self.point_feat_channels
        # first-order fusion
        x_fo = [None for _ in range(self.grid_points)]
        for i, points in enumerate(self.neighbor_points):
            x_fo[i] = x[:, i * c:(i + 1) * c]
            for j, point_idx in enumerate(points):
                x_fo[i] = x_fo[i] + self.forder_trans[i][j](
                    x[:, point_idx * c:(point_idx + 1) * c])

        # second-order fusion
        x_so = [None for _ in range(self.grid_points)]
        for i, points in enumerate(self.neighbor_points):
            x_so[i] = x[:, i * c:(i + 1) * c]
            for j, point_idx in enumerate(points):
                x_so[i] = x_so[i] + self.sorder_trans[i][j](x_fo[point_idx])

        # predicted heatmap with fused features
        x2 = torch.cat(x_so, dim=1)
        x2 = self.deconv1(x2)
        x2 = F.relu(self.norm1(x2), inplace=True)
        heatmap = self.deconv2(x2)

        # predicted heatmap with original features (applicable during training)
        if self.training:
            x1 = x
            x1 = self.deconv1(x1)
            x1 = F.relu(self.norm1(x1), inplace=True)
            heatmap_unfused = self.deconv2(x1)
        else:
            heatmap_unfused = heatmap

        return dict(fused=heatmap, unfused=heatmap_unfused)