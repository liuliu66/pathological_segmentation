from __future__ import division

import torch
import torch.nn as nn

from mmdet import ops
from mmdet.core import force_fp32
from ..registry import ROI_EXTRACTORS


@ROI_EXTRACTORS.register_module
class LocalActivatedRoIExtractor(nn.Module):
    """Extract RoI features from a single level feature map.

    If there are mulitple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56,
                 context=True,
                 self_attention=True):
        super(LocalActivatedRoIExtractor, self).__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale
        self.fp16_enabled = False

        self.context = context
        self.self_attention = self_attention

        self.conv_f = nn.Conv2d(self.out_channels, self.out_channels//8, kernel_size=1)
        self.conv_g = nn.Conv2d(self.out_channels, self.out_channels//8, kernel_size=1)
        self.conv_h = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    @property
    def num_inputs(self):
        """int: Input feature map levels."""
        return len(self.featmap_strides)

    def init_weights(self):
        pass

    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale: level 0
        - finest_scale <= scale < finest_scale * 2: level 1
        - finest_scale * 2 <= scale < finest_scale * 4: level 2
        - scale >= finest_scale * 4: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1] + 1) * (rois[:, 4] - rois[:, 2] + 1))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def roi_rescale(self, rois, scale_factor):
        cx = (rois[:, 1] + rois[:, 3]) * 0.5
        cy = (rois[:, 2] + rois[:, 4]) * 0.5
        w = rois[:, 3] - rois[:, 1] + 1
        h = rois[:, 4] - rois[:, 2] + 1
        new_w = w * scale_factor
        new_h = h * scale_factor
        x1 = cx - new_w * 0.5 + 0.5
        x2 = cx + new_w * 0.5 - 0.5
        y1 = cy - new_h * 0.5 + 0.5
        y2 = cy + new_h * 0.5 - 0.5
        new_rois = torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)
        return new_rois

    @force_fp32(apply_to=('feats',), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        out_size = self.roi_layers[0].out_size
        num_levels = len(feats)
        target_lvls = self.map_roi_levels(rois, num_levels)
        roi_feats = feats[0].new_zeros(rois.size()[0], self.out_channels,
                                       out_size, out_size)

        w_img = feats[0].shape[2] * self.featmap_strides[0] - 1
        h_img = feats[0].shape[3] * self.featmap_strides[0] - 1

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)
        for i in range(num_levels):
            inds = target_lvls == i
            if inds.any():
                rois_ = rois[inds, :]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] += roi_feats_t

                # add context roi
                if self.context:
                    rois_context = rois_.clone()
                    w = rois_context[:, 3] - rois_context[:, 1] + 1
                    h = rois_context[:, 4] - rois_context[:, 2] + 1
                    rois_context[:, 1] = torch.clamp(rois_context[:, 1] - 0.5 * w, min=0)
                    rois_context[:, 2] = torch.clamp(rois_context[:, 2] - 0.5 * h, min=0)
                    rois_context[:, 3] = torch.clamp(rois_context[:, 3] + 0.5 * w, max=w_img)
                    rois_context[:, 4] = torch.clamp(rois_context[:, 4] + 0.5 * h, max=h_img)
                    roi_feats_t_context = self.roi_layers[i](feats[i], rois_context)
                    roi_feats[inds] += roi_feats_t_context

        if self.self_attention:
            B, C, W, H = roi_feats.size()

            f = self.conv_f(roi_feats).view(B, -1, W * H).permute(0, 2, 1)
            g = self.conv_g(roi_feats).view(B, -1, W * H)  # B X C x (*W*H)
            energy = torch.bmm(f, g)  # transpose check
            attention = self.softmax(energy)  # BX (N) X (N)
            proj_value = self.conv_h(roi_feats).view(B, -1, W * H)  # B X C X N

            out = torch.bmm(proj_value, attention.permute(0, 2, 1))
            out = out.view(B, C, W, H)

            roi_feats = self.gamma * out + roi_feats

        return roi_feats
