import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import delta2bbox
from mmdet.ops import nms
from mmdet.ops import DeformConv, ModulatedDeformConv
from .anchor_head import AnchorHead
from ..registry import HEADS


@HEADS.register_module
class DeformRPNHead(AnchorHead):

    def __init__(self, in_channels, **kwargs):
        super(DeformRPNHead, self).__init__(2, in_channels, **kwargs)

    def _init_layers(self):
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        # self.rpn_cls = nn.Conv2d(self.feat_channels,
        # self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

        self.rpn_cls = nn.ModuleList()
        for _ in range(self.num_anchors):
            self.rpn_cls.append(DeformConv(self.feat_channels,
                                           self.cls_out_channels,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,
                                           deformable_groups=1))

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        for m in self.rpn_cls:
            normal_init(m, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        # rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)

        rpn_cls_score = rpn_bbox_pred.new_zeros(rpn_bbox_pred.size()[0],
                                                self.num_anchors * self.cls_out_channels,
                                                rpn_bbox_pred.size()[2],
                                                rpn_bbox_pred.size()[3])
        for i in range(self.num_anchors):
            # with torch.no_grad:
            offset_anchor = rpn_bbox_pred[:, i * 4: (i + 1) * 4, :, :].clone()
            off_x = (offset_anchor[:, 0, :, :] * offset_anchor[:, 2, :, :]).unsqueeze(1)
            off_y = (offset_anchor[:, 1, :, :] * offset_anchor[:, 3, :, :]).unsqueeze(1)
            offset = torch.cat((off_x, off_y), dim=1)
            rpn_cls_score[:, i * self.cls_out_channels: (i + 1) * self.cls_out_channels, :, :] = \
                self.rpn_cls[i](x, offset)

        return rpn_cls_score, rpn_bbox_pred

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        losses = super(DeformRPNHead, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            cfg,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            anchors = mlvl_anchors[idx]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means,
                                   self.target_stds, img_shape)
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
        return proposals
