import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import (AnchorGenerator, delta2bbox, anchor_target,
                        anchor_target_coarse, multi_apply)
from mmdet.ops import nms
from .anchor_head import AnchorHead
from ..registry import HEADS
from ..builder import build_loss


@HEADS.register_module
class CoarseRPNHead(AnchorHead):

    def __init__(self, in_channels, **kwargs):
        super(CoarseRPNHead, self).__init__(2, in_channels, **kwargs)
        self.anchor_scales_coarse = kwargs['coarse_anchor_scales']
        self.anchor_ratios_coarse = self.anchor_ratios \
            if kwargs['coarse_anchor_ratios'] is None else kwargs['coarse_anchor_ratios']
        self.anchor_base_sizes_coarse = list(kwargs['coarse_anchor_strides'])
        self.num_anchors_coarse = len(self.anchor_ratios_coarse) * len(self.anchor_scales_coarse)
        self.rpn_cls_coarse = nn.Conv2d(self.feat_channels,
                                        self.num_anchors_coarse * self.cls_out_channels, 1)

        self.anchor_generators_coarse = []
        for anchor_base in self.anchor_base_sizes_coarse:
            self.anchor_generators_coarse.append(
                AnchorGenerator(anchor_base, self.anchor_scales_coarse, self.anchor_ratios_coarse))

        self.loss_cls_coarse = build_loss(kwargs['loss_cls_coarse'])

    def _init_layers(self):
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)
        normal_init(self.rpn_cls_coarse, std=0.01)

    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        rpn_cls_score_coarse = self.rpn_cls_coarse(x)
        return rpn_cls_score, rpn_bbox_pred, rpn_cls_score_coarse

    def get_anchors(self, featmap_sizes, img_metas):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        multi_level_anchors_coarse = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            multi_level_anchors.append(anchors)
            anchors_coarse = self.anchor_generators_coarse[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            multi_level_anchors_coarse.append(anchors_coarse)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        anchor_list_coarse = [multi_level_anchors_coarse for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        valid_flag_list_coarse = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            multi_level_flags_coarse = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
                flag_coarse = self.anchor_generators_coarse[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags_coarse.append(flag_coarse)
            valid_flag_list.append(multi_level_flags)
            valid_flag_list_coarse.append(multi_level_flags_coarse)

        return anchor_list, valid_flag_list, anchor_list_coarse, valid_flag_list_coarse

    def loss_single(self, cls_score, bbox_pred, cls_score_coarse,
                    labels, label_weights, contains_ratio,
                    contains_ratio_weights, bbox_targets, bbox_weights,
                    num_total_samples, num_total_samples_coarse, cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        # coarse classification loss
        # labels_coarse = labels_coarse.reshape(-1)
        # label_weights_coarse = label_weights_coarse.reshape(-1)
        contains_ratio = contains_ratio.reshape(-1, 1)
        contains_ratio_weights = contains_ratio_weights.reshape(-1, 1)
        cls_score_coarse = cls_score_coarse.permute(0, 2, 3,
                                                    1).reshape(-1, self.cls_out_channels)
        loss_cls_coarse = self.loss_cls_coarse(
            cls_score_coarse, contains_ratio, contains_ratio_weights, avg_factor=num_total_samples_coarse)
        return loss_cls, loss_bbox, loss_cls_coarse

    def loss(self,
             cls_scores,
             bbox_preds,
             cls_scores_coarse,
             gt_bboxes,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators) == len(self.anchor_generators_coarse)

        anchor_list, valid_flag_list, anchor_list_coarse, valid_flag_list_coarse = self.get_anchors(
            featmap_sizes, img_metas)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=None,
            label_channels=label_channels,
            sampling=self.sampling)
        cls_reg_targets_coarse = anchor_target_coarse(
            anchor_list_coarse,
            valid_flag_list_coarse,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=None,
            label_channels=label_channels,
            sampling=self.sampling
        )
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # (labels_list_coarse, label_weights_list_coarse,
        # num_total_pos_coarse, num_total_neg_coarse) = cls_reg_targets_coarse

        (contains_ratio, contains_ratio_weights,
         num_total_pos_coarse, num_total_neg_coarse) = cls_reg_targets_coarse
        num_total_samples_coarse = (
            num_total_pos_coarse + num_total_neg_coarse if self.sampling else num_total_pos_coarse)
        losses_cls, losses_bbox, losses_cls_coarse = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            cls_scores_coarse,
            labels_list,
            label_weights_list,
            # labels_list_coarse,
            # label_weights_list_coarse,
            contains_ratio,
            contains_ratio_weights,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            num_total_samples_coarse=num_total_samples_coarse,
            cfg=cfg)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_cls_coarse=losses_cls_coarse)

    def get_bboxes(self, cls_scores, bbox_preds, cls_score_coarse, img_metas, cfg,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(cls_score_coarse)
        num_levels = len(cls_scores)

        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(cls_scores[i].size()[-2:],
                                                   self.anchor_strides[i])
            for i in range(num_levels)
        ]
        mlvl_anchors_coarse = [
            self.anchor_generators_coarse[i].grid_anchors(cls_scores[i].size()[-2:],
                                                          self.anchor_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            cls_score_coarse_list = [
                cls_score_coarse[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list, cls_score_coarse_list,
                                               mlvl_anchors, mlvl_anchors_coarse, img_shape,
                                               scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          cls_scores_coarse,
                          mlvl_anchors,
                          mlvl_anchors_coarse,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        mlvl_proposals = []
        mlvl_proposals_coarse = []
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

            rpn_cls_score_coarse = cls_scores_coarse[idx]
            anchors_coarse = mlvl_anchors_coarse[idx]
            rpn_cls_score_coarse = rpn_cls_score_coarse.permute(1, 2, 0)
            scores_coarse = rpn_cls_score_coarse.reshape(-1)
            _, topk_inds = scores_coarse.topk(100)
            proposals_coarse = anchors_coarse[topk_inds, :]
            scores_coarse = scores_coarse[topk_inds]
            proposals_coarse = proposals_coarse.clamp(min=0, max=img_shape[1] - 1)
            proposals_coarse = torch.cat([proposals_coarse, scores_coarse.unsqueeze(-1)], dim=-1)
            proposals_coarse, _ = nms(proposals_coarse, 0.5)
            mlvl_proposals_coarse.append(proposals_coarse)

        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]

        proposals_coarse = torch.cat(mlvl_proposals_coarse, 0)
        if cfg.nms_across_levels:
            proposals, _ = nms(proposals_coarse, 0.5)
            # proposals = proposals[:cfg.max_num, :]
        else:
            scores_coarse = proposals_coarse[:, 4]
            num = min(cfg.max_num, proposals_coarse.shape[0])
            _, topk_inds = scores_coarse.topk(num)
            proposals_coarse = proposals_coarse[topk_inds, :]
        return proposals, proposals_coarse

    @staticmethod
    def extract_coarse_data(img, img_metas, proposal_list_coarse, gt_labels, gt_bboxes, mode='train'):
        max_num = 3

        _, C, H, W = img.size()
        img_meta = img_metas[0]
        gt_label = gt_labels[0]
        gt_bbox = gt_bboxes[0]
        # proposals_coarse = proposal_list_coarse[:1, :]
        proposals_coarse = proposal_list_coarse
        # coarse_imgs = img.new_zeros((max_num, C, H, W))
        coarse_imgs = []
        coarse_gt_labels = []
        coarse_gt_bboxes = []
        coarse_scale_factors = []
        count = 0
        for proposal_coarse in proposals_coarse:
            if mode == 'train':
                gt_label = gt_labels[0]
                gt_bbox = gt_bboxes[0]

                lt = torch.ge(gt_bbox[:, None, 0], proposal_coarse[0]) * torch.ge(gt_bbox[:, None, 1], proposal_coarse[1])
                rb = torch.le(gt_bbox[:, None, 2], proposal_coarse[2]) * torch.le(gt_bbox[:, None, 3], proposal_coarse[3])
                object_contains = (lt * rb)[:, 0]
                gt_label_contains = gt_label[object_contains]
                gt_bbox_contains = gt_bbox[object_contains]

                if len(gt_label_contains) is not 0:
                    x1, y1, x2, y2 = proposal_coarse[:4].long()
                    new_img = img[:, :, y1:y2, x1:x2]
                    coarse_imgs.append(F.interpolate(new_img, size=(H, W), mode='nearest'))
                    scale_factor = [coarse_imgs[count].shape[3] / new_img.shape[3],
                                    coarse_imgs[count].shape[2] / new_img.shape[2]]
                    img_meta_copy = copy.copy(img_meta)
                    img_meta_copy['scale_factor'] = scale_factor
                    coarse_scale_factors.append(img_meta_copy)

                    new_gt_label = gt_label_contains
                    new_gt_bbox = gt_bbox_contains - torch.cat((proposal_coarse[:2], proposal_coarse[:2]))

                    new_gt_bbox[:, 0] *= coarse_scale_factors[count]['scale_factor'][0]
                    new_gt_bbox[:, 1] *= coarse_scale_factors[count]['scale_factor'][1]
                    new_gt_bbox[:, 2] *= coarse_scale_factors[count]['scale_factor'][0]
                    new_gt_bbox[:, 3] *= coarse_scale_factors[count]['scale_factor'][1]

                    coarse_gt_labels.append(new_gt_label)
                    coarse_gt_bboxes.append(new_gt_bbox)

                    count += 1
                if count == max_num:
                    break
            else:
                x1, y1, x2, y2 = proposal_coarse[:4].long()
                new_img = img[:, :, y1:y2, x1:x2]
                coarse_imgs.append(F.interpolate(new_img, size=(H, W), mode='nearest'))
                img_meta_copy = copy.copy(img_meta)
                img_meta_copy['scale_factor'] = scale_factor
                coarse_scale_factors.append(img_meta_copy)

                count += 1

                if count == max_num:
                    break

        coarse_imgs = torch.cat(coarse_imgs)
        return coarse_imgs, coarse_gt_labels, coarse_gt_bboxes, coarse_scale_factors

    @staticmethod
    def cat_coarse_data(x, x_coarse,
                        gt_labels, coarse_labels,
                        gt_bboxes, coarse_bboxes,
                        img_metas, coarse_scale_factors):

        for coarse_label, coarse_bbox, coarse_scale_factor in \
                zip(coarse_labels, coarse_bboxes, coarse_scale_factors):

            if coarse_label is not None:
                gt_labels.extend(coarse_label)
                gt_bboxes.extend(coarse_bbox)
                img_metas.extend({'scale_factor': coarse_scale_factor})
                for l in range(len(x)):
                    x[l] = torch.cat((x[l], x_coarse[l]))

        return x, gt_labels, gt_bboxes, img_metas
