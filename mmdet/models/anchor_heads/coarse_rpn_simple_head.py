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

def impad(img, shape, pad_val=0):
    """Pad an image to a certain shape.

    Args:
        img (ndarray): Image to be padded.
        shape (tuple): Expected padding shape.
        pad_val (number or sequence): Values to be filled in padding areas.

    Returns:
        ndarray: The padded image.
    """
    if not isinstance(pad_val, (int, float)):
        assert len(pad_val) == img.shape[-1]
    if len(shape) < len(img.shape):
        shape = shape + (img.shape[-1], )
    assert len(shape) == len(img.shape)
    for i in range(len(shape) - 1):
        assert shape[i] >= img.shape[i]
    pad = torch.empty(shape, dtype=img.dtype).to(img.device)
    pad[...] = pad_val
    pad[:img.shape[0], :img.shape[1], :img.shape[2]] = img
    return pad


def impad_to_multiple(img, divisor, pad_val=0):
    """Pad an image to ensure each edge to be multiple to some number.

    Args:
        img (ndarray): Image to be padded.
        divisor (int): Padded image edges will be multiple to divisor.
        pad_val (number or sequence): Same as :func:`impad`.

    Returns:
        ndarray: The padded image.
    """
    pad_h = int(np.ceil(img.shape[1] / divisor)) * divisor
    pad_w = int(np.ceil(img.shape[2] / divisor)) * divisor
    return impad(img, (img.shape[0], pad_h, pad_w), pad_val)


@HEADS.register_module
class CoarseRPNSimpleHead(AnchorHead):

    def __init__(self, in_channels, **kwargs):
        super(CoarseRPNSimpleHead, self).__init__(2, in_channels, **kwargs)
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

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls_coarse, std=0.01)

    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score_coarse = self.rpn_cls_coarse(x)
        return [rpn_cls_score_coarse]

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
            anchors_coarse = self.anchor_generators_coarse[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            multi_level_anchors_coarse.append(anchors_coarse)
        anchor_list_coarse = [multi_level_anchors_coarse for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list_coarse = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags_coarse = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flag_coarse = self.anchor_generators_coarse[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags_coarse.append(flag_coarse)

            valid_flag_list_coarse.append(multi_level_flags_coarse)

        return anchor_list_coarse, valid_flag_list_coarse

    def loss_single(self, cls_score_coarse,
                    contains_ratio,
                    contains_ratio_weights,
                    num_total_samples_coarse, cfg):
        # coarse classification loss
        # labels_coarse = labels_coarse.reshape(-1)
        # label_weights_coarse = label_weights_coarse.reshape(-1)
        contains_ratio = contains_ratio.reshape(-1, 1)
        contains_ratio_weights = contains_ratio_weights.reshape(-1, 1)
        cls_score_coarse = cls_score_coarse.permute(0, 2, 3, 1
                                                    ).reshape(-1, self.cls_out_channels)
        loss_cls_coarse = self.loss_cls_coarse(
            cls_score_coarse, contains_ratio, contains_ratio_weights, avg_factor=num_total_samples_coarse)
        return [loss_cls_coarse]

    def loss(self,
             cls_scores_coarse,
             gt_bboxes,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores_coarse]
        # assert len(featmap_sizes) == len(self.anchor_generators) == len(self.anchor_generators_coarse)

        anchor_list_coarse, valid_flag_list_coarse = self.get_anchors(featmap_sizes, img_metas)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

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

        # (labels_list_coarse, label_weights_list_coarse,
        # num_total_pos_coarse, num_total_neg_coarse) = cls_reg_targets_coarse

        (contains_ratio, contains_ratio_weights,
         num_total_pos_coarse, num_total_neg_coarse) = cls_reg_targets_coarse
        num_total_samples_coarse = (
            num_total_pos_coarse + num_total_neg_coarse if self.sampling else num_total_pos_coarse)
        losses_cls_coarse = multi_apply(
            self.loss_single,
            cls_scores_coarse,
            contains_ratio,
            contains_ratio_weights,
            num_total_samples_coarse=num_total_samples_coarse,
            cfg=cfg)
        return dict(loss_rpn_cls_coarse=losses_cls_coarse[0])

    def get_bboxes(self, cls_score_coarse, img_metas, cfg,
                   rescale=False):
        num_levels = len(cls_score_coarse)

        mlvl_anchors_coarse = [
            self.anchor_generators_coarse[i].grid_anchors(cls_score_coarse[i].size()[-2:],
                                                          self.anchor_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_coarse_list = [
                cls_score_coarse[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_coarse_list,
                                               mlvl_anchors_coarse, img_shape,
                                               scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores_coarse,
                          mlvl_anchors_coarse,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        mlvl_proposals_coarse = []
        for idx in range(len(cls_scores_coarse)):
            rpn_cls_score_coarse = cls_scores_coarse[idx]
            anchors_coarse = mlvl_anchors_coarse[idx]
            rpn_cls_score_coarse = rpn_cls_score_coarse.permute(1, 2, 0)
            scores_coarse = rpn_cls_score_coarse.reshape(-1)
            _, topk_inds = scores_coarse.topk(100)
            proposals_coarse = anchors_coarse[topk_inds, :]
            scores_coarse = scores_coarse[topk_inds]

            valid = proposals_coarse[:, 2] - proposals_coarse[:, 0] == \
                    proposals_coarse[:, 3] - proposals_coarse[:, 1]
            valid2 = valid & \
                           (proposals_coarse[:, 0] >= 0) & \
                           (proposals_coarse[:, 1] >= 0) & \
                           (proposals_coarse[:, 2] < img_shape[1]) & \
                           (proposals_coarse[:, 3] < img_shape[1])

            #proposals_coarse = proposals_coarse.clamp(min=0, max=img_shape[1] - 1)
            proposals_coarse = proposals_coarse[valid2]
            scores_coarse = scores_coarse[valid2]
            proposals_coarse = torch.cat([proposals_coarse, scores_coarse.unsqueeze(-1)], dim=-1)
            proposals_coarse, _ = nms(proposals_coarse, 0.3)
            mlvl_proposals_coarse.append(proposals_coarse)

        proposals_coarse = torch.cat(mlvl_proposals_coarse, 0)
        if cfg.nms_across_levels:
            proposals_coarse, _ = nms(proposals_coarse, 0.3)
            # proposals = proposals[:cfg.max_num, :]
        else:
            scores_coarse = proposals_coarse[:, 4]
            num = min(cfg.max_num, proposals_coarse.shape[0])
            _, topk_inds = scores_coarse.topk(num)
            proposals_coarse = proposals_coarse[topk_inds, :]
        return [_, proposals_coarse]

    @staticmethod
    def extract_coarse_data(img, img_metas, proposal_list_coarse, gt_labels, gt_bboxes,
                            max_num=3, mode='train'):
        n = min(proposal_list_coarse.size(0), max_num)

        _, C, H, W = img.size()
        img_meta = img_metas[0]
        proposals_coarse = proposal_list_coarse.detach()
        # proposals_coarse = proposal_list_coarse[:n, :]
        # coarse_imgs = img.new_zeros((n, C, H, W))
        coarse_imgs = []
        coarse_gt_labels = []
        coarse_gt_bboxes = []
        coarse_scale_factors = []
        count = 0
        for proposal_coarse in proposals_coarse:
            if mode == 'train':
                if count == n:
                    break
                x1, y1, x2, y2 = proposal_coarse[:4].long()
                if x1 == x2 or y1 == y2:
                    continue
                new_img = img[:, :, y1:y2, x1:x2]
                # print(new_img.shape)
                scale_factor = 800 / new_img.shape[2]
                # scale_factor = [coarse_imgs[count].shape[2] / new_img.shape[3],
                                # coarse_imgs[count].shape[1] / new_img.shape[2]]
                coarse_imgs.append(F.interpolate(new_img, scale_factor=scale_factor, mode='bilinear'))
                # coarse_imgs[count, :, :, :] = F.interpolate(new_img, scale_factor=scale_factor, mode='bilinear')

                img_meta_copy = copy.copy(img_meta)
                img_meta_copy['scale_factor'] = scale_factor
                coarse_scale_factors.append(img_meta_copy)

                gt_label = gt_labels[0]
                gt_bbox = gt_bboxes[0]

                lt = torch.ge(gt_bbox[:, None, 0], proposal_coarse[0]) * torch.ge(gt_bbox[:, None, 1], proposal_coarse[1])
                rb = torch.le(gt_bbox[:, None, 2], proposal_coarse[2]) * torch.le(gt_bbox[:, None, 3], proposal_coarse[3])
                object_contains = (lt * rb)[:, 0]
                gt_label_contains = gt_label[object_contains]
                gt_bbox_contains = gt_bbox[object_contains]

                if len(gt_label_contains) is not 0:
                    new_gt_label = gt_label_contains
                    new_gt_bbox = gt_bbox_contains - torch.cat((proposal_coarse[:2], proposal_coarse[:2]))

                    new_gt_bbox[:, 0] *= coarse_scale_factors[count]['scale_factor']
                    new_gt_bbox[:, 1] *= coarse_scale_factors[count]['scale_factor']
                    new_gt_bbox[:, 2] *= coarse_scale_factors[count]['scale_factor']
                    new_gt_bbox[:, 3] *= coarse_scale_factors[count]['scale_factor']

                    coarse_gt_labels.append(new_gt_label)
                    coarse_gt_bboxes.append(new_gt_bbox)
                else:
                    coarse_gt_labels.append(torch.Tensor([]).long().to(gt_labels[0].device))
                    coarse_gt_bboxes.append(torch.Tensor([]).to(gt_bboxes[0].device))

                count += 1

            else:
                x1, y1, x2, y2 = proposal_coarse[:4].long()
                new_img = img[:, :, y1:y2, x1:x2]
                coarse_imgs[count, :, :, :] = F.interpolate(new_img, size=(H, W), mode='bilinear')
                img_meta_copy = copy.copy(img_meta)
                img_meta_copy['scale_factor'] = scale_factor
                coarse_scale_factors.append(img_meta_copy)

        if len(coarse_scale_factors) != 0:
            for n in range(len(coarse_imgs)):
                coarse_imgs[n] = impad_to_multiple(coarse_imgs[n][0], 32, pad_val=0).unsqueeze(0)

            assert len(coarse_imgs) == len(coarse_gt_labels) == len(coarse_gt_bboxes) == len(coarse_scale_factors)

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
