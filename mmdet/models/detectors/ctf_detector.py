import torch.nn as nn

from .base import BaseDetector
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2result, coarse_bbox2result


@DETECTORS.register_module
class CTFDetector(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CTFDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(CTFDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)

        losses = dict()

        rpn_outs = self.rpn_head(x)
        rpn_loss_inputs = rpn_outs + (gt_bboxes, img_metas,
                                      self.train_cfg.rpn)
        rpn_losses = self.rpn_head.loss(
            *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        # losses.update(rpn_losses)

        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses.update(self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore))

        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                          self.test_cfg.rpn)
        proposal_inputs = rpn_outs + (img_metas, proposal_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)

        proposal_list_coarse = proposal_list[0][1]

        coarse_imgs, coarse_labels, coarse_bboxes, coarse_scale_factors \
            = self.rpn_head.extract_coarse_data(img,
                                                img_metas,
                                                proposal_list_coarse,
                                                gt_labels,
                                                gt_bboxes,
                                                max_num=3
                                                )
        if len(coarse_scale_factors) != 0:
            '''
            img_name = coarse_scale_factors[0]['filename'].split('/')[-1].split('.')[0]

            import pickle
            coarse_img_np = coarse_img.cpu().numpy()
            with open('ctf_results/coarse_img_'+img_name+'.pkl', 'wb') as f:
                pickle.dump(coarse_img_np, f)
            coarse_labels_np = []
            coarse_bboxes_np = []
            for i in range(len(coarse_labels)):
                coarse_labels_np.append(coarse_labels[i].cpu().numpy())
                coarse_bboxes_np.append(coarse_bboxes[i].cpu().numpy())
            with open('ctf_results/coarse_labels_'+img_name+'.pkl', 'wb') as f:
                pickle.dump(coarse_labels_np, f)
            with open('ctf_results/coarse_bboxes_'+img_name+'.pkl', 'wb') as f:
                pickle.dump(coarse_bboxes_np, f)
            with open('ctf_results/coarse_scale_factors_'+img_name+'.pkl', 'wb') as f:
                pickle.dump(coarse_scale_factors, f)
            '''
            
            for i in range(len(coarse_imgs)):
                x = coarse_imgs[i]

                x_coarse = self.extract_feat(x)
                outs_coarse = self.bbox_head(x_coarse)
                loss_inputs = outs_coarse + ([coarse_bboxes[i]], [coarse_labels[i]], [coarse_scale_factors[i]], self.train_cfg)
                losses_coarse = self.bbox_head.loss(
                    *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

                #for k in losses_coarse.keys():
                    #assert len(losses[k]) == len(losses_coarse[k])
                    #losses[k] = [losses[k][j] + losses_coarse[k][j] for j in range(len(losses[k]))]
            #for k in losses_coarse.keys():
                #losses[k] = [losses[k][j] / (len(coarse_imgs) + 1) for j in range(len(losses[k]))]

            #losses_coarse['loss_cls_coarse'] = losses_coarse.pop('loss_cls')
            #losses_coarse['loss_bbox_coarse'] = losses_coarse.pop('loss_bbox')

            #losses.update(losses_coarse)


        '''
        x, gt_labels, gt_bboxes, img_metas = self.rpn_head.cat_coarse_data(
            x, x_coarse,
            gt_labels, coarse_labels,
            gt_bboxes, coarse_bboxes,
            img_metas, coarse_scale_factors
        )
        '''

        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)

        proposal_list_coarse = self.simple_test_rpn(x, img_meta, self.test_cfg.rpn)

        coarse_img, _, _, coarse_scale_factors \
            = self.rpn_head.extract_coarse_data(img,
                                                img_meta,
                                                proposal_list_coarse,
                                                gt_labels=None,
                                                gt_bboxes=None,
                                                mode='test'
                                                )
        x_coarse = self.extract_feat(coarse_img)

        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]

        outs_coarse = self.bbox_head(x_coarse)
        bbox_inputs_coarse = outs_coarse + (coarse_scale_factors, self.test_cfg, rescale)
        bbox_list_coarse = self.bbox_head.get_bboxes(*bbox_inputs_coarse)
        bbox_results_coarse = [
            coarse_bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes, coarse_scale_factors)
            for det_bboxes, det_labels in bbox_list_coarse
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
