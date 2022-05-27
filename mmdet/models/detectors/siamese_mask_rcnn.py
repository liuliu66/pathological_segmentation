import torch
import torch.nn as nn

from .two_stage import TwoStageDetector
from ..registry import DETECTORS


@DETECTORS.register_module
class SiameseMaskRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(SiameseMaskRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        self.conv = nn.Conv2d(512, 256, 1)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def matching(self, If, Rf):
        out = []
        for i in range(len(Rf)):
            rf_avg = self.avg(Rf[i])
            delta = If[i] - rf_avg
            concat = torch.cat((If[i], delta.abs()), dim=1)
            out.append(self.conv(concat))
        return out

    # extract_feat in siamese way
    def extract_feat(self, img, img_meta):
        If = self.backbone(img)
        if self.with_neck:
            If = self.neck(If)
        Rf = []
        for i in range(len(img_meta)):
            Rf_ = torch.from_numpy(img_meta[i]['rf_img']).unsqueeze(0).to(
                img.device)
            Rf.append(Rf_)
        Rf = torch.cat(Rf, dim=0)
        Rf = self.backbone(Rf)
        if self.with_neck:
            Rf = self.neck(Rf)
        If = self.matching(If, Rf)
        return tuple(If)
