from mmdet.utils import Registry

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
ROI_EXTRACTORS = Registry('roi_extractor')
SHARED_HEADS = Registry('shared_head')
HEADS = Registry('head')
LOSSES = Registry('loss')
DETECTORS = Registry('detector')
DATASETS = Registry('datasets')
POSE_ESTIMATORS = Registry('pose_estimator')
SEGMENTORS = Registry('segmentor')
