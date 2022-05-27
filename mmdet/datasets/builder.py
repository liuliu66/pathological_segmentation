import copy

from mmdet.utils import build_from_cfg
from .dataset_wrappers import ConcatDataset, RepeatDataset, BalanceConcatDataset
from .registry import DATASETS


def _concat_dataset(cfg, default_args=None):
    data_types = cfg['type']
    ann_files = cfg['ann_file']
    img_prefixes = cfg.get('img_prefix', None)
    seg_prefixes = cfg.get('seg_prefixes', None)
    proposal_files = cfg.get('proposal_file', None)
    obj_model_dirs = cfg.get('obj_model_dir', None)
    sample_weights = cfg.get('sample_weight', [1] * len(ann_files))
    iters_per_epoch = cfg.get('iters_per_epoch', -1)
    pipelines = cfg.get('pipeline', None)
    intrinsics = cfg.get('intrinsics', None)
    label_maps = cfg.get('label_map', None)

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        data_cfg['type'] = data_types[i]
        data_cfg['ann_file'] = ann_files[i]
        if isinstance(img_prefixes, (list, tuple)):
            data_cfg['img_prefix'] = img_prefixes[i]
        if isinstance(seg_prefixes, (list, tuple)):
            data_cfg['seg_prefix'] = seg_prefixes[i]
        if isinstance(proposal_files, (list, tuple)):
            data_cfg['proposal_file'] = proposal_files[i]
        if isinstance(obj_model_dirs, (list, tuple)):
            data_cfg['obj_model_dir'] = obj_model_dirs[i]
        if isinstance(pipelines, (list, tuple)):
            data_cfg['pipeline'] = pipelines[i]
        if isinstance(intrinsics, (list, tuple)):
            data_cfg['intrinsics'] = intrinsics[i]
        if isinstance(label_maps, (list, tuple)):
            data_cfg['label_map'] = label_maps[i]
        datasets.append(build_dataset(data_cfg, default_args))

    if 'balance' in default_args.keys() and default_args['balance']:
        return BalanceConcatDataset(datasets, sample_weights=sample_weights, iters_per_epoch=iters_per_epoch)
    else:
        return ConcatDataset(datasets, sample_weights=sample_weights, iters_per_epoch=iters_per_epoch)


def build_dataset(cfg, default_args=None):
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif isinstance(cfg['ann_file'], (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset
