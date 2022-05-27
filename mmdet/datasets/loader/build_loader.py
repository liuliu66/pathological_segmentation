import platform
from functools import partial

from mmcv.parallel import collate
from mmcv.runner import get_dist_info
import torch
from torch.utils.data import DataLoader

from .sampler import (DistributedGroupSampler, DistributedSampler, GroupSampler, BalanceBatchSampler,
                      ConcatDatasetWeightedSampler, DistributedConcatDatasetWeightedSampler)

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def build_dataloader(dataset,
                     imgs_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     **kwargs):
    shuffle = kwargs.get('shuffle', True)
    # if dist:
    #     rank, world_size = get_dist_info()
    #     if shuffle:
    #         if hasattr(dataset, 'datasets'):
    #             sampler = DistributedConcatDatasetWeightedSampler(dataset, dataset.sample_weights,
    #                                                               world_size, rank, imgs_per_gpu)
    #         else:
    #             sampler = DistributedGroupSampler(dataset, imgs_per_gpu,
    #                                             world_size, rank)
    #     else:
    #         sampler = DistributedSampler(
    #             dataset, world_size, rank, shuffle=False)
    #     batch_size = imgs_per_gpu
    #     num_workers = workers_per_gpu
    # else:
    #     if hasattr(dataset, 'datasets'):
    #         sampler = ConcatDatasetWeightedSampler(dataset, dataset.sample_weights,
    #                                                dataset.iters_per_epoch, num_gpus, imgs_per_gpu)
    #         batch_size = num_gpus * imgs_per_gpu
    #         num_workers = num_gpus * workers_per_gpu
    #     else:
    #         sampler = GroupSampler(dataset, imgs_per_gpu) if shuffle else None
    #         batch_size = num_gpus * imgs_per_gpu
    #         num_workers = num_gpus * workers_per_gpu
    if dist:
        rank, world_size = get_dist_info()
        if shuffle:
            sampler = DistributedGroupSampler(dataset, imgs_per_gpu,
                                              world_size, rank)
        else:
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=False)
        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu
    else:
        if hasattr(dataset, 'balance') and dataset.balance:
            sampler = BalanceBatchSampler(dataset, imgs_per_gpu) if shuffle else None
            batch_size = num_gpus * imgs_per_gpu
            num_workers = num_gpus * workers_per_gpu
        else:
            sampler = GroupSampler(dataset, imgs_per_gpu) if shuffle else None
            batch_size = num_gpus * imgs_per_gpu
            num_workers = num_gpus * workers_per_gpu

    print('using {} workers'.format(num_workers))
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
        pin_memory=False,
        **kwargs)

    return data_loader
