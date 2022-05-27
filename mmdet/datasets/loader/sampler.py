from __future__ import division
import math

import numpy as np
import copy
import torch
from mmcv.runner.utils import get_dist_info
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import Sampler


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class GroupSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            self.num_samples += int(np.ceil(
                size / self.samples_per_gpu)) * self.samples_per_gpu

    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
            np.random.shuffle(indice)
            num_extra = int(np.ceil(size / self.samples_per_gpu)
                            ) * self.samples_per_gpu - len(indice)
            indice = np.concatenate([indice, indice[:num_extra]])
            indices.append(indice)
        indices = np.concatenate(indices)
        indices = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(
                range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


class ConcatDatasetWeightedSampler(Sampler):
    def __init__(self, dataset, sample_weights, iters_per_epoch, num_gpus=1, samples_per_gpu=1, replacement=True):
        self.dataset = dataset
        self.sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)
        self.iters_per_epoch = iters_per_epoch
        self.num_gpus = num_gpus
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.group_offsets = []
        for i in range(len(self.group_sizes)):
            s = self.group_sizes[:i].sum()
            self.group_offsets.append(s)
        assert len(self.group_sizes) == len(self.sample_weights)
        if self.iters_per_epoch > 0:
            self.num_samples = self.iters_per_epoch * self.samples_per_gpu / self.num_gpus
            self.num_samples = int(self.num_samples)
        else:
            self.num_samples = 0
            for i, size in enumerate(self.group_sizes):
                self.num_samples += int(np.ceil(
                    size / self.samples_per_gpu)) * self.samples_per_gpu

        self.replacement = replacement

    def __iter__(self):
        indices = []

        dataset_indices = torch.multinomial(self.sample_weights, self.num_samples // (self.samples_per_gpu * self.num_gpus), self.replacement).tolist()
        group_sizes = copy.deepcopy(self.group_sizes)
        for i in dataset_indices:
            indice = np.random.choice(group_sizes[i], self.samples_per_gpu * self.num_gpus)
            indice += self.group_offsets[i]
            indices.append(indice)
        indices = np.concatenate(indices)
        indices.astype(np.int64).tolist()

        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


class BalanceBatchSampler(Sampler):
    def __init__(self, dataset, samples_per_gpu=1):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        if self.dataset.iters_per_epoch > 0:
            self.num_samples = self.dataset.iters_per_epoch * self.samples_per_gpu
            self.num_samples = int(self.num_samples)
        else:
            self.num_samples = 0
            for i, size in enumerate(self.group_sizes):
                self.num_samples += int(np.ceil(
                    size / self.samples_per_gpu)) * self.samples_per_gpu

    def __iter__(self):
        indices = []
        samples_per_dataset_per_iter = int(self.samples_per_gpu / len(self.dataset.datasets))
        samples_per_dataset = int(samples_per_dataset_per_iter * self.dataset.iters_per_epoch)
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
            np.random.shuffle(indice)
            if len(indice) < samples_per_dataset:
                num_extra = samples_per_dataset - len(indice)
            else:
                num_extra = 0
            indice = np.concatenate([indice, indice[:num_extra]])
            indices.append(indice)
        indices_sample = []
        for i in np.random.permutation(range(self.dataset.iters_per_epoch)):
            batch_sample_inds = np.concatenate([indices[j][i * samples_per_dataset_per_iter:(i + 1) * samples_per_dataset_per_iter]
                                               for j in range(len(self.dataset.datasets))])
            indices_sample.append(batch_sample_inds)
        indices = np.concatenate(indices_sample)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        assert hasattr(self.dataset, 'flag')
        # self.flag = self.dataset.flag
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                indice = indice[list(torch.randperm(int(size),
                                                    generator=g))].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                # pad indice
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[:extra % size])
                indices.extend(indice)

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedConcatDatasetWeightedSampler(Sampler):
    def __init__(self, dataset, sample_weights, num_replicas=None,
                 rank=None, samples_per_gpu=1, replacement=True, ):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.replacement = replacement

        assert hasattr(self.dataset, 'flag')
        self.sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)
        self.samples_per_gpu = samples_per_gpu
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag.astype(np.int64))
        self.group_offsets = []
        for i in range(len(self.group_sizes)):
            s = self.group_sizes[:i].sum()
            self.group_offsets.append(s)
        assert len(self.group_sizes) == len(self.sample_weights)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []

        dataset_indices = torch.multinomial(self.sample_weights, self.num_samples * self.num_replicas // self.samples_per_gpu, self.replacement).tolist()
        group_sizes = copy.deepcopy(self.group_sizes)
        for i in dataset_indices:
            indice = np.random.choice(group_sizes[i], self.samples_per_gpu)
            indice += self.group_offsets[i]
            indices.append(indice)
        indices = np.concatenate(indices)
        indices.astype(np.int64).tolist()

        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
