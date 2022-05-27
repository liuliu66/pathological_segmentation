import numpy as np
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .registry import DATASETS


@DATASETS.register_module
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
    """

    def __init__(self, datasets, sample_weights, iters_per_epoch):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        if hasattr(datasets[0], 'flag'):
            flags = []
            for i in range(0, len(datasets)):
                flags.append(np.ones(len(datasets[i].flag)) * i)
            self.flag = np.concatenate(flags)
        self.sample_weights = sample_weights
        self.iters_per_epoch = iters_per_epoch
        self.balance = False


@DATASETS.register_module
class RepeatDataset(object):
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        if hasattr(self.dataset, 'flag'):
            self.flag = np.tile(self.dataset.flag, times)

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        return self.times * self._ori_len


@DATASETS.register_module
class BalanceConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
    """

    def __init__(self, datasets, sample_weights, iters_per_epoch):
        super(BalanceConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        self.flag = np.concatenate([datasets[i].flag for i in range(len(datasets))])
        self.sample_weights = sample_weights
        self.iters_per_epoch = iters_per_epoch
        self.balance = True
