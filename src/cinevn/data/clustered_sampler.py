"""
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.
"""

import math
from collections import defaultdict
from typing import Iterator

import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler, Dataset, RandomSampler, Sampler, SequentialSampler

from .mri_data import SliceDataset


class ListDataset(Dataset):
    def __init__(self, data: list):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class ClusteredBatchSampler(Sampler):
    """
    This sampler clusters batches based on k-space size.
    """
    def __init__(
            self,
            dataset: SliceDataset,
            batch_size: int,
            shuffle: bool,
            distributed: bool = False,
            seed: int = 0,
    ):
        super().__init__(None)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.distributed = distributed
        self.seed = seed
        if self.distributed:
            if not dist.is_available():
                raise RuntimeError('Requires distributed package to be available')
            self.num_replicas = dist.get_world_size()  # number of GPUs
            if not dist.is_available():
                raise RuntimeError('Requires distributed package to be available')
            self.rank = dist.get_rank()  # rank of current device
            if self.rank >= self.num_replicas or self.rank < 0:
                raise ValueError(
                    'Invalid rank {}, rank should be in the interval'
                    '[0, {}]'.format(self.rank, self.num_replicas - 1))
        else:
            self.num_replicas = 1
            self.rank = 0
        self.epoch = 0

        # cluster samples by k-space shape
        indices = defaultdict(list[int])
        for i, example in enumerate(dataset.examples):
            indices[example[2]].append(i)

        # create a batch sampler for every shape in the dataset
        self.datasets = []
        self.samplers = []
        for k, v in indices.items():
            ds = ListDataset(v)
            self.datasets.append(ds)

            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + sum(v))
                sampler = RandomSampler(ds, generator=g)
            else:
                sampler = SequentialSampler(ds)
            self.samplers.append(BatchSampler(sampler, batch_size=self.batch_size, drop_last=False))

        sum_len_samplers = sum([len(s) for s in self.samplers])
        self.num_batches = math.ceil(sum_len_samplers / self.num_replicas)
        self.total_size = self.num_batches * self.num_replicas

    def __iter__(self) -> Iterator[list[int]]:
        # create list of indices corresponding to the samplers
        which_sampler = []
        for i, sampler in enumerate(self.samplers):
            which_sampler.extend([i] * len(sampler))

        # create an iterator for each sampler
        iterators = [iter(sampler) for sampler in self.samplers]

        # shuffle
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(which_sampler), generator=g)
            which_sampler = [which_sampler[i] for i in indices]

        # iterate over samplers and create list of lists with batches
        batches = []
        for i in which_sampler:
            indices = next(iterators[i])
            # fill the last batch of each iterator with samples from the beginning
            if self.distributed and len(indices) < self.batch_size:
                iter_restarted = iter(self.samplers[i])
                first_batch = next(iter_restarted)
                indices += first_batch[:self.batch_size - len(indices)]
            batch = [self.datasets[i][b] for b in indices]
            batches.append(batch)

        # add extra batches to make it evenly divisible
        padding_size = self.total_size - len(batches)
        if padding_size <= len(batches):
            batches += batches[:padding_size]
        else:
            batches += (batches * math.ceil(padding_size / len(batches)))[:padding_size]
        assert len(batches) == self.total_size

        # distribute among devices
        batches = batches[self.rank:self.total_size:self.num_replicas]
        assert len(batches) == self.num_batches

        return iter(batches)

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
