import random
from functools import partial

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader

from vedacore.parallel import collate, get_dist_info
from vedacore.misc import registry, build_from_cfg


def build_dataset(cfg, default_args=None):
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    else:
        dataset = build_from_cfg(cfg, registry, 'dataset', default_args)

    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     shuffle=True,
                     pin_memory=True,
                     num_gpus=1,
                     dist=True,
                     seed=None,
                     **kwargs):
    """Build PyTorch DataLoader.
    Args:
        dataset (Dataset): A PyTorch dataset.
        batch_size (int): Number of training samples.
        num_workers (int): How many subprocesses to use for data loading.
        pin_memory (bool): Whether to copy Tensors into CUDA pinned memory
            before returning them.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=batch_size),
        shuffle=shuffle,
        pin_memory=pin_memory,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, seed):
    # The seed of each worker equals to worker_id + user_seed
    worker_seed = worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
