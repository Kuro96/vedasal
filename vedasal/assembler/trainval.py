import torch

from vedacore.hooks import HookPool
from vedacore.loopers import EpochBasedLooper
from vedacore.parallel import MMDataParallel

from ..datasets import build_dataset, build_dataloader
from ..engines import build_engine


def trainval(cfg, logger):
    for mode in cfg.modes:
        assert mode in ('train', 'val')

    dataloaders = dict()
    engines = dict()

    if 'train' in cfg.modes:
        dataset = build_dataset(cfg.data.train)
        dataloaders['train'] = build_dataloader(
            dataset,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            pin_memory=True)

        engine = build_engine(cfg.train_engine)
        engine = MMDataParallel(engine.cuda(),
                                device_ids=[torch.cuda.current_device()])

        engines['train'] = engine

    if 'val' in cfg.modes:
        # TODO implement validation
        dataset = build_dataset(cfg.data.val)
        dataloaders['val'] = build_dataloader(
            dataset,
            1,
            cfg.data.workers_per_gpu,
            pin_memory=True,
            shuffle=False)

        engine = build_engine(cfg.val_engine)
        engine = MMDataParallel(engine.cuda(),
                                device_ids=[torch.cuda.current_device()])

        engines['val'] = engine

    hook_pool = HookPool(cfg.hooks, cfg.modes, logger)

    looper = EpochBasedLooper(cfg.modes, dataloaders, engines, hook_pool,
                              logger, cfg.work_dir)
    if 'weights' in cfg:
        looper.load_weights(**cfg.weights)
    if 'train' in cfg.modes:
        if 'optimizer' in cfg:
            looper.load_optimizer(**cfg.optimizer)
        if 'meta' in cfg:
            looper.load_meta(**cfg.meta)
    else:
        if 'optimizer' in cfg:
            logger.warning('optimizer is not needed in train mode')
        if 'meta' in cfg:
            logger.warning('meta is not needed in train mode')
    looper.start(cfg.max_epochs)
