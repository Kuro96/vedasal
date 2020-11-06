from vedacore.misc import registry, build_from_cfg


def build_metrics(cfg):
    if isinstance(cfg, (list, tuple)):
        metrics = [build_metrics(c) for c in cfg]
    else:
        metrics = [build_from_cfg(cfg, registry, 'metrics')]

    return metrics
