from vedacore.misc import registry, build_from_cfg


def build_loss(cfg):
    loss = build_from_cfg(cfg, registry, 'loss')
    return loss
