from vedacore.misc import registry, build_from_cfg, singleton_arg


def build_backbone(cfg):
    """Build backbone."""
    return build_from_cfg(cfg, registry, 'backbone')


def build_neck(cfg):
    """Build neck."""
    return build_from_cfg(cfg, registry, 'neck')


def build_sequencer(cfg):
    """Build sequencer."""
    return build_from_cfg(cfg, registry, 'sequencer')


def build_head(cfg):
    """Build head."""
    return build_from_cfg(cfg, registry, 'head')


@singleton_arg
def build_model(cfg):
    return build_from_cfg(cfg, registry, 'sal_detector')
