from vedacore.misc import registry

from .losses import build_loss
from .base_criterion import BaseCriterion


@registry.register_module('criterion')
class PointwiseCriterion(BaseCriterion):
    def __init__(self,
                 loss=dict(typename='KLDLoss', reduction='batchmean')):
        super().__init__()
        if isinstance(loss, dict):
            self.loss_point = [build_loss(loss)]
        elif isinstance(loss, list):
            self.loss_point = [build_loss(x) for x in loss]
        else:
            raise TypeError('Loss config requires dict or list of dict, but'
                            f'got {type(loss)}.')

    def _loss(self, feats, targets):
        losses = dict()
        for i, l in enumerate(self.loss_point):
            losses[f'loss_{i}'] = l(feats, targets)
        return losses

    def loss(self,
             feats,
             targets):
        losses = self._loss(feats, targets)
        loss, log_vars = self._parse_losses(losses)
        return dict(loss=loss, log_vars=log_vars)
