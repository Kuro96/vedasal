from vedacore.optimizers import build_optimizer
from vedacore.misc import registry

from ..criteria import build_criterion
from .base_engine import BaseEngine


@registry.register_module('engine')
class TrainEngine(BaseEngine):
    def __init__(self, model, criterion, optimizer):
        super().__init__(model)
        self.criterion = build_criterion(criterion)
        self.optimizer = build_optimizer(self.model, optimizer)

    def extract_maps(self, frames, hidden_state=None):
        feat, hidden_state = self.model(frames, hidden_state, train=True)
        return feat, hidden_state

    def forward(self, data):
        return self.forward_impl(**data)

    def forward_impl(self, frames, maps, frame_metas):
        feat, hidden_state = self.extract_maps(frames)
        losses = self.criterion.loss(feat, maps)
        return losses
