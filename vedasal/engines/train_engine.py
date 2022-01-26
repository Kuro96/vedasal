import torch
from vedacore.optimizers import build_optimizer
from vedacore.misc import registry

from ..criteria import build_criterion
from .base_engine import BaseEngine


@registry.register_module('engine')
class TrainEngine(BaseEngine):
    def __init__(self, model, criterion, optimizer, split_size=None):
        super().__init__(model)
        self.criterion = build_criterion(criterion)
        self.optimizer = build_optimizer(self.model, optimizer)
        self.split_size = split_size

    def extract_maps(self, frames, hidden_state=None):
        feat, hidden_state = self.model(frames, hidden_state, train=True)
        return feat, hidden_state

    def forward(self, data):
        return self.forward_impl(**data)

    def forward_impl(self, frames, maps, frame_metas):
        if self.split_size:
            frames = torch.split(frames, self.split_size, dim=1)
            hidden_state = None
            preds = []
            for split in frames:
                split_preds, hidden_state = self.extract_maps(split,
                                                              hidden_state)
                preds.append(split_preds)
            preds = torch.cat(preds, dim=1)
        else:
            preds, hidden_state = self.extract_maps(frames)
        losses = self.criterion.loss(preds, maps)

        return losses
