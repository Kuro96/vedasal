import torch
from vedacore.misc import registry

from .infer_engine import InferEngine


@registry.register_module('engine')
class ValEngine(InferEngine):
    def __init__(self, model, split_size):
        super().__init__(model)
        self.split_size = split_size

    def extract_maps(self, frames, hidden_state):
        feat, hidden_state = self.model(frames, hidden_state, train=False)
        return feat, hidden_state

    def forward(self, data):
        return self.forward_impl(**data)

    def forward_impl(self, frames, maps, frame_metas):
        assert frames.size(0) == 1, \
            'samples per gpu in validation phase must be 1!'
        frames = torch.split(frames, self.split_size, dim=1)
        hidden_state = None
        preds = []
        for split in frames:
            split_preds, hidden_state = self.extract_maps(split, hidden_state)
            preds.append(split_preds)
        preds = torch.cat(preds, dim=1).cpu()
        maps = maps.cpu()

        return dict(preds=preds, maps=maps, frame_metas=frame_metas)
