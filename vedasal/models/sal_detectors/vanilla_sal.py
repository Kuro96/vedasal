import torch
from vedacore.misc import registry

from .base_sal_detector import BaseSalDetector


@registry.register_module('sal_detector')
class VanillaSal(BaseSalDetector):
    def __init__(self, backbone, neck, sequencer, head):
        super().__init__(backbone, neck, sequencer, head)

    def forward_impl(self, x, hidden_state=None):
        n, t, c, h, w = x.shape
        feat = x.view(n * t, c, h, w)
        feat = self.backbone(feat)

        feat = self.neck(feat)
        feat = feat.view(n, t, *feat.shape[1:])

        frames = []
        for i in range(x.size(1)):
            frame, hidden_state = self.sequencer(
                feat[:, i, ...], hidden_state)
            frames.append(frame)
        feat = torch.stack(frames, dim=1)
        feat = feat.view(-1, *feat.shape[-3:])

        feat = self.head(feat)
        feat = feat.view(n, t, *feat.shape[-2:])

        return (feat, hidden_state)
