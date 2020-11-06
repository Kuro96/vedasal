import torch
from torch import nn

from ..builder import build_backbone, build_neck, build_sequencer, build_head


class BaseSalDetector(nn.Module):
    def __init__(self, backbone, neck, sequencer, head):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.sequencer = build_sequencer(sequencer)
        self.head = build_head(head)

    def forward_impl(self, x, hidden_state=None):
        feat = self.backbone(x)
        feat = self.neck(feat)

        frames = []
        for i in range(x.size(0)):
            frame, hidden_state = self.sequencer(feat[i:i+1], hidden_state)
            frames.append(frame)
        feat = torch.cat(frames, dim=0)

        feat = self.head(feat)

        return (feat, hidden_state)

    def forward(self, x, hidden_state=None, train=True):
        if train:
            self.train()
            # x = x.squeeze()
            feats = self.forward_impl(x, hidden_state=hidden_state)
        else:
            self.eval()
            with torch.no_grad():
                # x = x.squeeze()
                feats = self.forward_impl(x, hidden_state=hidden_state)
        return feats
