import torch.nn as nn

from ..models import build_model


class BaseEngine(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = build_model(model)
