from torch import nn
from vedacore.misc import registry


@registry.register_module('head')
class VanillaHead(nn.Module):
    def __init__(self,
                 in_channels=256,
                 upscale_factor=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, 1, bias=True)
        self.upsample = nn.Upsample(scale_factor=upscale_factor)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.upsample(x)
        out = self.sigmoid(x)

        return out
