from torch import nn
from vedacore.misc import registry


@registry.register_module('neck')
class FPNFusion(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 out_channels=256):
        super().__init__()

        C2_size, C3_size, C4_size, C5_size = in_channels

        # upsample C5 to get P5 from the FPN paper
        self.P5_conv = nn.Conv2d(C5_size,
                                 out_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=False)
        self.P5_bn = nn.BatchNorm2d(out_channels)
        self.P5_upsample = nn.Upsample(scale_factor=2)

        # add P5 elementwise to C4
        self.P4_conv = nn.Conv2d(C4_size,
                                 out_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=False)
        self.P4_bn = nn.BatchNorm2d(out_channels)
        self.P4_upsample = nn.Upsample(scale_factor=2)

        # add P4 elementwise to C3
        self.P3_conv = nn.Conv2d(C3_size,
                                 out_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=False)
        self.P3_bn = nn.BatchNorm2d(out_channels)
        self.P3_upsample = nn.Upsample(scale_factor=2)

        self.P2_conv = nn.Conv2d(C2_size,
                                 out_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=False)
        self.P2_bn = nn.BatchNorm2d(out_channels)

        self.conv = nn.Conv2d(out_channels,
                              out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        C2, C3, C4, C5 = inputs

        P5 = self.P5_conv(C5)
        P5 = self.P5_bn(P5)
        P5 = self.P5_upsample(P5)

        P4 = self.P4_conv(C4)
        P4 = self.P4_bn(P4)
        P4 = P5 + P4
        P4 = self.P4_upsample(P4)

        P3 = self.P3_conv(C3)
        P3 = self.P3_bn(P3)
        P3 = P4 + P3
        P3 = self.P3_upsample(P3)

        P2 = self.P2_conv(C2)
        P2 = self.P2_bn(P2)
        P2 = P3 + P2

        out = self.conv(P2)
        out = self.bn(out)
        out = self.relu(out)

        return out
