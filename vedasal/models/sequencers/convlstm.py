import torch
from torch import nn
from vedacore.misc import registry


@registry.register_module('sequencer')
class ConvLSTMCell(nn.Module):
    def __init__(self,
                 in_channels=256,
                 hidden_channels=256,
                 kernel_size=3,
                 with_cell_state=True,
                 cell_shape=None,
                 dropout=0.1):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.padding = int((kernel_size-1)/2)

        self.conv_xi = nn.Conv2d(
            self.in_channels, self.hidden_channels,
            self.kernel_size, 1, self.padding, bias=True)
        self.conv_xf = nn.Conv2d(
            self.in_channels, self.hidden_channels,
            self.kernel_size, 1, self.padding, bias=True)
        self.conv_xc = nn.Conv2d(
            self.in_channels, self.hidden_channels,
            self.kernel_size, 1, self.padding, bias=True)
        self.conv_xo = nn.Conv2d(
            self.in_channels, self.hidden_channels,
            self.kernel_size, 1, self.padding, bias=True)

        self.conv_hi = nn.Conv2d(self.hidden_channels, self.hidden_channels,
                                 self.kernel_size, 1, self.padding, bias=False)
        self.conv_hf = nn.Conv2d(self.hidden_channels, self.hidden_channels,
                                 self.kernel_size, 1, self.padding, bias=False)
        self.conv_hc = nn.Conv2d(self.hidden_channels, self.hidden_channels,
                                 self.kernel_size, 1, self.padding, bias=False)
        self.conv_ho = nn.Conv2d(self.hidden_channels, self.hidden_channels,
                                 self.kernel_size, 1, self.padding, bias=False)

        self.dropout = nn.Dropout(dropout)

        if with_cell_state:
            self.w_ci = nn.Parameter(
                torch.zeros(1, self.hidden_channels, *cell_shape))
            self.w_cf = nn.Parameter(
                torch.zeros(1, self.hidden_channels, *cell_shape))
            self.w_co = nn.Parameter(
                torch.zeros(1, self.hidden_channels, *cell_shape))
            nn.init.xavier_normal_(self.w_ci)
            nn.init.xavier_normal_(self.w_cf)
            nn.init.xavier_normal_(self.w_co)
        else:
            self.w_ci = 0
            self.w_cf = 0
            self.w_co = 0

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, state=None):
        if state is None:
            h, c = self.init_hidden(x)
        else:
            h, c = state

        i = torch.sigmoid(self.conv_xi(self.dropout(x)) +
                          self.conv_hi(self.dropout(h)) +
                          self.w_ci * self.dropout(c))
        f = torch.sigmoid(self.conv_xf(self.dropout(x)) +
                          self.conv_hf(self.dropout(h)) +
                          self.w_cf * self.dropout(c))
        c = f * c + i * torch.tanh(self.conv_xc(self.dropout(x)) +
                                   self.conv_hc(self.dropout(h)))
        o = torch.sigmoid(self.conv_xo(self.dropout(x)) +
                          self.conv_ho(self.dropout(h)) +
                          self.w_co * self.dropout(c))
        h = o * torch.tanh(c)

        return h, (h, c)

    def init_hidden(self, x):
        b, _, h, w = x.size()

        state = (torch.zeros(b, self.hidden_channels, h, w).to(x.device),
                 torch.zeros(b, self.hidden_channels, h, w).to(x.device))

        return state
