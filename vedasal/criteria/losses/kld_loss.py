import sys

import torch
from torch import nn
from vedacore.misc import registry


def kl_div(inp, trg, reduction):
    eps = sys.float_info.epsilon

    d = trg*torch.log(eps+torch.div(trg, (inp+eps)))

    if reduction == 'sum':
        loss = torch.sum(d)
    elif reduction == 'mean':
        loss = torch.mean(d)
    elif reduction == 'batchmean':
        loss = torch.mean(torch.sum(d, dim=tuple(range(1, len(d.shape)))))
    elif reduction == 'none':
        loss = d
    else:
        raise ValueError(f'Reduction method "{reduction}" invalid!')

    return loss


@registry.register_module('loss')
class KLDLoss(nn.Module):
    """
    Args:
        reduction (string, optional):
            Specifies the reduction to apply to the output:
            'none' | 'batchmean' | 'sum' | 'mean'.
            'none': no reduction will be applied
            'batchmean': the sum of the output will be divided by the batchsize
            'sum': the output will be summed
            'mean': the output will be divided by the number of elements in the
                output
            Default: 'sum'
    """

    def __init__(self, reduction='sum'):
        super().__init__()
        self.reduction = reduction

    def forward(self, inp, trg):
        inp = inp / torch.sum(inp)
        trg = trg / torch.sum(trg)

        return kl_div(inp, trg, self.reduction)
