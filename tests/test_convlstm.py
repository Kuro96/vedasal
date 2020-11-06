import torch

from kurosal.models.sequencer import ConvLSTMCell


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    cell = ConvLSTMCell(3, 3, 3, 0.1).cuda()
    x = torch.Tensor(4, 3, 5, 5).cuda()

    out, state = cell(x, None)
    print(out, out.size())
    out, state = cell(x, state)
    print(out, out.size())

    out, state = cell(x, None)
    print(out.size())
    out, state = cell(x, state)
    print(out.size())
    import pdb
    pdb.set_trace()
