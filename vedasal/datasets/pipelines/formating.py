import torch
from vedacore.misc import registry

from .base import BasePipeline


@registry.register_module('pipeline')
class ToFloatTensor(BasePipeline):
    """Convert some dataflow to :obj:`torch.FloatTensor` by given keys.

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, dataflow):
        """Call function to convert data in dataflow to `torch.FloatTensor`.

        Args:
            dataflow (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data converted
                to `torch.FloatTensor`.
        """
        for key in self.keys:
            dataflow[key] = torch.tensor(dataflow[key], dtype=torch.float32)
        return dataflow


@registry.register_module('pipeline')
class FormatShape(BasePipeline):
    # TODO: docs
    _valid_dims = ['T', 'C', 'H', 'W']

    def __init__(self, in_format='THWC', out_format='TCHW'):
        self.in_format = in_format.upper()
        self.out_format = out_format.upper()

        assert all([x in self._valid_dims for x in self.in_format]), \
            f'Input format must be in {self._valid_dims}, ' \
            f'but got {self.in_format}.'
        assert len(out_format) == len(in_format) and \
            all([x in self.in_format] for x in self.out_format), \
            'Output format must match input format.'
        self.permute_order = [self.in_format.index(x) for x in self.out_format]
        self.permute_order_maps = self._get_order_maps()

    def _get_order_maps(self):
        in_format = list(self.in_format)  # also detach from :attr:
        out_format = list(self.out_format)
        in_format.remove('C')
        out_format.remove('C')

        return [in_format.index(x) for x in out_format]

    def __call__(self, dataflow):
        dataflow['frames'] = dataflow['frames'].permute(self.permute_order)
        dataflow['maps'] = dataflow['maps'].permute(self.permute_order_maps)

        return dataflow


@registry.register_module('pipeline')
class Collect(BasePipeline):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline.

    Args:
        keys (Sequence[str]): Keys of dataflow to be collected in `data`.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            `DataContainer` and collected in `data[frame_metas]`.
            Default: `('video_id', 'video_total_frames', 'frame_sample_rate',
            'start_idx', 'frame_inds', 'clip_length', 'frame_interval',
            'ori_frame_shape', 'frame_shape', 'pad_shape', 'scale_factor',
            'flip', 'flip_direction', 'frame_norm_cfg')`
    """
    def __init__(self,
                 keys,
                 meta_keys=('video_id', 'video_total_frames',
                            'frame_sample_rate', 'start_idx', 'frame_inds',
                            'clip_length', 'frame_interval', 'ori_frame_shape',
                            'frame_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'frame_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, dataflow):
        """Call function to collect keys in dataflow. The keys in `meta_keys`
        will be converted to `DataContainer`.

        Args:
            dataflow (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in `self.keys`
                - `frame_meta`
        """
        from vedacore.parallel import DataContainer
        data = {}
        frame_meta = {}
        for key in self.meta_keys:
            frame_meta[key] = dataflow[key]
        data['frame_metas'] = DataContainer(frame_meta, cpu_only=True)
        for key in self.keys:
            data[key] = dataflow[key]
        return data
