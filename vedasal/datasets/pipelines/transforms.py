# adapted mmdetection
import numpy as np
from vedacore import image
from vedacore.misc import registry

from .base import BasePipeline


@registry.register_module('pipeline')
class Resize(BasePipeline):
    """Resize frames to a specific size.

    Args:
        scale (float | Tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size:
            If it is a float number, frames will be rescaled by thisfactor,
            else if it is a tuple of 2 integers, frames will be rescaled as
            large as possible within the scale. Otherwise, it serves as (w, h)
            of output size.
        keep_ratio (bool): If set to True, frames will be resized without
            changing the aspect ratio. Otherwise, it will resize frames to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
        map_accord (bool): If set to True, maps will be resized in same manner
            with frames.

    Required keys:
        "frames", "ori_frame_shape",
        (optional) "maps"
    Modified keys:
        "frames", "frame_shape", "keep_ratio", "scale_factor",
        (optional) "maps"
    """
    def __init__(self,
                 scale,
                 keep_ratio=True,
                 interpolation='bilinear',
                 map_accord=False):
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
        else:
            raise TypeError(
                f'Scale must be float or tuple of int, but got {type(scale)}')
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation

    def __call__(self, dataflow):
        """Performs the Resize augmentation.

        Args:
            dataflow (dict): The dataflow dict to be modified and passed
                to the next transform in pipeline.
        """
        if 'scale_factor' not in dataflow:
            dataflow['scale_factor'] = np.array([1, 1], dtype=np.float32)
        img_h, img_w = dataflow['ori_frame_shape']

        if self.keep_ratio:
            new_w, new_h = image.rescale_size((img_w, img_h), self.scale)
        else:
            new_w, new_h = self.scale

        scale_factor = np.array([new_w / img_w, new_h / img_h],
                                dtype=np.float32)

        dataflow['frame_shape'] = (new_h, new_w)
        dataflow['keep_ratio'] = self.keep_ratio
        dataflow['scale_factor'] = dataflow['scale_factor'] * scale_factor
        dataflow['frames'] = [image.imresize(img, (new_w, new_h),
                                             interpolation=self.interpolation)
                              for img in dataflow['frames']]
        dataflow['maps'] = [image.imresize(img, (new_w, new_h),
                                           interpolation=self.interpolation)
                            for img in dataflow['maps']]

        return dataflow


@registry.register_module('pipeline')
class RandomFlip(BasePipeline):
    """Flip the input frames with a probability.

    Args:
        flip_ratio (float): Probability of implementing flip. Default: 0.5.
        direction (str): Flip imgs horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".

    Required keys:
        "frames", (optional) "maps"
    Modified keys:
        "frames", "flip", "flip_direction", (optional) "maps"
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self,
                 flip_ratio=0.5,
                 direction='horizontal'):
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported. '
                             f'Currently support ones are {self._directions}')
        self.flip_ratio = flip_ratio
        self.direction = direction

    def __call__(self, dataflow):
        """Performs the Flip augmentation.

        Args:
            dataflow (dict): The dataflow dict to be modified and passed
                to the next transform in pipeline.
        """
        if np.random.rand() < self.flip_ratio:
            flip = True
        else:
            flip = False
        if not dataflow.get('maps'):
            raise RuntimeError(
                '`RandomFlip` shouldn\'t be added in test phase!')

        dataflow['flip'] = flip
        dataflow['flip_direction'] = self.direction

        if flip:
            for img in dataflow['imgs']:
                image.imflip_(img, self.direction)
            for img in dataflow['maps']:
                image.imflip_(img, self.direction)

        return dataflow


@registry.register_module('pipeline')
class Normalize(BasePipeline):
    """Normalize frames.

    Args:
        mean (Sequence[float]): Mean values of different channels.
        std (Sequence[float]): Std values of different channels.
        to_bgr (bool): Whether to convert channels from RGB to BGR.
            Default: False.

    Required keys:
        "frames"
    Modified keys:
        "frames", "frame_norm_cfg"
    """
    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, dataflow):
        for frame in dataflow['frames']:
            frame = image.imnormalize(frame, self.mean, self.std, self.to_rgb)
        dataflow['frame_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return dataflow


@registry.register_module('pipeline')
class Pad(BasePipeline):
    """Pad frames & maps.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.

    Required keys:
        "frames", (optional) "maps"
    Modified keys:
        "frames", "pad_shape", (optional) "maps"
    """
    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert bool(size) ^ bool(size_divisor)

    def _pad_frames(self, dataflow):
        frames = []
        for frame in dataflow['frames']:
            if self.size is not None:
                frame = image.impad(
                    frame, shape=self.size, pad_val=self.pad_val)
            elif self.size_divisor is not None:
                frame = image.impad_to_multiple(
                    frame, self.size_divisor, pad_val=self.pad_val)
            frames.append(frame)
        dataflow['frames'] = frames
        dataflow['pad_shape'] = frame.shape[:2]

    def _pad_maps(self, dataflow):
        maps = []
        for m in dataflow['maps']:
            m = image.impad(m, shape=dataflow['pad_shape'])
            maps.append(m)
        dataflow['maps'] = maps

    def __call__(self, dataflow):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            dataflow (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_frames(dataflow)
        self._pad_maps(dataflow)
        return dataflow
