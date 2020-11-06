import os.path as osp
import random

import numpy as np

from vedacore.fileio import FileClient
from vedacore.image import imfrombytes
from vedacore.misc import registry

from .base import BasePipeline


@registry.register_module('pipeline')
class CountFrames(BasePipeline):
    def __init__(self, sample_rate=25):
        # TODO: if data is in video format, calculate its `total_frames`.
        pass


@registry.register_module('pipeline')
class SampleFrames(BasePipeline):
    """Randomly sample frames from the video.

    Args:
        clip_length (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.

    Required keys:
        "video_total_frames"
    Modified keys:
        "start_idx", "frame_inds", "frame_interval"
    """
    def __init__(self,
                 clip_length=None,
                 frame_interval=1):
        if clip_length is None and frame_interval != 1:
            raise ValueError(
                'frame interval must be 1 if no clip length is set!')

        self.clip_length = clip_length
        self.frame_interval = frame_interval
        self.margin = (clip_length - 1) * frame_interval + 1 if clip_length \
            else None

    def _sample_clip(self, total_frames):
        start_idx = random.randint(0, total_frames - self.margin)
        stop_idx = start_idx + self.margin  # exclusive
        frame_inds = np.arange(start_idx, stop_idx, step=self.frame_interval)
        return frame_inds

    def _full_sample_clip(self, total_frames):
        return np.arange(total_frames)

    def __call__(self, dataflow):
        """Perform the SampleFrames loading.

        Args:
            dataflow (dict): The dataflow dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = dataflow['video_total_frames']
        frame_inds = self._sample_clip(total_frames) if self.clip_length \
            else self._full_sample_clip(total_frames)

        dataflow['start_idx'] = int(frame_inds[0])
        dataflow['frame_inds'] = frame_inds.astype(np.int)
        dataflow['clip_length'] = self.clip_length if self.clip_length \
            else total_frames
        dataflow['frame_interval'] = self.frame_interval
        return dataflow


@registry.register_module('pipeline')
class RawFrameDecode(BasePipeline):
    """Load and decode frames with given indices.

    Args:
        io_backend (str): IO backend where frames are stored. Default: 'disk'.
        decoding_backend (str): Backend used for image decoding.
            Default: 'cv2'.
        kwargs (dict, optional): Arguments for FileClient.

    Required keys:
        "frame_prefix", "name_tmpl", "frame_inds"
    Modified keys:
        "frames", "frame_shape", "ori_frame_shape"
    """
    def __init__(self, io_backend='disk', **kwargs):
        self.file_client = FileClient(io_backend, **kwargs)

    def __call__(self, dataflow):
        """Perform the `RawFrameDecode` to pick frames given indices.

        Args:
            dataflow (dict): The dataflow dict to be modified and passed
                to the next transform in pipeline.
        """
        frame_prefix = dataflow['frame_prefix']
        name_tmpl = dataflow['name_tmpl']

        imgs = list()
        for frame_idx in dataflow['frame_inds']:
            filepath = osp.join(frame_prefix,
                                name_tmpl.format(frame_idx))
            img_bytes = self.file_client.get(filepath)
            cur_frame = imfrombytes(img_bytes)
            imgs.append(cur_frame)

        dataflow['frames'] = imgs
        dataflow['frame_shape'] = imgs[0].shape[:2]
        dataflow['ori_frame_shape'] = imgs[0].shape[:2]

        return dataflow


@registry.register_module('pipeline')
class RawMapDecode(BasePipeline):
    """Load and decode maps with given indices.

    Args:
        io_backend (str): IO backend where frames are stored. Default: 'disk'.
        decoding_backend (str): Backend used for image decoding.
            Default: 'cv2'.
        kwargs (dict, optional): Arguments for FileClient.

    Required keys:
        "map_prefix", "name_tmpl", "frame_inds",
    Modified keys:
        "maps"
    """
    def __init__(self, io_backend='disk', **kwargs):
        self.file_client = FileClient(io_backend, **kwargs)

    def __call__(self, dataflow):
        """Perform the `RawMapDecode` to pick maps given indices.

        Args:
            dataflow (dict): The dataflow dict to be modified and passed
                to the next transform in pipeline.
        """
        map_prefix = dataflow['map_prefix']
        name_tmpl = dataflow['name_tmpl']

        imgs = list()
        for frame_idx in dataflow['frame_inds']:
            filepath = osp.join(map_prefix,
                                name_tmpl.format(frame_idx))
            img_bytes = self.file_client.get(filepath)
            cur_frame = imfrombytes(img_bytes, flag='grayscale')
            imgs.append(cur_frame)

        dataflow['maps'] = imgs

        return dataflow
