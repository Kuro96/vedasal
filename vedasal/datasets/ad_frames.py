import os
import os.path as osp

from vedacore.misc import registry

from .custom import CustomDataset


@registry.register_module('dataset')
class AdFramesDataset(CustomDataset):
    _frame_sample_rate = 25
    _name_tmpl = '{:04d}.png'

    def __init__(self,
                 videos_path,
                 maps_path=None,
                 pipeline=None,
                 metrics=None,
                 test_mode=False):
        super().__init__(
            videos_path, maps_path, pipeline, metrics, test_mode)

        self._videos_list = sorted(os.listdir(self.videos_path))
        if not test_mode:
            self._maps_list = sorted(os.listdir(self.maps_path))
            assert len(self.videos_list) == len(self.maps_list)

    def prepare_test_data(self, idx):
        raise NotImplementedError

    def prepare_train_data(self, idx):
        """Generates one sample of train data."""
        video_id = self.videos_list[idx]

        # we assume gt frames are contiguous and le with video frames
        frame_prefix = osp.join(self.videos_path, video_id)
        map_prefix = osp.join(self.maps_path, video_id, 'maps')
        frame_list = [x for x in os.listdir(map_prefix) if x.endswith('.png')]
        video_total_frames = len(frame_list)

        dataflow = dict(
            video_id=video_id,
            frame_prefix=frame_prefix,
            name_tmpl=self._name_tmpl,
            map_prefix=map_prefix,
            frame_sample_rate=self._frame_sample_rate,
            video_total_frames=video_total_frames)

        data = self.pipeline(dataflow)

        return data
