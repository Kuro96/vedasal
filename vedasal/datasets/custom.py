import logging
from abc import ABCMeta, abstractmethod

from torch.utils.data import Dataset

from .pipelines import Compose
from .metrics import build_metrics


class CustomDataset(Dataset, metaclass=ABCMeta):
    def __init__(self,
                 videos_path,
                 maps_path,
                 pipeline,
                 metrics,
                 test_mode):
        self.logger = logging.getLogger('vedasal')

        self.videos_path = videos_path
        self.maps_path = maps_path
        self.pipeline = Compose(pipeline)
        self.metrics = build_metrics(metrics)
        self.test_mode = test_mode

    @property
    def videos_list(self):
        return self._videos_list

    @property
    def maps_list(self):
        return self._maps_list

    def __len__(self):
        """ Denotes the total number of samples. """
        return len(self.videos_list)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                self.logger.warning(f'No data with index: {idx}')
                continue
            return data

    @abstractmethod
    def prepare_test_data(self, idx):
        pass

    @abstractmethod
    def prepare_train_data(self, idx):
        pass
