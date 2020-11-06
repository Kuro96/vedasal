import os
import time
import random
import logging
from collections import OrderedDict

import numpy as np
import torch
from vedacore.misc import get_logger


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.
    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name='vedasal', log_file=log_file, log_level=log_level)
    return logger


def colorize(string, color='RED', bold=False, highlight=False):
    colors = dict(GRAY=30, RED=31, GREEN=32, YELLOW=33, BLUE=34,
                  MAGENTA=35, CYAN=36, WHITE=37, CRIMSON=38)
    num = colors[color.upper()]
    attr = []
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')

    return f'\x1b[{";".join(attr)}m{string}\x1b[0m'


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not os.path.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


class Timer:
    """
    A timer.

    Example:
        >>> import random

        >>> def type1():
        >>>     timer = Timer()
        >>>     print(f'{timer()}\tstart')
        >>>     for _ in range(5):
        >>>         time.sleep(random.random())
        >>>         print(f'{timer()}\tprocessed xxxx')
        >>>     print(timer.report())

        >>> def type2():
        >>>     logging.basicConfig(level=logging.DEBUG)
        >>>     timer = Timer()
        >>>     timer()
        >>>     for _ in range(5):
        >>>         time.sleep(random.random())
        >>>         timer()
        >>>     timer.report()

        >>> print(f'{"type1: print":=^60}')
        >>> type1()
        >>> print(f'{"type2: logger":=^60}')
        >>> type2()
    """

    def __init__(self,
                 msg_tmpl='time cost of {}:\t{:>7.3f} s',
                 level=logging.INFO):
        """
        `self.logger` will not take effect if no logger set before,
        since logging.basicConfig takes logging.WARNING as default level.

        Args:
            msg_tmpl: template for returned timer info
        """
        self.reset()
        self.msg_tmpl = msg_tmpl
        self.level = level
        self.logger = logging.getLogger()

    def reset(self):
        """
        initialize records
        """
        self.start_time = time.time()
        self.records = OrderedDict()
        self.idx = 0

    def __call__(self, step_name='', level=None):
        step_name = step_name if step_name else str(self.idx)
        self.idx += 1
        current_time = time.time()
        t = current_time - self.start_time
        self.start_time = current_time
        self.records[step_name] = t

        rpl = self.msg_tmpl.format(f'step {step_name}', t)
        self.logger.log(self.level if not level else level, rpl)
        return rpl

    def __repr__(self):
        return self.records.items().__repr__()

    def __str__(self):
        return '\n'.join([self.msg_tmpl.format(f'step {k}', v)
                          for k, v in self.records.items()])

    def show(self):
        """
        log and return all time consumption stored in `self.records`
        """
        self.logger.log(self.level, '='*50)
        for k, v in self.records.items():
            self.logger.log(self.level, self.msg_tmpl.format(f'step {k}', v))
        return '\n'.join(['='*50, str(self)])

    def total(self):
        """
        log and return total time consumption
        """
        total = round(sum(self.records.values()), 3)
        rpl = self.msg_tmpl.format('total', total)
        self.logger.log(self.level, rpl)
        return rpl

    def report(self):
        """
        log and return all time consumption stored in `self.records` and
        total time consumption
        """
        separate = self.show()
        total = self.total()
        rpl = '\n'.join((separate, total))
        return rpl
