import argparse

import _init_paths  # noqa: F401
from kurosal.models.sal_detectors import VanillaSal


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a video saliency detection model')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_fp = args.config

    runner = assemble(cfg_fp)
    runner.run()


if __name__ == "__main__":
    main()
