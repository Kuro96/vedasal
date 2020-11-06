import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Specify config file')
    parser.add_argument('config', help='config file path')
    args = parser.parse_args()
    return args
