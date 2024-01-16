import argparse

from module.utils import Config
from module.trainer import Trainer


def run():
    pass    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    
    config = Config(args)
    # run()