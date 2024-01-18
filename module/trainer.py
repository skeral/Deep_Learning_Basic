import torch

from module.utils import Config, seed_everything
from module.log import get_logger

class Trainer:
    def __init__(self, config: Config):
        self.config = config
    
    def setup(self, mode="train"):
        if mode == "train":
            seed_everything(self.config.seed)
            self.logger = get_logger(
                name="tensorboard",
                log_dir=f"{self.config.log_dir}",    
            )
        else:
            pass
        ## TODO ##
        # define dataset
        # define transforms
        # define self.dataloader
        
        
    
    def train(self):
        pass
    
    def _valid(self):
        pass
        
    
    