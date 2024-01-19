from glob import glob


from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
from torch.utils.data import DataLoader

from module.dataset import get_dataset
from module.models import get_model
from module.utils import Config, seed_everything
from module.log import get_logger

class Trainer:
    def __init__(self, config: Config):
        self.config = config
    
    def setup(self, mode="train"):
        """
        you need to code how to get data
        and define dataset, dataloader, transform in this function
        """
        if mode == "train":

            seed_everything(self.config.seed)

            self.logger = get_logger(
                name="tensorboard",
                log_dir=f"{self.config.log_dir}",    
            )
            
            # Train
            train_transform = A.Compose([
                # add augmentation
                A.Normalize(),
                ToTensorV2()
            ])

            train_dataset = get_dataset(
                "custom",
                img_paths=[],
                labels=[],
                transforms=train_transform
            )

            self.train_dataloader = DataLoader(
                dataset=train_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                shuffle=True,
            )
            
            # Validation
            val_transform = A.Compose([
                A.Normalize(),
                ToTensorV2()
            ])

            val_dataset = get_dataset(
                "custom",
                img_paths=[],
                labels=[],
                transforms=val_transform
            )         

            self.val_dataloader = DataLoader(
                dataset=val_dataset,
                batch_size=self.config.batch_size * 2,
                num_workers=self.config.num_workers,
                shuffle=False,
            )

            # Model
            self.model = get_model("custom")

            # Optimizer

            # LR Scheduler

        elif mode == "test":
            pass
    
    def train(self):
        for epoch in range(1, self.config.epochs+1):
            self.model.train()
            for batch in self.train_dataloader:
                self._valid()
    
    def _valid(self):
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_dataloader:
                pass
        
    
    