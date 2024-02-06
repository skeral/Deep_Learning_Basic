import sys
import os
from glob import glob

from tqdm import tqdm
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


from module.datasets import get_dataset
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

            ## TODO ##
            # Hint : get data by using pandas or glob 

            
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
            
            # load model
            
            # Loss function
            self.loss_fn = None

            # Optimizer
            self.optimizer = None

            # LR Scheduler
            self.lr_scheduler = None

        elif mode == "test":
            pass
    

    def train(self):
        self.model.to(self.config.device)
        
        # early stopping
        early_stopping = 0

        # metric
        best_acc = 0
        best_f1 = 0

        best_model = None
        
        for epoch in range(1, self.config.epochs+1):
            self.model.train()

            for batch in tqdm(self.train_dataloader):
                
                ## TODO ##
                # ----- Modify Example Code -----
                # following code is pesudo code
                # modify the code to fit your task 
                img = batch["img"]
                label = batch["label"]
                
                self.optimizer.zero_grad()
                pred = self.model(img)
                loss = self.loss_fn(pred, label)
                loss.backward()
                
                # calculate metric
                
                self.optimizer.step()
                # -------------------------------
                
            self._valid()
            # logging
            
            # save model
            
            if early_stopping >= 5:
                break
            
            
    def _valid(self):
        # metric

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader):
                img = batch["img"]
                label = batch["label"]

                pred = self.model(img)
                loss = self.loss_fn(pred, label)

                # logging
            

        
    
    