from typing import Union, List

import numpy as np

from torch.utils.data import Dataset

def get_dataset(name: str):
    if name == "custom":
        return CustomDataset()
    else:
        raise ValueError("Incorrect Name")
    

class CustomDataset(Dataset):
    """
    CustomDataset for Image Data
    """
    def __init__(
            self,
            img_paths: Union[List, np.ndarray], 
            labels: Union[List, np.ndarray], 
            transforms = None
        ):

        self.img_paths = img_paths
        self.labels = labels
        self.transforms = transforms
    
    def __getitem__(self, index):
        # Hint :
        # get image by using opencv-python or pillow library
        # return image and label(you can return as tuple or dictionary type)
        pass
    
    def __len__(self):
        # Hint : return labels or img_paths length
        pass


if __name__ == "__main__":
    # Check if CustomDataset is working
    pass