from typing import Union, List

import numpy as np
import cv2

from torch.utils.data import Dataset

def get_dataset(name: str, **kwargs):
    if name == "custom":
        return CustomImageDataset(**kwargs)
    else:
        raise ValueError("Incorrect Name")
    

class CustomImageDataset(Dataset):
    """
    CustomDataset for Image Data

    Args
    img_paths: image path list
    labels: image data label list
    transforms : transform instance made by albumentations
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

        ## TODO ##
        # ----- Modify Example Code -----
        img_path = self.img_paths[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
        
        return {
            "img": img,
            "label": self.labels[index]
        }
        # -------------------------------

    def __len__(self):
        # Hint : return labels or img_paths length
        return len(self.img_paths)


if __name__ == "__main__":
    # Check if CustomDataset is working
    pass