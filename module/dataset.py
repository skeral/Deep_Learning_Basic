from torch.utils.data import Dataset

def get_dataset(name: str):
    if name == "custom":
        pass
    else:
        raise ValueError("Incorrect Name")
    

class CustomDataset(Dataset):
    def __init__(self):
        pass
    
    def __getitem__(self, index):
        pass
    
    def __len__(self):
        pass