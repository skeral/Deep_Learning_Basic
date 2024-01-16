import torch.nn as nn

def get_model(name: str):
    if name == "custom":
        return CustomModel()
    else:
        raise ValueError("Incorrect Name")


class CustomModel(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, x):
        pass