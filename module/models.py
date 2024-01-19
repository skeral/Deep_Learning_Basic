import torch.nn as nn

def get_model(name: str, **kwargs):
    if name == "custom":
        return CustomModel(**kwargs)
    else:
        raise ValueError("Incorrect Name")


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # define DL layer
    
    def forward(self, x):
        # using module that you define in __init__ and check whether it's right sequence
        return x