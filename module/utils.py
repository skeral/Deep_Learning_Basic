import torch


class Config:
    def __init__(self, args):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # if you have more than 1 GPU, may be define variable for other GPU (actually using pytorch lightning is better)
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.lr = args.lr
        self.seed = args.seed
        self.data_path = args.data_path
        self.save_dir = args.save_dir
        
    def __str__(self):
        attr = vars(self)
        return "\n".join(f"{key}: {value}" for key, value in attr.items())


def seed_everything(seed):
    pass

        