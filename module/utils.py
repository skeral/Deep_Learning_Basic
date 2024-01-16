class Config:
    def __init__(self, args):
        self.epochs = args.epochs
        
    def __str__(self):
        attr = vars(self)
        return "\n".join(f"{key}: {value}" for key, value in attr.items())
        
        