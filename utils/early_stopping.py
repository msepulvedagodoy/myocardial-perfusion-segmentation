import torch

class EarlyStopping(object):

    def __init__(self, patience=10, delta=0) -> None:
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False


    def __call__(self):
        pass
    

    