import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, subject, seed):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, subject, seed)
        elif score > self.best_score + self.delta:
            self.counter += 1
            # print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, subject, seed)
            self.counter = 0
        
    def save_checkpoint(self, val_loss, model, path, subject, seed):
        # if self.verbose:
        #     print(
        #         f"Metric score decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...\n"
        #     )
        torch.save(model.state_dict(), path + "{}_seed{}_checkpoint.pth".format(subject,seed))
        self.val_loss_min = val_loss