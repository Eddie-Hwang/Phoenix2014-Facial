import torch
import torch.nn as nn

class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pred, target):
        '''
        Compute mean squared error between predicted output and targets.
        '''
        mse_loss = self.criterion(pred, target)
        loss = mse_loss
        
        return loss


class CrossEntropyLoss(nn.Module):

    def __init__(self, pad_idx):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_idx
        )

    def forward(self, pred, target):
        entropy_loss = self.criterion(pred, target)
        loss = entropy_loss

        return loss