import torch
import torch.nn as nn

class MSELoss(nn.Module):

    def __init__(self, use_custom_loss=True):
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.use_custom_loss = use_custom_loss

    def forward(self, pred, target):
        '''
        Compute custom mean squared error between predicted output and targets.
        '''
        n_element = pred.numel()
        
        # Mean squared error
        mse_loss = self.criterion(pred, target)

        if self.use_custom_loss:
            # Continuous motion
            # diff = [abs(pred[:, n, :] - pred[:, n-1, :]) for n in range(1, pred.shape[1])]
            # cont_loss = torch.sum(torch.stack(diff)) / n_element
            # cont_loss /= 100 # ~0.1 -> 0.001

            # Motion variance
            norm = torch.norm(pred, 2, 1) # B x S x dim
            var_loss = -torch.sum(norm) / n_element
            var_loss /= 1 # ~0.1 -> 0.1

            # Rotation loss
            rotation_diff = abs(pred[:, :, :1])
            rotation_loss = torch.sum(rotation_diff) / n_element
            rotation_loss /= 1
            
            # loss = mse_loss + cont_loss + var_loss + rotation_loss
            total_loss = mse_loss + norm + rotation_loss
            
            return total_loss, mse_loss, norm, rotation_loss
        
        else:
            return mse_loss, None, None, None


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