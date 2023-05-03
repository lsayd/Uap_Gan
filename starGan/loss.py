import torch
import torch.nn as nn

class L2RegularizationLoss(nn.Module):
    def __init__(self, alpha):
        super(L2RegularizationLoss, self).__init__()
        self.alpha = alpha

    def forward(self, inputs):
        l2_loss = torch.sum(torch.pow(inputs, 2))
        l2_loss *= self.alpha
        return l2_loss