import torch
from torch import nn

class BiFPN_Concat(nn.Module):
    """
    Weighted Feature Fusion for BiFPN
    """
    def __init__(self, dimension=1):
        super(BiFPN_Concat, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = torch.relu(self.w)
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        # x is a list of two tensors
        return weight[0] * x[0] + weight[1] * x[1]

class BiFPN_Concat_3(nn.Module):
    """
    Weighted Feature Fusion for BiFPN (3 inputs)
    """
    def __init__(self, dimension=1):
        super(BiFPN_Concat_3, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = torch.relu(self.w)
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        return weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]
