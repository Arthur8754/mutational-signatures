"""
https://github.com/havakv/pycox
"""

import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv

class GCNCoxModel(torch.nn.Module):
    """
    This class implements a Cox regression model with using Graph Convolutional Network.
    """
    def __init__(self, num_features: int) -> None:
        """
        - num_features : the number of features for each patient
        """
        super().__init__()

        # Convolutive layer :
        self.conv = GCNConv(in_channels=num_features, out_channels=num_features)

        # Regression layer :
        self.regression = Linear(in_features=num_features, out_features=1)

    def forward(self, x, edge_index):

        # Convolution
        h = torch.relu(self.conv(x, edge_index))

        # Regression layer
        out = self.regression(h)

        return out
    
    def npll_loss(self, out: torch.Tensor, status: torch.Tensor, time: torch.Tensor)->torch.Tensor:
        """
        Compute the Negative Partial Log Likelihood.

        ### Parameters :
        - out : the output of the GCN for each patient (g function in Cox Model)
        - status : the status event for each patient
        - time : the time event for each patient

        ### Returns :
        The NPLL loss.
        """
        loss = torch.zeros(1)

        for i in range(out.shape[0]):
            status_i, Ti, out_i = status[i], time[i], out[i]
            to_sum = torch.where((status==1) & (time>=Ti))[0]
            loss += status_i*torch.log(torch.sum(torch.exp(out[to_sum]-out_i)))/out.shape[0]
        
        return loss
