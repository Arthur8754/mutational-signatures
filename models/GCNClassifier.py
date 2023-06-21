import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv

class GCNClassifier(torch.nn.Module):
    """
    This class implements a Graph Convolutional Network.
    """
    def __init__(self, num_features: int) -> None:
        """
        - num_features : the number of features for each patient
        """
        super().__init__()

        # Convolutive layer
        self.conv = GCNConv(in_channels=num_features, out_channels=num_features)

        # Classifier layer
        self.linear = Linear(in_features=num_features, out_features=1)

    def forward(self, x, edge_index):

        # Convolution
        h = torch.relu(self.conv(x, edge_index))

        # Response probability
        out = torch.sigmoid(self.linear(h))

        return out

    def forward_conv(self, x, edge_index):
        return self.conv(x, edge_index)
