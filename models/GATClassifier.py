import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATv2Conv

class GATClassifier(torch.nn.Module):
    """
    This class implements a Graph Attention Networks.

    ### Parameters :
    - num_features : the number of features for each patient
    """
    def __init__(self, num_features: int) -> None:
        super().__init__()

        # Attention and convolutive layer
        self.attention = GATv2Conv(in_channels=num_features, out_channels=num_features)

        # Classifier layer
        self.linear = Linear(in_features=num_features, out_features=1)

    def forward(self, x, edge_index):

        # Attention and convolutive
        h = self.attention(x, edge_index)

        # Response probability
        out = torch.sigmoid(self.linear(h))

        return out
    
    def predict_class(self, x, edge_index):
        out = self.forward(x, edge_index)
        return torch.where(out>=0.5, 1, 0)

    def forward_conv(self, x, edge_index):
        return self.attention(x, edge_index)
