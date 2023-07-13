import torch
from torch.nn import Linear

class LogisticRegression(torch.nn.Module):
    """
    This class implements a simple logistic regression with PyTorch.

    ### Parameters :
    - num_features : the number of features for each patient
    """
    def __init__(self, num_features: int) -> None:
        super().__init__()

        # Classifier layer
        self.linear = Linear(in_features=num_features, out_features=1)

    def forward(self, x):

        # Response probability
        out = torch.sigmoid(self.linear(x))
        return out
    
    def predict_class(self, x):
        out = self.forward(x)
        return torch.where(out>=0.5, 1, 0)
    
    def train(self, n_epochs, x, edge_index, y, loss_function, optimizer):
        """ 
        Train the model for n_epochs.
        """
        train_loss = []

        for epoch in range(n_epochs):

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            out = self.forward(x, edge_index)

            # Compute loss
            loss = loss_function(out, y)
            train_loss.append(loss.item())

            # Backward pass (gradients computation)
            loss.backward()

            # Update parameters
            optimizer.step()

        return train_loss
