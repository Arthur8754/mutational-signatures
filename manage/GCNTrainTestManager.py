import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score

class GCNTrainTestManager:
    """ 
    This class manages the train/test process for a given model.
    """
    def __init__(self, model, trainset, loss_function, optimizer) -> None:
        """
        ### Parameters :
        - model : the GCN model to train.
        - train : the graph used for training
        - loss : the loss to optimize
        - optimizer : the optimizer algorithm to minimize the loss.
        """
        self.model = model
        self.trainset = trainset
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.train_loss = []

    def train(self, n_epochs: int):
        """ 
        Train the model for n_epochs.
        """
        self.train_loss = []

        for epoch in range(n_epochs):
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1} of {n_epochs}")

            # Clear gradients
            self.optimizer.zero_grad()

            # Forward pass
            out = self.model(self.trainset.x, self.trainset.edge_index)

            # Compute loss
            loss = self.loss_function(out, self.trainset.y)
            self.train_loss.append(loss.item())

            # Backward pass (gradients computation)
            loss.backward()

            # Update parameters
            self.optimizer.step()

        print("End of training.")

    def plot_loss(self, title: str, filename: str):
        """
        Plot the loss along epochs.
        """
        epochs = range(len(self.train_loss))
        fig, ax = plt.subplots(figsize=(10,7))
        ax.plot(epochs, self.train_loss, label='train loss')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
        plt.savefig(filename)