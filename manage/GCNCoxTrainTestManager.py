import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score

class GCNCoxTrainTestManager:
    """ 
    This class manages the train/test process for a given model.
    """
    def __init__(self, model, trainset, status, time) -> None:
        """
        ### Parameters :
        - model : the GCN model to train.
        - train : the graph used for training
        - status : the status event for each patient
        - time : the time event for each patient
        """
        self.model = model
        self.trainset = trainset
        self.status = status
        self.time = time
        self.train_loss = []

    def train(self, n_epochs: int):
        """ 
        Train the model for n_epochs.
        """
        self.train_loss = []

        for epoch in range(n_epochs):
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1} of {n_epochs}")

            # Forward pass
            out = self.model(self.trainset.x, self.trainset.edge_index)

            # Compute loss
            loss = self.model.npll_loss(out, self.status, self.time)
            self.train_loss.append(loss.item())

            # Backward pass (gradients computation)
            loss.backward()

            # Update parameters
            with torch.no_grad():
                for param in self.model.parameters():
                    new_param = param - 1*param.grad 
                    param.copy_(new_param)
                    param.grad.zero_()

        print("End of training.")

    def plot_loss(self):
        """
        Plot the loss along epochs.
        """
        epochs = range(len(self.train_loss))
        fig, ax = plt.subplots(figsize=(10,7))
        ax.plot(epochs, self.train_loss, label='train loss')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title("Train loss")
        ax.legend()
        plt.savefig("loss.png")