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

    def train(self, n_epochs: int):
        """ 
        Train the model for n_epochs.
        """
        for epoch in range(n_epochs):
            
            print(f"Epoch {epoch+1} of {n_epochs}")
            # Clear gradients
            self.optimizer.zero_grad()

            # Forward pass
            out, h = self.model(self.trainset.x, self.trainset.edge_index)

            # Compute loss
            loss = self.loss_function(out, self.trainset.y)

            # Backward pass (gradients computation)
            loss.backward()

            # Update parameters
            self.optimizer.step()

        print("End of training.")