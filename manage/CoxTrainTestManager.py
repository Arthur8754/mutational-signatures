import numpy as np
from sklearn.model_selection import KFold

class CoxTrainTestManager:
    """ 
    This class manages the train/test process for the Cox Model.

    ### Parameters :
    - model : the Cox Model
    """
    def __init__(self, model) -> None:
        self.model = model

    def leave_one_out_cross_validation(self, X: np.ndarray[np.ndarray[float]], y: np.ndarray[tuple[int, float]])->tuple[np.ndarray, np.ndarray]:
        """ 
        Make the one out cross validation to find the risk class for each sample.

        ### Parameters :
        - X : the features data of each sample.
        - y : the label of each sample, containing the event status and the time surviving for each sample.

        ### Returns :
        - The test risk class for each sample, after training.
        - The test risk score for each sample, after training.
        """
        # Sample array
        class_samples = np.zeros(y.shape)
        risk_score_samples = np.zeros(y.shape)

        # Split the index to n_splits folds
        n_samples = X.shape[0]
        folds = KFold(n_splits=n_samples, shuffle=True).split(X)
        
        # Train - cutoff - test for each fold
        for i, (train_index, test_index) in enumerate(folds):
            
            # Train
            self.model.train(X[train_index],y[train_index])

            # Predict train scores and find cutoff
            train_scores = self.model.predict_risk_score(X[train_index])
            cutoff = self.model.find_cutoff(train_scores)

            # Test
            risk_score_samples[test_index] = self.model.predict_risk_score(X[test_index])
            class_samples[test_index] = self.model.predict_class(risk_score_samples[test_index], cutoff)

        return class_samples, risk_score_samples

    # def train(self, n_epochs: int):
    #     """ 
    #     Train the model for n_epochs.
    #     """
    #     self.train_loss = []

    #     for epoch in range(n_epochs):

    #         # Clear gradients
    #         self.optimizer.zero_grad()

    #         # Forward pass
    #         out = self.model(self.trainset.x, self.trainset.edge_index)

    #         # Compute loss
    #         loss = self.loss_function(out, self.trainset.y)
    #         self.train_loss.append(loss.item())

    #         # Backward pass (gradients computation)
    #         loss.backward()

    #         # Update parameters
    #         self.optimizer.step()

    # def plot_loss(self, title: str, filename: str):
    #     """
    #     Plot the loss along epochs.
    #     """
    #     epochs = range(len(self.train_loss))
    #     fig, ax = plt.subplots(figsize=(10,7))
    #     ax.plot(epochs, self.train_loss, label='train loss')
    #     ax.set_xlabel("Epochs")
    #     ax.set_ylabel("Loss")
    #     ax.set_title(title)
    #     ax.legend()
    #     plt.savefig(filename)