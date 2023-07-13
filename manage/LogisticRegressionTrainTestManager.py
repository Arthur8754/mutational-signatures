import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split

class LogisticRegressionTrainTestManager:
    """ 
    This class manages the train/test process for the GCN classifier
    """
    def __init__(self, model) -> None:
        """
        ### Parameters :
        - model : the Logistic Regression classifier
        """
        self.model = model

    def train(self, X: np.ndarray, y: np.ndarray, n_epochs: int)->tuple[np.ndarray, np.ndarray]:
        """ 
        Train the model for n_epochs, with splitting in 80-20 train-validation set.  

        ### Parameters :
        - X (n_samples, n_features) : the features of each sample.
        - y (n_samples,) : the label of each sample.
        - n_epochs : the number of epochs.

        ### Returns :
        - The train loss for each epoch
        - The validation loss for each epoch
        """
        
        # Initialize train loss and validation loss arrays
        train_loss, val_loss = [],[]

        # Define loss function and optimizer
        loss_function = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(),lr=0.01)

        for epoch in range(n_epochs):

            ## 1 : SPLIT TRAIN VALIDATION SET (80-20) ## 
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

            # Convert in tensor
            X_train_torch, X_val_torch = torch.from_numpy(X_train).float(), torch.from_numpy(X_val).float()
            y_train_torch, y_val_torch = torch.from_numpy(y_train).float().unsqueeze(1), torch.from_numpy(y_val).float().unsqueeze(1)

            ## 2 : FORWARD PASS - BACKWARD PASS ON TRAINING SET ##

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            out_train = self.model.forward(X_train_torch)

            # Compute loss
            loss_train = loss_function(out_train, y_train_torch)
            train_loss.append(loss_train.item())

            # Backward pass (gradients computation)
            loss_train.backward()

            # Update parameters
            optimizer.step()

            ## 5 : FORWARD PASS IN VALIDATION SET AND STORE VALIDATION LOSS ## 

            out_val = self.model.forward(X_val_torch)

            loss_val = loss_function(out_val, y_val_torch)
            val_loss.append(loss_val.item())

        return train_loss, val_loss

    def leave_one_out_cross_validation(self, X: np.ndarray, y: np.ndarray, n_epochs: int)->tuple[np.ndarray,np.ndarray]:
        """ 
        Make a 1-fold CV to determine test labels and scores of the cohort.

        ### Parameters :
        - X (n_samples, n_features) : the features of each patient
        - y (n_samples,) : the class of each patient.
        - n_epoch : the number of epoch for each training.

        ### Returns :
        - The test score of each patient
        - The test class of each patient
        - The mean train loss along epochs on the CV steps
        - The mean validation loss along epochs on the CV steps
        - The standard deviation for train loss on the CV steps
        - The standard deviation for validation loss on the CV steps
        - The parameters of the logistic regression model.
        """

        # Split dataframe in n_samples groups
        n_samples = X.shape[0]
        folds = KFold(n_splits=n_samples, shuffle=True).split(X)

        # Initialize list of test risk scores and classes for each patient
        test_scores = np.zeros(X.shape[0])
        test_classes = np.zeros(X.shape[0])

        # Save weights
        params_linear = []

        # Train loss / val loss for each step of leave one out cross validation
        train_losses, val_losses = [],[]

        for i, (train_index, test_index) in enumerate(folds):
            
            # Select train set and test set
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            ## 1 : TRAIN ##

            # Training in train set
            train_loss, val_loss = self.train(X_train, y_train, n_epochs)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Save weights
            params_linear.append(list(self.model.parameters())[0])

            ## 2 : TEST ##
            
            # Convert in tensor
            X_test_torch = torch.from_numpy(X_test).float()

            ## Scores and response class prediction
            score_test = self.model.forward(X_test_torch).detach().numpy()[0]
            class_test = self.model.predict_class(X_test_torch).detach().numpy()[0]

            test_scores[test_index] = score_test
            test_classes[test_index] = class_test

        return test_scores, test_classes, np.mean(train_losses,axis=0), np.mean(val_losses,axis=0), np.std(train_losses, axis=0), np.std(val_losses, axis=0), params_linear