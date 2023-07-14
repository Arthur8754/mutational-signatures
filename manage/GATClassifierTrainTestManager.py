import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split
from models.BuildGraph import BuildGraph

class GATClassifierTrainTestManager:
    """ 
    This class manages the train/test process for the GAT classifier
    """
    def __init__(self, model) -> None:
        """
        ### Parameters :
        - model : the GAT classifier
        """
        self.model = model

    def train(self, X: np.ndarray, y: np.ndarray, group: np.ndarray, n_epochs: int)->tuple[np.ndarray, np.ndarray]:
        """ 
        Train the model for n_epochs, with splitting in 80-20 train-validation set.  

        ### Parameters :
        - X (n_samples, n_features) : the features of each sample.
        - y (n_samples,) : the label of each sample.
        - group (n_samples,) : the graph group of each patient.
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
            X_train, X_val, y_train, y_val, group_train, group_val, index_train, index_val = train_test_split(X, y, group, [i for i in range(len(y))], test_size=0.2)
                       
            ## 2 : BUILD TRAINING GRAPH ## 

            # Building graph
            build_train_graph = BuildGraph(X_train, y_train, group_train)
            build_train_graph.build_graph(None,None,False)
            pyg_graph_train = build_train_graph.pyg_graph

            ## 3 : FORWARD PASS - BACKWARD PASS ON TRAINING SET ##

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            out_train = self.model.forward(pyg_graph_train.x, pyg_graph_train.edge_index)

            # Compute loss
            loss_train = loss_function(out_train, pyg_graph_train.y)
            train_loss.append(loss_train.item())

            # Backward pass (gradients computation)
            loss_train.backward()

            # Update parameters
            optimizer.step()

            ## 4 : BUILD VALIDATION GRAPH ##

            # Building graph
            build_val_graph = BuildGraph(X, y, group)
            build_val_graph.build_graph(None, None, False)
            pyg_graph_val = build_val_graph.pyg_graph

            ## 5 : FORWARD PASS IN VALIDATION SET AND STORE VALIDATION LOSS ## 

            out_val = self.model.forward(pyg_graph_val.x, pyg_graph_val.edge_index)[index_val]

            loss_val = loss_function(out_val, pyg_graph_val.y[index_val])
            val_loss.append(loss_val.item())

        return train_loss, val_loss


    def leave_one_out_cross_validation(self, X: np.ndarray, y: np.ndarray, group: np.ndarray, n_epochs: int)->tuple[np.ndarray,np.ndarray]:
        """ 
        Make a 1-fold CV to determine test labels and scores of the cohort.

        ### Parameters :
        - X (n_samples, n_features) : the features of each patient
        - y (n_samples,) : the class of each patient.
        - group (n_samples,) : the group of each patient (for the graph).
        - n_epoch : the number of epoch for each training.
        - distance_measure (euclidean, cosine, or manhattan) : the distance used when we prune the graph
        - max_neighbors : the maximum of neighbors per node.

        ### Returns :
        - The test score of each patient
        - The test class of each patient
        - The mean train loss along epochs on the CV steps
        - The mean validation loss along epochs on the CV steps
        - The standard deviation for train loss on the CV steps
        - The standard deviation for validation loss on the CV steps
        - The parameters of the convolutive layer of the model.
        - The parameters of the fully connected layer of the model.
        """

        # Split dataframe in n_samples groups
        n_samples = X.shape[0]
        folds = KFold(n_splits=n_samples, shuffle=True).split(X)

        # Initialize list of test risk scores and classes for each patient
        test_scores = np.zeros(X.shape[0])
        test_classes = np.zeros(X.shape[0])

        # Save weights
        params_attention, params_conv, params_linear = [],[],[]

        # Train loss / val loss for each step of leave one out cross validation
        train_losses, val_losses = [],[]

        for i, (train_index, test_index) in enumerate(folds):
            
            # Select train set and test set
            X_train, y_train, group_train = X[train_index], y[train_index], group[train_index]
            X_test, y_test, group_test = X[test_index], y[test_index], group[test_index]

            ## 1 : TRAIN ##

            # Training in train set
            train_loss, val_loss = self.train(X_train, y_train, group_train, n_epochs)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Save weights
            params_attention.append(list(self.model.parameters())[2])
            params_conv.append(list(self.model.parameters())[4])
            params_linear.append(list(self.model.parameters())[6])

            ## 2 : TEST ##

            # Build test graph
            build_test_graph = BuildGraph(X, y, group)
            build_test_graph.build_graph(None, None, False)
            pyg_graph_test = build_test_graph.pyg_graph

            ## Scores and response class prediction
            score_test = self.model.forward(pyg_graph_test.x, pyg_graph_test.edge_index).detach().numpy().reshape((1,-1))[0]
            class_test = self.model.predict_class(pyg_graph_test.x, pyg_graph_test.edge_index).detach().numpy().reshape((1,-1))[0]

            test_scores[test_index] = score_test[test_index]
            test_classes[test_index] = class_test[test_index]

        return test_scores, test_classes, np.mean(train_losses,axis=0), np.mean(val_losses,axis=0), np.std(train_losses, axis=0), np.std(val_losses, axis=0), params_attention, params_conv, params_linear