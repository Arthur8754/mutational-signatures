import numpy as np
import torch
from sklearn.model_selection import KFold
from models.BuildGraph import BuildGraph
from torch_geometric.utils import from_networkx
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, manhattan_distances

class GCNClassifierTrainTestManager:
    """ 
    This class manages the train/test process for the GCN classifier
    """
    def __init__(self, model) -> None:
        """
        ### Parameters :
        - model : the GCN classifier
        """
        self.model = model

    def leave_one_out_cross_validation(self, X: np.ndarray, y: np.ndarray, group: np.ndarray, n_epoch: int, distance_measure: str, max_neighbors: int)->tuple[np.ndarray,np.ndarray]:
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
        - The test classes of each patient
        - The test scores of each patient
        """

        # Split dataframe in n_samples groups
        n_samples = X.shape[0]
        folds = KFold(n_splits=n_samples, shuffle=True).split(X)

        # Initialize list of test risk scores and classes for each patient
        test_scores = np.zeros(X.shape[0])
        test_classes = np.zeros(X.shape[0])

        # Save weights
        params_conv,params_linear = [],[]

        for i, (train_index, test_index) in enumerate(folds):

            # Select train set and test set
            X_train, y_train, group_train = X[train_index], y[train_index], group[train_index]
            X_test, y_test, group_test = X[test_index], y[test_index], group[test_index]

            ### 1 : TRAIN ###

            ## 1.1 : Build graph ##

            # Instanciate graph builder
            build_graph_train = BuildGraph(X_train, y_train, group_train)

            # Compute adjacency matrix
            build_graph_train.compute_adjacency_matrix()

            # Create graph
            build_graph_train.create_graph()

            # Prune graph
            if distance_measure == "euclidean":
                distance_matrix_train = euclidean_distances(X_train)
            elif distance_measure == "cosine":
                distance_matrix_train = cosine_distances(X_train)
            elif distance_measure == "manhattan":
                distance_matrix_train = manhattan_distances(X_train)
            else:
                raise ValueError(f"distance_measure = {distance_measure} is not valid. Possible values are 'euclidean', 'cosine' and 'manhattan'.")

            build_graph_train.prune_graph(distance_matrix_train, max_neighbors)

            # Convert graph to PyTorch geometric format
            pyg_graph_train = from_networkx(build_graph_train.G)

            ## 1.2 : Train the GNN classifier ##

            # Instanciate the train manager, with loss and optimizer
            loss_gnn = torch.nn.BCELoss()
            optimizer_gnn = torch.optim.Adam(self.model.parameters(),lr=0.01)

            # Training on num_epoch
            train_losses = self.model.train(n_epoch,pyg_graph_train.x, pyg_graph_train.edge_index, pyg_graph_train.y, loss_gnn, optimizer_gnn)

            # Save weights
            params_conv.append(list(self.model.parameters())[1])
            params_linear.append(list(self.model.parameters())[2])

            ### 2 : TEST ###

            ## 2.1 : Add patient to the graph (rebuild graph) ##

            # Instanciate graph builder
            build_graph_test = BuildGraph(X, y, group)

            # Compute adjacency matrix
            build_graph_test.compute_adjacency_matrix()

            # Create graph
            build_graph_test.create_graph()

            # Prune graph
            if distance_measure == "euclidean":
                distance_matrix_test = euclidean_distances(X)
            elif distance_measure == "cosine":
                distance_matrix_test = cosine_distances(X)
            elif distance_measure == "manhattan":
                distance_matrix_test = manhattan_distances(X)
            else:
                raise ValueError

            build_graph_test.prune_graph(distance_matrix_test, max_neighbors)

            # Convert graph to PyTorch geometric format
            pyg_graph_test = from_networkx(build_graph_test.G)

            ## 2.2 : scores and response class prediction
            score_test = self.model.forward(pyg_graph_test.x, pyg_graph_test.edge_index).detach().numpy().reshape((1,-1))[0]
            class_test = self.model.predict_class(pyg_graph_test.x, pyg_graph_test.edge_index).detach().numpy().reshape((1,-1))[0]

            test_scores[test_index] = score_test[test_index]
            test_classes[test_index] = class_test[test_index]

        return test_scores, test_classes, params_conv, params_linear