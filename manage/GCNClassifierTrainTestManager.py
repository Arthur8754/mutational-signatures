import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split
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

        # Train loss / val loss
        train_losses, val_losses = [],[]

        for i, (train_index, test_index) in enumerate(folds):
            
            train_loss_cv, val_loss_cv = [],[]

            # Select train set and test set
            X_train, y_train, group_train = X[train_index], y[train_index], group[train_index]
            X_test, y_test, group_test = X[test_index], y[test_index], group[test_index]

            # Split train learning - train validation (80-20)
            X_train_train, X_train_val, y_train_train, y_train_val, group_train_train, group_train_val, index_train, index_val = train_test_split(X_train, y_train, group_train, [i for i in range(len(y_train))], test_size=0.2)

            ### 1 : TRAIN ###

            ## 1.2 : Train the GNN classifier ##

            # Instanciate the train manager, with loss and optimizer
            loss_gnn = torch.nn.BCELoss()
            optimizer_gnn = torch.optim.Adam(self.model.parameters(),lr=0.01)

            for epoch in range(n_epoch):

                ## BUILD TRAIN GRAPH ## 

                build_train_graph = BuildGraph(X_train_train, y_train_train, group_train_train)
                build_train_graph.build_graph(euclidean_distances(X_train_train),3)
                pyg_graph_train = build_train_graph.pyg_graph

                # # Instanciate graph builder
                # build_graph_train = BuildGraph(X_train_train, y_train_train, group_train_train)

                # # Compute adjacency matrix
                # build_graph_train.compute_adjacency_matrix()

                # # Create graph
                # build_graph_train.create_graph()

                # # Prune graph
                # if distance_measure == "euclidean":
                #     distance_matrix_train = euclidean_distances(X_train_train)
                # elif distance_measure == "cosine":
                #     distance_matrix_train = cosine_distances(X_train_train)
                # elif distance_measure == "manhattan":
                #     distance_matrix_train = manhattan_distances(X_train_train)
                # else:
                #     raise ValueError(f"distance_measure = {distance_measure} is not valid. Possible values are 'euclidean', 'cosine' and 'manhattan'.")

                # build_graph_train.prune_graph(distance_matrix_train, max_neighbors)

                # # Convert graph to PyTorch geometric format
                # pyg_graph_train = from_networkx(build_graph_train.G)

                ## FP - BP IN TRAIN GRAPH ##

                # Clear gradients
                optimizer_gnn.zero_grad()

                # Forward pass
                out_train = self.model.forward(pyg_graph_train.x, pyg_graph_train.edge_index)

                # Compute loss
                loss_train = loss_gnn(out_train, pyg_graph_train.y)
                train_loss_cv.append(loss_train.item())

                # Backward pass (gradients computation)
                loss_train.backward()

                # Update parameters
                optimizer_gnn.step()

                ## BUILD VALIDATION GRAPH ##

                build_val_graph = BuildGraph(X_train, y_train, group_train)
                build_val_graph.build_graph(euclidean_distances(X_train),3)
                pyg_graph_val = build_val_graph.pyg_graph

                out_val = self.model.forward(pyg_graph_val.x, pyg_graph_val.edge_index)[index_val]

                loss_val = loss_gnn(out_val, pyg_graph_val.y[index_val])
                val_loss_cv.append(loss_val.item())

                # # Instanciate graph builder
                # build_graph_val = BuildGraph(X_train, y_train, group_train)

                # # Compute adjacency matrix
                # build_graph_val.compute_adjacency_matrix()

                # # Create graph
                # build_graph_val.create_graph()

                # # Prune graph
                # if distance_measure == "euclidean":
                #     distance_matrix_val = euclidean_distances(X_train)
                # elif distance_measure == "cosine":
                #     distance_matrix_val = cosine_distances(X_train)
                # elif distance_measure == "manhattan":
                #     distance_matrix_val = manhattan_distances(X_train)
                # else:
                #     raise ValueError(f"distance_measure = {distance_measure} is not valid. Possible values are 'euclidean', 'cosine' and 'manhattan'.")

                # build_graph_val.prune_graph(distance_matrix_val, max_neighbors)

                # # Convert graph to PyTorch geometric format
                # pyg_graph_val = from_networkx(build_graph_val.G)

                # ## COMPUTE VALIDATION LOSS ## 

                # # Forward pass for validation nodes
                # out = self.model.forward(pyg_graph_val.x, pyg_graph_val.edge_index)

                # # Compute loss
                # loss = loss_gnn(out, y)
                # val_losses.append(loss.item())

            train_losses.append(train_loss_cv)
            val_losses.append(val_loss_cv)

            # Save weights
            params_conv.append(list(self.model.parameters())[1])
            params_linear.append(list(self.model.parameters())[2])

            ### 2 : TEST ###
            build_test_graph = BuildGraph(X, y, group)
            build_test_graph.build_graph(euclidean_distances(X),3)
            pyg_graph_test = build_test_graph.pyg_graph

            # ## 2.1 : Add patient to the graph (rebuild graph) ##

            # # Instanciate graph builder
            # build_graph_test = BuildGraph(X, y, group)

            # # Compute adjacency matrix
            # build_graph_test.compute_adjacency_matrix()

            # # Create graph
            # build_graph_test.create_graph()

            # # Prune graph
            # if distance_measure == "euclidean":
            #     distance_matrix_test = euclidean_distances(X)
            # elif distance_measure == "cosine":
            #     distance_matrix_test = cosine_distances(X)
            # elif distance_measure == "manhattan":
            #     distance_matrix_test = manhattan_distances(X)
            # else:
            #     raise ValueError

            # build_graph_test.prune_graph(distance_matrix_test, max_neighbors)

            # # Convert graph to PyTorch geometric format
            # pyg_graph_test = from_networkx(build_graph_test.G)

            ## 2.2 : scores and response class prediction
            score_test = self.model.forward(pyg_graph_test.x, pyg_graph_test.edge_index).detach().numpy().reshape((1,-1))[0]
            class_test = self.model.predict_class(pyg_graph_test.x, pyg_graph_test.edge_index).detach().numpy().reshape((1,-1))[0]

            test_scores[test_index] = score_test[test_index]
            test_classes[test_index] = class_test[test_index]

        return test_scores, test_classes, np.mean(train_losses,axis=0), np.mean(val_losses,axis=0), np.std(train_losses, axis=0), np.std(val_losses, axis=0), params_conv, params_linear