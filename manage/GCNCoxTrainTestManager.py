import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import KFold
from models.BuildGraph import BuildGraph
from torch_geometric.utils import from_networkx

class GCNCoxTrainTestManager:
    """ 
    This class manages the train/test process for the GCN-Cox model.
    """
    def __init__(self, gcn_model, cox_model) -> None:
        """
        ### Parameters :
        - gcn_model
        - cox_model
        """
        self.gcn_model = gcn_model
        self.cox_model = cox_model

    def leave_one_out_cross_validation(self, df: pd.DataFrame, features_name: list[str], label_name: str, n_epoch: int):
        """ 
        Make the leave one out cross validation to train-test GNN+Cox model pipeline.

        ### Parameters :
        - df : the dataframe.
        - column_name : the name of the column used to choose graph connections.
        - features_name : the name of the features for each patient
        - label_name : the name of the label for each patient
        - n_epoch : the number of epochs for GNN training.

        ### Returns :
        - the risk scores for each test patient
        - the risk classes for each test patient
        - the Cox Model trained (for the metrics later)
        """

        # Split dataframe in n_samples groups
        n_samples = df.shape[0]
        folds = KFold(n_splits=n_samples, shuffle=True).split(df)

        # Initialize list of test risk scores and classes for each patient
        risk_scores = np.zeros(df.shape[0])
        risk_classes = np.zeros(df.shape[0])

        for i, (train_index, test_index) in enumerate(folds):

            # Select train set and test set
            df_train, df_test = df.iloc[train_index,:], df.iloc[test_index,:]

            ### 1 : TRAIN ###

            ## 1.1 : Build graph ##

            # Instanciate graph builder
            build_graph_train = BuildGraph(df_train)

            # Split patients per KMeans clusters.
            # kmeans = KMeans(n_clusters=10, n_init=10)
            # labels = build_graph_train.apply_clustering(kmeans, ["CD8+ T cell score","Exome mut per mb"])
            labels = df_train["Tumour type"].to_numpy()

            # Compute adjacency matrix
            build_graph_train.compute_adjacency_matrix(labels)

            # Create graph
            build_graph_train.create_graph(features_name, label_name)

            # Convert graph to PyTorch geometric format
            pyg_graph_train = from_networkx(build_graph_train.G)

            ## 1.2 : Train the GNN classifier ##

            # Instanciate the train manager, with loss and optimizer
            loss_gnn = torch.nn.BCELoss()
            optimizer_gnn = torch.optim.Adam(self.gcn_model.parameters(),lr=0.01)
            # train_manager = GCNTrainTestManager(gcn_classifier, pyg_graph_train, loss_gnn, optimizer_gnn)

            # Training on num_epoch
            train_losses = self.gcn_model.train(n_epoch,pyg_graph_train.x, pyg_graph_train.edge_index, pyg_graph_train.y, loss_gnn, optimizer_gnn)

            # Extract new embeddings
            df_learnt = pd.DataFrame(self.gcn_model.forward_conv(pyg_graph_train.x, pyg_graph_train.edge_index).detach().numpy(), columns=features_name, index=df_train.index)

            ## 1.3 : Train the Cox Model ##

            # Preprocessing data
            X_train = df_learnt.to_numpy()
            y_train = np.array(list((df_train[['Progression_1','Time to progression (days)']].itertuples(index=False, name=None))),dtype=[('Progression_1', '?'), ('Time to progression (days)', '<f8')])

            # Training 
            self.cox_model.train(X_train, y_train)

            # Find risk score cutoff between high risk and low risk
            risk_scores_train = self.cox_model.predict_risk_score(X_train)
            risk_cutoff = self.cox_model.find_cutoff(risk_scores_train)

            ### 2 : TEST ###

            ## 2.1 : Add patient to the graph (rebuild graph) ##

            # Instanciate graph builder
            build_graph_test = BuildGraph(pd.concat([df_train, df_test]))

            # Split patients per KMeans clusters.
            # kmeans = KMeans(n_clusters=10, n_init=10)
            # labels = build_graph_test.apply_clustering(kmeans, ["CD8+ T cell score","Exome mut per mb"])
            labels = pd.concat([df_train, df_test])["Tumour type"].to_numpy()

            # Compute adjacency matrix
            build_graph_test.compute_adjacency_matrix(labels)

            # Create graph
            build_graph_test.create_graph(features_name, label_name)

            # Convert graph to PyTorch geometric format
            pyg_graph_test = from_networkx(build_graph_test.G)

            ## 2.2 : GNN embedding prediction ##

            # Predict new embedding of test set
            new_test_embedding = pd.DataFrame(self.gcn_model.forward_conv(pyg_graph_test.x, pyg_graph_test.edge_index).detach().numpy()[-1:], columns=features_name, index=df_test.index)

            ## 2.3 : Cox Model prediction ##

            # Preprocessing data
            X_test = new_test_embedding.to_numpy()

            # Predict risk score and risk class
            risk_score_test = self.cox_model.predict_risk_score(X_test)
            risk_class_test = self.cox_model.predict_class(risk_score_test, risk_cutoff)

            risk_scores[test_index] = risk_score_test
            risk_classes[test_index] = risk_class_test

        return risk_scores, risk_classes

    # def train(self, n_epochs: int):
    #     """ 
    #     Train the model for n_epochs.
    #     """
    #     self.train_loss = []

    #     for epoch in range(n_epochs):
            
    #         if epoch % 10 == 0:
    #             print(f"Epoch {epoch+1} of {n_epochs}")

    #         # Forward pass
    #         out = self.model(self.trainset.x, self.trainset.edge_index)

    #         # Compute loss
    #         loss = self.model.npll_loss(out, self.status, self.time)
    #         print(loss)
    #         self.train_loss.append(loss.item())

    #         # Backward pass (gradients computation)
    #         loss.backward()

    #         # Update parameters
    #         with torch.no_grad():
    #             for param in self.model.parameters():
    #                 new_param = param - 0.01*param.grad 
    #                 param.copy_(new_param)
    #                 param.grad.zero_()

    #     print("End of training.")

    # def plot_loss(self):
    #     """
    #     Plot the loss along epochs.
    #     """
    #     epochs = range(len(self.train_loss))
    #     fig, ax = plt.subplots(figsize=(10,7))
    #     ax.plot(epochs, self.train_loss, label='train loss')
    #     ax.set_xlabel("Epochs")
    #     ax.set_ylabel("Loss")
    #     ax.set_title("Train loss")
    #     ax.legend()
    #     plt.savefig("loss.png")