import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans, AgglomerativeClustering

class BuildGraph:
    """ 
    This class creates a graph from an initial dataframe. The graph is an union of distinct complete sub graphs, where each sub graph 
    contains the patients with the same value of a specific column.
    """
    def __init__(self, df: pd.DataFrame) -> None:
        """ 
        - df : the dataframe used to build the graph.
        """
        self.df = df
        self.A = np.zeros((df.shape[0],df.shape[0]))
        self.G = None

    def compute_adjacency_matrix(self, column_name: str)->None:
        """ 
        Compute the adjacency matrix of the graph, with splitting per column_name values.

        ### Parameters :
        - column_name : the name of column used to make the split
        - min_rows : the minimum number of patients per sub graphs.

        ### Returns :
        None
        """
        # Initialize matrix with 0
        shape_A = self.df.shape[0]

        # For each patient, connect with patient with the same column_name value.
        column_values = self.df[column_name].to_numpy()
        for i in range(shape_A):
            value = column_values[i]
            self.A[i] = np.where(column_values==value,1,0)

        self.A = self.A - np.identity(shape_A)

    def compute_adjacency_matrix_kmeans(self, n_clusters: int, columns_names: list[str])->None:
        """ 
        Compute the adjacency matrix of the graph, with splitting per KMeans clusters. Each is a fully connected graph.

        ### Parameters :
        - n_clusters : the number of fully connected graphs.
        - columns_names : the names of the columns used as KMeans features.

        ### Returns :
        None
        """
        shape_A = self.df.shape[0]

        # Instanciate the KMeans model
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)

        # Select features columns for future clustering
        values = self.df.loc[:,columns_names].to_numpy()

        # Apply clusters and extract cluster labels.
        labels = kmeans.fit(values).labels_

        # Split per labels
        for i in range(shape_A):
            label = labels[i]
            self.A[i] = np.where(labels==label,1,0)

        self.A = self.A - np.identity(shape_A)

    def compute_adjacency_matrix_hierarchical(self, n_clusters: int, columns_names: list[str])->None:
        """ 
        Compute the adjacency matrix of the graph, with splitting per hierarchical clustering clusters. Each is a fully connected graph.

        ### Parameters :
        - n_clusters : the number of fully connected graphs.
        - columns_names : the names of the columns used as Hierarchical clustering features.

        ### Returns :
        None
        """
        shape_A = self.df.shape[0]

        # Instanciate the KMeans model
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters)

        # Select features columns for future clustering
        values = self.df.loc[:,columns_names].to_numpy()

        # Apply clusters and extract cluster labels.
        labels = agglomerative.fit(values).labels_

        # Split per labels
        for i in range(shape_A):
            label = labels[i]
            self.A[i] = np.where(labels==label,1,0)

        self.A = self.A - np.identity(shape_A)

    def create_graph(self, features_name: list[str], label_name: str)->None:
        """ 
        Create the networkx graph from the adjacency matrix.

        ### Parameters :
        - features_name : the list of the names of the patient's features to integrate in the graph
        - label_name : the name of the label column to integrate in the graph.

        ### Returns :
        None
        """
        X = torch.from_numpy(self.df.loc[:,features_name].to_numpy()).float()
        y = torch.from_numpy(self.df[label_name].to_numpy()).float().unsqueeze(1)

        # Initialize with empty graph
        self.G = nx.Graph()
        
        # Add nodes, with its features and label
        for i in range(self.A.shape[0]):
            self.G.add_nodes_from([(i,{"x":X[i],"y":y[i]})])

        # Add edges
        rows, cols = np.where(self.A == 1)
        edges = zip(rows.tolist(), cols.tolist())
        self.G.add_edges_from(edges)
    
    def show_graph(self, title: str, filename: str)->None:
        """ 
        Create the graph from the adjacency matrix and plot it.

        ### Parameters :
        - A (n_samples, n_samples) : the adjacency matrix of the graph
        - title : the title of the figure
        - filename : the location to store the figure

        ### Returns :
        None
        """
        fig, ax = plt.subplots(figsize=(10,7))
        nx.draw(self.G)
        plt.title(title)
        plt.savefig(filename)


