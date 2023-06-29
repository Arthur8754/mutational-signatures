import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch

class BuildGraph:
    """ 
    This class creates a graph from an initial dataframe.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, group: np.ndarray) -> None:
        """ 
        - X (n_samples, n_features) : the features of each node of the graph
        - y (n_samples, ) : the label of each node of the graph
        - group (n_samples, ) : the group of each node of the graph
        """
        self.X = X
        self.y = y
        self.group = group
        self.A = np.zeros((X.shape[0],X.shape[0]))
        self.G = None
    
    def compute_adjacency_matrix(self)->None:
        """ 
        Compute the adjacency matrix of the graph, with splitting per label.

        ### Parameters :
        - labels : the label of each patient

        ### Returns :
        None
        """
        shape_A = self.X.shape[0]

        for i in range(shape_A):
            group = self.group[i]
            self.A[i] = np.where(self.group==group,1,0)

    def create_graph(self)->None:
        """ 
        Create the networkx graph from the adjacency matrix.

        ### Parameters :
        - features_name : the list of the names of the patient's features to integrate in the graph
        - label_name : the name of the label column to integrate in the graph.

        ### Returns :
        None
        """
        X = torch.from_numpy(self.X).float()
        y = torch.from_numpy(self.y).float().unsqueeze(1)

        # Initialize with empty graph
        self.G = nx.Graph()
        
        # Add nodes, with its features and label
        for i in range(self.A.shape[0]):
            self.G.add_nodes_from([(i,{"x":X[i],"y":y[i]})])

        # Add edges
        rows, cols = np.where(self.A == 1)
        edges = zip(rows.tolist(), cols.tolist())
        self.G.add_edges_from(edges)

    def prune_graph(self, distance_matrix: np.ndarray, max_neighbors: int)->None:
        """ 
        Prune the graph, to have max max_neighbors neighbors per node.

        ### Parameters :
        - distance_matrix (n_samples, n_samples): the distance between nodes. If we need to drop nodes, we drop the edges between the most distant nodes.
        - max_neighbors : the expected max of neighbors per node.
        """

        for i in range(distance_matrix.shape[0]):
            
            # Get neighbors of node i
            neighbors_i = [n for n in self.G[i]]

            # Distances from i
            distance_i = distance_matrix[i][neighbors_i]

            # Get the number of i neighbors to drop
            number_to_drop = len(neighbors_i)-max_neighbors

            while number_to_drop>0:
                # Get the most distant node from i
                to_drop = np.argmax(distance_i)

                # Remove edge between i and to drop
                self.G.remove_edge(i,neighbors_i[to_drop])

                # Update adjacency matrix
                self.A = nx.to_numpy_array(self.G)

                # Update neighbors list, distance_i, to_drop
                neighbors_i = [n for n in self.G[i]]
                distance_i = distance_matrix[i][neighbors_i]
                number_to_drop = len(neighbors_i)-max_neighbors
                                
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


