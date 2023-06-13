import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import matplotlib.pyplot as plt

class BuildGraph:
    """ 
    This class creates a graph from an initial dataframe, with the k-NN algorithm. Each node is connected to its k nearest neighbors.
    """
    def __init__(self, k: int, metric: str) -> None:
        """
        - k : the number of neighbors for each node in the graph.
        - metric ("euclidean","cosine","manhattan") : the metric used to compute distance between points.
        """
        self.k = k+1
        self.metric=metric
        self.model = NearestNeighbors(n_neighbors=k+1, metric=metric)

    def apply_knn(self, X: np.ndarray)->np.ndarray:
        """
        Apply the k-NN algorithm to connect each node to its k nearest neighbors.

        ### Parameters :
        - X (n_samples, n_features) : the points to connect, with their features.

        ### Returns :
        The adjacency matrix of the graph.
        """
        # Need to have n_samples >= n_neighbors
        if X.shape[0]<self.k:
            return np.zeros((X.shape[0],X.shape[0]))
        
        # Find the k nearest neighbors of each patient
        self.model.fit(X)

        # Build the associated adjacency matrix
        A = self.model.kneighbors_graph(X).toarray()
        return A-np.identity(A.shape[0])
    
    def merge_adjacency_matrix(self, matrices:list[np.ndarray])->np.ndarray:
        """ 
        Merge the local adjacency matrices for subgraphs into global one for whole graph.

        ### Parameters :
        - matrices : the local adjacency matrices of the distinct subgraphs.

        ### Returns ;
        The global adjacency matrix for the whole graph.
        """
        #shape_A = np.sum([matrices[i].shape[0] for i in range(len(matrices))])
        A = matrices[0]

        # Iteratively merge each matrix
        for i in range(1,len(matrices)):
            shape_A, shape_i = A.shape[0], matrices[i].shape[0]

            # Concatenation of A and the next matrix along diagonal
            A = np.block([
                [A,np.zeros((shape_A, shape_i))],
                [np.zeros((shape_i, shape_A)),matrices[i]]
            ])

        return A
    
    def show_graph(self, A: np.ndarray)->None:
        """ 
        Create the graph from the adjacency matrix and plot it.

        ### Parameters :
        - A (n_samples, n_samples) : the adjacency matrix of the graph

        ### Returns :
        None
        """
        G = nx.Graph(A)
        fig, ax = plt.subplots(figsize=(10,7))
        nx.draw(G)
        plt.savefig('graph.png')

    def proportion_similarity_neighbors(self, A:np.ndarray, feature:np.ndarray)->float:
        """ 
        Estimate the proportion of similar neighbor patients for the given feature.

        ### Parameters :
        - A (n_samples,n_samples) : The adjacency matrix of the graph.
        - feature (n_samples,) : the feature value for each sample.

        ### Returns :
        The mean proportion of neighbor patients which are similar for the given feature.
        """
        similarity = 0
        n=0
        for i in range(A.shape[0]):
            # Feature of patient i
            feature_i = feature[i]

            # Feature extraction of the neighbors of i
            neighbors_i = feature[np.where(A[i] == 1)[0]]

            # Count similar feature
            if neighbors_i.shape[0]!=0:
                n+=1
                similarity += np.count_nonzero(np.where(neighbors_i == feature_i,True,False))/neighbors_i.shape[0]

        if n!=0:
            return np.round(similarity/n,2)
        return 0