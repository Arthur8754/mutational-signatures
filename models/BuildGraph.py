import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

class BuildGraph:
    """ 
    This class creates a graph from an initial dataframe.
    """
    def __init__(self, n_neighbors: int, metric: str) -> None:
        """
        - n_neighbors : the number of neighbors for each node.
        """
        self.n_neighbors = n_neighbors
        self.metric = metric

    def split_per_tumour(self, df: pd.DataFrame):
        """
        Split the patients depending on their tumour.
        """
        pass

    def build_adjacency_matrix(self, X: np.ndarray)->np.ndarray:
        """
        Compute the adjacency matrix with the NearestNeighbors method of sklearn.

        ### Parameters :
        - X (n_samples, n_features) : the dataframe to convert as graph

        ### Returns :
        The adjacency matrix of the built graph.
        """
        # Find the nearest neighbors of each patient
        neighbors_model = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric).fit(X)

        # Convert as adjacency matrix
        return neighbors_model.kneighbors_graph(X).toarray()
