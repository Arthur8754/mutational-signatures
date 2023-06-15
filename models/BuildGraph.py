import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

class BuildGraph:
    """ 
    This class creates a graph from an initial dataframe. The graph is an union of distinct complete sub graphs, where each sub graph 
    contains the patients with the same value of a specific column.
    """
    def __init__(self, df: pd.DataFrame) -> None:
        """ 
        - df : the source dataframe to build the graph 
        - A : the adjacency matrix of the graph
        - G : the networkx graph
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

    def create_graph(self, features_name: list[str], label_name: str)->None:
        """ 
        Create the networkx graph from the adjacency matrix.

        ### Parameters :
        - features_name : the list of the names of the patient's features to integrate in the graph
        - label_name : the name of the label column to integrate in the graph.

        ### Returns :
        None
        """
        X = self.df.loc[:,features_name].to_numpy()
        y = self.df[label_name].to_numpy()

        # Initialize with empty graph
        self.G = nx.Graph()
        
        # Add nodes, with its features and label
        for i in range(self.A.shape[0]):
            self.G.add_nodes_from([(i,{"x":X[i],"y":y[i]})])

        # Add edges
        rows, cols = np.where(self.A == 1)
        edges = zip(rows.tolist(), cols.tolist())
        self.G.add_edges_from(edges)

        # return G
    
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

    # def split_per_column(self, column_name: str, min_rows: int)->dict[str,pd.DataFrame]:
    #     """ 
    #     Split the dataframe df into distinct groups, depending on the column_name values. Merge if the sub dataframe contains less than min_rows.

    #     ### Parameters :
    #     - column_name : the name of the column used to make the split.
    #     - min_rows : the minimum number of rows required to 

    #     ### Returns :
    #     A dictionary, where key = column value, and value = sub dataframe.
    #     """
    #     column_values = np.unique(self.df[column_name].to_numpy())
    #     dico = {}
    #     for i in range(column_values.shape[0]):
    #         # Sub dataframe which contains the same specific values
    #         sub_df = self.df.loc[self.df[column_name] == column_values[i]]

    #         # Merge if not enough rows
    #         if sub_df.shape[0]<min_rows:
    #             if "merged" not in dico:
    #                 dico["merged"] = sub_df 
    #             else:
    #                 dico["merged"] = pd.concat([dico["merged"],sub_df])

    #         else:
    #             dico[column_values[i]] = self.df.loc[self.df[column_name] == column_values[i]]
    #     return dico
    
    # def local_adjacency_matrix(self, sub_df: pd.DataFrame)->np.ndarray:
    #     """ 
    #     Compute the local adjacency matrix of the sub_df fully connected graph.

    #     ### Parameters :
    #     - sub_df : the source sub dataframe to build the complete sub graph.

    #     ### Returns :
    #     - the adjacency matrix of this complete sub graph.
    #     """
    #     # The adjacency matrix of a complete graph is a matrix of 1.
    #     shape_A = sub_df.shape[0]
    #     return np.ones((shape_A,shape_A))-np.identity(shape_A)
    
    # def global_adjacency_matrix(self, dico_df: dict[str, pd.DataFrame])->np.ndarray:
    #     """ 
    #     Build the global adjacency matrix of the graph with merging the local matrices.
    #     ### Parameters :
    #     - matrices : the local adjacency matrices of the distinct subgraphs.

    #     ### Returns ;
        
    #     The global adjacency matrix for the whole graph.
    #     """
    #     keys = list(dico_df.keys())

    #     # Initialize the global adjacency matrix
    #     A_global = self.local_adjacency_matrix(dico_df[keys[0]])

    #     # Iteratively merge local adjacency matrices
    #     for i in range(1,len(keys)):
    #         # Local adjacency matrix
    #         A_local = self.local_adjacency_matrix(dico_df[keys[i]])

    #         shape_A_global, shape_A_local = A_global.shape[0], A_local.shape[0]

    #         # Concatenation of A_global and A_local along diagonal
    #         A_global = np.block([
    #             [A_global,np.zeros((shape_A_global, shape_A_local))],
    #             [np.zeros((shape_A_local, shape_A_global)),A_local]
    #         ])
    #     return A_global
    
    # def create_graph(self, A: np.ndarray, X: np.ndarray, y: np.ndarray):
    #     """ 
    #     Create the networkx graph from the adjacency matrix.

    #     ### Parameters :
    #     - A (n_samples, n_samples) : the adjacency matrix of the graph.
    #     - X (n_samples, n_features) : the features of each patient.
    #     - y (n_samples,) : the label of each patient.

    #     ### Returns :
    #     The networkx object containing the graph.
    #     """
    #     # Initialize with empty graph
    #     G = nx.Graph()

    #     # Add nodes, with its features and label
    #     for i in range(A.shape[0]):
    #         G.add_node((i,{"x":X[i],"y":y[i]}))

    #     # Add edges
    #     G.add_edges_from(zip(np.where(A==1)))

    #     return G
