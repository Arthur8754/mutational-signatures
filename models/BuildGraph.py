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
        """
        self.df = df

    def split_per_column(self, column_name: str, min_rows: int)->dict[str,pd.DataFrame]:
        """ 
        Split the dataframe df into distinct groups, depending on the column_name values. Merge if the sub dataframe contains less than min_rows.

        ### Parameters :
        - column_name : the name of the column used to make the split.
        - min_rows : the minimum number of rows required to 

        ### Returns :
        A dictionary, where key = column value, and value = sub dataframe.
        """
        column_values = np.unique(self.df[column_name].to_numpy())
        dico = {}
        for i in range(column_values.shape[0]):
            # Sub dataframe which contains the same specific values
            sub_df = self.df.loc[self.df[column_name] == column_values[i]]

            # Merge if not enough rows
            if sub_df.shape[0]<min_rows:
                if "merged" not in dico:
                    dico["merged"] = sub_df 
                else:
                    dico["merged"] = pd.concat([dico["merged"],sub_df])

            else:
                dico[column_values[i]] = self.df.loc[self.df[column_name] == column_values[i]]
        return dico
    
    def local_adjacency_matrix(self, sub_df: pd.DataFrame)->np.ndarray:
        """ 
        Compute the local adjacency matrix of the sub_df fully connected graph.

        ### Parameters :
        - sub_df : the source sub dataframe to build the complete sub graph.

        ### Returns :
        - the adjacency matrix of this complete sub graph.
        """
        # The adjacency matrix of a complete graph is a matrix of 1.
        shape_A = sub_df.shape[0]
        return np.ones((shape_A,shape_A))-np.identity(shape_A)
    
    def global_adjacency_matrix(self, dico_df: dict[str, pd.DataFrame])->np.ndarray:
        """ 
        Build the global adjacency matrix of the graph with merging the local matrices.
        ### Parameters :
        - matrices : the local adjacency matrices of the distinct subgraphs.

        ### Returns ;
        
        The global adjacency matrix for the whole graph.
        """
        keys = list(dico_df.keys())

        # Initialize the global adjacency matrix
        A_global = self.local_adjacency_matrix(dico_df[keys[0]])

        # Iteratively merge local adjacency matrices
        for i in range(1,len(keys)):
            # Local adjacency matrix
            A_local = self.local_adjacency_matrix(dico_df[keys[i]])

            shape_A_global, shape_A_local = A_global.shape[0], A_local.shape[0]

            # Concatenation of A_global and A_local along diagonal
            A_global = np.block([
                [A_global,np.zeros((shape_A_global, shape_A_local))],
                [np.zeros((shape_A_local, shape_A_global)),A_local]
            ])
        return A_global
    
    def show_graph(self, A: np.ndarray, title: str, filename: str)->None:
        """ 
        Create the graph from the adjacency matrix and plot it.

        ### Parameters :
        - A (n_samples, n_samples) : the adjacency matrix of the graph
        - title : the title of the figure
        - filename : the location to store the figure

        ### Returns :
        None
        """
        G = nx.Graph(A)
        fig, ax = plt.subplots(figsize=(10,7))
        nx.draw(G)
        plt.title(title)
        plt.savefig(filename)