import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from sklearn.cluster import KMeans, SpectralClustering

class Clustering:
    """
    This class implements some clustering methods.
    """
    def __init__(self, model) -> None:
        """
        Model which makes the clustering.
        """
        self.model = model

    def plot_2d_scatter(self, X_2d:np.ndarray[np.ndarray[float]], y:np.ndarray, title: str, filename:str)->None:
        """ 
        Make the 2D scatter plot of data, with color and label per value of y, and save the figure at filename.

        ### Parameters :
        - X_2d (n_samples, 2): the 2D vectors to print
        - y (n_samples,): the label vector, with we color and label
        - filename : the location where store the figure

        ### Returns :
        None
        """
        # Creation of the color mapping (each class associated to 1 color).
        dico_map = {}
        class_names = np.unique(y)
        colors = random.sample(list(mcolors.CSS4_COLORS),class_names.shape[0])
        for i in range(class_names.shape[0]):
            dico_map[class_names[i]] = colors[i]

        # Plot with label and color per class
        fig, ax = plt.subplots(figsize=(10,7))
        for classe in np.unique(y):
            idx_classe = np.where(y==classe)
            x_classe = X_2d[idx_classe].reshape((-1,2))
            ax.scatter(x_classe[:,0],x_classe[:,1],color=dico_map[classe],label=classe)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title(title)
        plt.legend()
        plt.savefig(filename)

    def apply_clustering(self, X: np.ndarray)->np.ndarray:
        """ 
        Apply the K-Means algorithm and associate each sample of X to its class.

        ### Parameters :
        - X (n_samples, n_features) : the data to group by similarity
        - K : the number of clusters for KMeans.

        ### Returns :
        - The cluster of each sample (n_samples,)
        """     
        return self.model.fit(X).labels_

    # def kmeans(self, X: np.ndarray, K: int):
    #     """ 
    #     Apply the K-Means algorithm and associate each sample of X to its class.

    #     ### Parameters :
    #     - X (n_samples, n_features) : the data to group by similarity
    #     - K : the number of clusters for KMeans.

    #     ### Returns :
    #     - The cluster of each sample (n_samples,)
    #     """
    #     return KMeans(K).fit(X).labels_
    
    # def spectral_clustering(self, X: np.ndarray, K: int):
    #     """
    #     Apply the spectral clustering algorithm and associate each sample of X to its class.

    #     ### Parameters :
    #     - X (n_samples, n_features) : the data to group by similarity
    #     - K : the number of clusters for KMeans.

    #     ### Returns :
    #     - The cluster of each sample (n_samples,)
    #     """
    #     return SpectralClustering(K).fit(X).labels_