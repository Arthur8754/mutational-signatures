import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

class featureSelection:
    """
    Class which studies correlation between features and select features.
    """
    def __init__(self) -> None:
        pass

    def correlation(self, df: pd.DataFrame, plot_heatmap: bool)->pd.DataFrame:
        """ 
        Compute the Pearson correlation between each column.

        ### Parameters :
        - df : the dataframe for computing the correlation
        - plot_heatmap : if True, print a heatmap of the correlation matrix.

        ### Returns :
        The correlation matrix.
        """
        # Compute the correlation matrix
        correl = np.round(df.corr(),2)

        # Plot the heatmap
        if plot_heatmap:
            fig, ax = plt.subplots(figsize=(12,12))
            ax = sns.heatmap(correl,linewidths=1, square=True, annot=True,cmap=sns.color_palette("Spectral", as_cmap=True))
            ax.set_title("Correlation between biomarkers")
            plt.savefig("heatmap-correlation-biomarkers.png")
        return np.round(df.corr(),2)
    
    def feature_importance(self, X: pd.DataFrame, y: np.ndarray, plot_hist: bool)->np.ndarray:
        """
        Compute the feature importance of each column of X dataframe for labelling y, with a Random Forest Classifier.

        ### Parameters :
        - X : the dataframe from which compute the feature importance
        - y : the label data for X
        - plot_hist : if True, print a histogram of each feature's importance.

        ### Returns :
        The relative importance for each feature.
        """
        # Fitting the RF classifier and compute the feature importances
        rf_clf = RandomForestClassifier()
        rf_clf.fit(X, y)
        feat_importance = rf_clf.feature_importances_

        # Sort the result
        indices_feat_importance = feat_importance.argsort()
        feat_importance_sorted = feat_importance[indices_feat_importance]
        biomarkers_per_importance = np.array(X.columns)[indices_feat_importance]

        # Plot the result
        if plot_hist:
            plt.figure(figsize=(10,15))
            plt.barh(biomarkers_per_importance,feat_importance_sorted)
            plt.xlabel("Importance")
            plt.title("Importance of each biomarker")
            plt.savefig("feature-importance.png")

        return feat_importance

