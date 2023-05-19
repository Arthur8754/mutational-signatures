import matplotlib.pyplot as plt
import pandas as pd

class BiomarkerDistribution:
    """ 
    This class studies the distribution of a given biomarker, between 2 groups.
    """
    def __init__(self) -> None:
        pass

    def boxplot_values(self, values_group1 : pd.Series, values_group2 : pd.Series, label_group1: str, label_group2: str, label_biomarker: str, title: str, filename: str) -> None:
        """ 
        Plot the biomarker values for each group using a boxplot.

        ### Parameters :
        - values_group1 : the values of the biomarker for the group 1.
        - values_group2 : the values of the biomarker for the group 2.
        - label_group1 : the name of the group 1.
        - label_group2 : the name of the group 2.
        - label_biomarker : the name of the biomarker measured.
        - title : the title of the figure.
        - filename : the location where to store the boxplot.
        """
        plt.boxplot([values_group1, values_group2],labels=[label_group1,label_group2])
        plt.ylabel(label_biomarker)
        plt.title(title)
        plt.savefig(filename)
