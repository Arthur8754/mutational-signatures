import pandas as pd
from lifelines.statistics import logrank_test
import numpy as np

class LogRankTest:
    """ 
    Code for implementing the Log-Rank Test method which compares 2 survival curves.
    """
    def __init__(self) -> None:
        pass
    
    def stat_test(self, survival_group1: pd.Series, status_group1: pd.Series, survival_group2: pd.Series, status_group2: pd.Series) -> float:
        """ 
        Make the log rank test : tests if the group1 and group2 curves are similar.

        ### Parameters :
        - survival_group1 : the time survival of group 1
        - status_group1 : the status survival of group 1 (1 if event observed)
        - survival_group2 : the time survival of group 2
        - status_group2 : the status survival of group 2 (1 if event observed)

        ### Returns :
        The p value of the log rank test.
        """
        test_results = logrank_test(survival_group1, survival_group2, status_group1, status_group2, alpha=0.95)
        return np.round(test_results.p_value,3)