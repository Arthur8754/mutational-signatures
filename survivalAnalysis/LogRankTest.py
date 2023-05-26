import pandas as pd
from lifelines.statistics import logrank_test
import numpy as np

class LogRankTest:
    """ 
    Code for implementing the Log-Rank Test method which compares 2 survival curves.
    """
    def __init__(self) -> None:
        pass
    
    def make_test(self, time_survival_1: np.ndarray, status_survival_1: np.ndarray, time_survival_2: np.ndarray, status_survival_2: np.ndarray) -> float:
        """ 
        Make the log rank test : tests if the group1 and group2 curves are similar.

        ### Parameters :
        - time_survival_1 : the time survival of group 1
        - status_survival_1 : the status survival of group 1 (1 if event observed)
        - time_survival_2 : the time survival of group 2
        - status_survival_2 : the status survival of group 2 (1 if event observed)

        ### Returns :
        The p value of the log rank test.
        """
        test_results = logrank_test(time_survival_1, time_survival_2, status_survival_1, status_survival_2, alpha=0.95)
        return np.round(test_results.p_value,3)