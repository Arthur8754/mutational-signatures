"""
https://medium.com/the-researchers-guide/survival-analysis-in-python-km-estimate-cox-ph-and-aft-model-5533843c5d5d
"""

from sksurv.linear_model import CoxPHSurvivalAnalysis
from lifelines import CoxPHFitter
import pandas as pd

class CoxRegression:
    """ 
    Code for implementing the Cox Regression Method which estimates the hazard function.
    """
    def __init__(self) -> None:
        pass

    def compute_hazard_function(self, df: pd.DataFrame, time_label: str, status_label: str) -> CoxPHFitter:
        """ 
        Compute the coefficients of the Cox Regression model.

        ### Parameters :
        - df : the dataframe used to learn the Cox model
        - time_label : the name of the column which contains the survival time
        - status_label : the name of the column which contains the status

        ### Returns :
        - the trained Cox regression model
        """
        cph = CoxPHFitter()
        cph.fit(df, duration_col=time_label, event_col=status_label)
        return cph
