import numpy as np
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival

class SurvivalMetrics:
    """
    This class implements some survival metrics methods.
    """
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def get_c_index(status: np.ndarray, time: np.ndarray, risk_scores: np.ndarray) -> float:
        """ 
        Compute the concordance index from the input samples.

        ### Parameters :
        - status (n_samples,) : the event status for each sample (1 if event happened, 0 otherwise)
        - time (n_samples,) : the surviving time for each sample.
        - risk_score (n_samples,) : the risk score for each sample.

        ### Returns :
        The associated concordance index.
        """
        return np.round(concordance_index_censored(status, time, risk_scores)[0],2)
    
    @staticmethod
    def kaplan_meier_estimation(status: np.ndarray[bool], time: np.ndarray)->tuple[float, float]:
        """
        Estimate the survival curve using the Kaplan Meier Estimator.

        ### Parameters :
        - status (n_samples,) : the event status for each sample (1 if event happened, 0 otherwise)
        - time (n_samples,) : the surviving time for each sample.

        ### Returns :
        - the time axis ;
        - the survival probability for each time point.
        """
        return kaplan_meier_estimator(status, time)
    
    @staticmethod
    def log_rank_test(status: np.ndarray, time: np.ndarray, group_indicator: np.ndarray)->float:
        """
        Make the log rank test between groups of group_indicator, and returns the associated p value.

        ### Parameters :
        - status (n_samples,) : the event status for each sample (1 if event happened, 0 otherwise)
        - time (n_samples,) : the surviving time for each sample.
        - group_indicator (n_samples,) : the label of the group for each sample.

        ### Returns :
        The p value after making log rank test.
        """
        # Structure array for log rank tester input
        y = np.array(list(zip(status, time)), dtype=[('status','?'),('time surviving','<f8')])
        return np.round(compare_survival(y, group_indicator)[1],2)