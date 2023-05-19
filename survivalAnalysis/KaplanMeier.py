from sksurv.nonparametric import kaplan_meier_estimator 
import numpy as np

class KaplanMeier:
    """ 
    Code for implementing the Kaplan-Meier method which estimates the survival function. 
    """
    def __init__(self) -> None:
        pass
    
    def estimate_survival_function(self, status : np.ndarray, survival: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ 
        Estimate the survival function from status and survival with the Kaplan-Meier Method.

        ### Parameters :
        - status : the status vector, 1 if event observed, 0 otherwise
        - survival : the patient survival time vector

        ### Returns :
        The probability to survive across time, with its time vector.
        """
        time, survival_prob = kaplan_meier_estimator(status, survival)
        return time, survival_prob
