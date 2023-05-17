from sksurv.nonparametric import kaplan_meier_estimator 

class KaplanMeier:
    """ 
    Code for implementing the Kaplan-Meier method which estimates the survival function. 
    """
    def __init__(self, status, survival) -> None:
        self.status = status
        self.survival = survival

    def estimate_survival_function(self):
        """ 
        Estimate the survival function from status and survival with the Kaplan-Meier Method.
        """
        time, survival_prob = kaplan_meier_estimator(self.status, self.survival)
        return time, survival_prob
