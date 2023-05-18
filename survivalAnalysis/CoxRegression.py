from sksurv.linear_model import CoxPHSurvivalAnalysis

class CoxRegression:
    """ 
    Code for implementing the Cox Regression Method which estimates the hazard function.
    """
    def __init__(self) -> None:
        pass

    def compute_hazard_function(self, X, y):
        """ 
        Compute the coefficients of the Cox Regression model.
        """
        cph = CoxPHSurvivalAnalysis()
        cph.fit(X, y)
        hazard_function = cph.predict_cumulative_hazard_function(X)
        return hazard_function
