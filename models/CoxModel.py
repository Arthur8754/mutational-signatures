from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival
from sklearn.model_selection import KFold
import numpy as np

class CoxModel:
    """
    Cox Proportional Hazard model.

    ### Parameters :
    None

    ### Attributes :
    - model : the Cox model from scikit-survival (logistic regression).
    """
    def __init__(self) -> None:
        self.model = CoxPHSurvivalAnalysis()

    def train(self, X: np.ndarray[np.ndarray[float]], y: np.ndarray[tuple[int, float]]) -> None:
        """
        Fit the model to estimate the Cox parameters.

        ### Parameters :
        - X : the matrix containing the variables values for each sample.
        - y : the event status and the time surviving for each sample.
        
        ### Returns :
        None
        """
        self.model = self.model.fit(X, y)

    def predict_risk_score(self, X: np.ndarray[np.ndarray[float]]) -> None:
        """ 
        Predict the risk for each sample.

        ### Parameters :
        - X : the matrix containing the variables for each sample.

        ### Returns :
        The array of risk scores for each patient of X.
        """
        # self.risk_scores = self.model.predict(X)
        return self.model.predict(X)
    
    def find_cutoff(self, risk_scores: np.ndarray)->float:
        """ 
        Determine the cutoff between high risk and low risk, with computing the median.

        ### Parameters :
        - risk_scores : the risk score for each sample.

        ### Returns :
        The threshold high risk / low risk
        """
        return np.median(risk_scores)
    
    def predict_class(self, risk_scores: np.ndarray, cutoff: float)->np.ndarray:
        """ 
        Predict the risk class (high or low) for each sample.

        ### Parameters :
        - X : the matrix containing the variables for each sample.
        - cutoff : the cutoff between high risk and low risk.

        ### Returns :
        The risk class for each sample, 1 if high, 0 otherwise.
        """
        risk_classes = np.copy(risk_scores)
        risk_classes[risk_scores>=cutoff] = 1
        risk_classes[risk_scores<cutoff] = 0
        return risk_classes
    
    ### FAUDRAIT DÉPLACER AILLEURS CE QUI EST EN-DESSOUS ###
    
    def get_c_index(self, status: np.ndarray, time: np.ndarray, risk_scores: np.ndarray) -> float:
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
    
    def kaplan_meier_estimation(self, status: np.ndarray[bool], time: np.ndarray)->tuple[float, float]:
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
    
    def log_rank_test(self, status: np.ndarray, time: np.ndarray, group_indicator: np.ndarray)->float:
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
