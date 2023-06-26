from sksurv.linear_model import CoxPHSurvivalAnalysis
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
