from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

class CoxModel:
    """
    Cox Proportional Hazard model.
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

    def predict_risk_score(self, X: np.ndarray[np.ndarray[float]]) -> np.ndarray:
        """ 
        Predict the risk for each sample.

        ### Parameters :
        - X : the matrix containing the variables for each sample.

        ### Returns :
        The risk score for each sample.
        """
        return self.model.predict(X)
    
    def find_cutoff(self, risk_scores: np.ndarray)->float:
        """ 
        Determine the cutoff between high risk and low risk, with computing the median.

        ### Parameters :
        - risk_scores : the risk score for each sample.

        ### Returns :
        The median of these risks, which is the high-low risk cutoff.
        """
        return np.median(risk_scores)
    
    def predict_class(self, X: np.ndarray[np.ndarray[float]], cutoff: float)->np.ndarray:
        """ 
        Predict the risk class (high or low) for each sample.

        ### Parameters :
        - X : the matrix containing the variables for each sample.
        - cutoff : the cutoff between high risk and low risk.

        ### Returns :
        The risk class for each sample, 1 if high, 0 otherwise.
        """
        risk_scores = self.predict_risk_score(X)
        risk_scores[risk_scores>=cutoff] = 1
        risk_scores[risk_scores<cutoff] = 0
        return risk_scores
    
    def leave_one_out_cross_validation(self, X: np.ndarray[np.ndarray[float]], y: np.ndarray[tuple[int, float]])->np.ndarray:
        """ 
        Make the one out cross validation to find the risk class for each sample.

        ### Parameters :
        - X : the train data, containing the variables values for each sample.
        - y : the train labels, containing the event status and the time surviving for each sample.

        ### Returns :
        The risk class for each sample, after training.
        """
        # Sample array
        class_samples = np.zeros(y.shape)

        # Split the index to n_splits folds
        n_samples = X.shape[0]
        folds = KFold(n_splits=n_samples, shuffle=True).split(X)
        
        # Train - cutoff - test for each fold
        for i, (train_index, test_index) in enumerate(folds):
            
            # Train
            self.train(X[train_index],y[train_index])

            # Predict train scores and find cutoff
            train_scores = self.predict_risk_score(X[train_index])
            cutoff = self.find_cutoff(train_scores)

            # Test
            class_samples[test_index] = self.predict_class(X[test_index], cutoff)

        return class_samples
    
    def predict_mean_survival_curve(self, X: np.ndarray[np.ndarray[float]])->tuple[np.ndarray,np.ndarray]:
        """ 
        Predict the mean survival curve from the samples, with the event time.

        ### Parameters :
        - X : the matrix containing the variables values for each sample.

        ### Returns :
        - the event times (x-axis)
        - the mean survival probability (y-axis)
        """
        mean_survival_curve = np.mean(self.model.predict_survival_function(X,return_array=True),axis=0)
        return self.model.event_times_, mean_survival_curve