from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival
from sklearn.model_selection import KFold
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
    
    def leave_one_out_cross_validation(self, X: np.ndarray[np.ndarray[float]], y: np.ndarray[tuple[int, float]])->tuple[np.ndarray, np.ndarray]:
        """ 
        Make the one out cross validation to find the risk class for each sample.

        ### Parameters :
        - X : the train data, containing the variables values for each sample.
        - y : the train labels, containing the event status and the time surviving for each sample.

        ### Returns :
        - The risk class for each sample, after training.
        - The risk score for each sample, after training.
        """
        # Sample array
        class_samples = np.zeros(y.shape)
        risk_score_samples = np.zeros(y.shape)

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
            risk_score_samples[test_index] = self.predict_risk_score(X[test_index])
            class_samples[test_index] = self.predict_class(risk_score_samples[test_index], cutoff)

        return class_samples, risk_score_samples
    
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

