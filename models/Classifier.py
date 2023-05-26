import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Classifier:
    """
    This class implements a binary classifier using scikit learn.
    """

    def __init__(self, model) -> None:
        """ 
        - classifier : the scikit learn model for classifying.
        """
        self.model = model

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the classifier.

        ### Parameters :
        - X : the features matrix (n_samples, n_features)
        - y : the labels vector (n_samples,)

        ### Returns :
        None
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class for each sample of X.

        ### Parameters :
        - X : the features matrix (n_samples, n_features)

        ### Returns :
        The predicted labels (n_samples,)
        """
        return self.model.predict(X)
    
    def kfold_cross_validation(self, X, y):
        pass
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray)->tuple[float, float, float, float]:
        """
        Compute the accuracy, precision, recall, and f1 score for the prediction made by y_pred.

        ### Parameters :
        - y_true : the right labels (n_samples,)
        - y_pred : the predicted labels (n_samples,)

        ### Returns :
        The accuracy, precision, recall and f1 score, under tuple format.
        """
        return accuracy_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)