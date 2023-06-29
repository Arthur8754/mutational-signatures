import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold

class BinaryClassifier:
    """
    The class implements a binary classifier model.
    """
    def __init__(self, model) -> None:
        """
        - model : the binary classifying model.
        """
        self.model = model
    
    def train(self, X: np.ndarray, y: np.ndarray)->None:
        """
        Fit the binary classifier.

        ### Parameters :
        - X (n_samples, n_features) : the features of each patient.
        - y (n_samples,) : the class of each patient.

        ### Returns :
        None
        """
        self.model.fit(X, y)

    def predict_score(self, X: np.ndarray)->np.ndarray:
        """
        Predict the output score for each patient.

        ### Parameters :
        - X (n_samples, n_features) : the features of each patient

        ### Returns :
        The output score of the model for each patient (class 1 probability).
        """
        return self.model.predict_proba(X).transpose()[0]

    def predict(self, X: np.ndarray)->np.ndarray:
        """
        Predict the class of each patient in parameter.

        ### Parameters :
        - X (n_samples, n_features) : the features of each patient

        ### Returns :
        The class of each patient.
        """
        return self.model.predict(X)
