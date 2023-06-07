import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold

class BinarySurvival:
    """
    The class predicts if a patient will survive or not.
    """
    def __init__(self, model) -> None:
        """
        - model : the binary classifying model.
        """
        self.model = model

    def drop_non_analyzable_patients(self, status: np.ndarray, time: np.ndarray, t: float)->np.ndarray:
        """
        Drop the patients for which status = 0 and time < t, which are non analyzable.

        ### Parameters :
        - status : the event status for each patient (1 if event observed, 0 otherwise)
        - time : the time of the event (if no event observed, time = censoring time)
        - t : the time when we look at.     

        ### Returns :
        The index of the patients to drop.   
        """
        indices_0_status = np.where(status == 0)[0]
        to_drop = indices_0_status[np.where(time[indices_0_status] < t)[0]]
        return to_drop

    def label_patients(self, time: np.ndarray, t: float)->np.ndarray:
        """
        Label the survival class depending on the status event, time event, and cutoff t.
        - status = 1 & time < t : class 1 (event observed before t)
        - status = 1 & time > t : class 0 (no event observed before t)
        - status = 0 & time > t : class 0 (no event observed before t)

        We suppose the non analyzable patients have already been deleted. So the only case to test is if time <t

        ### Parameters :
        - status : the event status for each patient (1 if event observed, 0 otherwise)
        - time : the time of the event (if no event observed, time = censoring time)
        - t : the time when we look at.

        ### Returns :
        The survival class of each patient (1 if event observed, 0 otherwise).
        """
        y = np.where(time<t, 1, 0)
        return y
    
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
    
    def leave_one_out_cross_validation(self, X: np.ndarray, y: np.ndarray)->tuple[np.ndarray, np.ndarray]:
        """ 
        Make a 1-fold CV to determine test labels and scores of the cohort.

        ### Parameters :
        - X (n_samples, n_features) : the features of each patient
        - y (n_samples,) : the class of each patient.

        ### Returns :
        - The test classes of each patient
        - The test scores of each patient
        """
        # Sample array
        classes = np.zeros(y.shape)
        scores = np.zeros(y.shape)

        # Split the index to n_splits folds
        n_samples = X.shape[0]
        folds = KFold(n_splits=n_samples, shuffle=True).split(X)
        
        # Train - test for each fold
        for i, (train_index, test_index) in enumerate(folds):
            
            # Train
            self.train(X[train_index],y[train_index])

            # Test
            scores[test_index] = self.predict_score(X[test_index])
            classes[test_index] = self.predict(X[test_index])

        return classes, scores
    
    def eval_metrics_from_conf_matrix(self, y_true: np.ndarray, y_pred: np.ndarray)->tuple[float, float, float, float]:
        """
        Compute the accuracy, precision, recall, and f1-score.

        ### Parameters :
        - y_true : the correct classes for each patient
        - y_pred : the predicted classes for each patient

        ### Returns :
        Accuracy, Precision, Recall, F1-score
        """
        return np.round([accuracy_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)],2)
    
    def compute_roc_curve(self, y_true: np.ndarray, y_score: np.ndarray)->tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the ROC curve associated to the prediction.

        ### Parameters :
        - y_true : the correct classes for each patient
        - y_score : the output score predicted by the model (class 1 probability)

        ### Returns :
        - The False Positive Rate (x-axis) for each threshold
        - The True Positive Rate (y-axis) for each threshold
        - The thresholds used.
        """
        return roc_curve(y_true, y_score)
    
    def compute_auc(self, y_true: np.ndarray, y_score: np.ndarray)->float:
        """
        Compute the AUC associated to the prediction.

        ### Parameters :
        - y_true : the correct classes for each patient
        - y_score : the output score predicted by the model (class 1 probability)

        ### Returns :
        The AUC score
        """
        return np.round(roc_auc_score(y_true, y_score),2)
    

    

