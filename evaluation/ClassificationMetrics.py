import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

class ClassificationMetrics:
    """
    This class implements some classification metrics methods.
    """
    def __init__(self) -> None:
        pass
    
    # @staticmethod
    # def eval_metrics_from_conf_matrix(y_true: np.ndarray, y_pred: np.ndarray)->tuple[float, float, float, float]:
    #     """
    #     Compute the accuracy, precision, recall, and f1-score.

    #     ### Parameters :
    #     - y_true : the correct classes for each patient
    #     - y_pred : the predicted classes for each patient

    #     ### Returns :
    #     Accuracy, Precision, Recall, F1-score
    #     """
    #     return np.round([precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)],2)

    @staticmethod
    def compute_roc_curve(y_true: np.ndarray, y_score: np.ndarray)->tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    # @staticmethod
    # def compute_auc(y_true: np.ndarray, y_score: np.ndarray)->float:
    #     """
    #     Compute the AUC associated to the prediction.

    #     ### Parameters :
    #     - y_true : the correct classes for each patient
    #     - y_score : the output score predicted by the model (class 1 probability)

    #     ### Returns :
    #     The AUC score
    #     """
    #     return np.round(roc_auc_score(y_true, y_score),2)
    
    @staticmethod
    def compute_sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray)->tuple[float, float]:
        """ 
        Compute the sensitivy (TPR) and the specificity (TNR) of the prediction.

        ### Parameters :
        - y_true : the correct classes for each patient
        - y_pred : the predicted classes for each patient

        ### Returns :
        Sensitivity, Specificity    
        """
        # Confusion matrix
        TP, FP, FN, TN = confusion_matrix(y_true, y_pred).ravel()

        # Sensitivity : TPR : given the real positive, how many the model found ? Same as recall
        sensitivity = np.round(TP / (TP + FN),2)

        # Specificity : TNR : given the real negative, how many the model found ?
        specificity = np.round(TN / (TN + FP),2)

        return sensitivity, specificity