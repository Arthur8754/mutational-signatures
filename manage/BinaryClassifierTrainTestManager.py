import numpy as np
from sklearn.model_selection import KFold

class BinaryClassifierTrainTestManager:
    """ 
    This class manages the train/test process for the binary classifier.
    """
    def __init__(self, model) -> None:
        """
        ### Parameters :
        - model : the binary classifier from the BinaryClassifier class.
        """
        self.model = model

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
            self.model.train(X[train_index],y[train_index])

            # Test
            scores[test_index] = self.model.predict_score(X[test_index])
            classes[test_index] = self.model.predict(X[test_index])

        return classes, scores