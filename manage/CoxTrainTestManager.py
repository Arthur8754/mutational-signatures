import numpy as np
from sklearn.model_selection import KFold

class CoxTrainTestManager:
    """ 
    This class manages the train/test process for the Cox Model.

    ### Parameters :
    - model : the Cox Model
    """
    def __init__(self, model) -> None:
        self.model = model

    def leave_one_out_cross_validation(self, X: np.ndarray[np.ndarray[float]], y: np.ndarray[tuple[int, float]])->tuple[np.ndarray, np.ndarray]:
        """ 
        Make the one out cross validation to find the risk class for each sample.

        ### Parameters :
        - X : the features data of each sample.
        - y : the label of each sample, containing the event status and the time surviving for each sample.

        ### Returns :
        - The test risk class for each sample, after training.
        - The test risk score for each sample, after training.
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
            self.model.train(X[train_index],y[train_index])

            # Predict train scores and find cutoff
            train_scores = self.model.predict_risk_score(X[train_index])
            cutoff = self.model.find_cutoff(train_scores)

            # Test
            risk_score_samples[test_index] = self.model.predict_risk_score(X[test_index])
            class_samples[test_index] = self.model.predict_class(risk_score_samples[test_index], cutoff)

        return class_samples, risk_score_samples