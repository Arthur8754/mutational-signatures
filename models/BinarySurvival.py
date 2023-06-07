import numpy as np

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

    def predict(self, X: np.ndarray)->np.ndarray:
        """
        Predict the class of each patient in parameter.

        ### Parameters :
        - X (n_samples, n_features) : the features of each patient

        ### Returns :
        The class of each patient.
        """
        return self.model.predict(X)
    
