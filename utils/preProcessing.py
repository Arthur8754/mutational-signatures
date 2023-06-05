import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class preProcessing:
    """
    Class which implements some useful preprocessings.
    """
    def __init__(self) -> None:
        pass

    def delete_nan_values(self, df: pd.DataFrame)->pd.DataFrame:
        """
        Delete rows which contains at least 1 NaN value.

        ### Parameters :
        - df : the dataframe to analyze

        ### Returns :
        The dataframe without NaN values.
        """
        return df.dropna()
    
    def filter_column(self, df: pd.DataFrame, column_name: str, expected_value)->pd.DataFrame:
        """
        Select df rows for which the column_name values are expected_value.

        ### Parameters :
        - df : the dataframe to filter
        - column_name : the name of the column to analyze for filtering
        - expected_value : the expected value of column_name

        ### Returns :
        The filtered dataframe.
        """
        return df.loc[df[column_name] == expected_value]
    
    def normalize_data(self, X: np.ndarray)->np.ndarray:
        """
        Normalize the dataframe using the Standard Scaler.

        ### Parameters :
        - the 2D numpy array to normalize

        ### Returns :
        The 2D normalized numpy array
        """
        return StandardScaler().fit_transform(X)