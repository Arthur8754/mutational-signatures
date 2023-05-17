import pandas as pd

class dataHandler:
    """
    Code for implementing the data reading and preprocessing.
    """
    def __init__(self, filename) -> None:
        self.filename = filename

    def readData(self):
        """ 
        Read the file and convert it into dataframe.
        """
        df = pd.read_excel(self.filename)
        return df
    


