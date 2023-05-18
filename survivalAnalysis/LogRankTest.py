import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

class LogRankTest:
    """ 
    Code for implementing the Log-Rank Test method which compares 2 survival curves.
    """
    def __init__(self, status, survival, group) -> None:
        self.status = status
        self.survival = survival
        self.group = group

    def split_per_group(self):
        """
        Split the dataset in 2 groups (one per kind of treatment).
        """
        df = pd.DataFrame({"status":self.status, "survival":self.survival, "group":self.group})
        group1, group2 = df.loc[df['group'] == 1], df.loc[df['group'] == 2]
        return group1, group2
    
    def stat_test(self):
        """ 
        Make the log rank test.
        """
        # Split per group
        group1, group2 = self.split_per_group()
        T_group1, T_group2 = group1["survival"], group2["survival"]
        E_group1, E_group2 = group1["status"], group2["status"]
        
        # Fit the data to determine features for log rank test.
        kmf = KaplanMeierFitter(alpha=0.05)
        kmf.fit(T_group1, event_observed=E_group1)
        kmf.fit(T_group2, event_observed=E_group2)

        # Make the test
        test_results = logrank_test(T_group1, T_group2, E_group1, E_group2, alpha=0.99)
        return test_results.p_value