import numpy as np
import scipy.stats as stats
import pandas as pd
from scipy.stats import shapiro, normaltest, anderson


class Check:
    def __init__(self):
        pass


    def isNormal(self, df, column, alpha=0.05, sample=False, sample_size=None):
        col_data = df[column].dropna()  
        n = col_data.size  
        if sample:
            if sample_size:
                col_data = col_data.sample(n=sample_size, random_state=42) 
            else:
                raise ValueError("Sample size must be provided if sample flag is True")
            result = anderson(col_data)
            if (result.statistic < result.critical_values[2]):
                return True
            else:
                return False
        if n < 30:
            stat, p = shapiro(col_data)
        else:
            stat, p = normaltest(col_data)  
        if p > alpha:
            return True  
        else:
            return False  

    def isSkew(self, df, column, threshold=0.5):
        if pd.api.types.is_numeric_dtype(df[column]):
            return df[column].skew() > threshold
        else:
            return False

    def isOutlier(self, df, column):
        iqr = df[column].quantile(0.75) - df[column].quantile(0.25)
        lower_bound = df[column].quantile(0.25) - 1.5 * iqr
        upper_bound = df[column].quantile(0.75) + 1.5 * iqr
        return df[column].between(lower_bound, upper_bound, inclusive=True).mean() < 0.75

    def isConstant(self, df, column):
        pass

    def isHighCardinality(self, df, column):
        pass

    def isLowFrequency(self, df, column):
        pass

    def isSparse(self, df, column):
        pass

df = pd.read_csv('testData/healthcare-dataset-stroke-data.csv')

check = Check()
print(check.isNormal(df, 'bmi'))