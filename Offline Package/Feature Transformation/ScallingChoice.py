import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Standardizaiton import Standardization
class ScallingChoice:
    def __init__ (self):
        pass

    def apply_Z_Score(self, df, column):
        df["Z-Score " + column] = (df[column] - df[column].mean()) / df[column].std()
        return df[["Z-Score " + column]]

    def apply_Robust(self, df, column):
        df["Robust " + column] = (df[column] - df[column].median()) / (df[column].quantile(0.75) - df[column].quantile(0.25))
        return df[["Robust " + column]]

    def apply_MinMax(self, df, column):
        df["MinMax " + column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        return df[["MinMax " + column]]
    
    def feature_scaling_choice(self, df, describe = False):
        st = Standardization()
        f_choices = {}
        for column in df.columns:
            if df[column].dtypes in ['float64', 'int64']:
                if st.is_column_normal(df, column):
                    f_choices[column] = 'Z'
                elif st.is_outliers(df, column):
                    f_choices[column] = 'R'
                else:
                    f_choices[column] = 'MM'
            else:
                f_choices[column] = 'Object'
        if (describe):
            self.see_ways_in_df(f_choices)
        else:
            f_choices_df = pd.DataFrame.from_dict(f_choices, orient='index', columns=['Scalling'])
            print(f_choices_df)
        return f_choices

    def see_ways_in_df(self, dict_value):
        messages = []    
        for key, value in dict_value.items():
            if value == 'R':
                messages.append(f'{key} is right skewed so using Robust')
            elif value == 'MM':
                messages.append(f'{key} not normal with no outliers so using MINMAX')
            elif value == 'Z':
                messages.append(f'{key} is normal so using Z-Score')
            else:
                messages.append(f'{key} is object')
            messages.append("---------------------------------------------------------------------------------------")
        print("\n".join(messages))
        return

sc = ScallingChoice()

df = pd.read_csv('Offline Package\churnData.csv')
sc.feature_scaling_choice(df)
