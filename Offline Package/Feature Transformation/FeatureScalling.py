from scipy.stats import shapiro, anderson, skew, kurtosis
import pandas as pd


class DifferenceCheck:
    def __init__(self):
        pass    

    def check_difference(self, column1, column2, threshold=1.0):
        difference = abs(column1 - column2)
        if difference > threshold:
            # Apply scaling logic here
            # For example, scaling column1 and column2
            scaled_column1 = column1 / difference
            scaled_column2 = column2 / difference
            return scaled_column1, scaled_column2
        return column1, column2


class Standardization:
    def __init__(self):
        pass

    def feature_scaling(self, df, column, method = None):
        f_set = self.feature_scaling_choice(df)
        if (method != None):
            return None
        else:
            for col in f_set.keys():
                if f_set[col] == 'Z':
                    self.apply_Z_Score(df, col)
                elif f_set[col] == 'R':
                    self.apply_Robust(df, col)
                elif f_set[col] == 'MM':
                    self.apply_MinMax(df, col)
                else:
                    pass
            return df
        
                
    def apply_Z_Score(self, df, column):
        df["Z-Score " + column] = (df[column] - df[column].mean()) / df[column].std()
        return df
    def apply_Robust(self, df, column):
        df["Robust " + column] = (df[column] - df[column].median()) / (df[column].quantile(0.75) - df[column].quantile(0.25))
        return df
    def apply_MinMax(self, df, column):
        df["MinMax " + column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        return df
    
    def feature_scaling_choice(self, df):
        f_choices = {}

        for column in df.columns:
            if df[column].dtypes in ['float64', 'int64']:
                if self.is_column_normal(df, column):
                    f_choices[column] = 'Z'
                elif self.is_outliers(df, column):
                    f_choices[column] = 'R'
                else:
                    f_choices[column] = 'MM'
            else:
                f_choices[column] = 'Object'
        return f_choices
    
    def see_in_df(self, dict_value):
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

                
    def is_column_normal(self, df, column, alpha = 0.05):
        data = df[column]
        stat_shapiro, p_shapiro = shapiro(data)
        if p_shapiro <= alpha:
            return False  # Reject normality if p-value is below alpha

        # 2. Anderson-Darling Test
        result_anderson = anderson(data, dist='norm')
        if result_anderson.statistic > result_anderson.critical_values[2]:  # Compare with 5% significance level
            return False

        data = df[column].dropna()
        # 3. Skewness and Kurtosis
        skewness = skew(data)
        kurt = kurtosis(data, fisher=False)  # Use non-Fisher for comparison with 3
        if abs(skewness) > 0.5 or abs(kurt - 3) > 1:
            return False

        return True  # Passes all tests

    def convert_to_binary(self, df):
        for col in df.select_dtypes(include=[bool]).columns:
            df[col] = df[col].map({True: 1, False: 0})
        for col in df.columns:
            if df[col].nunique() in [1, 2] and df[col].hasnans:
                df[col] = df[col].fillna(df[col].mode()[0])
            elif df[col].nunique() in [1, 2] and not df[col].hasnans:
                df[col] = df[col].mode()[0] 
        return df
    
    def is_outliers(self, df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return len(outliers) > 0

    def right_skewed(self, df, column):
        # check column is right skewed
        skew_value = df[column].skew()
        if skew_value > 0.5:
            return True
        else:
            return False
        

    def left_skewed(self, df, column):
        pass

df = pd.read_csv('Offline Package/churnData.csv')
std = Standardization()


df = std.feature_scaling(df, 'Contract')
print(df.columns)
df = std.convert_to_binary(df)
print(df.head())
