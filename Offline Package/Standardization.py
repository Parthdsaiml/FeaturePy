from scipy.stats import shapiro, anderson, skew, kurtosis
import pandas as pd
class Standardization:
    def __init__(self):
        pass

    def feature_scaling(self, df, column, method = None):
        pass

    def feature_scaling_choice(self, df):
        f_choices = {}

        for column in df.columns:
            if (df[column].dtypes == 'float64') | (df[column].dtypes == 'int64'):
                if (self.is_column_normal(df, column)):
                    f_choices[column] = 'Z'
                else:
                    if (self.is_otutliers(df, column)):
                        f_choices[column] = 'R'
                    else:
                        f_choices[column] = 'MM'
            else:
                f_choices[column] = 'Object'
        return f_choices
    
    def see_in_df(self, dict_value):
        for key, value in dict_value.items():
            if (value == 'R'):
                print(f'{key} is right skewed Robust')
            elif (value == 'MM'):
                print(f'{key} not normal with no outliers MINMAX')
            elif (value == 'Z'):
                print(f'{key} is normal Z-Score')
            else:
                print(f'{key} is object')
            print("---------------------------------------------------------------------------------------")

                
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

    def is_otutliers(self, df, column):
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

std.see_in_df(std.feature_scaling_choice(df))

