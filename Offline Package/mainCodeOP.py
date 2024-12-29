import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import seaborn as sns
from Trends import Trends

unique_identifiers = [
        "RowNumber",
        "ID",
        "UserID",
        "UserId",
        "CustomerId",
    
        "CustomerID",
        "ProductID",
        "EmployeeID",
        "StudentID",
        "AccountNumber",
        "TransactionID",
        "OrderID",
        "InvoiceNumber",
        "SessionID",
        "ActivityID",
        "VisitID",
        "UUID",
        "UUID(UniversallyUniqueIdentifier)",
        "SocialSecurityNumber(SSN)",
        "DriversLicenseNumber",
        "BusinessLicenseNumber",
        "LoyaltyProgramID",
        "MembershipID",
        "RegistrationNumber",
        "RollNumber",
        "User ID",
        "Customer ID",
        "Product ID",
        "Employee ID",
        "Student ID",
        "Account Number",
        "Transaction ID",
        "Order ID",
        "Invoice Number",
        "Session ID",
        "Activity ID",
        "Visit ID",
        "UUID (Universally Unique Identifier)",
        "Social Security Number (SSN)",
        "Driver's License Number",
        "Business License Number",
        "Loyalty Program ID",
        "Membership ID",
        "Registration Number",
        "Roll Number",
    ]
usefullColumns = [
    'Revenue',
    'Customer Satisfaction Score',
    'Net Promoter Score',
    'Sales Volume',
    'Churn Rate',
    'Patient Age',
    'Blood Pressure',
    'Cholesterol Level',
    'Body Mass Index',
    'Blood Sugar Level',
    'Temperature',
    'Humidity',
    'Air Quality Index',
    'Precipitation Levels',
    'Wind Speed',
    'Income Level',
    'Education Level',
    'Employment Status',
    'Age',
    'Marital Status',
    'Product Ratings',
    'User Engagement Metrics',
    'Inventory Levels',
    'Market Share',
    'Cost per Acquisition',
    'Geographic Location',
    'Date',
    'Product Category',
    'Customer Segment'
]

class IrrelvantColumns:
    def __init__(self):
        pass

    def constantValue(self, column):
        return column.nunique() == 1

    def check_high_cardinality_low_frequency(self, df, column, cardinality_threshold=0.1, frequency_threshold=0.05):
        # Calculate the number of distinct values
        num_distinct_values = df[column].nunique()
        num_rows = len(df)
    
        # High cardinality check: More distinct values than the threshold percentage of total rows
        if num_distinct_values / num_rows < cardinality_threshold:
            return False
    
        # Check frequency of values
        value_counts = df[column].value_counts(normalize=True)
    
        # Check if a significant portion of the values have a low frequency (below the threshold)
        low_frequency_count = sum(value_counts[value_counts < frequency_threshold])
    
        # High cardinality and low frequency condition
        if low_frequency_count > 0.5:  # At least 50% of the distinct values are low frequency
            return True
    
        return False

    def is_highly_skewed(self, df, column, threshold=1.0):
        # Check if the column exists in the DataFrame
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    
        # Ensure the column is numerical
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Column '{column}' is not a numerical column.")
    
        # Calculate skewness of the column, dropping any NaN values
        skewness_value = df[column].skew()  # .dropna() handles missing values
    
        # Return True if the absolute skewness is greater than the threshold, otherwise False
        return abs(skewness_value) > threshold

    def find_identical_columns_optimized(self, df):
        identical_column_pairs = []
        column_hashes = {}

        for col in df.columns:
            column_hash = hash(tuple(df[col].values))
        
            if column_hash in column_hashes:
                identical_column_pairs.append((column_hashes[column_hash], col))
            else:
                column_hashes[column_hash] = col

        return identical_column_pairs

    def check_sparse_data(self, df, column, threshold=0.9):
        """Check if a column has too many unique values compared to the total number of rows."""
        num_distinct_values = df[column].nunique()
        num_rows = len(df)
    
        # If the proportion of unique values exceeds the threshold, flag as sparse
        if num_distinct_values / num_rows > threshold:
            return True
    
        return False

    def removeColumns(self, df, targetColumn, threshold=1.0, cardinality_threshold=0.1, frequency_threshold=0.05, sparse_threshold=0.9):
        removalList = {
            'constant_values': [],
            'high_cardinality_low_frequency': [],
            'highly_skewed': [],
            'useless_columns': [],
            'identical_columns': [],
            'sparse_columns': [],  # Added for sparse columns
            "Outliers": []
        }

        # Identify identical columns first
        removalList['identical_columns'] = self.find_identical_columns_optimized(df)

        # Assuming OutlierDetection is defined elsewhere in your code
        ot = OutlierDetection(df, targetColumn)
        oList = ot.detectOutliers()
        removalList["Outliers"].append(oList)

        # Loop through each column and classify it based on the criteria
        for column in df.columns:
            if self.constantValue(df[column]):
                removalList['constant_values'].append(column)

            if self.check_high_cardinality_low_frequency(df, column, cardinality_threshold, frequency_threshold):
                removalList['high_cardinality_low_frequency'].append(column)

            if pd.api.types.is_numeric_dtype(df[column]):
                if self.is_highly_skewed(df, column, threshold):
                    removalList['highly_skewed'].append(column)

            # Check for sparse columns
            if self.check_sparse_data(df, column, sparse_threshold):
                removalList['sparse_columns'].append(column)

            # Assuming `unique_identifiers` is defined elsewhere, and its logic is correct
            if column in unique_identifiers:
                removalList['useless_columns'].append(column)

        return removalList
    
    
    def getCleanDFOfRemovedColumns(self, df, targetColumn, threshold=1.0, cardinality_threshold=0.1, frequency_threshold=0.05, sparse_threshold=0.9):
        removalList = self.removeColumns(df, targetColumn, threshold, cardinality_threshold, frequency_threshold, sparse_threshold)
        columns_to_remove = []
        for key in removalList:
            columns_to_remove.extend(removalList[key])
        df_clean = df.drop(columns=columns_to_remove)
        return df_clean
    
    
        

class OutlierDetection:
    def __init__(self, df, target_column):
        self.df = df.copy()  # Make a copy of the DataFrame to avoid modifying original data
        self.target_column = target_column
        
    def isolation_forest_outliers(self, column, contamination=0.1):
        """
        Detect outliers using the Isolation Forest method.
        """
        data = column.values.reshape(-1, 1)
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(data)
        return np.where(outliers == -1)[0].tolist()

    
    def getIQRRange(self, column, dynamicValue = -1):
        """
        Calculate the IQR (Interquartile Range) and dynamic range for outlier detection.
        """
        sortedData = np.sort(column)
        if len(sortedData) <= 1:
            return [sortedData[0], sortedData[0]]  # If only 1 or 0 elements, no IQR calculation

        Q1 = np.percentile(sortedData, 25)
        Q3 = np.percentile(sortedData, 75)
        IQR = Q3 - Q1
        lowerBound = Q1 - (dynamicValue if dynamicValue != -1 else 1.5) * IQR
        upperBound = Q3 + (dynamicValue if dynamicValue != -1 else 1.5) * IQR
        return [lowerBound, upperBound]

    def iqrOutliers(self, column, valueDynamic=-1):
        """
        Identify outliers in a column based on IQR.
        """
        iqrRange = self.getIQRRange(column, valueDynamic)
        outlier_indices = [idx for idx, value in enumerate(column) if value < iqrRange[0] or value > iqrRange[1]]
        return outlier_indices

    def sdRange(self, column, dynamicValue=-1):
        """
        Calculate the SD (Standard Deviation) range for outlier detection.
        """
        meanValue = column.mean()
        stdValue = column.std()
        lowerRange = meanValue - (dynamicValue if dynamicValue != -1 else 3) * stdValue
        upperRange = meanValue + (dynamicValue if dynamicValue != -1 else 3) * stdValue
        return [lowerRange, upperRange]

    def sdOutliers(self, column, valueDynamic=-1):
        """
        Identify outliers in a column based on Standard Deviation.
        """
        rangeSd = self.sdRange(column, valueDynamic)
        outlierIndices = [idx for idx, value in enumerate(column) if value < rangeSd[0] or value > rangeSd[1]]
        return outlierIndices




    

    def skewedDetection(self):
        """
        Detect skewed columns in the dataframe.
        Returns a list of skewness values for numeric columns.
        """
        skewedList = []
        for column in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[column]):
                if self.df[column].nunique() < 5:  # Ignore very low cardinality columns
                    skewedList.append(None)
                    continue
                skew_value = self.df[column].skew()
                if skew_value > 0.5:
                    skewedList.append(1)  # Right skewed
                elif skew_value < -0.5:
                    skewedList.append(-1)  # Left skewed
                else:
                    skewedList.append(0)  # Not skewed
            else:
                skewedList.append(None)  # For non-numeric columns
        return skewedList

    # Returns True if there are outliers, False otherwise

    # Existing methods from the previous part would be here...



    def showOutliers(self, plot_type="boxplot"):
    # Ensure plot_type is a string and is valid
        if not isinstance(plot_type, str):
            raise ValueError("plot_type must be a string")

        if plot_type not in ["boxplot", "scatter", "histogram"]:
            raise ValueError(f"Invalid plot_type: {plot_type}. Supported plot types are ['boxplot'].")
    
        for column in self.df.columns:
        # Only plot numeric columns
            if self.df[column].dtype in ['float64', 'int64']:
                sns.set_palette(["#FFA07A"])  # Light Salmon (light orange)

                sns.set_style("whitegrid")  # Adds a soft white grid background

                plt.figure(figsize=(5, 3))

                if plot_type == "boxplot":
                    sns.boxplot(y=self.df[column], color="#FF8C00")  # Skyblue for a calming look
                    plt.title(f"Box Plot of {column}", fontsize=14, fontweight='bold')
                    plt.ylabel(column, fontsize=12)  # Fixed the typo here
                elif plot_type == 'scatter':
                    plt.scatter(self.df.index, self.df[column], color='#FF8C00', alpha=0.7)  # Light coral for soothing color
                    plt.title(f'Scatter Plot of {column}', fontsize=14, fontweight='bold')
                    plt.ylabel(column, fontsize=12)
                    plt.xlabel('Index', fontsize=12)
                elif plot_type == 'histogram':
                    sns.histplot(self.df[column], bins=30, kde=True, color='#FF8C00')  # Lightseagreen for a calm histogram color
                    plt.title(f'Histogram of {column}', fontsize=14, fontweight='bold')
                    plt.xlabel(column, fontsize=12)
                    plt.ylabel('Frequency', fontsize=12)
        
# Adjust font sizes for readability
                plt.tight_layout()
                plt.show()
    


            
    def showColumnOutliers(self, column, plot_type='boxplot'):
        # Check if the column exists in the DataFrame
        if column not in self.df.columns:
            print(f"Column '{column}' does not exist in the DataFrame.")
            return
        
        plt.figure(figsize=(8, 6))
        
        if plot_type == 'boxplot':
            # Create a box plot
            sns.boxplot(y=self.df[column])
            plt.title(f'Box Plot of {column}')
            plt.ylabel(column)

        elif plot_type == 'scatter':
            # Create a scatter plot
            plt.scatter(self.df.index, self.df[column])
            plt.title(f'Scatter Plot of {column}')
            plt.ylabel(column)
            plt.xlabel('Index')

        elif plot_type == 'histogram':
            # Create a histogram
            sns.histplot(self.df[column], bins=30, kde=True)
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')

        else:
            print(f"Plot type '{plot_type}' is not supported.")
            return
        
        plt.show()
    def detectOutliersIndex(self, count=False):
        all_outliers = {}

        # Iterate over all columns
        for column in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[column]):
                data = self.df[column].dropna()

                # Skip columns with constant values (no variance)
                if data.nunique() == 1:
                    continue

                # Calculate IQR for the column
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1

                # Define the bounds for outliers (1.5 * IQR rule)
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Identify the outliers in the column based on index
                outlier_indices = data[(data < lower_bound) | (data > upper_bound)].index

                # If outliers exist, store either the count or the list of outliers' indices
                if not outlier_indices.empty:
                    if count:
                        all_outliers[column] = len(outlier_indices)  # Store count of outliers
                    else:
                        all_outliers[column] = outlier_indices.tolist()  # Store list of outlier indices
                else:
                    # If no outliers detected, add an empty list or a placeholder
                    all_outliers[column] = []

            elif pd.api.types.is_object_dtype(self.df[column]):
                # For categorical data, detect rare categories as outliers
                value_counts = self.df[column].value_counts(normalize=True)
                rare_categories = value_counts[value_counts < 0.01].index.tolist()  # Less than 1% frequency

                # Identify indices where rare categories appear
                rare_indices = self.df[column][self.df[column].isin(rare_categories)].index

                # If rare categories exist, add their indices as outliers
                if not rare_indices.empty:
                    if count:
                        all_outliers[column] = len(rare_indices)  # Store count of rare category indices
                    else:
                        all_outliers[column] = rare_indices.tolist()  # Store list of indices for rare categories
                else:
                    # If no rare categories, add an empty list or a placeholder
                    all_outliers[column] = []

        return all_outliers

    
    def detectOutliers(self, count = True):
        all_outliers = {}

        # Iterate over all columns
        for column in self.df.columns:
            # Skip non-numeric columns
            if pd.api.types.is_numeric_dtype(self.df[column]):
                data = self.df[column].dropna()

                # Skip columns with constant values (no variance)
                if data.nunique() <= 2:
                    continue
                # Calculate IQR for the column
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1

                # Define the bounds for outliers (1.5 * IQR rule)
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Identify the outliers in the column
                outliers = data[(data < lower_bound) | (data > upper_bound)]

                # If outliers exist, store either the count or the list of outliers
                if not outliers.empty:
                    if count:
                        all_outliers[column] = len(outliers)  # Store count of outliers
                    else:
                        all_outliers[column] = outliers.tolist()  # Store list of outliers
                else:
                    # If no outliers detected, add an empty list or a placeholder
                    all_outliers[column] = []

            elif pd.api.types.is_object_dtype(self.df[column]):
                # For categorical data, detect rare categories as outliers
                value_counts = self.df[column].value_counts(normalize=True)
                rare_categories = value_counts[value_counts < 0.01].index.tolist()  # Less than 1% frequency

                # If rare categories exist, add them as outliers
                if rare_categories:
                    if count:
                        all_outliers[column] = len(rare_categories)  # Store count of rare categories
                    else:
                        all_outliers[column] = rare_categories  # Store the list of rare categories
                else:
                    # If no rare categories, add an empty list or a placeholder
                    all_outliers[column] = []

        return all_outliers

    def removeOutliers(self):
        
        all_outliers = self.detectOutliersIndex(count=False)

        # Collect all the outlier indices across all columns
        outlier_indices_set = set()
        for outliers in all_outliers.values():
            outlier_indices_set.update(outliers)

        # Remove the rows with the outlier indices
        self.df = self.df.drop(index=outlier_indices_set)

        return self.df


        
    def detectCategoricalOutliers(self, column_name, threshold_percent=1):
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
        column = self.df[column_name]
        category_counts = column.value_counts()
        threshold = len(column) * (threshold_percent / 100)
        outliers = category_counts[category_counts < threshold]
        return outliers.index.tolist(), outliers.values.tolist()  # Return categories and their counts
        
    def detectColumnOutliers(self, column, boolean=False):
        outliers = {}

        # Check if the specified column is numeric
        if pd.api.types.is_numeric_dtype(self.df[column]):
            columnSkewness = self.df[column].skew()
        
            # If skewness is high, use the IQR method
            if abs(columnSkewness) > 0.5:
                outliers[column] = self.iqrOutliers(self.df[column])
            else:
                # If skewness is low, use the standard deviation method
                outliers[column] = self.sdOutliers(self.df[column])

            # If `boolean` is True, return True if outliers exist, otherwise False
            if boolean:
                return len(outliers[column]) > 0  # True if outliers are detected, False otherwise

        elif pd.api.types.is_object_dtype(self.df[column]):
            # If the column is categorical, use a categorical outlier detection method
            outliers[column] = self.detectCategoricalOutliers(column)
        
            # If `boolean` is True, return True if outliers exist, otherwise False
            print(outliers[column])
            if boolean:
                return len(outliers[column]) > 0  # True if outliers are detected, False otherwise

        # If `boolean` is False, return the outliers dictionary for that column
            
        return outliers

    def detectDynamicOutliers(self, boolean=False):
        allOutliers = {}

        # Check if the target column is numeric
        if pd.api.types.is_numeric_dtype(self.df[self.target_column]):
            targetSkewness = self.df[self.target_column].skew()
            if abs(targetSkewness) > 0.5:
                # Use IQR for skewed data
                targetOutliers = self.iqrOutliers(self.df[self.target_column])
            else:
                # Use SD method for data that's not skewed
                targetOutliers = self.sdOutliers(self.df[self.target_column])

            if len(targetOutliers) != 0:
                allOutliers[self.target_column] = targetOutliers
        else:
            # Categorical data outlier detection
            targetOutliers = self.detectCategoricalOutliers(self.target_column)
            if len(targetOutliers) != 0:
                allOutliers[self.target_column] = targetOutliers

        # Iterate through all columns except the target column
        for column in self.df.columns:
            if column == self.target_column:
                continue
        
            # For numeric columns
            if pd.api.types.is_numeric_dtype(self.df[column]):
                columnSkewness = self.df[column].skew()
                if abs(columnSkewness) > 0.5:
                    outliers = self.iqrOutliers(self.df[column])  # Skewed data -> IQR
                else:
                    outliers = self.sdOutliers(self.df[column])  # Normal data -> SD method
            elif pd.api.types.is_object_dtype(self.df[column]):
                # For categorical data, use a specific method
                outliers = self.detectCategoricalOutliers(column)
        
            # Only add columns with detected outliers
            if len(outliers) != 0:
                allOutliers[column] = outliers

        # If `boolean` is True, return True if any outliers were detected, otherwise False
        if boolean:
            return len(allOutliers) > 0  # Return True if there are any outliers, otherwise False

        # If `boolean` is False, return the dictionary of all detected outliers
        return allOutliers

    def handleOutliers(self, series, outliers, method="impute", lower_bound=None, upper_bound=None):
        if len(outliers) > 0:
            if method == "remove":
            # Option 1: Remove outliers
                series = series[~series.isin(outliers)]

            elif method == "cap":
            # Option 2: Cap outliers to a lower or upper bound (e.g., IQR or SD bounds)
                series = series.clip(lower=lower_bound, upper=upper_bound)

            elif method == "impute":
            # Option 3: Impute outliers with the mean of the surrounding values
                for outlier in outliers:
                    surrounding_values = series[series.index < outlier].tail(1).append(series[series.index > outlier].head(1))
                    imputed_value = surrounding_values.mean()
                    series.loc[series == outlier] = imputed_value  # This avoids the SettingWithCopyWarning

            else:
                print("Invalid method specified. Please use 'remove', 'cap', or 'impute'.")
        return series

        
    def automateOutliers(self, way = "impute"):
        allOutliers = self.detectOutliers(count = False)
        if self.target_column in allOutliers:
            if pd.api.types.is_numeric_dtype(self.df[self.target_column]):
                targetOutliers = allOutliers[self.target_column]
                self.df[self.target_column] = self.handleOutliers(self.df[self.target_column], targetOutliers, way)

        for column in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[column]):
                if column in allOutliers:
                    columnOutliers = allOutliers[column]
                    self.df[column] = self.handleOutliers(self.df[column], columnOutliers, way)
            
        return self.df
      
    # def automateOutliersAndNormalisation(self, target = False, columnC = False):
    #     allOutliers = self.detectOutliers()
        
    #     # Step 1: Handle outliers in the target column
    #     if self.target_column in allOutliers:
    #         targetOutliers = allOutliers[self.target_column]
    #         self.df[self.target_column] = self.handleOutliers(self.df[self.target_column], targetOutliers)

    #     # Step 2: Handle outliers in other columns
    #     for column in self.df.columns:
    #         if column != self.target_column and column in allOutliers:
    #             columnOutliers = allOutliers[column]
    #             self.df[column] = self.handleOutliers(self.df[column], columnOutliers)

    #     # Step 3: Apply transformations to normalize the data
    #     if (target and columnC):
    #         self.apply_transformation()
    #     elif (target):
    #         self.apply_transformationJustTarget()
    #     elif (columnC):
    #         self.apply_transformation()
    #     else:
    #         self.apply_transformation()
    
     
    # def apply_transformationJustTarget(self):
    #     data = self.df[self.target_column]  # Get the target column from the DataFrame
    #     skewness = stats.skew(data)
        
    #     # Step 1: Check for negative or zero values and apply Yeo-Johnson if needed
    #     if np.any(data <= 0):
    #         pt = PowerTransformer(method='yeo-johnson')
    #         self.df[self.target_column] = pt.fit_transform(data.values.reshape(-1, 1)).flatten()  # Apply Yeo-Johnson

    #     # After Yeo-Johnson, re-check skewness
    #     data = self.df[self.target_column]  # Re-get the target column after transformation
    #     skewness = stats.skew(data)
        
    #     # Step 2: Apply transformations based on skewness
    #     if skewness > 1:  # Positively skewed data
    #         self.df[self.target_column] = np.log(data + 1)  # Log transformation (adding 1 to handle zero values)
    #     elif skewness < -1:  # Negatively skewed data
    #         # Box-Cox requires strictly positive values
    #         self.df[self.target_column], _ = stats.boxcox(data[data > 0])  # Filter out non-positive values for Box-Cox
    #     elif 0 < skewness <= 1:  # Moderately positively skewed data
    #         self.df[self.target_column] = np.sqrt(data)  # Square root transformation
    #     elif -1 <= skewness < 0:  # Moderately negatively skewed data
    #         # Box-Cox transformation for moderately skewed negative data (requires positive values)
    #         self.df[self.target_column], _ = stats.boxcox(data[data > 0])
    #     # After transformation, re-check normality
    #     return self.check_normality()
        
    # def applyTransofmation(self):
    #     for column in self.df.columns:
    #         data = self.df[column]
    #         skewness = stats.skew(data)
    #         # Apply Yeo-Johnson if there are negative or zero values
    #         if np.any(data <= 0):
    #             pt = PowerTransformer(method='yeo-johnson')
    #             self.df[column] = pt.fit_transform(data.values.reshape(-1, 1)).flatten()
    #         # After transformation, re-check skewness and apply appropriate transformations
    #         data = self.df[column]
    #         skewness = stats.skew(data)
    #         # Apply transformations based on skewness
    #         if skewness > 1:  # Positively skewed data
    #             self.df[column] = np.log(data + 1)
    #         elif skewness < -1:  # Negatively skewed data
    #             self.df[column], _ = stats.boxcox(data[data > 0])
    #         elif 0 < skewness <= 1:  # Moderately positively skewed
    #             self.df[column] = np.sqrt(data)
    #         elif -1 <= skewness < 0:  # Moderately negatively skewed
    #             self.df[column], _ = stats.boxcox(data[data > 0])
    #     return self.check_normality()
        
    # def applyTransformationExceptTarget(self):
    # # Apply transformations to all columns, except the target column
    #     for column in self.df.columns:
    #         if column != self.target_column:  # Skip the target column
    #             data = self.df[column]
    #             skewness = stats.skew(data)

    #         # Apply Yeo-Johnson if there are negative or zero values
    #             if np.any(data <= 0):
    #                 pt = PowerTransformer(method='yeo-johnson')
    #                 self.df[column] = pt.fit_transform(data.values.reshape(-1, 1)).flatten()
    #         # After transformation, re-check skewness and apply appropriate transformations
    #             data = self.df[column]
    #             skewness = stats.skew(data)
    #         # Apply transformations based on skewness
    #             if skewness > 1:  # Positively skewed data
    #                 self.df[column] = np.log(data + 1)
    #             elif skewness < -1:  # Negatively skewed data
    #                 self.df[column], _ = stats.boxcox(data[data > 0])
    #             elif 0 < skewness <= 1:  # Moderately positively skewed
    #                 self.df[column] = np.sqrt(data)
    #             elif -1 <= skewness < 0:  # Moderately negatively skewed
    #                 self.df[column], _ = stats.boxcox(data[data > 0])
    #     return self.check_normality()
        
    def check_normality(self):
        for column in self.df.columns:
            data = self.df[column]
            normality_test = NormalityTest(data)
            is_normal = normality_test.check_normality()
            if is_normal:
                continue
            else:
                return False
        return True

class GetSummary:
    def __init__(self):
        pass

    def central_tendency(self, df, way="all"):
        total_dict = {}
        for col in df.columns:
            if df[col].dtype in ["int64", "float64"]:
                if way == "all":
                    total_dict[col] = {
                        "mean": df[col].mean(),
                        "median": df[col].median(),
                        "mode": df[col].mode()[0] if not df[col].mode().empty else np.nan
                    }
                elif way == "mean":
                    total_dict[col] = df[col].mean()
                elif way == "median":
                    total_dict[col] = df[col].median()
                elif way == "mode":
                    total_dict[col] = df[col].mode()[0] if not df[col].mode().empty else np.nan
                elif way == "mean_median":
                    total_dict[col] = {"mean": df[col].mean(), "median": df[col].median()}
                elif way == "mean_mode":
                    total_dict[col] = {"mean": df[col].mean(), "mode": df[col].mode()[0] if not df[col].mode().empty else np.nan}
                elif way == "median_mode":
                    total_dict[col] = {"median": df[col].median(), "mode": df[col].mode()[0] if not df[col].mode().empty else np.nan}
            else:
                total_dict[col] = {
                        "mean": np.nan,
                        "median": np.nan,
                        "mode": df[col].mode()[0] if not df[col].mode().empty else np.nan
                    }
        total_df = pd.DataFrame(total_dict)
        return total_df
    def print_central_tendency(self, df, way="all"):
        total_df = self.central_tendency(df, way)
        for col in total_df.columns:
            print(f"{col}: {total_df[col].to_dict()}")
            print("---------------------------------------------------------------------------------------")


    def _initDict(self, df):
        totalDict = {}
        for col in df.columns:
            totalDict[col] = None
        return totalDict

    def _centralTendency(self, df, col):
        centralTendency = {}
        centralTendency["mean"] = df[col].mean()
        centralTendency["median"] = df[col].median()
        centralTendency["mode"] = df[col].mode()[0] if not df[col].mode().empty else np.nan
        return centralTendency

    def CTMean(self, df):
        totalDict = self._initDict(df)
        for col in df.columns:
            totalDict[col] = self._mean(df, col)
        return totalDict
    
    def _mean(self, df, col):
        mean = df[col].mean()
        return mean
    
    def CTMedian(self, df):
        totalDict = self._initDict(df)
        for col in df.columns:
            totalDict[col] = self._median(df, col)
        return totalDict
    
    def _median(self, df, col):
        median = df[col].median()
        return median
    
    def CTMode(self, df):
        totalDict = self._initDict(df)
        for col in df.columns:
            totalDict[col] = self._mode(df, col)
        return totalDict
    
    def _mode(self, df, col):
        mode = df[col].mode()[0] if not df[col].mode().empty else np.nan
        return mode
    
    def spread_variability(self, df, way="all"):
        spread_dict = {}
        for col in df.columns:
            if (df[col].dtype in ["int64", "float64"]) and way == "all":
                spread_dict[col] = {
                    "range": df[col].max() - df[col].min(),
                    "standard_deviation": df[col].std(),
                    "variance": df[col].var(),
                    "coefficient_of_variation": df[col].std() / df[col].mean()
                }
            else:
                spread_dict[col] = {
                    "range": np.nan,
                    "standard_deviation": np.nan,
                    "variance": np.nan,
                    "coefficient_of_variation": np.nan
                }
        spread_df = pd.DataFrame(spread_dict)
        return spread_df
    
    def print_spread_variability(self, df, way="all"):
        spread_dict = self.spread_variability(df, way)
        for col in spread_dict.keys():
            print(f"{col}: {spread_dict[col]}")
            print("---------------------------------------------------------------------------------------")
   
    def covariance_matrix(self, df):
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        cov_matrix = df[numeric_cols].cov()
        return cov_matrix  

    def print_covariance_matrix(self, df):
        cov_matrix = self.covariance_matrix(df)
        print(cov_matrix)
        print("---------------------------------------------------------------------------------------")
    
    def see_covariance_matrix(self, df):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.covariance_matrix(df), annot=True, cmap="coolwarm")
        plt.title("Covariance Matrix")
        plt.show()

    def distribution(self, df):
        dist_dict = {}
        for col in df.columns:
            if (df[col].dtype in ["int64", "float64"]):
                dist_dict[col] = {
                    "skewness": df[col].skew(),
                    "kurtosis": df[col].kurtosis()
                }
            else:
                dist_dict[col] = {
                    "skewness": np.nan,
                    "kurtosis": np.nan
                }
        dist_df = pd.DataFrame(dist_dict)
        return dist_df
    def print_distribution(self, df):
        dist_df = self.distribution(df)
        for col in dist_df.columns:
            print(f"{col}: {dist_df[col].to_dict()}")
            print("---------------------------------------------------------------------------------------")
    

    def see_distribution(self, df):
        dist_df = self.distribution(df)
        print(dist_df)
        print("---------------------------------------------------------------------------------------")

   
    

    def histogram(self, df):
        for col in df.columns:
            plt.figure(figsize=(10, 8))
            sns.histplot(df[col], bins=30, kde=True)
            plt.title(f"Histogram of {col}")
            plt.show()
            print("---------------------------------------------------------------------------------------")
        return
    
 
    def relationships(self, df):
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        plt.figure(figsize=(10, 8))
        sns.pairplot(df[numeric_cols])
        plt.show()
        print("---------------------------------------------------------------------------------------")
        return

tr = Trends()
tr.intro()