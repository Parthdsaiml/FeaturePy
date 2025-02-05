class Trends:
    def __init__(self):
        pass

    def get_trends(self, df):        
        pass

    def get_counts(self, df):
        counts_dict = {}

        for column in df.columns:
            counts_dict[column] = df[column].value_counts().to_dict()
        return counts_dict
    
    def get_unique_values(self, df):
        unique_values_dict = {}

        for column in df.columns:
            unique_values_dict[column] = len(df[column].unique().tolist())
        return unique_values_dict
    
    def get_frequency(self, df):
        frequency_dict = {}
        for column in df.columns:
            frequency_dict[column] = df[column].value_counts(normalize=True).to_dict() * 100
        return frequency_dict
            
        

    

        
        
