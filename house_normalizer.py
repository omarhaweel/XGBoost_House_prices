import pandas as pd

class HouseNormalizer:
    """
    A class to normalize house data for prediction.
    It handles missing values, converts data types, and encodes categorical variables.
    """
    
    def __init__(self, columns):
        self.columns = columns

    def preprocess_house_data(self, house_values, X):
        # Create DataFrame for the house
        user_house = pd.DataFrame([house_values], columns=self.columns)

        # Handle missing values for numeric columns
        numeric_cols_user = user_house.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols_user:
            if col in X.columns:  # Only process if column exists in training
                user_house[col] = user_house[col].fillna(X[col].median())

        # Handle missing values for categorical columns  
        categorical_cols_user = user_house.select_dtypes(include=['object']).columns
        for col in categorical_cols_user:
            if col in X.columns:  # Only process if column exists in training
                user_house[col] = user_house[col].fillna('Missing')

        # Handle columns that should be numeric but are stored as object
        for col in user_house.columns:
            if col in X.columns and user_house[col].dtype == 'object':
                try:
                    user_house[col] = pd.to_numeric(user_house[col], errors='coerce')
                    user_house[col] = user_house[col].fillna(X[col].median())
                except:
                    pass

        # Apply label encoding for remaining categorical columns
        for col in user_house.select_dtypes(include=['object']).columns:
            if col in X.columns:
                unique_train_vals = X[col].unique() if X[col].dtype in ['int64', 'float64'] else list(range(len(X[col].unique())))
                if user_house[col].iloc[0] == 'NA':
                    user_house[col] = unique_train_vals[0] if len(unique_train_vals) > 0 else 0
                else:
                    user_house[col] = unique_train_vals[0] if len(unique_train_vals) > 0 else 0

        return user_house