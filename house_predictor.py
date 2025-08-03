import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder
from house_normalizer import HouseNormalizer
train_df = pd.read_csv('houses-train.csv')
test_df = pd.read_csv('houses-test.csv')

# 2. Separate target and features in train
X = train_df.drop(['SalePrice', 'Id'], axis=1)
y = train_df['SalePrice']

# Keep test Ids for final submission or output
test_ids = test_df['Id']
X_test = test_df.drop(['Id'], axis=1)

# Handle missing values differently for numeric and categorical columns
# For numeric columns, fill with median
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    X[col] = X[col].fillna(X[col].median())
    X_test[col] = X_test[col].fillna(X[col].median())

# For object columns, fill with 'Missing' and then encode
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    X[col] = X[col].fillna('Missing')
    X_test[col] = X_test[col].fillna('Missing')

# Handle columns that should be numeric but are stored as object due to missing values
for col in X.columns:
    if X[col].dtype == 'object':
        # Try to convert to numeric, if it fails, keep as categorical
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
            # Fill any remaining NaN values with median
            X[col] = X[col].fillna(X[col].median())
            X_test[col] = X_test[col].fillna(X[col].median())
        except:
            pass

# encode categorical data
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    # Fit on training data
    X[col] = le.fit_transform(X[col].astype(str))
    
    # Transform test data, handling unseen categories
    X_test_col = X_test[col].astype(str)
    # Replace unseen categories with a known category (first class)
    X_test_col = X_test_col.apply(lambda x: x if x in le.classes_ else le.classes_[0])
    X_test[col] = le.transform(X_test_col)

# train XGBoost model
model = XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01)
model.fit(X, y)

predictions = model.predict(X_test)

output = pd.DataFrame({'Id': test_ids, 'SalePrice': predictions})
output.to_csv('submission.csv', index=False)
print("Model training complete and predictions saved to 'submission.csv'.")

# Predict on a specific house from user data
print("\n--- User's Specific House Prediction ---")


# column names for the house data
columns = ['MSSubClass','MSZoning','LotFrontage','LotArea','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','OverallQual','OverallCond','YearBuilt','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','Heating','HeatingQC','CentralAir','Electrical','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond','PavedDrive','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','PoolQC','Fence','MiscFeature','MiscVal','MoSold','YrSold','SaleType','SaleCondition']

# Example house values to predict
house_values = [20,'RL',69,11302,'Pave','NA','IR1','Lvl','AllPub','Inside','Gtl','StoneBr','Norm','Norm','1Fam','1Story',8,5,2005,2006,'Gable','CompShg','VinylSd','Other','BrkFace',238,'Gd','TA','PConc','Gd','TA','Gd','GLQ',1422,'Unf',0,392,1814,'GasA','Ex','Y','SBrkr',1826,0,0,1826,1,0,2,0,3,1,'Gd',7,'Typ',1,'TA','Attchd',2005,'Fin',3,758,'TA','TA','Y',180,75,0,0,120,0,'NA','NA','NA',0,8,2006,'New','Partial']

# get the normalizer instance
normalizer = HouseNormalizer(columns)
# normalize the house data
user_house = normalizer.preprocess_house_data(house_values, X)

# Make prediction
user_prediction = model.predict(user_house)
print(f"Predicted price for your house: ${user_prediction[0]:,.2f}")

# this model works nicely with the provided house data and can be used to predict the price of a specific house based on its features. The normalization process ensures that the input data is consistent with the training data, allowing for accurate predictions.