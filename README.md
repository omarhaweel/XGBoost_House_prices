# 🏠 XGBoost House Price Predictor

A machine learning project that predicts house prices using XGBoost regression algorithm. This project processes real estate data and provides accurate price predictions based on various house features.

## 📊 Project Overview

This project uses **XGBoost (eXtreme Gradient Boosting)** to predict house sale prices based on features like:
- Lot area and living space
- Overall quality and condition
- Year built and neighborhood
- Number of bedrooms, bathrooms, and garage capacity
- And many more architectural and location features

## 🚀 Quick Start

### Installation & Setup

1. **Clone or download the project**
   ```bash
   cd house_price_evaluator
   ```

2. **Activate the virtual environment**
   ```bash
   source .venv/bin/activate  # On macOS/Linux
   # OR
   .venv\Scripts\activate     # On Windows
   ```

3. **Install required packages** (if not already installed)
   ```bash
   pip install pandas xgboost scikit-learn
   ```

4. **Verify installation**
   ```bash
   python -c "import pandas, xgboost, sklearn; print('All packages installed successfully!')"
   ```

### Running the Program

1. **Make sure virtual environment is activated**
   ```bash
   source .venv/bin/activate
   ```

2. **Run the prediction script**
   ```bash
   python house_predictorr.py
   ```


## 📈 What the Program Does

1. **Data Preprocessing**:
   - Handles missing values (median for numeric, 'Missing' for categorical)
   - Converts mixed data types to appropriate formats
   - Encodes categorical variables to numbers
   - Processes both training and test data consistently

2. **Model Training**:
   - Uses XGBoost regression with 1000 estimators
   - Conservative learning rate (0.01) for stability
   - Trains on 1460 houses from the training dataset

3. **Predictions**:
   - Generates predictions for all test houses
   - Saves results to `submission.csv`
   - Shows example prediction for a specific house

4. **Output Files**:
   - `submission.csv`: Contains Id and predicted SalePrice for test data

## 🔧 Model Configuration

The XGBoost model uses these parameters:
- **Objective**: `reg:squarederror` (regression with squared error)
- **Estimators**: 1000 trees
- **Learning Rate**: 0.01 (conservative approach)



### Common Issues

1. **Virtual environment not activated**:
   ```bash
   source .venv/bin/activate  # Ensure this runs first
   ```

2. **Missing packages**:
   ```bash
   pip install pandas xgboost scikit-learn
   ```

3. **OpenMP issues on macOS**:
   ```bash
   brew install libomp
   ```
## 🤝 Contributing

Feel free to fork this project and experiment with:
- Different model parameters
- Additional feature engineering
- Alternative algorithms for comparison
- Enhanced data visualization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.