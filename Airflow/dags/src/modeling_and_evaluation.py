
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to train the model and make predictions
def train_and_predict(augmented_data, test_data, combined_features):
    """
    Trains a linear regression model and predicts on the test set.

    Parameters:
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training target.
    - X_test (pd.DataFrame): Test features.

    Returns:
    - np.ndarray: Predicted values for the test set.
    """
    X_train = augmented_data[combined_features]
    y_train = augmented_data['SalePrice']
    X_test = test_data.drop[combined_features]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

# Function to evaluate the model's performance
def evaluate_model(test_data, y_pred):
    """
    Evaluates model performance using various metrics.

    Parameters:
    - y_test (pd.Series): Actual target values for the test set.
    - y_pred (np.ndarray): Predicted values from the model.

    Returns:
    - dict: A dictionary containing evaluation metrics.
    """
    y_test = test_data['SalePrice']

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

# Function to compare model performance on original and augmented data
def compare_models(train_data, augmented_data, test_data):
    """
    Compares model performance using original and augmented training data.

    Parameters:
    - train_data (pd.DataFrame): Original training dataset with 'SalePrice' target.
    - augmented_data (pd.DataFrame): Augmented training dataset with 'SalePrice' target.
    - test_data (pd.DataFrame): Test dataset with 'SalePrice' target.

    Returns:
    - pd.DataFrame: Comparison of evaluation metrics for original and augmented data.
    """
    # Train and evaluate on original training data
    y_pred_original = train_and_predict(train_data, test_data)
    original_metrics = evaluate_model(test_data, y_pred_original)

    # Train and evaluate on augmented training data
    y_pred_augmented = train_and_predict(augmented_data, test_data)
    augmented_metrics = evaluate_model(test_data, y_pred_augmented)

    # Compare the metrics in a DataFrame
    comparison_df = pd.DataFrame([original_metrics, augmented_metrics], index=["Original Data", "Augmented Data"])

    return comparison_df