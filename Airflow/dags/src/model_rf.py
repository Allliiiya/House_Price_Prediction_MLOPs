from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn


# Function to train the model and make predictions
def train_and_predict_rf(augmented_data, test_data, combined_features):
    """
    Trains a Random Forest Regressor model and predicts on the test set.

    Parameters:
    - augmented_data (pd.DataFrame): Training dataset with features and target.
    - test_data (pd.DataFrame): Test dataset with features.
    - combined_features (list): List of feature columns.

    Returns:
    - np.ndarray: Predicted values for the test set.
    """
    X_train = augmented_data[combined_features]
    y_train = augmented_data['SalePrice']
    X_test = test_data[combined_features]
    y_test = test_data['SalePrice']

    # Set the experiment    
    mlflow.set_experiment("Randomforest Regressor Experiment")
    # Start MLflow run
    with mlflow.start_run(): 
        model = RandomForestRegressor(random_state=42, n_estimators=100)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred)         
        mlflow.log_metric("rmse", rmse)         
        mlflow.log_param("model_type", "Randomforest Regressor")                  
        # Log model        
        mlflow.sklearn.log_model(model, "model")
 
        return y_pred, model

# Function to evaluate the model's performance
def evaluate_model(test_data, y_pred):
    """
    Evaluates model performance using various metrics.

    Parameters:
    - test_data (pd.DataFrame): Test dataset with actual target values.
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
def compare_models_rf(train_data, augmented_data, test_data, combined_features):
    """
    Compares model performance using original and augmented training data.

    Parameters:
    - train_data (pd.DataFrame): Original training dataset with 'SalePrice' target.
    - augmented_data (pd.DataFrame): Augmented training dataset with 'SalePrice' target.
    - test_data (pd.DataFrame): Test dataset with 'SalePrice' target.
    - combined_features (list): List of feature columns.

    Returns:
    - pd.DataFrame: Comparison of evaluation metrics for original and augmented data.
    """
    # Train and evaluate on original training data
    y_pred_original, _ = train_and_predict_rf(train_data, test_data, combined_features)
    original_metrics = evaluate_model(test_data, y_pred_original)

    # Train and evaluate on augmented training data
    y_pred_augmented, _ = train_and_predict_rf(augmented_data, test_data, combined_features)
    augmented_metrics = evaluate_model(test_data, y_pred_augmented)

    # Compare the metrics in a DataFrame
    comparison_df = pd.DataFrame([original_metrics, augmented_metrics], index=["Original Data", "Augmented Data"])

    return comparison_df