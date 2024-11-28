from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from src.model_linear_regression import evaluate_model

# Function to train the model and make predictions
def train_and_predict_elastic_net(augmented_data, test_data, combined_features):
    """
    Trains an elastic net Regressor model and predicts on the test set.

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
    mlflow.set_experiment("Elastic Net Regressor Experiment")

   # Start MLflow run
    with mlflow.start_run():
        model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42) # l1=0.5 is the balance of ridge and lasso
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Log metrics, parameters, and model
        mlflow.log_metric("rmse", rmse)
        mlflow.log_param("model_type", "Elastic Net Regressor")
        mlflow.log_param("alpha", 1.0)
        mlflow.log_param("l1_ratio", 0.5)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        return y_pred, model

def compare_models_elastic_net(train_data, augmented_data, test_data, combined_features):
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
    y_pred_original, _ = train_and_predict_elastic_net(train_data, test_data, combined_features)
    original_metrics = evaluate_model(test_data, y_pred_original)

    # Train and evaluate on augmented training data
    y_pred_augmented, _ = train_and_predict_elastic_net(augmented_data, test_data, combined_features)
    augmented_metrics = evaluate_model(test_data, y_pred_augmented)

    # Compare the metrics in a DataFrame
    comparison_df = pd.DataFrame([original_metrics, augmented_metrics], index=["Original Data", "Augmented Data"])

    return comparison_df