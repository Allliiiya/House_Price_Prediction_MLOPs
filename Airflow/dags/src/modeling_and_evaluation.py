import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow
import matplotlib.pyplot as plt
from pyngrok import ngrok
import os

def initialize_mlflow():
    os.system("lsof -t -i:5000 | xargs kill -9")  # Terminate any process using port 5000
    ngrok.kill()  # Kill existing ngrok process

    ngrok.set_auth_token("2olR2n16bop4IlGsEvuzfBQiQl7_2Y4vTZhSn6GqqJkjHMZa1")
    get_ipython().system_raw("mlflow ui --port 5000 &")  # Start MLflow UI on port 5000
    public_url = ngrok.connect(5000, "http")  # Establish ngrok tunnel
    print("MLflow Tracking UI:", public_url)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.end_run()  # Ensure no active run

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
# Function to log metrics and model to MLflow
def log_model_to_mlflow(data_type, model, metrics, X_test, run_name):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("Training Data", data_type)
        mlflow.log_metrics(metrics)
        input_example = X_test.iloc[:1].to_dict(orient="records")[0]
        mlflow.sklearn.log_model(model, f"house_price_model_{data_type.lower()}", input_example=input_example)

# Function to compare models and log results to MLflow
def compare_models_with_mlflow(train_data, augmented_data, test_data):
    # Initialize MLflow and ngrok
    initialize_mlflow()
    mlflow.set_experiment("House Price Prediction")

    # Train and evaluate on original training data
    model, y_pred_original, X_test = train_and_predict(train_data, test_data)
    original_metrics = evaluate_model(test_data, y_pred_original)
    log_model_to_mlflow("Original", model, original_metrics, X_test, "Original Data")

    # Train and evaluate on augmented training data
    model, y_pred_augmented, X_test = train_and_predict(augmented_data, test_data)
    augmented_metrics = evaluate_model(test_data, y_pred_augmented)
    log_model_to_mlflow("Augmented", model, augmented_metrics, X_test, "Augmented Data")

    # Compare the metrics in a DataFrame
    comparison_df = pd.DataFrame([original_metrics, augmented_metrics], index=["Original Data", "Augmented Data"])
    
    # Plot and log comparison visualization for each metric to MLflow
    for metric in comparison_df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        comparison_df[metric].plot(kind='bar', ax=ax, color=['skyblue', 'orange'])
        ax.set_title(f'Comparison of {metric}')
        ax.set_xlabel('Data Type')
        ax.set_ylabel(metric)
        ax.set_ylim(0, comparison_df[metric].max() * 1.2) 

        # Save and log the plot image to MLflow
        plot_filename = f"{metric}_comparison_plot.png"
        fig.savefig(plot_filename)
        mlflow.log_artifact(plot_filename)
        plt.close(fig)

    return comparison_df
