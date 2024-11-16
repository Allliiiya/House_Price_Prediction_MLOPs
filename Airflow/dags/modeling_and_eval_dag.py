from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import logging
import mlflow
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from airflow.models import Variable
from src.bias_detection import detect_model_bias, evaluate_bias_disparity
from matplotlib import pyplot as plt

# Define default arguments for your DAG
default_args = {
    'owner': 'House_Price_Prediction Team',
    'start_date': datetime(2024, 11, 6),
    'retries': 0,  # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5),
}

# Create a new DAG instance named 'DAG_Model_Training_and_Evaluation'
dag3 = DAG(
    'DAG_Model_Training_and_Evaluation',
    default_args=default_args,
    description='DAG for training, evaluating, and comparing model performance',
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
)
# Initialize MLflow and ngrok
def initialize_mlflow_callable():
    os.system("lsof -t -i:5000 | xargs kill -9")  # Terminate any process using port 5000
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.end_run()  # Ensure no active run

initialize_mlflow_task = PythonOperator(
    task_id='initialize_mlflow_task',
    python_callable=initialize_mlflow_callable,
    dag=dag3,
)
# Task to train the model and make predictions
def train_and_predict_callable(**kwargs):
    ti = kwargs['ti']

    # Access the configuration passed from dag1
    conf = kwargs.get("dag_run").conf
    # Retrieve encoded_result from the conf dictionary
    augmented_data = conf.get('augmented_data')
    test_data = conf.get('test_data')
    combined_features = conf.get('combined_features')

    # Pull combined features from XCom
    # combined_features = ti.xcom_pull(task_ids='feature_selection_task', key='combined_features')

    augmented_data = pd.read_json(augmented_data, orient='split')
    test_data = pd.read_json(test_data, orient='split')
    # combined_features = pd.read_json(combined_features)

    print(combined_features)

    print(augmented_data.head())

    # Train and predict
    X_train = augmented_data[combined_features]
    y_train = augmented_data['SalePrice']
    X_test = test_data[combined_features]
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Push predictions to XCom
    ti.xcom_push(key='y_pred', value=y_pred.tolist())
    ti.xcom_push(key='model', value=model)
    print(combined_features)

train_and_predict_task = PythonOperator(
    task_id='train_and_predict_task',
    python_callable=train_and_predict_callable,
    provide_context=True,
    dag=dag3,
)

# Task to evaluate the model's performance
def evaluate_and_compare_models_callable(**kwargs):
    ti = kwargs['ti']
    y_pred = np.array(ti.xcom_pull(task_ids='train_and_predict_task', key='y_pred'))

    # Access the configuration
    conf = kwargs.get("dag_run").conf
    # Retrieve encoded_result from the conf dictionary
    test_data = conf.get('test_data')
    augmented_data = conf.get('augmented_data')

    if y_pred is None or test_data is None:
        raise ValueError("Required data for model evaluation not found in XCom")

    test_data = pd.read_json(test_data, orient='split')
    augmented_data = pd.read_json(augmented_data, orient='split')

    # Evaluate model
    y_test = test_data['SalePrice']
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        "Metric": ["MAE", "MSE", "RMSE", "R2"],
        "Value": [mae, mse, rmse, r2]
    })

    # Log and push results
    logging.info(f"Evaluation metrics:\n{comparison_df}")
    ti.xcom_push(key='comparison_metrics', value=comparison_df.to_json(orient='split'))

    # print(augmented_data.head())
# Log metrics to MLflow
    mlflow.set_experiment("House Price Prediction")
    model = ti.xcom_pull(task_ids='train_and_predict_task', key='model')
    with mlflow.start_run(run_name="Evaluation"):
        mlflow.log_metrics({"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2})
        mlflow.sklearn.log_model(model, "linear_regression_model")
        # Save comparison plot
        fig, ax = plt.subplots(figsize=(6, 4))
        comparison_df.set_index("Metric").plot(kind="bar", ax=ax)
        plt.tight_layout()
        plot_path = "/tmp/comparison_plot.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)

    # Push comparison metrics to XCom
    ti.xcom_push(key='comparison_metrics', value=comparison_df.to_json(orient='split'))

evaluate_and_compare_task = PythonOperator(
    task_id='evaluate_and_compare_task',
    python_callable=evaluate_and_compare_models_callable,
    provide_context=True,
    dag=dag3,
)

def detect_model_bias_callable(**kwargs):

    ti = kwargs['ti']

    # Pull test data and predictions from XCom
    conf = kwargs.get("dag_run").conf
    test_data_json = conf.get('test_data')
    y_pred = np.array(ti.xcom_pull(task_ids='train_and_predict_task', key='y_pred'))

    if test_data_json is None or y_pred is None:
        raise ValueError("Required data not found in XCom for 'test_data' or 'y_pred'")

    test_data = pd.read_json(test_data_json, orient='split')
    group_feature = 'Bldg Type'
    target = 'SalePrice'

    # Call the bias detection function
    bias_metrics = detect_model_bias(test_data, y_pred, group_feature, target)

    # Push bias metrics to XCom
    ti.xcom_push(key='bias_metrics', value=bias_metrics.to_json(orient='split'))

bias_detection_task = PythonOperator(
task_id='detect_model_bias_task',
python_callable=detect_model_bias_callable,
provide_context=True,
dag=dag3,)

def evaluate_bias_disparity_callable(**kwargs):

    ti = kwargs['ti']
    bias_metrics_json = ti.xcom_pull(task_ids='detect_model_bias_task', key='bias_metrics')

    if bias_metrics_json is None:
        raise ValueError("Bias metrics not found in XCom")

    bias_metrics = pd.read_json(bias_metrics_json, orient='split')
    metric_columns = ['MAE', 'MSE', 'RMSE', 'R2']

    # Call the disparity evaluation function
    disparities = evaluate_bias_disparity(bias_metrics, metric_columns)

    # Log disparities
    for metric, values in disparities.items():
        logging.info(f"{metric}: Max = {values['max']}, Min = {values['min']}, Disparity = {values['disparity']}")

    # Push disparities to XCom
    ti.xcom_push(key='bias_disparities', value=disparities)

bias_disparity_evaluation_task = PythonOperator(
task_id='evaluate_bias_disparity_task',
python_callable=evaluate_bias_disparity_callable,
provide_context=True,
dag=dag3,)


# Set dependencies for tasks within dag3

initialize_mlflow_task >> train_and_predict_task >> evaluate_and_compare_task >> bias_detection_task >> bias_disparity_evaluation_task

