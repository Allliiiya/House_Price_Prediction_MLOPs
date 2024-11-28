from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from airflow.models import Variable
from src.bias_detection import detect_model_bias, evaluate_bias_disparity
from src.model_rf import train_and_predict_rf
from src.model_linear_regression import train_and_predict_linear_regression
from src.model_rf import evaluate_model
from src.model_elastic_net import train_and_predict_elastic_net

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

# Task to train the linear regression model and make predictions
def train_and_predict_linear_regression_callable(**kwargs):
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

    # # Train and predict
    # X_train = augmented_data[combined_features]
    # y_train = augmented_data['SalePrice']
    # X_test = test_data[combined_features]
    # model = LinearRegression()
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    y_pred, model = train_and_predict_linear_regression(augmented_data, test_data, combined_features)

    # Push predictions to XCom
    ti.xcom_push(key='linear_y_pred', value=y_pred.tolist())

    print(combined_features)

train_and_predict_linear_regression_task = PythonOperator(
    task_id='train_and_predict_linear_regression_task',
    python_callable=train_and_predict_linear_regression_callable,
    provide_context=True,
    dag=dag3,
)

# Task to train the random forest regression model
def train_and_predict_rf_callable(**kwargs):
    ti = kwargs['ti']

    # Access the configuration passed from dag1
    conf = kwargs.get("dag_run").conf
    augmented_data = conf.get('augmented_data')
    test_data = conf.get('test_data')
    combined_features = conf.get('combined_features')

    augmented_data = pd.read_json(augmented_data, orient='split')
    test_data = pd.read_json(test_data, orient='split')

    # Train and predict with Random Forest Regressor
    y_pred, model = train_and_predict_rf(augmented_data, test_data, combined_features)

    # Push predictions and model to XCom
    ti.xcom_push(key='rf_y_pred', value=y_pred.tolist())
    # ti.xcom_push(key='rf_model', value=model)


train_and_predict_rf_task = PythonOperator(
    task_id='train_and_predict_rf_task',
    python_callable=train_and_predict_rf_callable,
    provide_context=True,
    dag=dag3,
)

# Task to train the Elastic Net regression model
def train_and_predict_elastic_net_callable(**kwargs):
    ti = kwargs['ti']

    # Access the configuration passed from dag1
    conf = kwargs.get("dag_run").conf
    augmented_data = conf.get('augmented_data')
    test_data = conf.get('test_data')
    combined_features = conf.get('combined_features')

    #alpha = conf.get('alpha', 0.1) 
    #l1_ratio = conf.get('l1_ratio', 0.5)  # Default value if not provided

    augmented_data = pd.read_json(augmented_data, orient='split')
    test_data = pd.read_json(test_data, orient='split')

    # Train and predict with Elastic Net Regressor
    y_pred, model = train_and_predict_elastic_net(augmented_data, test_data, combined_features)

    # Push predictions and model to XCom
    ti.xcom_push(key='elastic_net_y_pred', value=y_pred.tolist())
    # Note: Model objects are not pushed directly to XCom to avoid serialization issues.

train_and_predict_elastic_net_task = PythonOperator(
    task_id='train_and_predict_elastic_task',
    python_callable=train_and_predict_elastic_net_callable,
    provide_context=True,
    dag=dag3,
)

# Task to evaluate the model's performance
def evaluate_and_compare_models_callable(**kwargs):
    ti = kwargs['ti']
    # y_pred = np.array(ti.xcom_pull(task_ids='train_and_predict_task', key='y_pred'))

     # Pull predictions for both models
    linear_y_pred = ti.xcom_pull(task_ids='train_and_predict_linear_regression_task', key='linear_y_pred')  # Linear regression
    rf_y_pred = ti.xcom_pull(task_ids='train_and_predict_rf_task', key='rf_y_pred')  # Random Forest
    elastic_y_pred = ti.xcom_pull(task_ids='train_and_predict_elastic_task', key='elastic_net_y_pred')  # Elastic net


    # Access the configuration
    conf = kwargs.get("dag_run").conf
    # Retrieve encoded_result from the conf dictionary
    test_data = conf.get('test_data')
    augmented_data = conf.get('augmented_data')

    # if y_pred is None or test_data is None:
    #     raise ValueError("Required data for model evaluation not found in XCom")

    test_data = pd.read_json(test_data, orient='split')
    augmented_data = pd.read_json(augmented_data, orient='split')

    # Evaluate model
    y_test = test_data['SalePrice']
    # mae = mean_absolute_error(y_test, y_pred)
    # mse = mean_squared_error(y_test, y_pred)
    # rmse = np.sqrt(mse)
    # r2 = r2_score(y_test, y_pred)

    metrics = []

    # Evaluate Linear Regression
    if linear_y_pred:
        linear_metrics = evaluate_model(test_data, np.array(linear_y_pred))
        linear_metrics["Model"] = "Linear Regression"
        metrics.append(linear_metrics)

    # Evaluate Random Forest Regressor
    if rf_y_pred:
        rf_metrics = evaluate_model(test_data, np.array(rf_y_pred))
        rf_metrics["Model"] = "Random Forest Regressor"
        metrics.append(rf_metrics)

        # Evaluate elastic net Regressor
    if elastic_y_pred:
        elastic_net_metrics = evaluate_model(test_data, np.array(elastic_y_pred))
        elastic_net_metrics["Model"] = "Elastic Net Regressor"
        metrics.append(elastic_net_metrics)    

    # # Create comparison DataFrame
    # comparison_df = pd.DataFrame({
    #     "Metric": ["MAE", "MSE", "RMSE", "R2"],
    #     "Value": [mae, mse, rmse, r2]
    # })

    # # Log and push results
    # logging.info(f"Evaluation metrics:\n{comparison_df}")
    # ti.xcom_push(key='comparison_metrics', value=comparison_df.to_json(orient='split'))

       # Combine metrics into a DataFrame
    metrics_df = pd.DataFrame(metrics)
    logging.info(f"Evaluation metrics:\n{metrics_df}")

    # Push metrics to XCom
    ti.xcom_push(key='comparison_metrics', value=metrics_df.to_json(orient='split'))

    # print(augmented_data.head())

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
    # y_pred = np.array(ti.xcom_pull(task_ids='train_and_predict_task', key='y_pred'))

    # Pull predictions from XCom
    linear_y_pred = ti.xcom_pull(task_ids='train_and_predict_linear_regression_task', key='linear_y_pred')  # Linear Regression
    rf_y_pred = ti.xcom_pull(task_ids='train_and_predict_rf_task', key='rf_y_pred')  # Random Forest
    elastic_y_pred = ti.xcom_pull(task_ids='train_and_predict_elastic_task', key='elastic_net_y_pred')  # Random Forest


    if test_data_json is None or (linear_y_pred is None and rf_y_pred and elastic_y_pred is None):
        raise ValueError("Required data not found in XCom for 'test_data' or predictions")

    test_data = pd.read_json(test_data_json, orient='split')
    target = 'SalePrice'

    # Identify one-hot encoded `Bldg Type` features
    encoded_features_prefix = 'Bldg Type_'  # Adjust prefix if necessary
    encoded_features = [col for col in test_data.columns if col.startswith(encoded_features_prefix)]
    if not encoded_features:
        raise ValueError(f"No encoded features found with prefix '{encoded_features_prefix}'")


    # # Call the bias detection function
    # bias_metrics = detect_model_bias(test_data, y_pred, encoded_features, target)

    # Initialize a dictionary to hold bias metrics for both models
    bias_metrics_all = {}

    # Detect bias for Linear Regression
    if linear_y_pred:
        linear_y_pred = np.array(linear_y_pred)
        bias_metrics_linear = detect_model_bias(test_data, linear_y_pred, encoded_features, target)
        bias_metrics_all['Linear Regression'] = bias_metrics_linear

    # Detect bias for Random Forest Regressor
    if rf_y_pred:
        rf_y_pred = np.array(rf_y_pred)
        bias_metrics_rf = detect_model_bias(test_data, rf_y_pred, encoded_features, target)
        bias_metrics_all['Random Forest'] = bias_metrics_rf

    # Detect bias for Elastic Net Regressor
    if elastic_y_pred:
        elastic_y_pred = np.array(elastic_y_pred)
        bias_metrics_elastic_net = detect_model_bias(test_data, elastic_y_pred, encoded_features, target)
        bias_metrics_all['Elastic Net'] = bias_metrics_elastic_net

    # logging.info(f"Bias metrics:\n{bias_metrics}")
    # # Push bias metrics to XCom
    # ti.xcom_push(key='bias_metrics', value=bias_metrics.to_json(orient='split'))

    # Log bias metrics for both models
    for model_name, metrics in bias_metrics_all.items():
        logging.info(f"Bias metrics for {model_name}:\n{metrics}")

    # Push combined bias metrics to XCom
    ti.xcom_push(key='bias_metrics_all', value={k: v.to_json(orient='split') for k, v in bias_metrics_all.items()})


bias_detection_task = PythonOperator(
task_id='detect_model_bias_task',
python_callable=detect_model_bias_callable,
provide_context=True,
dag=dag3,)


def evaluate_bias_disparity_callable(**kwargs):

    ti = kwargs['ti']
    # bias_metrics_json = ti.xcom_pull(task_ids='detect_model_bias_task', key='bias_metrics')
    # Pull bias metrics for all models from XCom
    bias_metrics_json_all = ti.xcom_pull(task_ids='detect_model_bias_task', key='bias_metrics_all')


    # if bias_metrics_json is None:
        # raise ValueError("Bias metrics not found in XCom")
    if not bias_metrics_json_all:
        raise ValueError("Bias metrics for models not found in XCom")


    # bias_metrics = pd.read_json(bias_metrics_json, orient='split')
    # metric_columns = ['MAE', 'MSE', 'RMSE', 'R2']

    # Convert bias metrics back to DataFrames
    bias_metrics_all = {
        model_name: pd.read_json(metrics_json, orient='split')
        for model_name, metrics_json in bias_metrics_json_all.items()
    }
    metric_columns = ['MAE', 'MSE', 'RMSE', 'R2']
    disparities_all = {}

    # # Call the disparity evaluation function
    # disparities = evaluate_bias_disparity(bias_metrics, metric_columns)

    # # Log disparities
    # for metric, values in disparities.items():
    #     logging.info(f"{metric}: Max = {values['max']}, Min = {values['min']}, Disparity = {values['disparity']}")

    # # Push disparities to XCom
    # ti.xcom_push(key='bias_disparities', value=disparities)

    # Evaluate disparities for each model
    for model_name, bias_metrics in bias_metrics_all.items():
        disparities = evaluate_bias_disparity(bias_metrics, metric_columns)
        disparities_all[model_name] = disparities

        # Log disparities for the current model
        logging.info(f"Disparities for {model_name}:")
        for metric, values in disparities.items():
            logging.info(f"{metric}: Max = {values['max']}, Min = {values['min']}, Disparity = {values['disparity']}")

    # Push all disparities to XCom
    ti.xcom_push(key='bias_disparities_all', value=disparities_all)

bias_disparity_evaluation_task = PythonOperator(
task_id='evaluate_bias_disparity_task',
python_callable=evaluate_bias_disparity_callable,
provide_context=True,
dag=dag3,)



# Set dependencies for tasks within dag3
[train_and_predict_linear_regression_task, train_and_predict_rf_task, train_and_predict_elastic_net_task]>> evaluate_and_compare_task >> bias_detection_task >> bias_disparity_evaluation_task
