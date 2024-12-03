import pandas as pd
import logging
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.data_prep import data_preprocessing, load_data, data_clean_test
from src.datashift import data_validation_tfdv


# Define default arguments for your DAG
default_args = {
    'owner': 'House_Price_Prediction Team',
    'start_date': datetime(2024, 10, 29),
    'retries': 0,  # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5),  # Delay before retries
}

# Create a DAG instance named 'DAG 1' with the defined default arguments
dag = DAG(
    'DAG_Data_Preprocessing',
    default_args=default_args,
    description='Dag data prep example for Project',
    schedule_interval=None,  # Set the schedule interval or use None for manual triggering
    catchup=False,
)

# Define PythonOperators for each function

# Task to load data, calls the 'load_data' Python function
def load_data_callable(**kwargs):
    data = load_data()
    # logging.info(data.head())
    # Push data to XCom
    kwargs['ti'].xcom_push(key='data', value=data)

# load_data_task = PythonOperator(
#     task_id='load_data_task',
#     python_callable=load_data_callable,
#     provide_context=True,
#     dag=dag,
# )

load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data_callable,  # Direct reference to your updated load_data function
    dag=dag,
)

# Task to validate data, calls the 'data_validation_tfdv' Python function
def validate_data_callable(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids='load_data_task', key='data')
    validated_data=data_validation_tfdv(data)
    kwargs['ti'].xcom_push(key='validated_data', value=validated_data)

data_validation_task = PythonOperator(
    task_id='data_validation_task',
    python_callable=validate_data_callable,
    provide_context=True,
    dag=dag,
)
   
# Task to perform data preprocessing
def data_preprocessing_callable(**kwargs):
    # Pull data from XCom
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids='data_validation_task', key='validated_data')
    processed_data = data_preprocessing(data)
    # Push processed data to XCom
    ti.xcom_push(key='processed_data', value=processed_data)


data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing_task',
    python_callable=data_preprocessing_callable,
    provide_context=True,
    dag=dag,
)

def data_clean_test_callable(**kwargs):
    # Pull data from XCom
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids='data_preprocessing_task', key='processed_data')

    if data is None:
        logging.error("No data found in XCom for key 'processed_data'")
    else:
        data_clean_test(data)
    
    data_clean_test(data)
    # Push processed data to XCom
    # ti.xcom_push(key='processed_data', value=processed_data)

dag_clean_test_task = PythonOperator(
    task_id='dag_clean_test_task',
    python_callable=data_clean_test_callable,  # Direct reference to your updated load_data function
    dag=dag,
)


# Set task dependencies
load_data_task >> data_validation_task >> data_preprocessing_task >> dag_clean_test_task
