import pandas as pd
import logging
import json
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.data_prep import load_data, data_overview, data_validation, data_cleaning
from src.encoding import encode_data
from src.data_splitting import split_data


# Define default arguments for your DAG
default_args = {
    'owner': 'House_Price_Prediction Team',
    'start_date': datetime(2024, 10, 29),
    'retries': 0,  # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5),  # Delay before retries
}

# Create a DAG instance named 'DAG_Data_Preprocessing' with the defined default arguments
dag = DAG(
    'DAG_Data_Preprocessing',
    default_args=default_args,
    description='DAG for data preprocessing tasks in House Price Prediction Project',
    schedule_interval=None,  # Set the schedule interval or use None for manual triggering
    catchup=False,
)

# Define PythonOperators for each function

# Task to load data, calls the 'load_data' Python function
def load_data_callable(**kwargs):
    data = load_data()
    # Push data to XCom
    kwargs['ti'].xcom_push(key='data', value=data)

load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data_callable,
    dag=dag,
)

# Task to perform data overview
def data_overview_callable(**kwargs):
    # Pull data from XCom
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids='load_data_task', key='data')
    if data is None:
        logging.error("No data found in XCom for key 'data'")
    else:
        overview_data = data_overview(data)
        # Push overview data to XCom
        ti.xcom_push(key='overview_data', value=overview_data)

data_overview_task = PythonOperator(
    task_id='data_overview_task',
    python_callable=data_overview_callable,
    provide_context=True,
    dag=dag,
)

# Task to perform data validation
def data_validation_callable(**kwargs):
    # Pull data from XCom
    ti = kwargs['ti']
    overview_data = ti.xcom_pull(task_ids='data_overview_task', key='overview_data')
    if overview_data is None:
        logging.error("No data found in XCom for key 'overview_data'")
    else:
        validated_data = data_validation(overview_data)
        # Push validated data to XCom
        ti.xcom_push(key='validated_data', value=validated_data)

data_validation_task = PythonOperator(
    task_id='data_validation_task',
    python_callable=data_validation_callable,
    provide_context=True,
    dag=dag,
)

# Task to perform data cleaning
def data_cleaning_callable(**kwargs):
    # Pull data from XCom
    ti = kwargs['ti']
    validated_data = ti.xcom_pull(task_ids='data_validation_task', key='validated_data')
    if validated_data is None:
        logging.error("No data found in XCom for key 'validated_data'")
    else:
        cleaned_data = data_cleaning(validated_data)
        # Push cleaned data to XCom
        ti.xcom_push(key='cleaned_data', value=cleaned_data)

data_cleaning_task = PythonOperator(
    task_id='data_cleaning_task',
    python_callable=data_cleaning_callable,
    provide_context=True,
    dag=dag,
)

# Task to perform all encoding using encode_data
def encode_data_callable(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids='data_cleaning_task', key='data')
    if data is None:
        logging.error("No data found in XCom for key 'data'")
    else:
        serialized_data, remaining_mappings = encode_data(data)
        ti.xcom_push(key='encoded_data', value=serialized_data)
        ti.xcom_push(key='remaining_mappings', value=json.dumps(remaining_mappings))

encode_data_task = PythonOperator(
    task_id='encode_data_task',
    python_callable=encode_data_callable,
    provide_context=True,
    dag=dag,
)

# Task to split data
def split_data_callable(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids='encode_data_task', key='encoded_data')
    if data is None:
        logging.error("No encoded data found in XCom for key 'encoded_data'")
    else:
        # Call the split_data function with encoded data
        split_result = split_data(data, test_size=0.15)
        
        # Push split data back to XCom for later use
        ti.xcom_push(key='train_data', value=split_result['train_data'])
        ti.xcom_push(key='test_data', value=split_result['test_data'])

data_split_task = PythonOperator(
    task_id='data_split_task',
    python_callable=split_data_callable,
    provide_context=True,
    dag=dag,
)

# Set the updated task dependencies
load_data_task >> data_overview_task >> data_validation_task >> data_cleaning_task >> encode_data_task >> data_split_task
