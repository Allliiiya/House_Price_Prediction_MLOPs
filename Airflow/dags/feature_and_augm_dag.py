from airflow import DAG
import logging
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import pandas as pd
from src.data_splitting import split_data
from src.feature_select import rank_features_by_lasso, select_correlated_features, select_categorical_features_by_rf
from src.data_augment import augment_data_with_perturbations

# Define default arguments for your DAG
default_args = {
    'owner': 'House_Price_Prediction Team',
    'start_date': datetime(2024, 11, 2),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

# Create a DAG instance
dag2 = DAG(
    'DAG_feature_select_and_data_augmentation',
    default_args=default_args,
    description=(
        'DAG for splitting data, feature selection, and '
        'data augmentation tasks'
    ),
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
)


def split_data_callable(**kwargs):
    """Task to split data."""
    ti = kwargs['ti']
    conf = kwargs.get("dag_run").conf
    # Retrieve encoded_result from the conf dictionary
    data = conf.get('encoded_result')
    if data is None:
        logging.error("No encoded data found in XCom for key 'encoded_result'")
    else:
        split_result = split_data(data, test_size=0.15)
        ti.xcom_push(key='train_data', value=split_result['train_data'])
        ti.xcom_push(key='test_data', value=split_result['test_data'])


data_split_task = PythonOperator(
    task_id='data_split_task',
    python_callable=split_data_callable,
    provide_context=True,
    dag=dag2,
)


def feature_selection_callable(**kwargs):
    """Task to select features."""
    ti = kwargs['ti']
    conf = kwargs.get("dag_run").conf
    encoded_data_json = conf.get('encoded_result')
    if encoded_data_json is None:
        raise ValueError("No encoded data found in XCom for 'encoded_data'")
    # Parameters
    numerical_features = [
        "Order", "PID", "MS SubClass", "Lot Frontage", "Lot Area",
        "Overall Qual", "Overall Cond", "Year Built", "Year Remod/Add",
        "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF",
        "Total Bsmt SF", "1st Flr SF", "2nd Flr SF", "Low Qual Fin SF",
        "Gr Liv Area", "Bsmt Full Bath", "Bsmt Half Bath", "Full Bath",
        "Half Bath", "Bedroom AbvGr", "Kitchen AbvGr", "TotRms AbvGrd",
        "Fireplaces", "Garage Cars", "Garage Area", "Wood Deck SF",
        "Open Porch SF", "Enclosed Porch", "3Ssn Porch", "Screen Porch",
        "Pool Area", "Misc Val", "Mo Sold", "Yr Sold"
    ]
    # Load the encoded data
    df = pd.read_json(encoded_data_json)
    target = 'SalePrice'
    prefix_for_bldg_type = 'Bldg Type_'

    # Identify one-hot encoded categorical columns
    categorical_features = [col for col in df.columns if col not in numerical_features + [target]]
    # Step 1: Select features based on correlation
    selected_features = select_correlated_features(encoded_data_json, numerical_features, target, threshold=0.3)
    # Step 2: Rank features using Lasso and further refine the selection
    final_features = rank_features_by_lasso(encoded_data_json, selected_features, target, threshold=0.01)
    # Step 3: Select categorical features using Random Forest
    selected_categorical_features = select_categorical_features_by_rf(
        encoded_data=encoded_data_json,  # Pass JSON-encoded dataset
        selected_features=categorical_features,  # Focus on categorical features
        target=target,
        threshold=0.01
    )

     # Step 4: Include one-hot encoded features for `Bldg Type`
    bldg_type_encoded_features = [col for col in df.columns if col.startswith(prefix_for_bldg_type)]
    # Log encoded Bldg Type features
    logging.info(f"Encoded Bldg Type features: {bldg_type_encoded_features}")

    # Step 5: Combine numerical and categorical features
    combined_features = list(set(final_features + selected_categorical_features + bldg_type_encoded_features))  # Remove duplicates
    
    # Push combined features to XCom
    ti.xcom_push(key='combined_features', value=combined_features)
    # Push categorical features to XCom
    ti.xcom_push(key='selected_categorical_features', value=selected_categorical_features)
    # Push final selected numerical features to XCom for use in subsequent tasks
    ti.xcom_push(key='final_features', value=final_features)


feature_selection_task = PythonOperator(
    task_id='feature_selection_task',
    python_callable=feature_selection_callable,
    provide_context=True,
    dag=dag2,
)


# Data augmentation task
def data_augmentation_callable(**kwargs):
    """Task to augment data."""
    ti = kwargs['ti']

    train_data_json = ti.xcom_pull(task_ids='data_split_task', key='train_data')
    final_features = ti.xcom_pull(task_ids='feature_selection_task', key='final_features') # final feature means numerical features
    combined_features = ti.xcom_pull(task_ids='feature_selection_task', key='combined_features')

    if train_data_json is None or final_features is None:
        raise ValueError("Required data not found in XCom")
    train_data = pd.read_json(train_data_json, orient='split')
    
    augmented_data = augment_data_with_perturbations(
        train_data, final_features, perturbation_percentage=0.02
    )
    ti.xcom_push(key='augmented_data', value=augmented_data.to_json(orient='split'))
    ti.xcom_push(key='combined_features', value=combined_features)


data_augmentation_task = PythonOperator(
    task_id='data_augmentation_task',
    python_callable=data_augmentation_callable,
    provide_context=True,
    dag=dag2,
)


def trigger_dag3_with_conf(**kwargs):
    ti = kwargs['ti']
    # Retrieve `augmented_data` and `test_data` from XCom
    augmented_data = ti.xcom_pull(task_ids='data_augmentation_task', key='augmented_data')
    test_data = ti.xcom_pull(task_ids='data_split_task', key='test_data')
    combined_features = ti.xcom_pull(task_ids='feature_selection_task', key='combined_features')
    if augmented_data is None or test_data is None:
        raise ValueError("Required data not found in XCom for 'augmented_data' or 'test_data'")
    # Dynamically set up TriggerDagRunOperator to trigger `dag3`
    TriggerDagRunOperator(
        task_id="trigger_model_training_and_evaluation",
        trigger_dag_id="DAG_Model_Training_and_Evaluation",  # The ID of DAG 3
        conf={"augmented_data": augmented_data, "test_data": test_data,
              "combined_features": combined_features, },  # Pass data to DAG 3
        trigger_rule="all_success",  # Only trigger if all previous tasks in DAG2 are successful
    ).execute(kwargs)  # Pass Airflow context to execute method


# Trigger the DAG3 from DAG2
trigger_dag3_task = PythonOperator(
    task_id="trigger_model_training_and_evaluation",
    python_callable=trigger_dag3_with_conf,
    provide_context=True,
    dag=dag2,
)

# Set dependencies
data_split_task >> feature_selection_task >> data_augmentation_task >> trigger_dag3_task

