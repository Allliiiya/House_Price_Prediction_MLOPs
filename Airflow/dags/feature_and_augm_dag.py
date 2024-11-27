from airflow import DAG
import logging
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import pandas as pd
from src.data_splitting import split_data
from src.feature_select import rank_features_by_lasso, select_correlated_features, select_categorical_features_by_rf
from src.data_augment import augment_data_with_perturbations
from src.sensitivity_analysis import generate_sensitivity_report

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
    # Identify one-hot encoded categorical columns
    categorical_features = [col for col in df.columns if col not in numerical_features + [target]]
    # Step 1: Select features based on correlation
    selected_features = select_correlated_features(encoded_data_json, numerical_features, target, threshold=0.3)
    # Step 2: Rank features using Lasso and further refine the selection
    final_features = rank_features_by_lasso(encoded_data_json, selected_features, target, threshold=0.1)
    # Step 3: Select categorical features using Random Forest
    selected_categorical_features = select_categorical_features_by_rf(
        encoded_data=encoded_data_json,  # Pass JSON-encoded dataset
        selected_features=categorical_features,  # Focus on categorical features
        target=target,
        threshold=0.01
    )
    # Step 4: Combine numerical and categorical features
    combined_features = list(set(final_features + selected_categorical_features))  # Remove duplicates
    # Push combined features to XCom
    ti.xcom_push(key='combined_features', value=combined_features)
    # Push categorical features to XCom
    ti.xcom_push(key='selected_categorical_features', value=selected_categorical_features)
    # Push final selected features to XCom for use in subsequent tasks
    ti.xcom_push(key='final_features', value=final_features)


feature_selection_task = PythonOperator(
    task_id='feature_selection_task',
    python_callable=feature_selection_callable,
    provide_context=True,
    dag=dag2,
)


def sensitivity_analysis_callable(**kwargs):
    """Task to perform sensitivity analysis including SHAP values on selected features."""
    try:
        ti = kwargs['ti']
        
        # Get encoded data from conf
        conf = kwargs.get("dag_run").conf
        encoded_data_json = conf.get('encoded_result')
        
        # Get features from feature selection task
        combined_features = ti.xcom_pull(task_ids='feature_selection_task', key='combined_features')
        final_features = ti.xcom_pull(task_ids='feature_selection_task', key='final_features')
        
        if encoded_data_json is None or combined_features is None:
            raise ValueError("Required data not found in XCom")
        
        # Load the encoded data
        data = pd.read_json(encoded_data_json)
        
        # Generate sensitivity report for both combined and final features
        combined_features_report = generate_sensitivity_report(
            data=data,
            features=combined_features,
            target='SalePrice'
        )
        
        final_features_report = generate_sensitivity_report(
            data=data,
            features=final_features,
            target='SalePrice'
        )
        
        # Prepare reports for XCom (excluding SHAP values due to serialization limitations)
        sensitivity_results = {
            'combined_features_analysis': {
                'feature_importance': combined_features_report['feature_importance'].to_json(orient='split'),
                'shap_summary': combined_features_report['shap_analysis']['shap_summary'].to_json(orient='split'),
                'feature_sensitivity': combined_features_report['feature_sensitivity'].to_json(orient='split'),
                'feature_stability': combined_features_report['feature_stability'].to_json(orient='split'),
                'baseline_model_score': float(combined_features_report['baseline_model_score'])
            },
            'final_features_analysis': {
                'feature_importance': final_features_report['feature_importance'].to_json(orient='split'),
                'shap_summary': final_features_report['shap_analysis']['shap_summary'].to_json(orient='split'),
                'feature_sensitivity': final_features_report['feature_sensitivity'].to_json(orient='split'),
                'feature_stability': final_features_report['feature_stability'].to_json(orient='split'),
                'baseline_model_score': float(final_features_report['baseline_model_score'])
            }
        }
        
        # Push results to XCom
        ti.xcom_push(key='sensitivity_results', value=sensitivity_results)
        
        # Use SHAP values to identify highly important features
        shap_important_features = final_features_report['shap_analysis']['shap_summary'][
            final_features_report['shap_analysis']['shap_summary']['mean_shap_value'] > 
            final_features_report['shap_analysis']['shap_summary']['mean_shap_value'].median()
        ]['feature'].tolist()
        
        # Combine SHAP importance with stability scores
        stable_and_important_features = list(set(shap_important_features).intersection(
            final_features_report['feature_stability'][
                final_features_report['feature_stability']['stability_score'] > 0.5
            ]['feature'].tolist()
        ))
        
        # Push important features to XCom
        ti.xcom_push(key='stable_and_important_features', value=stable_and_important_features)
        
        logging.info("Sensitivity analysis with SHAP completed successfully")
        
    except Exception as e:
        logging.error(f"Error in sensitivity analysis task: {str(e)}")
        raise

# Create the sensitivity analysis task operator
sensitivity_analysis_task = PythonOperator(
    task_id='sensitivity_analysis_task',
    python_callable=sensitivity_analysis_callable,
    provide_context=True,
    dag=dag2,
)

# Update data augmentation task to use sensitivity results
def data_augmentation_callable(**kwargs):
    """Task to augment data using sensitivity analysis results."""
    ti = kwargs['ti']
    train_data_json = ti.xcom_pull(task_ids='data_split_task', key='train_data')
    final_features = ti.xcom_pull(task_ids='feature_selection_task', key='final_features')
    combined_features = ti.xcom_pull(task_ids='feature_selection_task', key='combined_features')
    highly_stable_features = ti.xcom_pull(task_ids='sensitivity_analysis_task', key='highly_stable_features')
    
    if train_data_json is None or final_features is None:
        raise ValueError("Required data not found in XCom")
    
    train_data = pd.read_json(train_data_json, orient='split')
    
    # Use highly stable features for augmentation if available
    features_for_augmentation = highly_stable_features if highly_stable_features else final_features
    
    augmented_data = augment_data_with_perturbations(
        train_data, 
        features_for_augmentation,
        perturbation_percentage=0.02
    )
    
    ti.xcom_push(key='augmented_data', value=augmented_data.to_json(orient='split'))
    ti.xcom_push(key='combined_features', value=combined_features)
    ti.xcom_push(key='features_used_for_augmentation', value=features_for_augmentation)

# Update the data augmentation task operator
data_augmentation_task = PythonOperator(
    task_id='data_augmentation_task',
    python_callable=data_augmentation_callable,
    provide_context=True,
    dag=dag2,
)

# Update trigger_dag3 to include sensitivity results
def trigger_dag3_with_conf(**kwargs):
    ti = kwargs['ti']
    augmented_data = ti.xcom_pull(task_ids='data_augmentation_task', key='augmented_data')
    test_data = ti.xcom_pull(task_ids='data_split_task', key='test_data')
    combined_features = ti.xcom_pull(task_ids='feature_selection_task', key='combined_features')
    sensitivity_results = ti.xcom_pull(task_ids='sensitivity_analysis_task', key='sensitivity_results')
    features_used_for_augmentation = ti.xcom_pull(task_ids='data_augmentation_task', key='features_used_for_augmentation')
    
    if augmented_data is None or test_data is None:
        raise ValueError("Required data not found in XCom")
    
    TriggerDagRunOperator(
        task_id="trigger_model_training_and_evaluation",
        trigger_dag_id="DAG_Model_Training_and_Evaluation",
        conf={
            "augmented_data": augmented_data, 
            "test_data": test_data,
            "combined_features": combined_features,
            "sensitivity_results": sensitivity_results,
            "features_used_for_augmentation": features_used_for_augmentation
        },
        trigger_rule="all_success",
    ).execute(kwargs)

# Update the trigger_dag3 task operator
trigger_dag3_task = PythonOperator(
    task_id="trigger_model_training_and_evaluation",
    python_callable=trigger_dag3_with_conf,
    provide_context=True,
    dag=dag2,
)

# The task dependencies remain the same
data_split_task >> feature_selection_task >> sensitivity_analysis_task >> data_augmentation_task >> trigger_dag3_task
