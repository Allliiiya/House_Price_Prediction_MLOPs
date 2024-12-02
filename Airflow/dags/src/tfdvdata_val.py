import os
import tensorflow_data_validation as tfdv
import pandas as pd
from sklearn.model_selection import train_test_split
from google.cloud import storage
import tempfile
from google.oauth2 import service_account # type: ignore
from airflow.models import Variable

def save_schema_to_gcs(schema, bucket_name, schema_path_in_gcs):
    """Serialize schema to pbtxt and upload to GCS."""
    # Initialize the GCS client
    # Fetch the credentials from Airflow Variables
    gcp_credentials = Variable.get("GOOGLE_APPLICATION_CREDENTIALS", deserialize_json=True)
    # Authenticate using the fetched credentials
    credentials = service_account.Credentials.from_service_account_info(gcp_credentials)
    
    # Create a storage client with specified credentials
    storage_client = storage.Client(credentials=credentials)
    #storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(schema_path_in_gcs)

    # Use a temporary file to store the schema before uploading
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_path = temp_file.name
        tfdv.write_schema_text(schema, temp_path)

    # Upload the schema from the temporary file to GCS
    blob.upload_from_filename(temp_path)
    print(f"Schema uploaded to gs://{bucket_name}/{schema_path_in_gcs}")

def data_validation_tfdv(data):
    # Set up output directory
    #output_dir = "/opt/airflow/data/validation_reports"
    #os.makedirs(output_dir, exist_ok=True)
    output_dir = "/tmp/validation_reports"
    os.makedirs(output_dir, exist_ok=True)

    # Load data and split into training and evaluation sets
    df = pd.read_json(data)
    train_df, eval_df = train_test_split(df, test_size=0.2, shuffle=False)

    # Generate statistics
    train_stats = tfdv.generate_statistics_from_dataframe(train_df)
    eval_stats = tfdv.generate_statistics_from_dataframe(eval_df)

    # Compare statistics and save report
    comparison_report_path = os.path.join(output_dir, "training_vs_evaluation_comparison_report.txt")
    with open(comparison_report_path, "w") as f:
        f.write("Training vs Evaluation Dataset Comparison Report\n")
        f.write("="*50 + "\n\n")
        
        for train_feature, eval_feature in zip(train_stats.datasets[0].features, eval_stats.datasets[0].features):
            feature_name = train_feature.path.step[0]
            f.write(f"Feature: {feature_name}\n")

            # Compare numerical stats
            if train_feature.HasField("num_stats") and eval_feature.HasField("num_stats"):
                f.write("  Type: Numerical\n")
                f.write(f"    Train Mean: {train_feature.num_stats.mean}\n")
                f.write(f"    Eval Mean: {eval_feature.num_stats.mean}\n")
                f.write(f"    Train Std Dev: {train_feature.num_stats.std_dev}\n")
                f.write(f"    Eval Std Dev: {eval_feature.num_stats.std_dev}\n")
                f.write(f"    Train Min: {train_feature.num_stats.min}\n")
                f.write(f"    Eval Min: {eval_feature.num_stats.min}\n")
                f.write(f"    Train Max: {train_feature.num_stats.max}\n")
                f.write(f"    Eval Max: {eval_feature.num_stats.max}\n")

            # Compare categorical stats
            elif train_feature.HasField("string_stats") and eval_feature.HasField("string_stats"):
                f.write("  Type: Categorical\n")
                f.write(f"    Train Unique Values: {train_feature.string_stats.unique}\n")
                f.write(f"    Eval Unique Values: {eval_feature.string_stats.unique}\n")

            f.write("\n" + "-"*50 + "\n\n")

    # Infer schema from training data statistics
    schema = tfdv.infer_schema(statistics=train_stats)
    # Upload the schema to GCS
    # Define your GCS bucket and schema path
    bucket_name = "bucket_data_mlopsteam2"  # Your GCS bucket
    schema_path_in_gcs = "schema/schema.pbtxt"  # Desired GCS path
    # Save schema pbtxt directly to GCS
    # Save schema to GCS in pbtxt format
    save_schema_to_gcs(schema, bucket_name, schema_path_in_gcs)
    print(f"Schema saved directly to GCS: gs://{bucket_name}/{schema_path_in_gcs}")

    # Validate evaluation stats against the training schema
    initial_anomalies = tfdv.validate_statistics(statistics=eval_stats, schema=schema)
    
    # Save anomalies in a plain text report
    anomalies_path = os.path.join(output_dir, "initial_anomalies_report.txt")
    with open(anomalies_path, "w") as f:
        if initial_anomalies.anomaly_info:
            f.write("Anomalies Detected:\n")
            f.write("="*50 + "\n")
            for anomaly_name, anomaly in initial_anomalies.anomaly_info.items():
                f.write(f"Feature: '{anomaly_name}'\n")
                f.write(f"  Anomaly short description: {anomaly.short_description}\n")
                f.write(f"  Anomaly long description: {anomaly.description}\n\n")
        else:
            f.write("No anomalies found. The evaluation data is consistent with the training schema.\n")
    
    # Further validation after potential schema updates
    updated_anomalies = tfdv.validate_statistics(eval_stats, schema)
    
    # Save updated anomalies report
    updated_anomalies_path = os.path.join(output_dir, "updated_anomalies_report.txt")
    with open(updated_anomalies_path, "w") as f:
        if updated_anomalies.anomaly_info:
            f.write("Updated Anomalies Detected:\n")
            f.write("="*50 + "\n")
            for anomaly_name, anomaly in updated_anomalies.anomaly_info.items():
                f.write(f"Feature: '{anomaly_name}'\n")
                f.write(f"  Anomaly short description: {anomaly.short_description}\n")
                f.write(f"  Anomaly long description: {anomaly.description}\n\n")
        else:
            f.write("No anomalies found. The evaluation data is consistent with the updated training schema.\n")

    # Serialize data and return
    serialized_data = df.to_json()
    return serialized_data
