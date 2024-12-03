import os
import tensorflow_data_validation as tfdv
import pandas as pd
from sklearn.model_selection import train_test_split
from google.cloud import storage
import tempfile
from google.oauth2 import service_account  # type: ignore
from airflow.models import Variable


def upload_to_gcs(local_path, bucket_name, gcs_path):
    """Uploads a file to a GCS bucket."""
    # Fetch the credentials from Airflow Variables
    gcp_credentials = Variable.get("GOOGLE_APPLICATION_CREDENTIALS", deserialize_json=True)
    # Authenticate using the fetched credentials
    credentials = service_account.Credentials.from_service_account_info(gcp_credentials)

    # Create a storage client with specified credentials
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)

    # Upload the file to GCS
    blob.upload_from_filename(local_path)
    print(f"File uploaded to gs://{bucket_name}/{gcs_path}")


def save_reference_stats_to_gcs(stats, bucket_name, stats_path_in_gcs):
    """Save reference statistics to GCS in Protobuf format."""
    # Use a temporary file to store the statistics before uploading
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
        tfdv.write_stats_text(stats, temp_path)

    # Upload the statistics file to GCS
    upload_to_gcs(temp_path, bucket_name, stats_path_in_gcs)


def data_validation_tfdv(data):
    # Set up output directory
    output_dir = "/tmp/validation_reports"
    os.makedirs(output_dir, exist_ok=True)

    # Load data and split into training and evaluation sets
    df = pd.read_json(data)
    train_df, eval_df = train_test_split(df, test_size=0.2, shuffle=False)

    # Generate statistics
    train_stats = tfdv.generate_statistics_from_dataframe(train_df)
    eval_stats = tfdv.generate_statistics_from_dataframe(eval_df)

    # Save the training stats as reference stats
    bucket_name = "bucket_data_mlopsteam2"  # Your GCS bucket
    reference_stats_path_in_gcs = "stats/reference_stats.pb"  # Desired GCS path for stats
    save_reference_stats_to_gcs(train_stats, bucket_name, reference_stats_path_in_gcs)
    print(f"Reference statistics saved to GCS: gs://{bucket_name}/{reference_stats_path_in_gcs}")

    # Compare statistics and save report
    comparison_report_path = os.path.join(output_dir, "training_vs_evaluation_comparison_report.txt")
    with open(comparison_report_path, "w") as f:
        f.write("Training vs Evaluation Dataset Comparison Report\n")
        f.write("=" * 50 + "\n\n")

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

            f.write("\n" + "-" * 50 + "\n\n")

    # Infer schema from training data statistics
    schema = tfdv.infer_schema(statistics=train_stats)

    # Upload the schema to GCS
    schema_path_in_gcs = "schema/schema.pbtxt"
    #save_schema_to_gcs(schema, bucket_name, schema_path_in_gcs)
    print(f"Schema saved to GCS: gs://{bucket_name}/{schema_path_in_gcs}")

    # Validate evaluation stats against the training schema
    initial_anomalies = tfdv.validate_statistics(statistics=eval_stats, schema=schema)

    # Save anomalies in a plain text report
    anomalies_path = os.path.join(output_dir, "initial_anomalies_report.txt")
    with open(anomalies_path, "w") as f:
        if initial_anomalies.anomaly_info:
            f.write("Anomalies Detected:\n")
            f.write("=" * 50 + "\n")
            for anomaly_name, anomaly in initial_anomalies.anomaly_info.items():
                f.write(f"Feature: '{anomaly_name}'\n")
                f.write(f"  Anomaly short description: {anomaly.short_description}\n")
                f.write(f"  Anomaly long description: {anomaly.description}\n\n")
        else:
            f.write("No anomalies found. The evaluation data is consistent with the training schema.\n")

    # Serialize data and return
    serialized_data = df.to_json()
    return serialized_data
