import pandas as pd
from google.oauth2 import service_account
from airflow.models import Variable
from google.cloud import storage
import logging


def load_data():
    # Fetch the credentials from Airflow Variables
    gcp_credentials = Variable.get("GOOGLE_APPLICATION_CREDENTIALS", deserialize_json=True)
    # Authenticate using the fetched credentials
    credentials = service_account.Credentials.from_service_account_info(gcp_credentials)

    # Create a storage client with specified credentials
    storage_client = storage.Client(credentials=credentials)
    bucket_name = 'bucket_data_mlopsteam2'
    blob_name = 'AmesHousing.csv'

    # Get the bucket and blob
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    # Download the blob to a temporary file
    blob.download_to_filename('/tmp/AmesHousing.csv')

    # Load the dataset into a DataFrame
    data = pd.read_csv('/tmp/AmesHousing.csv')
    serialized_data = data.to_json()
    # pickle.dumps(data)
    logging.info(data.head())
    return serialized_data


def data_overview(data):
    """
    Deserializes the dataset, provides an overview, and returns serialized data.
    """
    df = pd.read_json(data)

    # Data overview: shape, info, and descriptive statistics
    logging.info(f"Data shape: {df.shape}")
    print("Data shape:", df.shape)
    logging.info(df.info())
    print("Data Info:", df.info())
    logging.info(df.describe(include="all"))
    print("Data Description:", df.describe(include="all"))

    return df.to_json()


def data_validation(data):
    """
    Validates the dataset by handling outliers and correcting anomalies, then re-serializes.
    """
    df = pd.read_json(data)

    # Drop outliers for 'Gr Liv Area'
    df.drop(df[df['Gr Liv Area'] > 4000].index, inplace=True)
    logging.info("Outliers removed where 'Gr Liv Area' > 4000.")

    # Correcting Garage Year Built anomaly
    df['Garage Yr Blt'] = df['Garage Yr Blt'].replace({2207: 2007})
    logging.info("Corrected 'Garage Yr Blt' from 2207 to 2007.")

    return df.to_json()


def data_cleaning(data):
    """
    Cleans the dataset by removing duplicates and filling missing values, then re-serializes.
    """
    df = pd.read_json(data)

    # Remove duplicates
    duplicate_rows = df.duplicated()
    if duplicate_rows.any():
        logging.info(f"Number of duplicate rows: {duplicate_rows.sum()}")
        df.drop_duplicates(inplace=True)
    else:
        logging.info("No duplicate rows found.")

    # Handle missing values in specific columns
    df['Lot Frontage'] = df['Lot Frontage'].fillna(df['Lot Frontage'].median())
    df['Mas Vnr Area'] = df['Mas Vnr Area'].fillna(1)
    df['Bsmt Full Bath'] = df['Bsmt Full Bath'].fillna(0)
    df['Bsmt Half Bath'] = df['Bsmt Half Bath'].fillna(0)
    df['BsmtFin SF 1'] = df['BsmtFin SF 1'].fillna(0)
    df['Garage Cars'] = df['Garage Cars'].fillna(df['Garage Cars'].median())
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    df['Total Bsmt SF'] = df['Total Bsmt SF'].fillna(0)
    df['Bsmt Unf SF'] = df['Bsmt Unf SF'].fillna(0)
    df['BsmtFin SF 2'] = df['BsmtFin SF 2'].fillna(0)
    df['Garage Area'] = df['Garage Area'].fillna(df['Garage Area'].median())
    df['Garage Yr Blt'] = df['Garage Yr Blt'].fillna(0)

    # Replace NaN and empty values with "Missing" in selected categorical columns
    columns_replace_na_and_empty = [
        'Garage Cond', 'Garage Finish', 'Garage Qual',
        'Bsmt Exposure', 'BsmtFin Type 2', 'Bsmt Qual', 'Bsmt Cond', 'BsmtFin Type 1'
    ]
    for col in columns_replace_na_and_empty:
        df[col] = df[col].replace('', 'Missing').fillna('Missing')

    columns_replace_na_only = [
        'Pool QC', 'Misc Feature', 'Alley', 'Fence', 'Garage Type', 'Fireplace Qu', 'Mas Vnr Type'
    ]
    df[columns_replace_na_only] = df[columns_replace_na_only].fillna('Missing')

    return df.to_json()
