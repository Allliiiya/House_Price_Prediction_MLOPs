import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator # type: ignore
from google.oauth2 import service_account # type: ignore
from airflow.models import Variable
import json
from google.cloud import storage
import pickle
import os
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


def data_preprocessing(data):
    """
    Deserializes data, performs data preprocessing, and returns serialized clustered data.
    """
    df = pd.read_json(data)
    logging.info(f"Initial data shape: {df.shape}")  # Log initial shape
    print("Initial data:", df.head())  # Print initial data
    
    # Check for and handle missing data appropriately
    if df.isnull().values.any():
        logging.info("Missing values found")
        df = df.fillna(method='ffill')  # Example: forward fill missing values
    
    logging.info(f"Data shape after handling missing values: {df.shape}")
    print("Data after handling missing:", df.head())

    serialized_data = df.to_json()
    return serialized_data

def data_clean_test(data):
    # df_clean = pickle.loads(data)
    df_clean = pd.read_json(data)
    print(df_clean.isna())
