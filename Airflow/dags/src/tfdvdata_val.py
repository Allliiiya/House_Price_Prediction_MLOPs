import tensorflow_data_validation as tfdv
import pandas as pd
#import matplotlib.pyplot as plt
#import os
from sklearn.model_selection import train_test_split
from tensorflow_metadata.proto.v0 import schema_pb2

def data_validation_tfdv(data):
    print('TFDV Version: {}'.format(tfdv.__version__))
    """
    Data Validation using TensorFlow Data Validation
    """
    # Load data and split into train and evaluation sets
    df = pd.read_json(data)
    train_df, eval_df = train_test_split(df, test_size=0.2, shuffle=False)
    
    # Generate training and evaluation dataset statistics
    train_stats = tfdv.generate_statistics_from_dataframe(train_df)
    eval_stats = tfdv.generate_statistics_from_dataframe(eval_df)

    print(train_stats)
    print(eval_stats)
    
    # Serialize the data and return
    serialized_data = df.to_json()
    return serialized_data
