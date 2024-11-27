import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(encoded_data, test_size):
    """
    Splits the preprocessed encoded dataset into training and testing sets.

    Parameters:
    - serialized_data (str): The encoded data in JSON format (from XCom).
    - test_size (float): The proportion of the dataset to include in the test split.

    Returns:
    - dict: Dictionary with 'train_data' and 'test_data' as JSON strings.
    """
    # Load the JSON-encoded data into a DataFrame
    df = pd.read_json(encoded_data)

    # Perform train-test split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=27)

    # Convert train and test data back to JSON strings for XCom
    return {
        'train_data': train_df.to_json(orient='split'),
        'test_data': test_df.to_json(orient='split')
    }
