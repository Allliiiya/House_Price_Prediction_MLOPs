import pandas as pd


def encode_one_hot_columns(df):
    """
    Applies One-Hot Encoding to categorical columns in the dataset.
    """
    # Identify categorical columns that need One-Hot encoding
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    print("Categorical columns:", categorical_columns)
    # Perform One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=categorical_columns)

    print("Encoded columns:", df_encoded.columns.tolist())

    print("Data after One-Hot Encoding:")
    print(df_encoded.head())
    return df_encoded


def encode_data(data):
    """
    Main function to perform encoding on the dataset.
    """
    # Load the dataset from the file path
    df = pd.read_json(data)
    # Apply One-Hot Encoding to the dataset
    df_encoded = encode_one_hot_columns(df)
    
    # Log or print the number of features after encoding
    print(f"Number of features after encoding: {len(df_encoded.columns)}")
    print(f"Final encoded column names: {df_encoded.columns.tolist()}")


    # Save the encoded data back to the same file path
    serialized_data = df_encoded.to_json()
    return serialized_data
