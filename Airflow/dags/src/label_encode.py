import pandas as pd
import logging


def encode_quality_columns(df):
    """
    Encodes quality/condition-related columns with a custom mapping.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.

    Returns:
    - pd.DataFrame: DataFrame with encoded quality columns.
    """
    custom_mapping = {
        'Missing': 0,
        'Po': 1,
        'Fa': 2,
        'TA': 3,  # Typical/Average
        'Gd': 4,
        'Ex': 5,
    }

    quality_cols = [
        'Exter Qual', 'Exter Cond', 'Bsmt Qual', 'Bsmt Cond',
        'Heating QC', 'Kitchen Qual', 'Fireplace Qu', 'Garage Qual',
        'Garage Cond', 'Pool QC',
    ]

    for col in quality_cols:
        if col in df.columns:
            df[col] = df[col].map(custom_mapping)
            logging.info(f"Mapping applied for column '{col}': {custom_mapping}")
            print(f"Mapping for column '{col}':")
            mapping_df = pd.DataFrame(
                list(custom_mapping.items()),
                columns=[f"{col}_Category", f"{col}_Encoded"],
            )
            print(mapping_df, "\n")

    return df


def encode_other_categorical_columns(df):
    """
    Encodes other specific categorical columns based on predefined mappings.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.

    Returns:
    - pd.DataFrame: DataFrame with encoded categorical columns.
    """
    custom_mappings = {
        'Land Slope': {'Gtl': 1, 'Mod': 2, 'Sev': 3},
        'Bsmt Exposure': {'Gd': 4, 'No': 1, 'Mn': 2, 'Av': 3, 'Missing': 0},
        'Lot Shape': {'IR1': 1, 'Reg': 4, 'IR2': 2, 'IR3': 3},
        'Functional': {
            'Typ': 1, 'Mod': 2, 'Min1': 3, 'Min2': 4,
            'Maj1': 5, 'Maj2': 6, 'Sev': 7, 'Sal': 8,
        },
        'Garage Finish': {'Fin': 3, 'Unf': 1, 'RFn': 2, 'Missing': 0},
        'Paved Drive': {'P': 1, 'Y': 2, 'N': 0},
        'Central Air': {'Y': 1, 'N': 0},
    }

    for col, mapping in custom_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            logging.info(f"Encoding applied for column '{col}': {mapping}")
            print(f"Encoding for column '{col}':")
            mapping_df = pd.DataFrame(
                list(mapping.items()),
                columns=[f"{col}_Category", f"{col}_Encoded"],
            )
            print(mapping_df, "\n")

    return df


def encode_remaining_categorical_columns(df):
    """
    Identifies remaining categorical columns and applies a unique encoding.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.

    Returns:
    - tuple: (DataFrame with encoded columns, dictionary of mappings).
    """
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    remaining_mappings = {}

    for col in categorical_cols:
        unique_values = df[col].unique()
        mapping = {
            val: (0 if val == 'Missing' else i + 1)
            for i, val in enumerate(unique_values)
        }
        df[col] = df[col].map(mapping)
        remaining_mappings[col] = mapping
        logging.info(f"Encoding applied for column '{col}': {mapping}")
        print(f"Encoding for remaining column '{col}':")
        mapping_df = pd.DataFrame(
            list(mapping.items()),
            columns=[f"{col}_Category", f"{col}_Encoded"],
        )
        print(mapping_df, "\n")

    return df, remaining_mappings


def encode_data(data):
    """
    Main function to perform encoding on the dataset.

    Parameters:
    - data (str): JSON string containing the dataset.

    Returns:
    - tuple: (Serialized JSON of the encoded DataFrame, mappings dictionary).
    """
    df = pd.read_json(data)

    # Encode quality/condition-related columns
    df = encode_quality_columns(df)

    # Encode other specified categorical columns
    df = encode_other_categorical_columns(df)

    # Encode any remaining categorical columns
    df, remaining_mappings = encode_remaining_categorical_columns(df)

    # Serialize the encoded DataFrame to JSON
    serialized_data = df.to_json()

    return serialized_data, remaining_mappings
