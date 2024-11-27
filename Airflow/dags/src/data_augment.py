import pandas as pd
import numpy as np
# from src.feature_select import rank_features_by_lasso


def augment_data_with_perturbations(data, final_features, perturbation_percentage=0.02):
    """
    Generate synthetic records by applying random perturbations to numeric features.
    Parameters:
    - data (pd.DataFrame): Original training data.
    - numeric_features (list): List of numerical feature column names to perturb.
    - num_synthetic_records (int): Number of synthetic records to generate.
    - perturbation_percentage (float): Percentage of the standard deviation to use for noise (default is 5%).
    Returns:
    - pd.DataFrame: Augmented training dataset containing both original and synthetic records.
    """

    # adding 2000 synthetic records
    num_synthetic_records = 2000

    # Randomly sample rows from the data to generate synthetic records
    synthetic_data = data.sample(n=num_synthetic_records, replace=True).copy()

    # Apply perturbation to each selected numerical feature
    for feature in final_features:
        # Calculate noise based on feature's standard deviation and perturbation percentage
        noise = np.random.normal(0, perturbation_percentage * synthetic_data[feature].std(), num_synthetic_records)
        synthetic_data[feature] += noise

    # Combine synthetic data with original data
    augmented_data = pd.concat([data, synthetic_data], ignore_index=True)

    return augmented_data
