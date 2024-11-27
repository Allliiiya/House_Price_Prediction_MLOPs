def detect_model_bias(test_data, y_pred, encoded_features, target):
    """
    Detect model bias by evaluating performance metrics across groups.

    Parameters:
    - test_data (pd.DataFrame): Test dataset containing predictions and group features.
    - y_pred (np.ndarray): Model predictions for the test data.
    - group_feature (str): Feature name to group by (e.g., 'Overall Qual').
    - target (str): Target variable name (e.g., 'SalePrice').

    Returns:
    - pd.DataFrame: A DataFrame with metrics calculated for each group.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    import pandas as pd

    y_true = test_data[target]

    group_metrics = []
    for feature in encoded_features:
        group_indices = test_data[feature] == 1  # Rows where the feature is active
        y_true_group = y_true[group_indices]
        y_pred_group = y_pred[group_indices]

        if len(y_true_group) == 0:  # Skip empty groups
            continue

        mae = mean_absolute_error(y_true_group, y_pred_group)
        mse = mean_squared_error(y_true_group, y_pred_group)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_group, y_pred_group)

        group_metrics.append({
            'Group': feature,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
        })

    return pd.DataFrame(group_metrics)


def evaluate_bias_disparity(bias_metrics, metric_columns):
    """
    Evaluate disparities in model performance metrics across groups.

    Parameters:
    - bias_metrics (pd.DataFrame): DataFrame with metrics calculated for each group.
    - metric_columns (list): List of metric column names to evaluate disparities (e.g., ['MAE', 'MSE', 'RMSE', 'R2']).

    Returns:
    - dict: Disparity information including max, min, and range for each metric.
    """
    disparities = {}
    for metric in metric_columns:
        max_val = bias_metrics[metric].max()
        min_val = bias_metrics[metric].min()
        disparities[metric] = {
            'max': max_val,
            'min': min_val,
            'disparity': max_val - min_val,
        }
    return disparities
