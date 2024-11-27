# src/sensitivity_analysis.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import shap
import logging
from typing import Dict, Any

def analyze_shap_values(model, X_data, feature_names, sample_size=1000):
    """
    Analyze feature importance using SHAP values.
    
    Parameters:
    - model: Trained linear model
    - X_data: Feature matrix
    - feature_names: List of feature names
    - sample_size: Number of samples to use for SHAP analysis
    
    Returns:
    - Dict containing SHAP values and summary statistics
    """
    try:
        # Sample data for SHAP analysis if needed
        if len(X_data) > sample_size:
            indices = np.random.choice(len(X_data), sample_size, replace=False)
            X_sample = X_data[indices]
        else:
            X_sample = X_data

        # Calculate SHAP values
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)
        
        # Calculate mean absolute SHAP values for each feature
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        
        # Create SHAP summary
        shap_summary = pd.DataFrame({
            'feature': feature_names,
            'mean_shap_value': mean_shap_values
        }).sort_values('mean_shap_value', ascending=False)
        
        logging.info("SHAP analysis completed successfully")
        
        return {
            'shap_values': shap_values,
            'shap_summary': shap_summary,
            'explainer': explainer
        }
        
    except Exception as e:
        logging.error(f"Error in SHAP analysis: {str(e)}")
        raise

def generate_sensitivity_report(data: pd.DataFrame, features: list, target: str = 'SalePrice') -> Dict[str, Any]:
    """
    Generates a comprehensive sensitivity analysis report including SHAP values.
    
    Parameters:
    - data: Input dataset
    - features: List of feature names to analyze
    - target: Target variable name
    
    Returns:
    - Dictionary containing all sensitivity analysis results
    """
    try:
        # Standardize features
        X = data[features]
        y = data[target]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=features)
        
        # Train model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Regular feature importance analysis
        importance_results = pd.DataFrame({
            'feature': features,
            'importance': abs(model.coef_)
        }).sort_values('importance', ascending=False)
        
        # SHAP analysis
        shap_results = analyze_shap_values(model, X_scaled_df, features)
        
        # Feature sensitivity analysis (leave-one-out)
        sensitivity_results = []
        baseline_score = model.score(X_scaled, y)
        
        for i, feature in enumerate(features):
            X_reduced = np.delete(X_scaled, i, axis=1)
            model_reduced = LinearRegression()
            model_reduced.fit(X_reduced, y)
            reduced_score = model_reduced.score(X_reduced, y)
            impact = baseline_score - reduced_score
            
            sensitivity_results.append({
                'feature': feature,
                'impact_score': impact
            })
        
        sensitivity_df = pd.DataFrame(sensitivity_results)
        
        # Feature stability analysis
        stability_results = []
        n_iterations = 10
        
        for _ in range(n_iterations):
            sample_indices = np.random.choice(len(data), size=int(0.8*len(data)), replace=True)
            sample_data = data.iloc[sample_indices]
            
            X_sample = sample_data[features]
            y_sample = sample_data[target]
            
            X_sample_scaled = scaler.transform(X_sample)
            model_sample = LinearRegression()
            model_sample.fit(X_sample_scaled, y_sample)
            
            stability_results.append(abs(model_sample.coef_))
        
        stability_matrix = np.array(stability_results)
        stability_df = pd.DataFrame({
            'feature': features,
            'mean_coef': np.mean(stability_matrix, axis=0),
            'std_coef': np.std(stability_matrix, axis=0),
            'stability_score': np.mean(stability_matrix, axis=0) / (np.std(stability_matrix, axis=0) + 1e-10)
        })
        
        # Combine all results
        report = {
            'feature_importance': importance_results,
            'shap_analysis': {
                'shap_summary': shap_results['shap_summary'],
                'shap_values': shap_results['shap_values']
            },
            'feature_sensitivity': sensitivity_df,
            'feature_stability': stability_df,
            'baseline_model_score': baseline_score
        }
        
        logging.info("Comprehensive sensitivity report generated successfully")
        
        return report
        
    except Exception as e:
        logging.error(f"Error in generate_sensitivity_report: {str(e)}")
        raise