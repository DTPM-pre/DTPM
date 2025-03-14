import numpy as np
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr
import pandas as pd


def calculate_combined_correlations(features, targets):
    feature_names = features.columns.tolist()
    num_features = features.shape[1]
    combined_correlations = []

    # 
    linear_correlations = np.zeros(num_features)
    nonlinear_correlations = np.zeros(num_features)

    # 
    for i in range(num_features):
        feature = features.iloc[:, i].values
        corr, _ = pearsonr(feature, targets)
        linear_correlations[i] = abs(corr)
        nonlinear_correlations[i] = mutual_info_regression(feature.reshape(-1, 1), targets)

    # 
    for i in range(num_features):
        combined_correlations.append(linear_correlations[i] + abs(nonlinear_correlations[i]))

    # 
    combined_correlations_df = pd.DataFrame({
        'Feature': feature_names,
        'Combined Correlation': combined_correlations
    })
    return combined_correlations_df
