# feature_functions.py 

"""
Submodule containing tools to extract features from the data.
"""

import pandas as pd
from sklearn.model_selection import cross_val_score

def perform_cross_validation(X, y, model, cv=5):
    """
    Perform cross-validation using the given model.
    Parameters:
        X (array-like): The input features.
        y (array-like): The target variable.
        model: The machine learning model to use for cross-validation.
        cv (int, optional): The number of folds for cross-validation. Default is 5.
    Returns:
        scores (array-like): The cross-validation scores.
    """
    
    # X = dataframe.drop('target', axis=1)  # Assuming 'target' is the target variable column
    # y = dataframe['target']
    
    scores = cross_val_score(model, X, y, cv=cv)  # Perform 5-fold cross-validation
    
    return scores
