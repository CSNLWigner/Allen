import pandas as pd
from sklearn.model_selection import cross_val_score

def perform_cross_validation(X, y, model, cv=5):
    # X = dataframe.drop('target', axis=1)  # Assuming 'target' is the target variable column
    # y = dataframe['target']
    
    scores = cross_val_score(model, X, y, cv=cv)  # Perform 5-fold cross-validation
    
    return scores
