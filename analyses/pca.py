# analyses/pca.py

"""
This module contains functions for performing PCA on neural data.

Functions:
- pca(neural_activity) -> dict: Perform PCA on the given neural activity data.
"""

from sklearn.decomposition import PCA


def pca(neural_activity):

    # Create an instance of PCA
    pca = PCA()

    # Fit the PCA model to the reshaped data
    pca.fit(neural_activity)

    # Get the principal components
    principal_components = pca.components_  # shape (Neurons, PCs)

    # Get the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_ # shape (PCs,)

    # Perform dimensionality reduction by transforming the data
    reduced_data = pca.transform(neural_activity) # shape (Trials, PCs)
    
    # Return the results
    return {
        'principal_components': principal_components,
        'explained_variance_ratio': explained_variance_ratio,
        'components': reduced_data
    }