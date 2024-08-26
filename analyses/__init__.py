# analyses/__init__.py

"""
This package contains tools for analyzing neural data.

Submodules
----------

- `cca`: Canonical Correlation Analysis (CCA) tools.
- `corr`: Correlation analysis tools.
- `pca`: Principal Component Analysis (PCA) tools.
- `cv-lag-time-selection`: tools for selecting the maximum mean across the CV, lag, and time dimensions.
- `rrr_time_slice`: tools for performing RRRR analysis on time slices.
- `layer_rank`: tools for performing rank-based analysis on neural data.
- `data_preprocessing`: tools for preprocessing neural data.
- `imbalanced_preprocessing`: tools for preprocessing imbalanced data.
- `rrr`: Reduced Rank Regression Regularization (RRRR) tools.
- `layer_interaction_maxValues`: tools for calculating the maximum values of the layer interaction matrix.
- `machine_learning_models`: machine learning models for regression and feature selection.
"""

# Package metadata

__version__ = "1.0"
__author__ = "Zsombor Szabó"

# Import submodules

from . import cca, corr, pca, cv_lag_time_selection, rrr_time_slice, layer_rank, data_preprocessing, imbalanced_preprocessing, rrr, layer_interaction_maxValues, machine_learning_models

__all__ = ['cca', 'corr', 'pca', 'cv_lag_time_selection', 'rrr_time_slice', 'layer_rank', 'data_preprocessing', 'imbalanced_preprocessing', 'rrr', 'layer_interaction_maxValues', 'machine_learning_models']
