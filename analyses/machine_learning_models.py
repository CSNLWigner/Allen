# analyses/machine_learning_models.py

"""
This module contains machine learning models for regression and feature selection.

Reduced rank regression class.
Requires scipy to be installed.

Implemented by Chris Rayner (2015)
dchrisrayner AT gmail DOT com

Optimal linear 'bottlenecking' or 'multitask learning'.

Functions:
- ReducedRankRegressor(X, Y, rank, reg=None) -> object: Create a ReducedRankRegressor object.
- custom_feature_selection(X_data, Y_data, rank, n_splits=5) -> int: Custom feature selection for multioutput regression using ReducedRankRidgeRegression.
- _fit_rrr_no_intercept_all_ranks(X: np.ndarray, Y: np.ndarray, alpha: float, solver: str) -> tuple: Fit the Reduced Rank Ridge Regression model without intercept for all ranks.
- _fit_rrr_no_intercept(X: np.ndarray, Y: np.ndarray, alpha: float, rank: int, solver: str, memory=None) -> np.ndarray: Fit the Reduced Rank Ridge Regression model without intercept.
- ReducedRankRidgeRegression(alpha=1.0, fit_intercept=True, rank=None, ridge_solver='auto', memory=None) -> object: Reduced Rank Ridge Regression.
"""

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import numpy as np
import sklearn.datasets
import sklearn.linear_model
from matplotlib import pyplot as plt
from scipy import sparse


class ReducedRankRegressor(object):
    """
    Analytical method
    
    Reduced Rank Regressor (linear 'bottlenecking' or 'multitask learning')
        X is an n-by-x matrix of features.
        Y is an n-by-y matrix of targets.
        rrank is a rank constraint.
        reg is a regularization parameter (optional).
    
    ```
    def ideal_data(num, dimX, dimY, rrank, noise=1):
        "Low rank data"
        X = np.random.randn(num, dimX)
        W = np.dot(np.random.randn(dimX, rrank), np.random.randn(rrank, dimY))
        Y = np.dot(X, W) + np.random.randn(num, dimY) * noise
        return X, Y
    ```
    
    Source: https://github.com/riscy/machine_learning_linear_models/tree/master?tab=readme-ov-file
    """

    def __init__(self, X, Y, rank, reg=None):
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        if np.size(np.shape(Y)) == 1:
            Y = np.reshape(Y, (-1, 1))
        if reg is None:
            reg = 0
        self.rank = rank

        CXX = np.dot(X.T, X) + reg * sparse.eye(np.size(X, 1))
        CXY = np.dot(X.T, Y)
        _U, _S, V = np.linalg.svd(
            np.dot(CXY.T, np.dot(np.linalg.pinv(CXX), CXY)))
        self.W = V[0:rank, :].T
        self.A = np.dot(np.linalg.pinv(CXX), np.dot(CXY, self.W)).T

    def __str__(self):
        return f'Reduced Rank Regressor (rank = {self.rank})'

    def predict(self, X):
        """Predict Y from X."""
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        return np.dot(X, np.dot(self.A.T, self.W.T))





def _fit_rrr_no_intercept_all_ranks(X: np.ndarray, Y: np.ndarray, alpha: float, solver: str):
    ridge = sklearn.linear_model.Ridge(
        alpha=alpha, fit_intercept=False, solver=solver)
    beta_ridge = ridge.fit(X, Y).coef_
    Lambda = np.eye(X.shape[1]) * np.sqrt(np.sqrt(alpha))
    X_star = np.concatenate((X, Lambda))
    Y_star = X_star @ beta_ridge.T
    _, _, Vt = np.linalg.svd(Y_star, full_matrices=False)
    return beta_ridge, Vt


def _fit_rrr_no_intercept(X: np.ndarray, Y: np.ndarray, alpha: float, rank: int, solver: str, memory=None):
    memory = sklearn.utils.validation.check_memory(memory)
    fit = memory.cache(_fit_rrr_no_intercept_all_ranks)
    beta_ridge, Vt = fit(X, Y, alpha, solver)
    return Vt[:rank, :].T @ (Vt[:rank, :] @ beta_ridge)


class ReducedRankRidgeRegression(sklearn.base.MultiOutputMixin, sklearn.base.RegressorMixin, sklearn.linear_model._base.LinearModel):
    """
    Reduced Rank Ridge Regression.

    This class implements the Reduced Rank Ridge Regression algorithm, which is a scikit-learn compatible Python implementation of reduced rank ridge regression.
    It is based on the `rrs.fit` method of the R package `rrpack`, which is in turn based on the research paper by Mukherjee and Zhu [^1].

    The `ReducedRankRidgeRegression` estimator supports gradient descent method and does not support missing values.

    Parameters:
        alpha (float, default=1.0): Regularization parameter.
        fit_intercept (bool, default=True): Whether to calculate the intercept for this model.
        rank (int, default=None): Rank of the coefficient matrix.
        ridge_solver (str, default='auto'): Solver to use for ridge regression.
        memory (joblib.Memory, default=None): Memory object to cache the computation of the rank parameter.

    References:
    [^1] Mukherjee, A. and Zhu, J. (2011). Reduced rank ridge regression and its kernel extensions.

    Source: https://github.com/krey/rrpy

    Examples:
    ```python
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from machine_learning_models import ReducedRankRidgeRegression

    # Generate synthetic data
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit the ReducedRankRidgeRegression model
    model = ReducedRankRidgeRegression(alpha=0.1, rank=2)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    ```
    """

    def __init__(self, alpha=1.0, fit_intercept=True, rank=None, ridge_solver='auto', memory=None):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.rank = rank
        self.ridge_solver = ridge_solver
        self.memory = memory

    def fit(self, X, y):
        """
        Fit the Reduced Rank Ridge Regression model to the training data.

        Parameters:
            X (array-like of shape (n_samples, n_features)): Training data.
            y (array-like of shape (n_samples, n_targets)): Target values.

        Returns:
            self (object): Returns self.
        """
        if self.fit_intercept:
            X_offset = np.average(X, axis=0)
            y_offset = np.average(y, axis=0)
            X = X - X_offset
            y = y - y_offset
        self.coef_ = _fit_rrr_no_intercept(
            X, y, self.alpha, self.rank, self.ridge_solver, self.memory)
        self.rank_ = np.linalg.matrix_rank(self.coef_)
        if self.fit_intercept:
            self.intercept_ = y_offset - X_offset @ self.coef_.T
        else:
            self.intercept_ = np.zeros(y.shape[1])
        return self

    def predict(self, X):
        """
        Predict target values for the given test data.

        Parameters:
            X (array-like of shape (n_samples, n_features)): Test data.

        Returns:
            y_pred (array-like of shape (n_samples, n_targets)): Predicted target values.
        """
        return np.dot(X, self.coef_.T)

    def feature_importances_(self):
        # Return the absolute values of the coefficients as the importance scores
        return np.abs(self.coef_)

    def rfe(self, X, y, n_features_to_select):
        # Initialize the full set of features
        features = list(range(X.shape[1]))
        while len(features) > n_features_to_select:
            # Fit the model
            self.fit(X[:, features], y)
            # Get feature importances
            importances = self.feature_importances_()
            # Find the least important feature and remove it
            least_important = np.argmin(importances)
            features.pop(least_important)
        # Finally, fit the model with the selected features
        self.fit(X[:, features], y)
        # Store the selected features for prediction use
        self.selected_features_ = features

    def predict_with_rfe(self, X):
        # Ensure prediction is done using only the selected features
        if hasattr(self, 'selected_features_'):
            return self.predict(X[:, self.selected_features_])
        else:
            raise ValueError("Model has not been fitted using RFE.")


def custom_feature_selection(X_data, Y_data, rank, n_splits=5):
    """
    Custom feature selection for multioutput regression using ReducedRankRidgeRegression.

    Args:
        X_data (np.ndarray): The input features.
        Y_data (np.ndarray): The target outputs.
        rank (int): The rank for the ReducedRankRidgeRegression model.
        n_splits (int): Number of folds for cross-validation.

    Returns:
        int: The optimal number of features.
    """
    n_features = X_data.shape[1]
    scores = np.zeros(n_features)
    kf = KFold(n_splits=n_splits)

    for i in range(n_features):
        current_scores = []
        for train_index, test_index in kf.split(X_data):
            X_train, X_test = X_data[train_index], X_data[test_index]
            Y_train, Y_test = Y_data[train_index], Y_data[test_index]

            # Use only a subset of features
            X_train_subset = X_train[:, :i+1]
            X_test_subset = X_test[:, :i+1]

            model = ReducedRankRidgeRegression(rank=rank)
            model.fit(X_train_subset, Y_train)
            Y_pred = model.predict(X_test_subset)

            # Evaluate the model
            score = r2_score(Y_test, Y_pred, multioutput='uniform_average')
            current_scores.append(score)

        # Store the average score for the current number of features
        scores[i] = np.mean(current_scores)

    # Find the number of features with the best average score
    # Adding 1 because index 0 means 1 feature
    optimal_features = np.argmax(scores) + 1

    print(f"Optimal number of features: {optimal_features}")
    return optimal_features
