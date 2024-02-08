"""
Reduced rank regression class.
Requires scipy to be installed.

Implemented by Chris Rayner (2015)
dchrisrayner AT gmail DOT com

Optimal linear 'bottlenecking' or 'multitask learning'.
"""
import sklearn.datasets
import sklearn.linear_model
import numpy as np
from scipy import sparse




class ReducedRankRegressor(object):
    """
    Analytical method
    
    Reduced Rank Regressor (linear 'bottlenecking' or 'multitask learning')
    - X is an n-by-x matrix of features.
    - Y is an n-by-y matrix of targets.
    - rrank is a rank constraint.
    - reg is a regularization parameter (optional).
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
    # rrpy
    
    Gradient descent method.

    **rrpy** is a scikit-learn compatible Python implementation of reduced rank ridge regression.
    It is based on `rrs.fit` method of the R package **rrpack**, which is in turn based on [[1]](#1).

    It does not support missing values, though such a feature could be added using https://github.com/aksarkar/wlra.

    The `ReducedRankRidge` estimator has a `memory` parameter which allows rapid tuning of the `rank` parameter:
    ```python
    import sklearn.datasets
    import joblib
    from rrpy import ReducedRankRidge
    X, Y = sklearn.datasets.make_regression(n_samples=1000, n_features=500, n_targets=50, random_state=1, n_informative=25)
    memory = joblib.Memory(location='/tmp/rrpy-test/', verbose=2)
    estimator = ReducedRankRidge(memory=memory, rank=10)
    estimator.fit(X, Y)
    estimator.rank = 20
    estimator.fit(X, Y) # cached
    memory.clear(warn=False)
    ```

    ## References
    [^1]:
    Mukherjee, A. and Zhu, J. (2011)
    Reduced rank ridge regression and its kernel extensions.

    source: https://github.com/krey/rrpy
    """
    def __init__(self, alpha=1.0, fit_intercept=True, rank=None, ridge_solver='auto', memory=None):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.rank = rank
        self.ridge_solver = ridge_solver
        self.memory = memory

    def fit(self, X, y):
        if self.fit_intercept:
            X_offset = np.average(X, axis=0)
            y_offset = np.average(y, axis=0)
            # doesn't modify inplace, unlike -=
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
        """Predict Y from X."""
        # if np.size(np.shape(X)) == 1:
        #     X = np.reshape(X, (-1, 1))
        return np.dot(X, self.coef_.T)


"""
from analyses.machine_learning_models import ReducedRankRidge, ReducedRankRegressor

X, Y = sklearn.datasets.make_regression(
    n_samples=500, n_features=100, n_targets=30, random_state=1, n_informative=10)
print(Y[0])

estimator = ReducedRankRidge(rank=10)
estimator.fit(X, Y)
Y_hat_1 = estimator.predict(X)
print('\nY_hat_1')
print(Y_hat_1[0])

model = ReducedRankRegressor(X, Y, rank=10)
Y_hat_2 = model.predict(X)
print('\nY_hat_2')
print(Y_hat_2[0])

plt.plot(Y[0])
plt.plot(Y_hat_1[0])
plt.plot(Y_hat_2[0])
plt.show()
"""
