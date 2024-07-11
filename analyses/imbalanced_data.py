import numpy as np
from joblib import Parallel, delayed
from sklearn import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


def evaluate_fold(model, X_train, X_test, y_train, y_test, sample_size, replace):
    """
    Train and evaluate the model on a single fold.
    """
    # Perform undersampling on the training data using random selecting 'sample_size' samples.
    indices = np.random.choice(len(y_train), sample_size, replace=replace)
    X_train_undersampled = X_train[indices]
    y_train_undersampled = y_train[indices]

    # Train the model on the undersampled data
    model.fit(X_train_undersampled, y_train_undersampled)

    # Evaluate the model on the held-out fold (non-undersampled)
    y_pred = model.predict(X_test)

    # Calculate R^2 score and Mean Squared Error
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    return r2, mse, model

def undersampled_cross_validation(estimator, X, y, sample_size, k_folds=5, replace=False, log=False, n_jobs=-1, warn=True) -> dict:
    """
    Perform undersampled cross-validation on the given dataset using the specified model, in parallel.

    Parameters:
    - X (array-like): The input features.
    - y (array-like): The target variable.
    - model: The machine learning model to be trained and evaluated.
    - k_folds (int): The number of folds for cross-validation. Default is 5.
    - sample_size (int): The number of samples to be randomly selected for undersampling.
    - replace (bool): Whether to allow replacement when undersampling. Default is False.
    - log (bool): Whether to log the overall performance metrics. Default is False.
    - n_jobs (int): The number of CPUs to use to do the computation. -1 means using all processors.
    - warn (bool): Whether to log a warning if sample_size is greater than the number of samples. Default is True.

    Returns:
    dict: A dictionary containing the test scores, F1 scores, and the trained model.
    """

    kf = KFold(n_splits=k_folds, shuffle=True)
    
    # If sample_size is greater than the number of samples, then log a warning and return aret dictionary filled with corresponding number of values.
    n_features = X.shape[1]
    if sample_size > n_features:
        if warn:
            print(f"Waring: sample_size ({sample_size}) is greater than the number of samples ({n_features}). Returning empty results.")
        return {
            'test_score': np.array([np.nan] * k_folds),
            'mse_score': np.array([np.nan] * k_folds),
            'estimator': [None] * k_folds
        }

    # Parallelize the cross-validation using joblib
    results = Parallel(n_jobs=n_jobs)(delayed(evaluate_fold)(
        clone(estimator), X[train_index], X[test_index], y[train_index], y[test_index], sample_size, replace
    ) for train_index, test_index in kf.split(X))

    # Unpack results
    r2_scores, mse_scores, fitted_estimators = zip(*results)

    # Optionally log the results
    if log:
        print(f"R2 scores: {r2_scores}")
        print(f"MSE scores: {mse_scores}")

    # return {"r2_scores": r2_scores, "mse_scores": mse_scores, "model": model}
    
    ret = {}
    ret['test_score'] = r2_scores
    ret['mse_score'] = mse_scores
    ret['estimator'] = fitted_estimators
    
    return ret
