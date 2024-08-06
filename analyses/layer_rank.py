

import numpy as np
import yaml

from analyses.rrr import RRRR

preproc = yaml.safe_load(open('params.yaml'))['preprocess']
params = yaml.safe_load(open('params.yaml'))['layer-rank']

def cv_search(predictor, target):
    """
    Perform a RRRR search for the best cv.
    """

    # Define the cross-validation, and time
    cvs = params['cv']
    rank = params['bestRank']
    timepoint = params['timepoint']

    # Get the time index from the timepoint (ms) and 'step-size' (s)
    t = int(timepoint / 1000 / preproc['step-size'])

    # Create a results array
    results = np.full((10+1), np.nan)

    # Calculate the RRRR for each cv
    for cv in cvs:

        result = RRRR(predictor[:, :, t].T,
                      target[:, :, t].T,
                      rank=rank, cv=cv)

        # Save the result averaged over the folds
        results[cv] = result['test_score'].mean()

    # Cut off the negative values
    results[results < -0] = np.nan

    # Get the maximum value (max_r2) and its index (best_cv)
    max_r2 = np.nanmax(results).round(3)
    best_cv = np.nanargmax(results)

    return best_cv


def rank_search(predictor, target, cv, log=False) -> int:
    """
    Perform a RRRR search for the best rank.

    Parameters:
    predictor (ndarray): The predictor data. Shape: (n_samples, n_features).
    target (ndarray): The target data. Shape: (n_samples, n_features).
    cv (int): The number of cross-validation folds.
    log (bool): Whether to log the results or not.

    Returns:
    int: The best rank found during the search.
    """

    # Define the ranks
    ranks = np.arange(params['minRank'], params['maxRank'], params['stepRank'])

    # Create a results array
    results = np.full((len(ranks)), np.nan)

    # Calculate the RRRR for each rank
    for i, rank in enumerate(ranks):

        result = RRRR(predictor[:, :].T,
                      target[:, :].T,
                      rank=rank, cv=cv,
                      success_log=log) # Switch to False!

        # Save the result averaged over the folds
        results[i] = result['test_score'].mean()

    # Cut off the negative values
    results[results < -0] = np.nan
    
    # If all nan, return nan
    if np.all(np.isnan(results)):
        return np.nan

    # Get the maximum value (max_r2) and its index (max_idx)
    max_r2 = np.nanmax(results).round(3)
    max_idx = np.nanargmax(results)

    return max_idx * params['stepRank'] + params['minRank']

# Some calculations


def calc_ranks(V1_data, LM_data, timepoints, log=False):
    '''
    Calculate the time lag between two time series.

    Parameters:
    - V1_data: numpy array, time series data for V1 area
    - LM_data: numpy array, time series data for LM area
    - timepoints: list, time points to calculate ranks for
    - log: bool, whether to log the results or not (default: False)

    Returns:
    - results: numpy array, array of calculated ranks. Shape: (nAreas(2), nLayers(6+1), nLayers(6+1), nTimepoints)
    '''

    # Calculate the time length after the preprocessing by the time step and the stimulus duration
    time_length = int(preproc['stimulus-duration'] / preproc['step-size'])

    # Create a results array
    results = np.full((2, 6+1, 6+1, time_length), np.nan)

    for i, sourceArea, targetArea in zip([0, 1], [V1_data, LM_data], [LM_data, V1_data]):

        # Parameters
        sourceLayers = sourceArea['layer-assignments'].unique()
        targetLayers = targetArea['layer-assignments'].unique()

        best_cv = cv_search(sourceArea['activity'], targetArea['activity'])

        # Iterate over layer combinations
        for output in sourceLayers:
            for input in targetLayers:

                # Iterate over the time points
                for t in timepoints:

                    best_rank = rank_search(sourceArea['activity'][:, :, t],
                                            targetArea['activity'][:, :, t],
                                            best_cv,
                                            log=log)

                    # Save the result averaged over the folds
                    results[i, output, input, t] = best_rank

    return results
