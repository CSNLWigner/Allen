# analyses/rrr.py

"""
This module contains tools for performing Reduced Rank Regression (RRR) analysis on neural data.

Functions:

- getCoeffs(model, log=False) -> numpy.ndarray: Calculate the mean coefficients of a model over cross-validation folds.
- RRRR(X_data, Y_data, dataBalancing='none', rank=None, cv=None, log=False, success_log=True, warn=True) -> dict: Make Reduced Rank Regression (RRR) analysis.
- RFE_CV(X_data, Y_data, rank=None, cv=None) -> dict: Perform Recursive Feature Elimination (RFE) cross-validation.
- compare_two_areas(area_X_responses:np.ndarray, area_Y_responses:np.ndarray, log=False) -> dict: Compare the responses of units in two brain areas using Reduced Rank Regression (RRR).
- control_models(predictor_names=['V1', 'movement', 'pupil'], response_name='V2', log=False) -> np.ndarray: Perform control models analysis using Reduced Rank Regression (RRR).
- rrr_rank_analysis(V1_activity, V2_activity, max_rank=15, cv=params['cv'], log=False) -> np.ndarray: Perform Reduced Rank Regression (RRR) rank analysis.
- calculate_cross_time_correlation(areaX, areaY, log=False) -> np.ndarray: Calculate the cross-time correlation RRR between the responses of two brain areas.
- cross_time_rrr_coeffs(V1_activity, V2_activity, cv=None, rank=None) -> np.ndarray: Calculate the cross-time RRR coefficients between two sets of activities.
- crosstime_analysis(predictor, target, cv, rank, scaling_factor=10, dataBalancing='none', ProgressBar=True) -> np.ndarray: Perform cross-time analysis based on timpoints of rrr-param-search lag.
"""

import numpy as np
import pandas as pd
import yaml
from allensdk.brain_observatory.ecephys.behavior_ecephys_session import \
    BehaviorEcephysSession
from sklearn.model_selection import cross_validate

from analyses.data_preprocessing import preprocess_area_responses
from analyses.imbalanced_data import undersampled_cross_validation
from analyses.machine_learning_models import (ReducedRankRidgeRegression,
                                              custom_feature_selection)
from utils.data_io import load_pickle
from utils.utils import manager

preprocess = yaml.safe_load(open('params.yaml'))['preprocess']
params = yaml.safe_load(open('params.yaml'))['rrr']

def getCoeffs(model, log=False):
    """
    Calculate the mean coefficients of a model over cross-validation folds.

    Parameters:
        model (list): A list of estimators representing the model.
        log (bool, optional): Whether to print the shape of the coefficients. Default is False.

    Returns:
        numpy.ndarray: The mean coefficients of the model.

    Note:
        If the model is None, None is returned.
    """

    # If the model is None, return None
    if model[0] is None:
        return None

    # Concatenate the coefficients over the cross-validation folds
    coefficients = np.array(
        [estimator.coef_ for estimator in model])
    if log:
        print('coefficients.shape', coefficients.shape)

    # Calculate the mean of the coefficients
    mean_coefficients = np.mean(coefficients, axis=0)

    # Append the mean coefficients to the results
    return mean_coefficients.T

def RRRR(X_data, Y_data, dataBalancing='none', rank=None, cv=None, log=False, success_log=True, warn=True) -> dict:
    """
    Make Reduced Rank Regression (RRR) analysis.

    Args:
        X_data (np.ndarray): The data of the first brain area. Shape (n_samples, n_features_X)
        Y_data (np.ndarray): The data of the second brain area. Shape (n_samples, n_features_Y)
        dataBalancing (str, optional): The type of the RRR analysis. Either 'none' or 'undersampled'. Defaults to 'none'.
        log (bool, optional): Whether to log the progress. Defaults to True.
        rank (int, optional): The rank of the RRR model. Defaults to None.
        cv (int, optional): The number of cross-validation folds. Defaults to None.

    Returns:
        dict: A dictionary containing the results of the RRR analysis.
            - mean_coefficients (np.ndarray): The mean coefficients of the RRR model. Shape (n_features_X, n_features_Y)
            - test_score (np.ndarray): The test scores of the RRR model. Shape (n_splits,)
            - estimator (np.ndarray): The estimators of the RRR model. Shape (n_splits,)
    """
    
    # Set default values
    if rank is None:
        rank = params['rank']
    if cv is None:
        cv = params['cv']

    # Make RRR model
    model = ReducedRankRidgeRegression(rank=rank)
    
    # Set the undersampling sample size
    sample_size = params['sample-size']
    
    # Perform cross-validation
    if dataBalancing == 'none':
        results = cross_validate(model, X_data, Y_data, cv=cv, scoring='r2', return_estimator=True)
    elif dataBalancing == 'undersampled':
        results = undersampled_cross_validation(model, X_data, Y_data, sample_size, k_folds=cv, warn=warn)
    else:
        raise ValueError('Invalid type. Use either "none" or "undersampled".')
    
    # Log the cross-validation scores
    if log:
        print('Cross-validation scores:', results['test_score'])
    
    # Get the coefficients
    results['mean_coefficients'] = getCoeffs(results['estimator'], log=log)
    
    # Get the adjusted R2 score
    results['adjusted_r2'] = 1 - (1 - results['test_score']) * (Y_data.shape[0] - 1) / (Y_data.shape[0] - X_data.shape[1] - 1)
    if log:
        print('R2:', results['test_score'].mean())
        print('Adjusted R2:', results['adjusted_r2'].mean())
    
    # If mean of the scores is not negative, then print the cv, rank and the mean of the scores
    if success_log and np.mean(results['test_score']) > 0:
        print(f'CV: {cv}, Rank: {rank}, Mean test score: {np.mean(results["test_score"])}')
        
    # Negative scores are not meaningful, so set them to nan
    # results['test_score'][results['test_score'] < 0] = np.nan
    if np.mean(results['test_score']) < 0:
        results['test_score'] = np.array([np.nan])
    
    return results

def RFE_CV(X_data, Y_data, rank=None, cv=None):

    # Set default values
    if rank is None:
        rank = params['rank']
    if cv is None:
        cv = params['cv']
    
    # Make RFE model
    optimal_features = custom_feature_selection(X_data, Y_data, rank, n_splits=cv)
    
    return optimal_features

def compare_two_areas(area_X_responses:np.ndarray, area_Y_responses:np.ndarray, log=False) -> dict:
    """
    Compare the responses of units in two brain areas using Reduced Rank Regression (RRR).

    Args:
        session (Session): The session object containing the spike times and stimulus presentations.
        area_X (str): The name of the first brain area.
        area_Y (str): The name of the second brain area.
        session_block (int): The stimulus block number.
            0: change detection task
            2: receptive field mapping by gabor stimuli
            4: full-flash
            5: passive replay
        log (bool, optional): Whether to log the progress. Defaults to True.

    Returns:
        dict: A dictionary containing the results of the CCA analysis.
            - coefficients (np.ndarray): The coefficients of the CCA model. Shape (Y_features, X_features)
    """
    
    # Parameters
    binSize = preprocess['bin-size']
    duration = preprocess['stimulus-duration']
    time_length = int(duration/binSize)
    n_area_X_units, n_area_Y_units = area_X_responses.shape[0], area_Y_responses.shape[0]

    # Make RRR
    if log:
        print('Make RRR...')
    coefficients = np.zeros((n_area_X_units, n_area_Y_units, time_length))
    for time in range(time_length):
        coefficients[:, :, time]= RRRR(area_X_responses[:, :, time].T, area_Y_responses[:, :, time].T, log=False).T

    if log:
        print('coefficients.shape', coefficients.shape)

    return {
        'coefficients': coefficients
    }


def control_models(predictor_names=['V1', 'movement', 'pupil'], response_name='V2', log=False) -> np.ndarray:
    '''
    Perform control models analysis using Reduced Rank Regression (RRR).

    This function loads the necessary data, transforms the behavioral data to match the neuronal data,
    and performs RRR analysis at each time point. The analysis is performed using the V1 neuronal activity
    as predictors and V2 neuronal activity as the target variable. The behavioral data (movement and pupil)
    can also be included as predictors.
    
    Params:
        - predictor names (list): Define, which data to concatenate into the predictors. The options are: 'V1', 'movement', 'pupil'.
        - outcome name (str): The name of the outcome variable. Default is 'V2'.

    Returns:
        - results (numpy.ndarray): Array of shape (T, cv) containing the RRR test scores at each time point.
    '''
    
    # Load the data
    V1 = load_pickle("5_block_VISp-activity", path="data/area-responses") # shape (Neurons, Trials, Time)
    V2 = load_pickle("5_block_VISl-activity", path="data/area-responses") # shape (Neurons, Trials, Time)
    movement = load_pickle("5_block_running-speed", path="data/behav-responses") # shape (Trials, Time)
    pupil = load_pickle("5_block_pupil-area", path="data/behav-responses") # shape (Trials, Time)
    
    # Transform the behav data to amtch the V1 and V2 data
    movement = movement[np.newaxis, :, :]
    pupil = pupil[np.newaxis, :, :]
    
    # Get the number of neurons, trials, and time
    N_1, K, T = V1.shape
    N_2 = V2.shape[0]
    if log:
        print(f'V1 shape: {V1.shape}, V2 shape: {V2.shape}, movement shape: {movement.shape}, pupil shape: {pupil.shape}')
    
    # Init the results bs shape (T, cv)
    results = np.zeros((params['cv'], T))
    
    # Loop through the time
    for t in range(T):
    
        # Make DataFrame from the data
        X_V1  = pd.DataFrame(V1[:, :, t].T, columns=[f'V1_{i}' for i in range(V1.shape[0])])
        X_mov = pd.DataFrame(movement[:, :, t].T, columns=['movement'])
        X_pup = pd.DataFrame(pupil[:, :, t].T, columns=['pupil'])
        Y_V2  = pd.DataFrame(V2[:, :, t].T, columns=[f'V2_{i}' for i in range(V2.shape[0])])
                
        # Calculate mean values
        X_V1_mean = X_V1.mean()
        X_mov_mean = X_mov.mean()
        X_pup_mean = X_pup.mean()
        Y_V2_mean = Y_V2.mean()
        
        # If any of the mean values are NaN, print a warning specifying which one
        if log:
            if X_V1_mean.isna().any():
                print('X_V1_mean contains NaNs')
            if X_mov_mean.isna().any():
                print('X_mov_mean contains NaNs')
            if X_pup_mean.isna().any():
                print('X_pup_mean contains NaNs')
            if Y_V2_mean.isna().any():
                print('Y_V2_mean contains NaNs')

        # Replace the NaNs with the mean
        X_V1.fillna(X_V1_mean, inplace=True)
        X_mov.fillna(X_mov_mean, inplace=True)
        X_pup.fillna(X_pup_mean, inplace=True)
        Y_V2.fillna(Y_V2_mean, inplace=True)
        
        # Create a dictionary of the dataframes
        dfs = {'V1': X_V1, 'movement': X_mov, 'pupil': X_pup, 'V2': Y_V2}
        
        # Create a list from the dataframes if they are listed in the predictor names list
        predictors = [df for name, df in dfs.items() if name in predictor_names]
        
        # Concatenate the data
        X = pd.concat(predictors, axis=1).values
        
        # Get the outcome variable
        Y = dfs[response_name].values
    
        # if x contains nan, raise an error
        if np.isnan(X).any():
            raise ValueError('X contains NaNs')
    
        # Make Reduced Rank Reegression
        scores = RRRR(X, Y, rank=params['rank'], cv=params['cv'], log=log)
        
        # Append the scores to the results
        results[:, t] = scores['test_score']
    
    return results

def rrr_rank_analysis(V1_activity, V2_activity, max_rank=15, cv=params['cv'], log=False):
    """
    Perform Reduced Rank Regression (RRR) rank analysis.
    """
    
    # Get the number of neurons, trials, and time points
    N, K_V1, T = V1_activity.shape
    
    # Define the range of ranks to iterate over
    ranks = range(1, max_rank+1)

    # Initialize the errors
    test_scores = np.zeros((max_rank, T))

    # Iterate over time
    for t in range(T):
        for rank in ranks:
            X = V1_activity[:, :, t].T
            Y = V2_activity[:, :, t].T
        
            # Calculate rrr ridge using your rrrr function
            models = RRRR(X, Y, rank=rank, cv=cv)
            
            # Calculate the mean of the test scores above the cv-folds
            test_score = np.mean(models['test_score'])

            # Save the test score
            test_scores[rank-1, t] = test_score

            if log:
                print(f"Rank: {rank}, Time: {t}, Test Score: {test_score}")

    # If test scores are negative, set them to nan
    test_scores[test_scores < 0] = np.nan
    
    return test_scores

def calculate_cross_time_correlation(areaX, areaY, log=False) -> np.ndarray:
    """
    Calculate the cross-time correlation RRR between the responses of two brain areas. The output will be a matrix of shape (T, T) where T is the number of time points.
    """
    
    # Get the number of neurons, trials, and time points
    N, K, T = areaX.shape
    
    # Initialize the cross-time correlation coefficients
    cross_time_r2 = np.zeros((T, T))
    
    # Iterate over time
    for t in range(T):
        for s in range(T):
            X = areaX[:, :, t].T
            Y = areaY[:, :, s].T
        
            # Calculate rrr ridge using your rrrr function
            models = RRRR(X, Y, rank=params['rank'], cv=params['cv'])
            
            # Calculate the mean of the test scores above the cv-folds
            cross_time_r2[t, s] = np.mean(models['test_score'])

    if log:
        print('cross_time_r2\n', cross_time_r2)

    # The values must be nan where the time of V1 is greater than V2
    cross_time_r2[np.tril_indices(cross_time_r2.shape[0], k=-1)] = np.nan

    # The negative values must be nan, because they are not meaningful
    cross_time_r2[cross_time_r2 < 0] = np.nan
    
    return cross_time_r2


def cross_time_rrr_coeffs(V1_activity, V2_activity, cv=None, rank=None) -> np.ndarray:
    """
    Calculate the cross-time RRR coefficients between two sets of activities.

    Parameters:
        V1_activity (np.ndarray): Array of activities for the first set.
        V2_activity (np.ndarray): Array of activities for the second set.

    Returns:
        np.ndarray: Array of cross-time RRR coefficients. Shape (timelength_X, timelength_Y)

    """
    return RRRR(V1_activity.mean(axis=0), V2_activity.mean(
        axis=0), cv=cv, rank=rank, log=True)


def crosstime_analysis(predictor, target, cv, rank, scaling_factor=10, dataBalancing='none', ProgressBar=True):
    """
    Perform cross-time analysis based on timpoints of rrr-param-search lag.
    
    Args:
        predictor (ndarray): Array of shape (neurons, trials, timepoints) representing the activity of the predictor area.
        target (ndarray): Array of shape (neurons, trials, timepoints) representing the activity of the target area.
        cv (int): Number of cross-validation folds to use.
        rank (int): Rank of the reduced-rank regression model.
        scaling_factor (int, optional): Scaling factor for the time points. Defaults to 1.
        dataBalancing (str, optional): Type of data balancing to use. Defaults to 'none'.
        ProgressBar (bool, optional): Whether to display a progress bar. Defaults to True.
    
    Returns:
        ndarray: Array of shape (len(xseries), len(yseries)) containing the results of the cross-time analysis.
    """
    predictor_orig = predictor
    target_orig = target
    
    # Get the time bin
    time_bin = int(preprocess["bin-size"] * 1000) # in ms
    
    # Define the parameters
    xseries = np.arange(0, 200, scaling_factor)
    yseries = np.arange(0, 200, scaling_factor)
    
    # Init results
    results = np.full((len(xseries), len(yseries)), fill_value=np.nan)
    
    # If sample_size is greater than the number of samples, then log a warning and return empty results
    if dataBalancing == 'undersampled':
        y_length = target_orig.shape[0]
        if params['sample-size'] > y_length:
            print(f"Waring: sample_size ({params['sample-size']}) is greater than the number of samples ({y_length}). Returning empty results.")
            return results
    
    # In case of undersampling lower boundary of layer 5 -> do not calculate the other layers
    # if dataBalancing == 'none': # TODO: wipe this out
    #     return results
    
    # Print progressbar
    progress_bar_id = 'crosstime analysis'
    if type(ProgressBar) == str:
        progress_bar_id = ProgressBar
        ProgressBar = True
    if ProgressBar: manager.progress_bar(progress_bar_id, 0, len(xseries))
    
    for x, t_x in enumerate(xseries):
        for y, t_y in enumerate(yseries):
            
            # Preprocess the data (the trial duration is only one bin now).
            predictor = preprocess_area_responses(predictor_orig[:, :, t_x : t_x + time_bin], 
                                                  stimulus_duration=preprocess["bin-size"], 
                                                  step_size=preprocess["step-size"]).squeeze()
            target    = preprocess_area_responses(target_orig[:, :, t_y : t_y + time_bin], 
                                                  stimulus_duration=preprocess["bin-size"], 
                                                  step_size=preprocess["step-size"]).squeeze()
            
            # Calculate the RRRR
            res = RFE_CV(predictor.T, target.T, rank=rank, cv=cv)
            
            # Save results
            results[x, y] = res
            
        # Print progressbar
        if ProgressBar: manager.progress_bar(progress_bar_id, x+1, len(xseries))
    
    return results