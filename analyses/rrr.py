

import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate
from analyses.machine_learning_models import ReducedRankRidgeRegression
import yaml
import numpy as np
from allensdk.brain_observatory.ecephys.behavior_ecephys_session import BehaviorEcephysSession
from utils.data_io import load_pickle

from utils.utils import MSE

preprocess = yaml.safe_load(open('params.yaml'))['preprocess']
params = yaml.safe_load(open('params.yaml'))['rrr']

def RRRR(X_data, Y_data, rank=None, cv=None, log=False):
    """
    Make Reduced Rank Regression (RRR) analysis.

    Args:
        X_data (np.ndarray): The data of the first brain area. Shape (n_samples, n_features)
        Y_data (np.ndarray): The data of the second brain area. Shape (n_samples, n_features)
        log (bool, optional): Whether to log the progress. Defaults to True.
        rank (int, optional): The rank of the RRR model. Defaults to None.
        cv (int, optional): The number of cross-validation folds. Defaults to None.

    Returns:
        dict: A dictionary containing the results of the RRR analysis.
            - mean_coefficients (np.ndarray): The mean coefficients of the RRR model. Shape (n_features, n_features)
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
    
    # Perform cross-validation
    results = cross_validate(model, X_data, Y_data, cv=cv, return_estimator=True, scoring='r2')
    if log:
        print('Cross-validation scores:', results['test_score'])
    
    # Concatenate the coefficients over the cross-validation folds
    coefficients = np.array([estimator.coef_ for estimator in results['estimator']])
    if log:
        print('coefficients.shape', coefficients.shape)
    
    # Calculate the mean of the coefficients
    mean_coefficients = np.mean(coefficients, axis=0)
    
    # Append the mean coefficients to the results
    results['mean_coefficients'] = mean_coefficients
    
    return results

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
    
    # Init the results bs shape (T, cv)
    results = np.zeros((T, params['cv']))
    
    # Loop through the time
    for t in range(1):
    
        # Make DataFrame from the data
        X_V1  = pd.DataFrame(V1[:, :, t].T, columns=[f'V1_{i}' for i in range(V1.shape[0])])
        X_mov = pd.DataFrame(movement[:, :, t].T, columns=['movement'])
        X_pup = pd.DataFrame(pupil[:, :, t].T, columns=['pupil'])
        Y_V2  = pd.DataFrame(V2[:, :, t].T, columns=[f'V2_{i}' for i in range(V2.shape[0])])
        
        # Replace the NaNs with the mean
        X_V1.fillna(X_V1.mean(), inplace=True)
        X_mov.fillna(X_mov.mean(), inplace=True)
        X_pup.fillna(X_pup.mean(), inplace=True)
        Y_V2.fillna(Y_V2.mean(), inplace=True)
        
        # print()
        # print('X_pup', X_pup)
        # print()
        
        # Create a dictionary of the dataframes
        dfs = {'V1': X_V1, 'movement': X_mov, 'pupil': X_pup, 'V2': Y_V2}
        
        # Create a list from the dataframes if they are listed in the predictor names list
        predictors = [df for name, df in dfs.items() if name in predictor_names]
        
        # Concatenate the data
        X = pd.concat(predictors, axis=1).values
        
        # Get the outcome variable
        Y = dfs[response_name].values
    
        # Make Reduced Rank Reegression
        scores = RRRR(X, Y, rank=params['rank'], cv=params['cv'], log=log)
    
        # Print the scores
        # print(scores)
        
        # Append the scores to the results
        results[t, :] = scores['test_score']
    
    return results