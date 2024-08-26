# analyses/rrr_time_slice.py

"""
This module contains tools for performing RRRR analysis on time slices.

Functions:
- RRRR_time_slice(predictor, target, predictor_time, cv, rank, log=True) -> dict: Calculate the RRRR for each time slice.
- bidirectional_time_slice(V1_activity, LM_activity, session_params:pd.DataFrame, predictor_time, log=False) -> dict: Perform bidirectional time slice analysis using RRRR.
"""

import numpy as np
import pandas as pd
from analyses.data_preprocessing import preprocess_area_responses, z_score_normalize
from analyses.rrr import RRRR
from scipy.stats import sem as SEM
from utils.utils import printProgressBar

import yaml
preprocess = yaml.safe_load(open("params.yaml"))["preprocess"]

def RRRR_time_slice(predictor, target, predictor_time, cv, rank, log=True):
    """
    Calculate the RRRR (Reduced Rank Regression Regularization) for each time slice.

    Args:
        predictor (ndarray): The predictor data with shape (N, M, T), where N is the number of samples,
                             M is the number of features, and T is the number of time slices.
        target (ndarray): The target data with shape (N, P, T), where N is the number of samples,
                          P is the number of target variables, and T is the number of time slices.
        predictor_time (float): The time point of interest in milliseconds.
        cv (int): The number of cross-validation folds.
        rank (int): The rank of the RRRR model.

    Returns:
        dict: A dictionary containing the mean and standard error of the RRRR scores for each time slice.
              The dictionary has the following keys:
              - 'mean': ndarray of shape (T), containing the mean RRRR scores for each time slice.
              - 'sem': ndarray of shape (T), containing the standard error of the RRRR scores for each time slice.
    """
    
    T = predictor.shape[2]
    t_pred = int(predictor_time / preprocess["step-size"])
    
    if log:
        print('predictor_time', predictor_time)

    # Init results
    mean = np.full((T), fill_value=np.nan)
    sem  = np.full((T), fill_value=np.nan)
    
    # Print progress bar
    if log:
        printProgressBar(0, T, prefix = 'RRR analysis:', length = 50)

    for x in range(T):
        
        predictor_t = predictor[:, :, t_pred]
        target_t = target[:, :, x]

        # Calculate the RRRR
        model = RRRR(predictor_t.T, target_t.T,
                     rank=rank, cv=cv, success_log=False)

        # Save results
        mean[x]  =   model['test_score'].mean()
        sem[x] = SEM(model['test_score'])
        
        # Update progress bar
        if log:
            printProgressBar(x + 1, T, prefix = 'RRR analysis:', length = 50)
        
    # Return the results
    return {
        'mean': mean,
        'sem': sem
    }


def bidirectional_time_slice(V1_activity, LM_activity, session_params:pd.DataFrame, predictor_time, log=False):
    """
    Perform bidirectional time slice analysis using RRRR.

    Args:
        V1_activity (numpy.ndarray): The activity data for V1.
        LM_activity (numpy.ndarray): The activity data for LM.
        session_params (pd.DataFrame): The best parameters for each prediction direction and session.
        predictor_time (numpy.ndarray): The time points for the predictor activity.

    Returns:
        dict: A dictionary containing the results for each prediction direction.

    """
    
    # Define the parameters
    abstract_areas = {
        'top-down': {
            'predictor': LM_activity,
            'target': V1_activity
        },
        'bottom-up': {
            'predictor': V1_activity,
            'target': LM_activity
        },
        'V1': {
            'predictor': V1_activity,
            'target': V1_activity
        },
        'LM': {
            'predictor': LM_activity,
            'target': LM_activity
        }
    }

    # Init results
    results = {}

    # Iterate through the prediction directions
    for prediction_direction in ['top-down', 'bottom-up', 'V1', 'LM']:
        
        # Extract the data
        predictor_activity = abstract_areas[prediction_direction]['predictor']
        target_activity = abstract_areas[prediction_direction]['target']
        
        # Extract the session parameters
        if prediction_direction in ['top-down', 'bottom-up']:
            session_key = prediction_direction
        if prediction_direction is 'V1':
            session_key = 'bottom-up'
        if prediction_direction is 'LM':
            session_key = 'top-down'
        params = session_params[session_params['direction'] == session_key]
        cv = params['cv'].values[0]
        rank = params['rank'].values[0]

        # Calculate the RRRR
        result = RRRR_time_slice(
            predictor_activity, target_activity, predictor_time, cv, rank, log=log)

        # Save the results
        results[prediction_direction] = result

    return results