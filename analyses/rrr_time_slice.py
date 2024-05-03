import numpy as np
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
        printProgressBar(x + 1, T, prefix = 'RRR analysis:', length = 50)
        
    # Return the results
    return {
        'mean': mean,
        'sem': sem
    }
