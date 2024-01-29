

from sklearn.model_selection import cross_val_score
from utils.neuropixel import get_area_units, get_stimulus_presentations, get_unit_responses
from analyses.machine_learning_models import ReducedRankRidgeRegression
import yaml
import numpy as np
from allensdk.brain_observatory.ecephys.behavior_ecephys_session import BehaviorEcephysSession

from utils.utils import MSE

preprocess = yaml.safe_load(open('params.yaml'))['preprocess']
params = yaml.safe_load(open('params.yaml'))['rrr']

def RRRR(X_data, Y_data, log=False):
    """
    Make Reduced Rank Regression (RRR) analysis.

    Args:
        X_data (np.ndarray): The data of the first brain area.
        Y_data (np.ndarray): The data of the second brain area.
        log (bool, optional): Whether to log the progress. Defaults to True.

    Returns:
        np.ndarray: The coefficients of the RRR model. Shape (Y_features, X_features)
    """

    model = ReducedRankRidgeRegression(rank=params['rank'])
    # scores = cross_val_score(model, area_X_responses.T, area_Y_responses.T,
    #                          cv=params['cv'], error_score='raise')
    model.fit(X_data, Y_data)
    prediction = model.predict(X_data)
    accuracy = MSE(Y_data, prediction)
    score = model.score(X_data, Y_data)
    if log:
        print('Train accuracy', accuracy)
    coefficients = model.coef_
    if log:
        print('coefficients.shape', coefficients.shape)
    
    return coefficients

def compare_two_areas(area_X_responses:np.ndarray, area_Y_responses:np.ndarray, log=False) -> dict:
    """
    Compare the responses of units in two brain areas using Canonical Correlation Analysis (CCA).

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
