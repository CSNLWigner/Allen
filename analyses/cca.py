# Here will be the functions to be used to CCA analysis
# They are called by the cca_analysis.py, which is the main program of this analysis

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import cross_val_score
from utils.neuropixel import get_area_units, get_stimulus_presentations, get_unit_responses

import yaml
params = yaml.safe_load(open('params.yaml'))['cca']


def cca(X_train, Y_train, X_test, Y_test):
    """
    Perform Canonical Correlation Analysis (CCA) on two sets of variables.

    Parameters:
    X_train (numpy.ndarray): First set of variables for training, shape (n_samples, n_features1).
    Y_train (numpy.ndarray): Second set of variables for training, shape (n_samples, n_features2).
    X_test (numpy.ndarray): First set of variables for testing, shape (n_samples, n_features1).
    Y_test (numpy.ndarray): Second set of variables for testing, shape (n_samples, n_features2).

    Returns:
    dict: A dictionary containing the following items:
        - 'model' (sklearn.cross_decomposition.CCA): CCA object fitted on the training data.
        - 'X_train_r' (numpy.ndarray): Transformed X_train data using CCA.
        - 'Y_train_r' (numpy.ndarray): Transformed Y_train data using CCA.
        - 'X_test_r' (numpy.ndarray): Transformed X_test data using CCA.
        - 'Y_test_r' (numpy.ndarray): Transformed Y_test data using CCA.
    """
    cca = CCA()
    cca.fit(X_train, Y_train)
    
    X_train_r, Y_train_r = cca.transform(X_train, Y_train)
    X_test_r, Y_test_r = cca.transform(X_test, Y_test)

    return {
        'model': cca,
        'X_train_r': X_train_r,
        'Y_train_r': Y_train_r,
        'X_test_r': X_test_r,
        'Y_test_r': Y_test_r
    }

"""
We can then consider two layers, L1 and L2 of a neural network as two sets of observations, to which we can then apply CCA to determine the similarity between the two layers.
"""


def compare_two_areas(session, area_X, area_Y, log=True) -> dict:
    """
    Compare the responses of units in the VISp and VISpm brain areas using Canonical Correlation Analysis (CCA).

    Args:
        session: The session object containing the spike times and stimulus presentations.
        log (bool, optional): Whether to log the progress. Defaults to True.

    Returns:
        dict: A dictionary containing the results of the CCA analysis.
    """

    stimulus_block = 2
    """
    stimulus block:
    0: change detection task
    2: receptive field mapping by gabor stimuli
    4: full-flash
    5: passive replay
    """

    # Get area units
    if log:
        print('Get area units')
    area_X_units = get_area_units(session, area_X) # shape (units)
    area_Y_units = get_area_units(session, area_Y) # shape (units)
    print('area_X_units number', area_X_units.shape[0])  # (98)
    print('area_Y_units number', area_Y_units.shape[0])  # (111)

    # Get the responses
    if log:
        print('Get the responses')
    stimulus_presentations = get_stimulus_presentations(session)
    # print(stimulus_presentations[stimulus_presentations['stimulus_block'] == stimulus_block].head(1))
    trial_start = stimulus_presentations[stimulus_presentations['stimulus_block']
                                         == stimulus_block]['start_time'].values
    print('trial_start', trial_start)
    area_X_responses = get_unit_responses(
        area_X_units, session.spike_times, trial_start) # shape (units, timestep)
    area_Y_responses = get_unit_responses(
        area_Y_units, session.spike_times, trial_start) # shape (units, timestep)
    print('area_X_responses.shape', area_X_responses.shape)  # (98, 30)
    print('area_Y_responses.shape', area_Y_responses.shape)  # (111, 30)

    # Make CCA
    if log:
        print('Make CCA')
    # result = cca(area_X_responses.T, area_Y_responses.T)
    model = CCA(n_components=params['n_components'])
    scores = cross_val_score(model, area_X_responses.T, area_Y_responses.T, cv=params['cv'], scoring=params['scoring'])
    
    print('Scores', scores)
    
    return {
        'scores': scores
    }
