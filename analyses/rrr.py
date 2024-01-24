

from sklearn.model_selection import cross_val_score
from utils.neuropixel import get_area_units, get_stimulus_presentations, get_unit_responses
from analyses.machine_learning_models import ReducedRankRidge
import yaml
import numpy as np

params = yaml.safe_load(open('params.yaml'))['rrr']

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
    area_X_units = get_area_units(session, area_X)  # shape (units)
    area_Y_units = get_area_units(session, area_Y)  # shape (units)
    print('area_X_units number', area_X_units.shape[0])  # (98)
    print('area_Y_units number', area_Y_units.shape[0])  # (92)

    # Get the responses
    if log:
        print('Get the responses')
    stimulus_presentations = get_stimulus_presentations(session)
    # print(stimulus_presentations[stimulus_presentations['stimulus_block'] == stimulus_block].head(1))
    trial_start = stimulus_presentations[stimulus_presentations['stimulus_block']
                                         == stimulus_block]['start_time'].values
    # Calculate the average difference between neighboring elements
    average_difference = np.mean(np.diff(trial_start))
    print("Average difference between neighboring elements in trial_start:", average_difference)
    
    area_X_responses = get_unit_responses(
        area_X_units, session.spike_times, trial_start, duration=0.25, binSize=params['bin-size'])  # shape (units, timestep)
    area_Y_responses = get_unit_responses(
        area_Y_units, session.spike_times, trial_start, duration=0.25, binSize=params['bin-size'])  # shape (units, timestep)
    print('area_X_responses.shape', area_X_responses.shape)  # (98, 30)
    print('area_Y_responses.shape', area_Y_responses.shape)  # (92, 30)


    # # Make RRR
    # Remember, to make it in each timestep!
    # if log:
    #     print('Make RRR')
    # # result = cca(area_X_responses.T, area_Y_responses.T)
    # model = ReducedRankRidge(rank=params['rank'])
    # # scores = cross_val_score(model, area_X_responses.T, area_Y_responses.T,
    # #                          cv=params['cv'], error_score='raise')
    # model.fit(area_X_responses.T, area_Y_responses.T)
    # results = model.predict(area_X_responses.T)
    # print('\nX_data')
    # print(area_X_responses.T[0])
    # print('\nresults')
    # print(results[0])

    # print('Scores', scores)

    # return {
    #     'scores': scores
    # }
