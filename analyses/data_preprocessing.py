from allensdk.brain_observatory.ecephys.behavior_ecephys_session import BehaviorEcephysSession
import numpy as np
import yaml

from utils.neuropixel import get_area_units, get_stimulus_presentations, get_unit_responses, stimulus_duration

params = yaml.safe_load(open('params.yaml'))['preprocess']

def get_area_responses(session: BehaviorEcephysSession, area: str, session_block: int, log=False) -> np.ndarray:
    """
    Compare the responses of units in a brain area using Canonical Correlation Analysis (CCA).

    Args:
        session (BehaviorEcephysSession): The session object containing the spike times and stimulus presentations.
        area (str): The name of the brain area.
        session_block (int): The stimulus block number.
            0: change detection task
            2: receptive field mapping by gabor stimuli
            4: full-flash
            5: passive replay
        log (bool, optional): Whether to log the progress. Defaults to False.

    Returns:
        Numpy array containing the area responses of the units. Shape (units, trials, time)

    Raises:
        None

    Example:
        >>> session = BehaviorEcephysSession()
        >>> area = "V1"
        >>> session_block = 0
        >>> log = True
        >>> get_area_responses(session, area, session_block, log)
        array([[[0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]],
               [[0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2]]])

    """

    # Parameters
    stimulus_block = session_block
    binSize = params['bin-size']
    duration = params['stimulus-duration']
    time_length = duration/binSize

    # Get area units
    if log:
        print('Get area units...')
    area_units = get_area_units(session, area)  # shape (units)
    n_area_units = area_units.shape[0]
    if log:
        print('area_units number', n_area_units)  # shape (units)

    # Get the responses
    if log:
        print('Get responses...')
    stimulus_presentations = session.stimulus_presentations
    trial_start = stimulus_presentations[stimulus_presentations['stimulus_block']
                                         == stimulus_block]['start_time'].values
    
    # Average difference between trial start times and the duration of the stimulus
    if log:
        
        average_difference = np.mean(np.diff(trial_start))
        print("Average difference between neighboring elements in trial_start:",
              average_difference)
        
        stimulus_duration(session, stimulus_block)

    area_responses = get_unit_responses(
        area_units, session.spike_times, trial_start, duration=duration, binSize=binSize)  # shape (units, trials, time)
    if log:
        print('area_responses.shape', area_responses.shape)  # shape (units, trials, time)

    return area_responses



def stimulus_log(session: BehaviorEcephysSession, trial_start, stimulus_block: int):
        
    average_difference = np.mean(np.diff(trial_start))
    print("Average difference between neighboring elements in trial_start:",
            average_difference)
    
    stimulus_duration(session, stimulus_block)
    
    
    
def calculate_residual_activity(full_activity: np.ndarray) -> np.ndarray:
    """
    Calculate the residual activity by subtracting the baseline (PSTH) from the full activity.

    Args:
        full_activity (np.ndarray): The full activity of the units. Shape (units, trials, time)

    Returns:
        Numpy array containing the residual activity of the units. Shape (units, trials, time)

    Raises:
        None

    Example:
        >>> full_activity = np.array([[[0.1, 0.2, 0.3],
                                       [0.4, 0.5, 0.6]],
                                      [[0.7, 0.8, 0.9],
                                       [1.0, 1.1, 1.2]]])
        >>> calculate_residual_activity(full_activity)
        array([[[0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]],
               [[0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2]]])
    """

    # Get trial average activity
    baseline = full_activity.mean(axis=1)  # shape (units, time)

    # Make sure the baseline is the same shape as the full activity by repeating it
    baseline = np.repeat(baseline[:, np.newaxis, :], full_activity.shape[1], axis=1)

    # Subtract baseline (PSTH) from responses to get the residual activity
    residual_activity = full_activity - baseline

    return residual_activity