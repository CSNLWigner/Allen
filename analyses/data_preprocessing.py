from allensdk.brain_observatory.ecephys.behavior_ecephys_session import BehaviorEcephysSession
import numpy as np
import pandas as pd
import yaml

from utils.neuropixel import get_area_units, get_unit_responses, stimulus_duration
from utils.utils import printProgressBar
from scipy.signal import convolve

# Load the parameters
load = yaml.safe_load(open('params.yaml'))['load']
params = yaml.safe_load(open('params.yaml'))['preprocess']

def get_behav_responses(behav_data:pd.DataFrame, value_name:str, trial_start, duration=0.250, stepSize=0.010, binSize=0.050) -> np.ndarray:
    """
    Calculate the unit responses for each trial in the given timestamps.

    Parameters:
    behav_data (pd.DataFrame): DataFrame containing the timestamps and values.
    value_name (str): Name of the column in behav_data that contains the values.
    trial_start (list or array-like): List or array-like object containing the start time of each trial.
    duration (float): Total duration of trial for PSTH in seconds. Default is 0.250.
    binSize (float): Bin size for PSTH in seconds. Default is 0.050.

    Returns:
    numpy.ndarray: Array containing the unit responses, shape (n_trial, n_bin).
    """
    
    # Get timestamps, values from behav_data
    timestamps = behav_data['timestamps']
    values = behav_data[value_name]

    # Get the number of trials and bins
    n_trial = len(trial_start)
    n_step = int(duration / stepSize)
    n_bin = int(duration / binSize)

    # Create an empty array to store the data
    data = np.zeros((n_trial, n_step))

    # Calculate the unit responses for each trial
    for j, start in enumerate(trial_start):  # Trials

        # Calculate the average value for each bin
        for k, time in enumerate(np.arange(start, start + duration, stepSize)):  # Time
            
            # If due to floating point rounding, the last bin is not complete, break the loop
            if k == n_bin:
                break

            # Get the indices of the timestamps in the time bin
            bin_start_idx = np.searchsorted(timestamps, time)
            bin_end_idx = np.searchsorted(timestamps, time + binSize)

            # timestamps_in_timebin = timestamps[bin_start_idx:bin_end_idx]
            
            # Calculate the average value for the bin
            bin_value = np.mean(values[bin_start_idx:bin_end_idx])
            
            # Store the value in the data array
            data[j, k] = bin_value

    return data

def get_area_responses(session: BehaviorEcephysSession, area: str, trials:pd.DataFrame, log=False) -> np.ndarray:
    """
    Get the responses of the units in a specific brain area to a specific stimulus block.

    Args:
        session (BehaviorEcephysSession): The session object containing the spike times and stimulus presentations.
        area (str): The name of the brain area.
        trials (pd.DataFrame): DataFrame containing the trial information.

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
    stepSize = load['step-size']
    binSize = load['step-size']
    duration = load['stimulus-duration']

    # Get area units
    if log:
        print('Get area units...')
    area_units = get_area_units(session, area)  # shape (units)
    n_area_units = area_units.shape[0]
    if log:
        print('area_units number', n_area_units)  # shape (units)

    # Get the time of the start of each trial
    trial_start = trials['start_time'].values
    
    # Average difference between trial start times and the duration of the stimulus
    if log:
        average_difference = np.mean(np.diff(trial_start))
        print("Average difference between neighboring elements in trial_start:",
              average_difference)

    area_responses = get_unit_responses(
        area_units, session.spike_times, trial_start, duration=duration, stepSize=stepSize, binSize=binSize)  # shape (units, trials, time)
    if log:
        print('area_responses.shape', area_responses.shape)  # shape (units, trials, time)

    return area_responses



def transform_behav_data(behav_data: pd.DataFrame, behav_data_type: str, stimulus_presentations:pd.DataFrame, session_block: int, log=False) -> np.ndarray:

    # Parameters
    stimulus_block = session_block
    stepSize = params['step-size']
    binSize = params['bin-size']
    duration = params['stimulus-duration']
    time_length = duration/binSize

    # Get the time of the start of each trial
    trial_start = stimulus_presentations[stimulus_presentations['stimulus_block']
                                         == stimulus_block]['start_time'].values
    
    transformed_data = get_behav_responses(
        behav_data, behav_data_type, trial_start, duration=duration, stepSize=stepSize, binSize=binSize)  # shape (trials, time)
    if log:
        print('behav_data.shape', transformed_data.shape)  # shape (units, trials, time)

    return transformed_data



def stimulus_log(session: BehaviorEcephysSession, trial_start, stimulus_block: int):
        
    average_difference = np.mean(np.diff(trial_start))
    print("Average difference between neighboring elements in trial_start:",
            average_difference)
    
    stimulus_duration(session, stimulus_block)
    
    
    
def calculate_residual_activity(full_activity: np.ndarray, axis=None) -> np.ndarray:
    """
    Calculate the residual activity by subtracting the baseline (PSTH) from the full activity.

    Args:
        full_activity (np.ndarray): The full activity of the units. Shape (units, trials, time)
        axis (int, tuple, None): The axis along which to calculate the mean. Default is None.

    Returns:
        Numpy array containing the residual activity of the units. Shape (units, trials, time)

    Raises:
        None

    Example:
        >>> full_activity = np.array([[[0.1, 0.2, 0.3],
                                       [0.4, 0.5, 0.6]],
                                      [[0.7, 0.8, 0.9],
                                       [1.0, 1.1, 1.2]]])
        >>> print(calculate_residual_activity(full_activity, axis=0))
            [[[-0.25 -0.15 -0.05]
            [ 0.05  0.15  0.25]]

            [[-0.25 -0.15 -0.05]
            [ 0.05  0.15  0.25]]]
    """
    
    # Get the inverse of the axis
    axis = tuple(np.delete(np.arange(full_activity.ndim), axis))

    # Get trial average activity
    baseline = full_activity.mean(axis=axis, keepdims=True)

    # Subtract baseline (PSTH) from responses to get the residual activity
    residual_activity = full_activity - baseline

    return residual_activity

def z_score_normalize(activity: np.ndarray, dims=(2)) -> np.ndarray:
    """
    Perform Z-score normalization on the activity.

    Args:
        activity (np.ndarray): The activity to be normalized. Shape (units, trials, time)
        dims (tuple): Dimensions along which to compute the mean and standard deviation for Z-score normalization. Default is (2).

    Returns:
        np.ndarray: Numpy array containing the normalized activity. Shape (units, trials, time)
    """
    mean = np.mean(activity, axis=dims, keepdims=True)
    std = np.std(activity, axis=dims, keepdims=True)
    normalized_activity = (activity - mean) / std

    return normalized_activity


def min_max_normalize(activity: np.ndarray, dims=(0, 1)) -> np.ndarray:
    """
    Perform min-max normalization on the activity.

    Args:
        activity (np.ndarray): The activity to be normalized. Shape (units, trials, time)
        dims (tuple): Dimensions along which to compute the minimum and maximum values for min-max normalization. Default is (0,1).

    Returns:
        np.ndarray: Numpy array containing the normalized activity. Shape (units, trials, time)
    """
    nominator = activity - np.min(activity, axis=dims, keepdims=True)
    denominator = np.max(activity, axis=dims, keepdims=True) - np.min(activity, axis=dims, keepdims=True)
    activity = nominator / denominator

    return activity


def convolve_spike_train(spike_times: np.ndarray, step_size=0.010, kernel='Gaussian', kernel_size=50) -> np.ndarray:
    """
    Convolve the spike train with a boxcar or Gaussian kernel to get a continuous signal.

    Args:
        spike_times (np.ndarray): The spike times. Shape (n_spikes)
        step_size (float, optional): The step size for the time vector. Default is 0.010.
        kernel (str, optional): The type of kernel to use for convolution. Can be 'boxcar' or 'Gaussian'. Default is 'Gaussian'.

    Returns:
        np.ndarray: Numpy array containing the continuous signal. Shape (time)

    Raises:
        None

    Example:
        >>> spike_times = np.array([0.1, 0.2, 0.3])
        >>> convolve_spike_train(spike_times)
        array([0., 0.13533528, 0.60653066, 0.13533528, 0.])
    """

    # Create a time vector
    time_vector = np.arange(0, np.max(spike_times), step_size)

    # Create an empty spike train
    spike_train = np.zeros_like(time_vector)

    # Fill in the spike train
    for spike in spike_times:
        spike_train[int(spike / step_size)] = 1

    # Convolve with the kernel
    kernel_types = {
        'boxcar': np.ones(kernel_size),
        'Gaussian': np.exp(-np.linspace(-2, 2, kernel_size)**2)
        }
    continuous_signal = convolve(spike_train, kernel_types[kernel], mode='same')
    
    return continuous_signal

def recalculate_neural_activity(neural_activity: np.ndarray, duration: float, time_step: float, time_bin: float, orig_time_step=load['step-size']) -> np.ndarray:
    """
    Recalculate the neural activity based on the given time step and time bin.

    Args:
        neural_activity (np.ndarray): The neural activity. Shape (neurons, trials, time).
        time_step (float): The new time step.
        time_bin (float): The new time bin.
        orig_time_step (float, optional): The original time step. Default is 0.001.

    Returns:
        np.ndarray: Numpy array containing the recalculated neural activity. Shape (neurons, trials, new_time)

    Raises:
        None

    Example:
        >>> neural_activity = np.array([[[0.1, 0.2, 0.3, 0.6],
                                         [0.4, 0.5, 0.6, 0.6]],
                                        [[0.7, 0.8, 0.9, 0.6],
                                         [1.0, 1.1, 1.2, 0.6]]])
        >>> time_step = 0.001
        >>> time_bin = 0.002
        >>> recalculate_neural_activity(neural_activity, time_step, time_bin)
        array([[[0.3, 0.5, 0.9, 0.6],
                [0.9, 1.1, 1.2, 0.6]],
               [[1.5, 1.7, 1.5, 0.6],
                [2.1, 2.3, 1.8, 0.6]]])
    """
    
    N, K, T_orig = neural_activity.shape
    
    # Calculate the new time length
    T_new = int(duration / time_step)
    
    # Create an empty array to store the recalculated neural activity
    recalculated_activity = np.zeros((N, K, T_new))
    
    # TODO: Progress bar
    # printProgressBar(0, T_new, prefix='Recalculating neural activity:', length=30)
    
    for t in range(T_new):

        # Calculate the start and end indices for the original time step
        start_idx = int(t * time_step / orig_time_step)
        end_idx = int((t * time_step + time_bin) / orig_time_step)
        
        # Recalculate the neural activity
        recalculated_activity[:, :, t] = np.sum(neural_activity[:, :, start_idx:end_idx], axis=2)
        
        # TODO: Update the progress bar
        # printProgressBar(t + 1, T_new, prefix='Recalculating neural activity:', length=30)
                
    return recalculated_activity


def residual_for_each_stimulus_identity(full_activity: np.ndarray, stimulus_identities: pd.core.series.Series) -> np.ndarray:
    """
    Calculate the residual activity for each stimulus identity.

    Args:
        full_activity (np.ndarray): The full activity of the units. Shape (units, trials, time)
        stimulus_presentations (pd.core.series.Series): The stimulus presentations.

    Returns:
        np.ndarray: Numpy array containing the residual activity for each stimulus identity. Shape (units, n_identities, time)

    Raises:
        None

    Example:
        >>> full_activity = np.array([[[0.1, 0.2, 0.3],
                                       [0.4, 0.5, 0.6]],
                                      [[0.7, 0.8, 0.9],
                                       [1.0, 1.1, 1.2]]])
        >>> stimulus_identities = np.array([0, 1])
        >>> residual_for_each_stimulus_identity(full_activity, stimulus_identities)
        array([[[0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]],
               [[0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2]]])
    """
    
    # Get the unique stimulus identities
    unique_identities = stimulus_identities.unique()

    # Create an empty array to store the residual activity for each stimulus identity
    residual_activities = []
    
    print('idx of first stimulus', np.where(stimulus_identities == unique_identities[0])[0])
    print('stimulus_identities shape', stimulus_identities.shape)
    print('idx shape of first stimulus', np.where(stimulus_identities == unique_identities[0])[0].shape)
    print('full activity', full_activity.shape)
    print('unique identities', unique_identities)

    # Calculate the residual activity for each stimulus identity
    for identity in unique_identities:
        print('identity', identity)
        idx = np.where(stimulus_identities == identity)[0]
        residual_activity = calculate_residual_activity(full_activity[:, idx, :], axis=1)
        residual_activities.append(residual_activity)
    
    residual_activity = np.concatenate(residual_activities, axis=1)

    return residual_activity


def preprocess_area_responses(raw_activity, method='z-score', stimulus_duration=params['stimulus-duration'], step_size=params['step-size'], bin_size=params['bin-size']):
    """
    Preprocesses the raw neural activity of a specific brain area.

    Args:
        raw_activity (ndarray): The raw neural activity.
        method (str, optional): The method to use for normalization. Default is 'z-score'.

    Returns:
        ndarray: The normalized and preprocessed neural activity.
    """
    
    # Recalculate time steps and time bins of the full activity
    full_activity = recalculate_neural_activity(raw_activity,
                                                stimulus_duration, step_size, bin_size,
                                                orig_time_step=load['step-size'])

    # Get residual activity To get rid of high-frequency noise neurons, times and stimuli
    full_activity = calculate_residual_activity(full_activity, axis=(0, 2))  # Neuron-wise AND time-wise

    # Normalize the responses to better model performance
    if method == 'z-score':
        normalized_activity = z_score_normalize(full_activity, dims=(0, 1, 2))  # TODO: Normalize based on ITI activity?

    return normalized_activity
