from analyses.data_preprocessing import get_area_responses
from utils.download_allen import cache_allen
from utils.data_io import load_pickle, save_pickle
import yaml

# Load parameters
load = yaml.safe_load(open('params.yaml'))['load']
preproc = yaml.safe_load(open('params.yaml'))['preprocess']
rrr_params = yaml.safe_load(open('params.yaml'))['rrr']
main_params = yaml.safe_load(open('params.yaml'))['rrr-cv-rank-time']

# Load the activity
full_VISp_activity = load_pickle(f'{load["stimulus-block"]}_block_VISp-activity', path='data/raw-area-responses')
full_VISl_activity = load_pickle(f'{load["stimulus-block"]}_block_VISl-activity', path='data/raw-area-responses')

# Import preprocessing functions
from analyses.data_preprocessing import calculate_residual_activity, recalculate_neural_activity, z_score_normalize

# Preprocess the area responses
def preprocess_area_responses(raw_activity):
    '''
    Preprocess the area responses.
    '''
    
    # Recalculate time steps and time bins of the full activity
    full_activity = recalculate_neural_activity(raw_activity,
        preproc['stimulus-duration'], preproc['step-size'], preproc['bin-size'],
        orig_time_step=0.001)
    
    # Get residual activity
    residual_activity = calculate_residual_activity(full_activity) # Neuron-wise AND time-wise
    
    # Normalize the responses
    normalized_activity = z_score_normalize(residual_activity, dims=(0,1,2)) # TODO: Normalize based on ITI activity?
    
    return normalized_activity
    

# Import utile functions
import numpy as np

# Import analysis functions
from analyses.rrr import RRRR

# Some calculations
def calculate_something(cv, time_lag):
    '''
    Calculate the time lag between two time series.
    '''
    
    # Calculate the time length after the preprocessing by the time step and the stimulus duration
    time_length = int(preproc['stimulus-duration'] / preproc['step-size'])
    # Print the timelength calculation with the parameters
    # print(f'{preproc["stimulus-duration"]} / {preproc["step-size"]} = {time_length}')
    
    # Print the time length
    # print(f'Time length: {time_length}')

    # Create a results array
    results = np.zeros((len(cv), len(time_lag), time_length))

    # Loop through the cross-validation and rank
    for j, lag in enumerate(time_lag):
        
        # Preprocess the area responses
        V1 = preprocess_area_responses(full_VISp_activity)
        V2 = preprocess_area_responses(full_VISl_activity)
        
        # Move the activity of V2 back in time by the actual time lag
        V2 = np.roll(V2, -lag, axis=2)
                
        for i, c in enumerate(cv):
            for t, time in zip(timepoint_indices, timepoints):
                
                # Reduced Rank Regression
                # print(f'Cross-validation: {c}, Time lag: {lag}')
                # result = RRRR(V1.mean(axis=0), V2.mean(axis=0), params['rank'], cv=c) # cross-time RRRR
                result = RRRR(V1[:,:,t].T, V2[:,:,t].T, rrr_params['rank'], cv=c)
                
                # Save the result averaged over the folds
                results[i, j, t] = result['test_score'].mean()

    # Cut off the negative values
    results[results < -0] = np.nan
    
    # Get rid of the slices that contain only zeros
    results = results[:, :, np.all(results, axis=(0,1))]

    print(results)
    
    return results


# Define the cross-validation, and time
cv = main_params['cv']
time_lag = main_params['lag']
timepoints = main_params['timepoints'] # ms
# Get the correpsponding time indices of the activity matrix (shape neurons X trials X time) based on the preprocess params (time-step (in s), stimulus-duration (in s))
timepoint_indices = [int(t / preproc['step-size'] / 1000) for t in timepoints]
# Print timepoints and timepoint_indices
print(f'Timepoints: {timepoints}')
print(f'Timepoint indices: {timepoint_indices}')
    
# Print the time
print(f'Calculating something')

# Calculate the results
result = calculate_something(cv, time_lag)

print('result.shape:', result.shape)

# Save the results
save_pickle(result, f'CV-lag-time')

# Get the maximum
max = np.nanmax(result)

# Print the maximum
print(f'Maximum: {max}')

