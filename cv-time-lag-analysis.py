from analyses.data_preprocessing import get_area_responses
from utils.download_allen import cache_allen
from utils.data_io import load_pickle, save_pickle
import yaml

# Load parameters
load = yaml.safe_load(open('params.yaml'))['load']
preproc = yaml.safe_load(open('params.yaml'))['preprocess']
rrr_params = yaml.safe_load(open('params.yaml'))['rrr']
main_params = yaml.safe_load(open('params.yaml'))['rrr-param-search']

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

# Define the cross-validation, and time
cv = main_params['cv']
time_lag = main_params['lag']
rank = main_params['rank']
timepoints = main_params['timepoints'] # ms
timepoint_indices = [int(t / preproc['step-size'] / 1000) for t in timepoints]

# Some calculations
def calculate_something():
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
    results = np.zeros((len(cv), len(time_lag), len(rank), time_length))

    # Loop through the cross-validation and rank
    for j, lag in enumerate(time_lag):
        
        # Preprocess the area responses
        V1 = preprocess_area_responses(full_VISp_activity)
        V2 = preprocess_area_responses(full_VISl_activity)
        
        # Move the activity of V2 back in time by the actual time lag
        V2 = np.roll(V2, -lag, axis=2)
                
        for i, c in enumerate(cv):
            for k, r in enumerate(rank):
                for t, time in zip(timepoint_indices, timepoints):
                    
                    # Reduced Rank Regression
                    # print(f'Cross-validation: {c}, Time lag: {lag}')
                    # result = RRRR(V1.mean(axis=0), V2.mean(axis=0), params['rank'], cv=c) # cross-time RRRR
                    result = RRRR(V1[:,:,t].T, V2[:,:,t].T, rank=r, cv=c)
                    
                    # Save the result averaged over the folds
                    results[i, j, k, t] = result['test_score'].mean()

    # Cut off the negative values
    results[results < -0] = np.nan
    
    # Get rid of the slices that contains only zeros
    results = results[:, :, :, np.all(results, axis=(0,1,2))]
    
    return results


    
# Print the time
print(f'Calculating something')

# Calculate the results
result = calculate_something()

print('result.shape:', result.shape)

# Save the results
save_pickle(result, f'CV-lag-time')

# Get the maximum
max = np.nanmax(result)

# Get the indices of the maximum value
max_idx = np.unravel_index(np.nanargmax(result), result.shape)
print(f'maximum value ({max.round(3)}) is at cv={cv[max_idx[0]]}, lag={time_lag[max_idx[1]]}, rank={rank[max_idx[2]]}, time={timepoints[max_idx[3]]}ms')


