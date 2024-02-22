from analyses.data_preprocessing import get_area_responses
from utils.download_allen import cache_allen
from utils.data_io import load_pickle, save_pickle
import yaml

# Load parameters
params = yaml.safe_load(open('params.yaml'))['preprocess']

# Load the activity
full_VISp_activity = load_pickle(f'{params["stimulus-block"]}_block_VISp-activity', path='data/raw-area-responses')
full_VISl_activity = load_pickle(f'{params["stimulus-block"]}_block_VISl-activity', path='data/raw-area-responses')

# Import preprocessing functions
from analyses.data_preprocessing import calculate_residual_activity, recalculate_neural_activity, z_score_normalize

# Preprocess the area responses
def preprocess_area_responses(raw_activity):
    '''
    Preprocess the area responses.
    '''
    
    # Recalculate time steps and time bins of the full activity
    full_activity = recalculate_neural_activity(raw_activity,
        params['stimulus-duration'], params['step-size'], params['bin-size'],
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

    # Create a results array
    results = np.zeros((len(cv), len(time_lag)))

    # Loop through the cross-validation and rank
    for i, c in enumerate(cv):
        for j, lag in enumerate(time_lag):
            
            # Preprocess the area responses
            V1 = preprocess_area_responses(full_VISp_activity)
            V2 = preprocess_area_responses(full_VISl_activity)
            
            # Move the activity of V2 back in time by the actual time lag
            V2 = np.roll(V2, -lag, axis=2)
            
            # Reduced Rank Regression
            result = RRRR(V1, V2, c, lag)
            
            # Save the result
            results[i, j] = result['test_score'].mean()

    # Cut off the negative values
    results[results < -0] = np.nan

    print(results)
    
    return results


# Define the cross-validation, and time
cv = [2, 3, 4]
time_lag = list(range(0, 10, 2))
    
# Print the time
print(f'Calculating something')

# Calculate the results
result = calculate_something(cv, time_lag)
max = np.nanmax(result)

# Print the maximum
print(f'Maximum: {max}')

# Save the results
save_pickle(result, f'CV-timelag')
