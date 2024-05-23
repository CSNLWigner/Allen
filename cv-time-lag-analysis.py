from analyses.data_preprocessing import get_area_responses, preprocess_area_responses
from utils.download_allen import cache_allen
from utils.data_io import load_pickle, save_pickle
import yaml
import sys

# Get the arguments
opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
log=False
if "-l" in opts:
    log = True

# Load parameters
load = yaml.safe_load(open('params.yaml'))['load']
preproc = yaml.safe_load(open('params.yaml'))['preprocess']
rrr_params = yaml.safe_load(open('params.yaml'))['rrr']
main_params = yaml.safe_load(open('params.yaml'))['rrr-param-search']

# Load the activity
full_activity_predictor = load_pickle(f'{load["stimulus-block"]}_block_{rrr_params["predictor"]}-activity', path='data/raw-area-responses')
full_activity_target    = load_pickle(f'{load["stimulus-block"]}_block_{rrr_params["target"]}-activity', path='data/raw-area-responses')
#print(full_activity_predictor.shape, full_activity_target.shape)

# Get the image names
image_names = load_pickle(f'{load["stimulus-block"]}_block_{rrr_params["target"]}-image-names', path='data/stimulus-presentations')
#print(image_names)

# Import utile functions
import numpy as np

# Import analysis functions
from analyses.rrr import RRRR

# Define the cross-validation, and time
cv = main_params['cv']
time_lag = main_params['lag']
rank = main_params['rank']
timepoints = main_params['timepoints'] # ms

#cv, rank, timepoints, time_lag = [4], [30], [50], [0]

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
        predictor = preprocess_area_responses(full_activity_predictor, image_names)
        target = preprocess_area_responses(full_activity_target, image_names)
        
        # Move the activity of V2 back in time by the actual time lag
        lagged_target = np.roll(target, -lag, axis=2)
                
        for i, c in enumerate(cv):
            for k, r in enumerate(rank):
                for t, time in zip(timepoint_indices, timepoints):
                    
                    # Reduced Rank Regression
                    # result = RRRR(V1.mean(axis=0), V2.mean(axis=0), params['rank'], cv=c) # cross-time RRRR
                    result = RRRR(predictor[:,:,t].T, lagged_target[:,:,t].T, rank=r, cv=c, log=True)
                    
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
#debug(result)

# Save the results
save_pickle(result, f'CV-lag-time')

# Get the maximum
max = np.nanmax(result).round(3)

# Get the indices of the maximum value
max_idx = np.unravel_index(np.nanargmax(result), result.shape)

# Print the maximum value and the corresponding parameters
print(f'maximum value({max}) is at time={timepoints[max_idx[3]]} s, cv={cv[max_idx[0]]} fold, lag={time_lag[max_idx[1]]} ms, and rank={rank[max_idx[2]]}')

# Append the maximum value and the corresponding parameters to a csv file
with open('best-rrr-params.csv', 'a') as f:
    f.write(f'{load["session"]},{timepoints[max_idx[3]]},{max},{cv[max_idx[0]]},{time_lag[max_idx[1]]},{rank[max_idx[2]]}\n')


