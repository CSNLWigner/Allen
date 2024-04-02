# Import 
import numpy as np
import yaml

from analyses.data_preprocessing import preprocess_area_responses
from analyses.rrr import RRRR
from utils.data_io import load_pickle, save_pickle
from utils.utils import printProgressBar

# Load parameters
load = yaml.safe_load(open('params.yaml'))['load']
preproc = yaml.safe_load(open('params.yaml'))['preprocess']
rrr_params = {**yaml.safe_load(open('params.yaml'))
              ['rrr'], **yaml.safe_load(open('params.yaml'))['best-rrr-params']}
search_params = yaml.safe_load(open('params.yaml'))['rrr-param-search']

# Load the raw activity
full_activity_predictor = load_pickle(
    f'{load["stimulus-block"]}_block_{rrr_params["predictor"]}-activity', path='data/raw-area-responses')
full_activity_target = load_pickle(
    f'{load["stimulus-block"]}_block_{rrr_params["target"]}-activity', path='data/raw-area-responses')

# Define the parameters
prediction_direction = 'top-down' if rrr_params['predictor'] == 'VISl' else 'bottom-up'
session = load['session']
cv = rrr_params[session][prediction_direction]['cv']
rank = rrr_params[session][prediction_direction]['rank']
timepoints = np.arange(0, preproc['stimulus-duration'], preproc['step-size']) # in seconds
timepoint_indices = [int(timepoint / preproc['step-size']) for timepoint in timepoints]
time_lags = search_params['lag']

# Initialize the results array (timepoints x time_lags)
results = np.full((len(timepoints), len(time_lags)), fill_value=np.nan)

# Progress bar
printProgressBar(0, len(time_lags), prefix='Progress:', suffix='Complete', length=50)

for l, lag in enumerate(time_lags):
        
    # print(f'Processing lag {lag} ({l+1}/{len(time_lags)})')
    
    # Get the activity
    predictor = preprocess_area_responses(full_activity_predictor[:, :, : full_activity_predictor.shape[2] - lag], stimulus_duration=preproc['stimulus-duration']-lag/1000)
    target   = preprocess_area_responses(full_activity_target[:, :, lag : lag + full_activity_target.shape[2]], stimulus_duration=preproc['stimulus-duration']-lag/1000)
    
    for t, time in zip(timepoint_indices, timepoints):
                
        # If the timepoint is too close to the end of the stimulus duration, skip (to avoid index out of bounds error) (in ms)
        if time + lag/1000 + preproc['bin-size'] >= preproc['stimulus-duration']-lag/1000:
            continue
        
        # Calculate the RRRR
        result = RRRR(predictor[:,:,t].T, target[:,:,t].T, rank=rank, cv=cv, success_log=False)
        
        # Save the results
        results[t, l] = result['test_score'].mean()
        
    # Update the progress bar
    printProgressBar(l + 1, len(time_lags), prefix='Progress:', suffix='Complete', length=50)
    


# Save the results
save_pickle(results, 'lags-along-time')

# Cut off the slices with all NaNs
results = results[~np.isnan(results).all(axis=1)]

# Max lags along time
max_lag_idx = np.nanargmax(results, axis=1)
max_lag_time = [time_lags[i] for i in max_lag_idx]

# Save the max lags
save_pickle(max_lag_time, 'max-lags-along-time')