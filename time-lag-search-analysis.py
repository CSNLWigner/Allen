import numpy as np
from analyses.rrr import RRRR
from analyses.data_preprocessing import preprocess_area_responses
from utils.data_io import load_pickle, save_pickle
import yaml

from utils.utils import shift_with_nans

# Load parameters
load = yaml.safe_load(open('params.yaml'))['load']
preproc = yaml.safe_load(open('params.yaml'))['preprocess']
rrr_params = {**yaml.safe_load(open('params.yaml'))['rrr'], **yaml.safe_load(open('params.yaml'))['best-rrr-params']}
search_params = yaml.safe_load(open('params.yaml'))['rrr-param-search']

# Load the activity
full_activity_predictor = load_pickle(
    f'{load["stimulus-block"]}_block_{rrr_params["predictor"]}-activity', path='data/raw-area-responses')
full_activity_target = load_pickle(
    f'{load["stimulus-block"]}_block_{rrr_params["target"]}-activity', path='data/raw-area-responses')

# Define the parameters
prediction_direction = 'top-down' if rrr_params['predictor'] == 'VISl' else 'bottom-up'
session = load['session']
cv   = rrr_params[session][prediction_direction]['cv']
rank = rrr_params[session][prediction_direction]['rank']
time = int(rrr_params['timepoint'] / preproc['step-size'])
lags = search_params['lag']

# Initialize the results array
results = []

# Max lag
max_lag = max(lags)

# Length of the activity after the end of the chosen timewindow (which is start at the timepoint of the rrr_params and has a duration of bin-size of the preproc params)
remaining_activity_length = full_activity_target.shape[2] - time - preproc['bin-size']*1000

# If the lag is larger than the activity length, cut off the last lag elements
if max_lag > remaining_activity_length:
    lags = [lag for lag in lags if lag <= remaining_activity_length]

# print(f'Remaining activity length: {remaining_activity_length}')
# print(f'Max lag: {max_lag}')
# print(f'Lags: {lags}')

# Loop over the lags
for cnt, lag in enumerate(lags):
    
    if cnt % 5 == 0:
        print(f'Processing lag {lag} ({cnt+1}/{len(lags)})')
    
    # Move the activity of V2 back in time by the actual time lag
    # Better to use shift_with_nans() function, if RRRR could handle NaNs: lagged_target = shift_with_nans(full_activity_target, -lag, axis=2)
    lagged_target = np.roll(full_activity_target, -lag, axis=2)
    
    # Preprocess the area responses
    predictor = preprocess_area_responses(full_activity_predictor)
    target    = preprocess_area_responses(lagged_target)

    # Reduced Rank Regression
    result = RRRR(predictor[:, :, time].T,
                     target[:, :, time].T, rank=rank, cv=cv, success_log=False)

    results.append(result['test_score'].mean())

# Get the index of the maximum value
max_idx = np.nanargmax(np.array(results))
lag = lags[max_idx]
print(f'Best lag: {lag}')

# Save the results
save_pickle(results, 'time-lag-search')

