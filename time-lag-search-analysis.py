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
# search_params = yaml.safe_load(open('params.yaml'))['rrr-param-search']

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
# lags = search_params['lag']
lags = np.arange(0, 50, 1)

# Initialize a 2 row array for the results
results = []

for cnt, lag in enumerate(lags):
    
    # Move the activity of V2 back in time by the actual time lag
    # Better to use shift_with_nans() function, if RRRR could handle NaNs
    # lagged_target = np.roll(full_activity_target, -lag, axis=2)
    lagged_target = shift_with_nans(full_activity_target, -lag, axis=2)
    
    # Cut off the last lag elements, which contains nan values
    lagged_target = lagged_target[:, :, :-lag]
    
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

