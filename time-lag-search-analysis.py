import numpy as np
from analyses.rrr import RRRR
from analyses.data_preprocessing import preprocess_area_responses
from utils.data_io import load_pickle, save_pickle
import yaml

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
cv = rrr_params[session][prediction_direction]['cv']
rank = rrr_params[session][prediction_direction]['rank']
lags = search_params['lag']

# Preprocess the area responses
predictor = preprocess_area_responses(full_activity_predictor)
target = preprocess_area_responses(full_activity_target)

# Initialize a 2 row array for the results
results = np.zeros((2, len(lags)))

for t, time in enumerate(lags):

    # Reduced Rank Regression
    result = RRRR(predictor[:, :, t].T,
                    target[:, :, t].T, rank=rank, cv=cv, success_log=False)

    results[0, t] = result['test_score'].mean()
    results[1, t] = time

# Get the index of the maximum value
max_idx = np.nanargmax(np.array(results[0, :]))
lag = lags[max_idx]
print(f'Best lag: {lag}')

# Print the results rounded to 3 decimal places
print(results.round(3))

# Save the results
save_pickle(results[0,:], 'time-lag-search')

