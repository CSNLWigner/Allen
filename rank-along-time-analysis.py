from analyses.rrr import RRRR
import numpy as np
from analyses.data_preprocessing import calculate_residual_activity, recalculate_neural_activity, z_score_normalize
from analyses.data_preprocessing import get_area_responses, preprocess_area_responses
from utils.download_allen import cache_allen
from utils.data_io import load_pickle, save_pickle
import yaml

# Load parameters
load = yaml.safe_load(open('params.yaml'))['load']
preproc = yaml.safe_load(open('params.yaml'))['preprocess']
rrr_params = yaml.safe_load(open('params.yaml'))['rrr']
search_params = yaml.safe_load(open('params.yaml'))['rrr-param-search']

# Load the activity
full_activity_predictor = load_pickle(
    f'{load["stimulus-block"]}_block_{rrr_params["predictor"]}-activity', path='data/raw-area-responses')
full_activity_target = load_pickle(
    f'{load["stimulus-block"]}_block_{rrr_params["target"]}-activity', path='data/raw-area-responses')

# Define the parameters
cv = rrr_params['cv']
time_lag = preproc['lag-time']
ranks = search_params['rank']

# Define the timepoints
timepoints = np.arange(0, preproc['stimulus-duration'], preproc['step-size']) # in seconds
timepoint_indices = [int(t / preproc['step-size'] / 1000) for t in timepoints]
time_length = int(preproc['stimulus-duration'] / preproc['step-size'])

# Preprocess the area responses
predictor = preprocess_area_responses(full_activity_predictor)
target = preprocess_area_responses(full_activity_target)

# Initialize the list for the results
max_ranks = []

for t, time in zip(timepoint_indices, timepoints):
    results = []
    for k, r in enumerate(ranks):

        # Reduced Rank Regression
        # print(f'Cross-validation: {c}, Time lag: {lag}')
        # result = RRRR(V1.mean(axis=0), V2.mean(axis=0), params['rank'], cv=c) # cross-time RRRR
        result = RRRR(predictor[:, :, t].T,
                      target[:, :, t].T, rank=r, cv=cv, success_log=False)
        
        results.append(result['test_score'].mean())

    # Get the index of the maximum value
    max_idx = np.nanargmax(np.array(results))

    # Save the result averaged over the folds
    max_ranks.append(ranks[max_idx])

# Save the results
save_pickle(max_ranks, 'max-ranks')