import pickle
import numpy as np
import matplotlib.pyplot as plt
from analyses.rrr import RRRR
from utils.data_io import load_pickle, save_pickle
import yaml

# Load parameters
preprocess = yaml.safe_load(open('params.yaml'))['preprocess']
params = yaml.safe_load(open('params.yaml'))['rrr']

# Load V1 and V2 activity
V1_activity = load_pickle(f'{preprocess["stimulus-block"]}_block_VISp-activity', path='data/area-responses')
V2_activity = load_pickle(f'{preprocess["stimulus-block"]}_block_VISl-activity', path='data/area-responses')

# Set the maximum rank to iterate over
max_rank = params['rank']

# Get the number of neurons, trials, and time points
N, K_V1, T = V1_activity.shape

# Define the range of ranks to iterate over
ranks = range(1, max_rank+1)

# Initialize the errors
test_scores = np.zeros((max_rank, T))

# Iterate over time
for t in range(T):
    
    for rank in ranks:
    
        # Calculate rrr ridge using your rrrr function
        models = RRRR(V1_activity[:, :, t].T, V2_activity[:, :, t].T, rank=rank, cv=params['cv'])
        
        # Calculate the mean of the test scores above the cv-folds
        test_score = np.mean(models['test_score'])

        # Save the test score
        test_scores[rank-1, t] = test_score

# Save the errors
save_pickle(test_scores, f'VISp_VISl_cross-time-test-scores', path='results')