import pickle
import numpy as np
import matplotlib.pyplot as plt
from analyses.rrr import RRRR, rrr_rank_analysis
from utils.data_io import load_pickle, save_pickle
import yaml

# Load parameters
load = yaml.safe_load(open('params.yaml'))['load']
preprocess = yaml.safe_load(open('params.yaml'))['preprocess']
params = yaml.safe_load(open('params.yaml'))['rrr']

# Load V1 and V2 activity
V1_activity = load_pickle(f'{load["stimulus-block"]}_block_VISp-activity', path='data/area-responses')
V2_activity = load_pickle(f'{load["stimulus-block"]}_block_VISl-activity', path='data/area-responses')

# Set the maximum rank to iterate over
max_rank = 15

cvs = [2, 3, 4]

# Get the number of neurons, trials, and time points
N, K_V1, T = V1_activity.shape

# Init the results array
test_scores = np.full((max_rank, T, cvs[-1]+1), np.nan)

for cv in cvs:
    
    test_scores[:,:,cv] = rrr_rank_analysis(V1_activity, V2_activity, cv=cv, log=False)


# Save the errors
save_pickle(test_scores, f'VISp_VISl_cross-time-test-scores', path='results')