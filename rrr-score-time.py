
from matplotlib import pyplot as plt
import numpy as np
import yaml
from analyses.rrr import RRRR
from scipy.stats import sem as standard_error
from utils.data_io import load_pickle, save_pickle

# Load the params
preprocess = yaml.safe_load(open('params.yaml'))['preprocess']
rrr = yaml.safe_load(open('params.yaml'))['rrr']

# Load the data
X = load_pickle("5_block_VISp-activity", path="data/area-responses") # shape (Neurons, Trials, Time)
Y = load_pickle("5_block_VISl-activity", path="data/area-responses") # shape (Neurons, Trials, Time)

# Get the shape of the data
N_X, K, T = X.shape

# Initialize the result
rrr_result = []
mean = []
sem  = []

# Loop through the time steps
for time in range(T):
    
    # Get the time step
    X_t = X[:, :, time].T
    Y_t = Y[:, :, time].T
    
    # Perform RRR
    result = RRRR(X_t, Y_t, rank=rrr['rank'], cv=rrr['cv'])['test_score']
    
    # Save the mean and standard error
    mean.append(result.mean())
    sem.append(standard_error(result))

# Save the result
save_pickle({'rrr-result': rrr_result,
             'mean': mean,
             'sem': sem},
             "rrr-score-time")
