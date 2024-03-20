
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
predictor = load_pickle(f"5_block_{rrr['predictor']}-activity", path="data/area-responses") # shape (Neurons, Trials, Time)
target = load_pickle(f"5_block_{rrr['target']}-activity", path="data/area-responses") # shape (Neurons, Trials, Time)

# Get the shape of the data
N_predictor, K, T = predictor.shape

# Initialize the result
rrr_result = []
mean = []
sem  = []

# Move the activity of V2 back in time by the actual time lag
target = np.roll(target, -rrr['lag'], axis=2)

# Loop through the time steps
for time in range(T):
    
    # Get the time step
    predictor_t = predictor[:, :, time].T
    target_t = target[:, :, time].T
    
    # Perform RRR
    result = RRRR(predictor_t, target_t, rank=rrr['rank'], cv=rrr['cv'])['test_score']
    
    # Save the mean and standard error
    mean.append(result.mean())
    sem.append(standard_error(result))

# Save the result
save_pickle({'rrr-result': rrr_result,
             'mean': mean,
             'sem': sem},
             "rrr-score-time")
