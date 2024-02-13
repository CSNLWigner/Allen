
import numpy as np
import yaml

from utils.data_io import load_pickle, save_pickle
from analyses.data_preprocessing import min_max_normalize
from analyses.rrr import RRRR

# Load parameters
params = yaml.safe_load(open('params.yaml'))['preprocess']

# Load V1 and V2 activity
V1_activity = load_pickle(f'{params["stimulus-block"]}_block_VISp-activity', path='data/area-responses')
V2_activity = load_pickle(f'{params["stimulus-block"]}_block_VISl-activity', path='data/area-responses')

# Average over trials
V1_activity = np.mean(V1_activity, axis=0)
V2_activity = np.mean(V2_activity, axis=0)

# Normalize activity
V1_activity = min_max_normalize(V1_activity, dims=(0, 1))
V2_activity = min_max_normalize(V2_activity, dims=(0, 1))

# Calculate RRR between V1 and V2_activity
coeffs = RRRR(V1_activity, V2_activity, log=True)['mean_coefficients']
index = np.unravel_index(np.argmax(coeffs, axis=None), coeffs.shape)
time_lag = np.abs(index[1] - index[0]) * params['bin-size']

# Print the time lag
print("Time lag:", time_lag*1000, "ms")

# Save the coeffs
save_pickle(coeffs, f'VISp_VISl_cross-time-coeffs', path='results')

# Save the time lag
save_pickle(time_lag, f'VISp_VISl_cross-time-lag', path='results')