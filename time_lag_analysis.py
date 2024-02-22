
import numpy as np
import yaml

from utils.data_io import load_pickle, save_pickle
from analyses.data_preprocessing import z_score_normalize
from analyses.rrr import RRRR, calculate_cross_time_correlation

# Load parameters
load = yaml.safe_load(open('params.yaml'))['load']
params = yaml.safe_load(open('params.yaml'))['preprocess']

# Load V1 and V2 activity
V1_activity = load_pickle(f'{load["stimulus-block"]}_block_VISp-activity', path='data/area-responses')
V2_activity = load_pickle(f'{load["stimulus-block"]}_block_VISl-activity', path='data/area-responses')

# Check whether the activity contains NaNs
assert not np.isnan(V1_activity).any(), "V1 activity contains NaNs"
assert not np.isnan(V2_activity).any(), "V2 activity contains NaNs"

# Calculate RRR between V1 and V2_activity (averaged over the neurons)
result = RRRR(V1_activity.mean(axis=0), V2_activity.mean(axis=0), log=True)['mean_coefficients'] # You can mean without normalizing, because the mean is the same bcs of the normalization
# result = calculate_cross_time_correlation_coefficients(V1_activity, V2_activity, log=False)
index = np.unravel_index(np.argmax(result, axis=None), result.shape)
time_lag = np.abs(index[1] - index[0]) * params['bin-size']

# Print the time lag
print("Time lag:", time_lag*1000, "ms")

# Save the coeffs
save_pickle(result, f'VISp_VISl_cross-time-coeffs', path='results')

# Save the time lag
save_pickle(time_lag, f'VISp_VISl_cross-time-lag', path='results')