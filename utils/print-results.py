

# Load the params
import numpy as np
import yaml
from utils.data_io import load_pickle

params = yaml.safe_load(open('params.yaml'))['rrr-param-search']

# Define the cross-validation, and time
cv = params['cv']
lag = params['lag']
rank = np.array(params['rank'], dtype=int)
time = params['timepoints']

# Load the results
result = load_pickle('CV-lag-time')
print(result)
