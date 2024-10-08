# max-lags-along-time-plot.py

"""

This module plots the max lag along time.

**Parameters**:

- `preprocess`: Preprocess parameters.
- `rrr-param-search`: RRR parameter search.

**Input**:

- `results/max-lags-along-time.pickle`: Max lag along time.

**Output**:

- `figures/max-lags-along-time.png`: Plot of the max lag along time.

**Submodules**:

- `utils.data_io`: Data I/O.

"""

import numpy as np
import yaml
from matplotlib import pyplot as plt

from utils.data_io import load_pickle

# Load parameters
preproc = yaml.safe_load(open('params.yaml'))['preprocess']
search_params = yaml.safe_load(open('params.yaml'))['rrr-param-search']

# Load lags
lags = load_pickle('max-lags-along-time')

# Define the timepoints
timepoints = np.arange(0, len(lags)*preproc['step-size'], preproc['step-size']) # in seconds

# Plot the max ranks
plt.plot(timepoints, lags)
plt.xlabel('Time (s)')
plt.ylabel('Max lag')
plt.xticks(timepoints[::5], timepoints[::5])

# Save the figure
plt.savefig('figures/max-lags-along-time.png')
