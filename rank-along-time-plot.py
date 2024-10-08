# rank-along-time-plot.py

"""
This module plots the optimal rank along time.

**Parameters**:

- `preprocess`: Preprocess parameters.

**Input**:

- `results/max-ranks.pickle`: Optimal rank along time.

**Output**:

- `figures/max-ranks.png`: Plot of the optimal rank along time.

**Submodules**:

- `utils.data_io`: Data I/O.
"""
import numpy as np
import yaml
from matplotlib import pyplot as plt

from utils.data_io import load_pickle

# Load parameters
preproc = yaml.safe_load(open('params.yaml'))['preprocess']

# Define the timepoints
timepoints = np.arange(0, preproc['stimulus-duration'], preproc['step-size']) # in seconds

# Load max ranks
max_ranks = load_pickle('max-ranks')

print(max_ranks)

# Plot the max ranks
plt.plot(timepoints, max_ranks)
plt.xlabel('Time (s)')
plt.ylabel('Optimal rank')
plt.xticks(timepoints[::2], timepoints[::2])

# Save the figure
plt.savefig('figures/max-ranks.png')