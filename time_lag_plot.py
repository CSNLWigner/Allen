# time_lag_plot.py

"""
This module plots the time lag between V1 and LM.

**Parameters**:

- `preprocess`: Preprocess parameters.

**Input**:

- `results/VISp_VISl_cross-time-coeffs.pickle`: Cross time correlation coefficients.

**Output**:

- `figures/Time_lag_between_V1_LM.png`: Plot of the time lag between V1 and LM.

**Submodules**:

- `utils.data_io`: Module for loading and saving data.
- `utils.plots`: Module for plotting data.
"""

import yaml
from matplotlib import pyplot as plt

from utils.data_io import load_pickle, save_fig
from utils.plots import cross_time_correlation_coefficients_plot

# Load parameters
params = yaml.safe_load(open('params.yaml'))['preprocess']

# Load results from time_lag_analysis.py
coeffs = load_pickle('VISp_VISl_cross-time-coeffs', path='results')

# Plot the coefficients
fig = cross_time_correlation_coefficients_plot(coeffs, title='Cross time correlation', first_dim_label='V1 time (s)', second_dim_label='V2 time (s)')

# Show the plot
plt.show()

# Save the figure
save_fig(fig, f'Time_lag_between_V1_LM', path='figures')
