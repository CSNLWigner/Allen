

from matplotlib import pyplot as plt
from utils.data_io import load_pickle, save_fig
from utils.plots import cross_time_correlation_coefficients_plot
import yaml

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
