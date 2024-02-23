

from utils.data_io import load_pickle, save_fig
from utils.plots import cv_rank_time_plot

# Define the cross-validation, and time
cv = [2, 3, 4]
time_lag = list(range(0, 10, 2))

# Load the results
result = load_pickle('CV-timelag')

# Plot the results
fig = cv_rank_time_plot(result, cv, time_lag,
                        title=f'CV-timelag')

# Save the figure
save_fig(fig, f'CV-timelag')