

from utils.data_io import load_pickle, save_fig
from utils.plots import score_time

# Load the rrr-score-time data
mean = load_pickle("rrr-score-time")['mean']
sem  = load_pickle("rrr-score-time")['sem']

# Plot the result
fig = score_time(mean, sem)

# Save the figure
save_fig(fig, "rrr-score-time")