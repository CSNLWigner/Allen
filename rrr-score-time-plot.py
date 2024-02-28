

from utils.data_io import load_pickle, save_fig
from utils.plots import score_time
import yaml

# Load the params
rrr = yaml.safe_load(open('params.yaml'))['rrr']

# Load the rrr-score-time data
mean = load_pickle("rrr-score-time")['mean']
sem  = load_pickle("rrr-score-time")['sem']

# Plot the result
fig = score_time(mean, sem, title=f'Predict {rrr["target"]} activity from {rrr["predictor"]} activity', ylabel='accuracy (R^2)')

# Save the figure
save_fig(fig, "rrr-score-time")