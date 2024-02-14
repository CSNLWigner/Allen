
from matplotlib import pyplot as plt
import numpy as np
import yaml

from utils.data_io import load_pickle, save_fig, save_pickle
from utils.plots import rrr_rank_plot, rrr_rank_plot_over_time

# Load params
preprocess = yaml.safe_load(open('params.yaml'))['preprocess']
params = yaml.safe_load(open('params.yaml'))['rrr-plot']

# Load rrr rank analysis results
scores = load_pickle(f'VISp_VISl_cross-time-test-scores', path='results') # Shape (max_rank, T)

# Set first time point to nan
if params['discard-first-500-ms']:
    scores[:, 0] = np.nan

# Plot the results
fig = rrr_rank_plot = rrr_rank_plot(scores)
save_fig(fig, f'V1-V2_cross-time_RRR-rank-analysis-2DIM', path='figures')
plt.close(fig)
fig = rrr_rank_plot_over_time(scores)
save_fig(fig, f'V1-V2_cross-time_RRR-rank-analysis-timewise', path='figures')
plt.close(fig)

scores = np.nanmean(scores, axis=1)
plt.plot(scores)
plt.xlabel('Rank')
plt.ylabel('Test score (r2)')
plt.title('RRR rank analysis averaged over time')
plt.savefig(f'figures/V1-V2_cross-time_RRR-rank-analysis-averaged-over-time.png')

# Print at which rank is the test score the lowest
optimal_rank = np.argmax(scores)+1
print(f'Optimal rank: {optimal_rank}')

# Save the optimal rank to the results
save_pickle(optimal_rank, f'VISp_VISl_cross-time-optimal-rank', path='results')