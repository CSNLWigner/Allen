
from matplotlib import pyplot as plt
import numpy as np
import yaml

from utils.data_io import load_pickle, save_fig, save_pickle
from utils.plots import rrr_rank_plot, rrr_rank_plot_over_time

# Load params
preprocess = yaml.safe_load(open('params.yaml'))['preprocess']
params = yaml.safe_load(open('params.yaml'))['rrr-plot']

# Load rrr rank analysis results
scores = load_pickle(f'VISp_VISl_cross-time-test-scores', path='results') # Shape (max_rank, T, cvs)

# 2D plot of the first 30 ranks
print('Plotting 2D plot')
fig = rrr_rank_plot(scores[:30, :, 3], title='RRR test scores (r2) 3-fold')
save_fig(fig, f'V1-V2_cross-time_RRR-rank-analysis-2DIM', path='figures')
plt.close(fig)

# Plot every time point
print('Plotting every time point')
fig, axs = plt.subplots(1, scores.shape[1], figsize=(15, 3))
for cv in [2,3,4]:
    rrr_rank_plot_over_time(scores[:,:,cv], axs=axs, label=f'{cv}-fold')
axs[0].legend()
save_fig(fig, f'V1-V2_cross-time_RRR-rank-analysis-timewise', path='figures')
plt.close(fig)

# Average over time
print('Averaging over time')
fig, ax = plt.subplots()
for cv in [2,3,4]:
    scores_over_time = np.nanmean(scores[:,:,cv], axis=1)
    plt.plot(scores_over_time, label=f'{cv}-fold')
plt.xlabel('Rank')
plt.ylabel('Test score (r2)')
plt.legend()
plt.title('RRR rank analysis averaged over time')
plt.savefig(f'figures/V1-V2_cross-time_RRR-rank-analysis-averaged-over-time.png')

# Optimal rank over time
optimal_rank = np.nanargmax(np.nanmean(scores, axis=1))
print(f'Optimal rank: {optimal_rank}')

# Get the indices of the optimal rank in the scores array
optimal_rank_indices = np.unravel_index(optimal_rank, scores.shape)

# Get the cv of the optimal rank
optimal_cv = optimal_rank_indices[2]
print(f'cv of the optimal rank: {optimal_cv}')
