

from matplotlib import pyplot as plt
import numpy as np
from utils.data_io import load_pickle, save_fig
from utils.plots import cv_rank_time_plot
import yaml

# Load the params
preproc = yaml.safe_load(open('params.yaml'))['preprocess']
params = yaml.safe_load(open('params.yaml'))['rrr-param-search']

# Define the cross-validation, and time
cv = params['cv']
lag = params['lag']
rank = np.array(params['rank'], dtype=int)
time = params['timepoints']

# Get the rank idx of the value 8
# rank_idx = np.where(rank == 8)[0][0]

# Load the results
result = load_pickle('CV-lag-time')#[:,:,rank_idx, :] # Shape: (cv, lag, rank, time)
print(result.shape)

# Dimension names
dim_names = ['cv', 'lag', 'rank', 'time']

# Create fig, axs with 
fig, axs = plt.subplots(6, 1, figsize=(20, 20))

# Permuations of the dimensions (select 2 from 4) with built-in function
from itertools import combinations
perm = list(combinations(dim_names, 2))
for p in perm:
    
    # Select the dimensions
    x, y = p
    
    # Index of the rest dimensions in tuple
    rest = [d for d in dim_names if d not in [x, y]]
    rest_idx = [dim_names.index(r) for r in rest]
    
    # Plot the mean of the result above the rest dimension indices
    mean = np.nanmean(result, axis=tuple(rest_idx))
    
    im = cv_rank_time_plot(mean,
                           #title=f'Averaged over {p}', 
                           ax=axs[perm.index(p)],
                           xlabel=x, ylabel=y,
                           xticks=eval(x), yticks=eval(y))
    
    fig.colorbar(im, ax=axs[perm.index(p)])
    

# Save the figure
save_fig(fig, f'rrr-param-search')