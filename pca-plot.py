# pca-plot.py

"""
This module plots the PCA analysis results.

**Parameters**:

None

**Input**:

- `results/pca-analysis.pickle`: PCA analysis results.

**Output**:

- `figures/pca-analysis_explained_variance_ratio.png`: Plot of the explained variance ratio.
- `figures/pca-analysis_principal-components.png`: Plot of the principal components.

**Submodules**:

- `utils.data_io`: Module for loading and saving data.

"""

import matplotlib.pyplot as plt

from utils.data_io import load_pickle, save_fig

# Load the PCA results dict of dicts
pca_results = load_pickle('pca-analysis', path='results')

# areas = list(pca_results.keys())
# times = list(pca_results[areas[0]].keys())

# print(pca_results.keys())
# print(pca_results['VISp'].keys())

# Create a subplot for explained variance ratio iterating through brain areas and time
fig_EVR, axs_EVR = plt.subplots(len(pca_results), len(pca_results['VISp']), figsize=(10,4))

# Create a figure for the principal components iterating through brain areas and time
fig_PC, axs_PC = plt.subplots(len(pca_results), len(pca_results['VISp']), figsize=(10,4))

max_row = len(pca_results) -1

# Get the min and max values for 1st and 2nd components for all areas and times
min_PC1 = min([pca_results[area][time]['components'][:,0].min() for area in pca_results for time in pca_results[area]])
max_PC1 = max([pca_results[area][time]['components'][:,0].max() for area in pca_results for time in pca_results[area]])
min_PC2 = min([pca_results[area][time]['components'][:,1].min() for area in pca_results for time in pca_results[area]])
max_PC2 = max([pca_results[area][time]['components'][:,1].max() for area in pca_results for time in pca_results[area]])

# Loop through the areas and times
for i, area in enumerate(pca_results):
    for j, time in enumerate(pca_results[area]):
        
        # Plot the first 5 explained variance ratio
        axs_EVR[i,j].bar(range(5), pca_results[area][time]['explained_variance_ratio'][:5])
        
        # Plot the first two principal components
        axs_PC[i,j].scatter(pca_results[area][time]['components'][:,0], pca_results[area][time]['components'][:,1])
        
        # Set the xlim and ylim to [-2,5], [-2,5]
        axs_PC[i,j].set_xlim(min_PC1, max_PC1)
        axs_PC[i,j].set_ylim(min_PC2, max_PC2)

    # Create suptitle for fig_EVR
    fig_EVR.suptitle('Explained Variance Ratio')

    # Create suptitle for fig_PC
    fig_PC.suptitle('Principal Components')
    
    # Set the labels for EVR and PC
    axs_EVR[0, 0].set_ylabel(f'VISp')
    axs_EVR[1, 0].set_ylabel(f'VISl')
    axs_EVR[max_row, 0].set_xlabel('0 ms')
    axs_EVR[max_row, 1].set_xlabel('50 ms')
    axs_EVR[max_row, 2].set_xlabel('100 ms')
    axs_EVR[max_row, 3].set_xlabel('150 ms')
    axs_EVR[max_row, 4].set_xlabel('200 ms')
    axs_PC[0, 0].set_ylabel(f'VISp')
    axs_PC[1, 0].set_ylabel(f'VISl')
    axs_PC[max_row, 0].set_xlabel('0 ms')
    axs_PC[max_row, 1].set_xlabel('50 ms')
    axs_PC[max_row, 2].set_xlabel('100 ms')
    axs_PC[max_row, 3].set_xlabel('150 ms')
    axs_PC[max_row, 4].set_xlabel('200 ms')

# Save the EVR figure
save_fig(fig_EVR, 'pca-analysis_explained_variance_ratio', path='figures')

# Save the PC figure
save_fig(fig_PC, 'pca-analysis_principal-components', path='figures')
