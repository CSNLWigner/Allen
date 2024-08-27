"""
This script plots the rank of the bottom-up and top-down connections between layers.

**Arguments**:

- `-rank`: Rank switch.
- `-r2`: R2 switch.

**Output**:

- `layer-rank`: Layer rank results.
"""

import sys

import numpy as np

from utils.data_io import load_pickle
from utils.megaplot import megaplot
from utils.plots import plot_3d_scatter_with_color
from utils.utils import get_args

# Get the arguments
opts, args = get_args(sys.argv)

# Log switch
rank_r2 = None
if "-rank" in opts:
    rank_r2 = 'rank'
elif "-r2" in opts:
    rank_r2 = 'r2'
else:
    raise ValueError("Please specify '-rank' or '-r2'.")

# ATTENTION! 'preprocess' parameters are used in the subfunctions!

# Load the results
result = load_pickle(f'layer-{rank_r2}') # Shape: (nAreas(2), nLayers(6+1), nLayers(6+1), nTimepoints)

# Initialize megaplot with 2 rows and 2 columns
mp = megaplot(nrows=1, ncols=2, constrained_layout=True)

# Plot 4D data in each subplot
for i, name in enumerate(['bottom-up', 'top-down']):
    ax = mp.ax(0, i, projection='3d')
    plot_3d_scatter_with_color(ax, result[i, :, :, :], title=name,
                               xlabel='Source Layer', ylabel='Target Layer', zlabel='Rank',
                               xticks=np.arange(1,7), yticks=np.arange(1,7))

# Save the megaplot
mp.save('layer-rank', path='figures/')