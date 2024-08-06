import numpy as np
from matplotlib import pyplot as plt

from utils.data_io import load_pickle
from utils.megaplot import megaplot
from utils.plots import plot_3d_scatter_with_color

# ATTENTION! 'preprocess' parameters are used in the subfunctions!

# Load the results
result = load_pickle('layer-rank') # Shape: (nAreas(2), nLayers(6+1), nLayers(6+1), nTimepoints)

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