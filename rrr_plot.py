# rrr_plot.py

"""
This module plots the RRR results.

**Parameters**:

- `rrr`: RRR parameters.
- `preprocess`: Preprocess parameters.

**Input**:

- `results/rrr_coefficients.pickle`: RRR coefficients.

**Output**:

- `figures/VISl-VISp_block-<stimulus-block>_rrr-coefficients_along_time.png`: Plot of the RRR coefficients along time.

**Submodules**:

- `utils.data_io`: Module for loading and saving data.
- `utils.plots`: Module for plotting data.

"""
import matplotlib.pyplot as plt
import yaml

from utils.data_io import load_pickle, save_fig
from utils.plots import simple_rrr_plot

params = yaml.safe_load(open('params.yaml'))['rrr']
preprocess = yaml.safe_load(open('params.yaml'))['preprocess']

# Load the results
results_path = "results"
name = "rrr_coefficients"
result = load_pickle(f"{results_path}/{name}")

# Print the shape of the results
print('result.shape', result.shape)

# Visualize the coefficients
fig = simple_rrr_plot(result, params)

# Save the plots
figname = f"VISl-VISp_block-{preprocess['stimulus-block']}_rrr-coefficients_along_time"
# fig.savefig(f"figures/{figname}.png")
save_fig(fig, figname, path="figures")

# Display the figure
plt.show()



