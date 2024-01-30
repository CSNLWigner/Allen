# Main for plot RRR results (saved in the results folder)
# DVC file. 

from utils.data_io import load_pickle, save_fig
import yaml
import matplotlib.pyplot as plt

from utils.plots import simple_rrr_plot_mean
from utils.data_io import load_pickle
from utils.plots import simple_rrr_plot_mean

params = yaml.safe_load(open('params.yaml'))['rrr']
preprocess = yaml.safe_load(open('params.yaml'))['preprocess']

# Load the results
results_path = "results"
name = "rrr_coefficients"
result = load_pickle(f"{results_path}/{name}")

# Print the shape of the results
print('result.shape', result.shape)

# Visualize the coefficients
fig = simple_rrr_plot_mean(result)

# Save the plots
figname = f"VISl-VISp_block-{preprocess['stimulus-block']}_rrr-coefficients"
# fig.savefig(f"figures/{figname}.png")
save_fig(fig, figname, path="figures")

# Display the figure
plt.show()



