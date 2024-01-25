# Main for plot CCA results (saved in the results folder)
# DVC file. 

from utils import plots
import numpy as np

from utils.data_io import load_pickle

results_path = "results"
name = "rrr_coefficients"

result = load_pickle(f"{results_path}/{name}")

print(result.shape)

# plt.show()

