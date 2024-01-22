# Main for plot CCA results (saved in the results folder)
# DVC file. 

from plots import cca
import numpy as np

# Load the results from the results folder
results_path = "results/cca_results.npy"
results = np.load(results_path)

cca.line(results)

