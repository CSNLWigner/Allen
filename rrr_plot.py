# Main for plot CCA results (saved in the results folder)
# DVC file. 

from utils import plots
import numpy as np

from utils.data_io import load_csv

results_path = "results/rrr_scores.csv"

result = load_csv(results_path)

print(result)

# plt.show()

