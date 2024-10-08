# cca_plot.py

"""
This module plots the CCA scores.

**Parameters**:

None

**Input**:

- `results/cca_scores.csv`: CCA scores.

**Output**:

None

**Submodules**:

None

"""

import numpy as np
from matplotlib import pyplot as plt

# Load the results from the results folder
results_path = "results/cca_scores.csv"
with open(results_path, 'r') as file:
    vmi = np.genfromtxt(file)
    # reader = csv.reader(file)
    # print(reader)
    # results = list(reader)[0][0]

print(vmi)

# plt.plot(results.tolist())
# plt.plot([-2.5, -1.5, 3.01])
# plt.plot([-0.40501441 -0.41737124 -0.50881126 -0.66061258 -0.47925625])
plt.show()
# cca.line(results)

