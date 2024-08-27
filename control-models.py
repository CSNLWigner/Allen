# control-models.py

"""
This module compares the control models.

**Parameters**:

None

**Input**:

- `results/rrr-control.pickle`: RRR control models.

**Output**:

- `figures/control-models.png`: Plot of the control models.

**Submodules**:

- `analyses.rrr`: Module for RRR analysis.
- `utils.data_io`: Module for loading and saving data.
- `utils.plots`: Module for plotting data.
"""

import matplotlib.pyplot as plt

from analyses.rrr import control_models
from utils.data_io import save_fig
from utils.plots import score_plot_by_time

# Create Figure
fig, ax = plt.subplots()  # 1, 1, figsize=(10, 10)

predictors = ['V1']

result = control_models(predictor_names=predictors,
                        response_name='V2', log=True)

score_plot_by_time(result, ax=ax, label=', '.join(predictors))

predictors = ['V1', 'movement']

result = control_models(predictor_names=predictors,
                        response_name='V2')

score_plot_by_time(result, ax=ax, label=', '.join(predictors))

# predictors = ['V1', 'pupil']

# result = control_models(predictor_names=predictors,
#                         response_name='V2')

# score_plot_by_time(result, ax=ax, label=', '.join(predictors))

# predictors = ['V1', 'movement', 'pupil']

# result = control_models(predictor_names=predictors,
#                         response_name='V2')

# score_plot_by_time(result, ax=ax, label=', '.join(predictors))

predictors = ['V1', 'movement', 'V2']

result = control_models(predictor_names=predictors,
                        response_name='V2')

score_plot_by_time(result, ax=ax, label=', '.join(predictors))

predictors = ['V2']

result = control_models(predictor_names=predictors,
                        response_name='V2')

score_plot_by_time(result, ax=ax, label=', '.join(predictors))

# Generate legend
ax.legend()

plt.show()

# Save the figure
save_fig(fig, 'control-models', path='figures')

# Save the results
# save_pickle(results, f'RRR_control', path='data/rrr-results')

    
    
    
    




