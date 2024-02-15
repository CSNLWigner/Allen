from utils.data_io import save_fig
from analyses.rrr import control_models
import matplotlib.pyplot as plt
from utils.plots import score_plot_by_time


# Create Figure
fig, ax = plt.subplots()  # 1, 1, figsize=(10, 10)

predictors = ['V1']

result = control_models(predictor_names=predictors,
                        response_name='V2')

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

    
    
    
    




