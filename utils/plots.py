# Matplotlib plot for the results of the cca analysis which has compared two brain areas by CCA and saved the results in the results folder.
# Save the plots in the figures folder.

import matplotlib.pyplot as plt
import numpy as np
from allensdk.brain_observatory.ecephys.visualization import _VlPlotter
import yaml

preprocess = yaml.safe_load(open('params.yaml'))['preprocess']

def line(result):
    """
    expects result to be a one-dimensional array-like object (such as a list or a numpy array). Each element in the array represents a "CCA Score" at a certain "Time". The index of the element in the array corresponds to the "Time"
    """

    # Plot the results
    plt.plot(result)
    # plt.xlabel("Time")
    # plt.ylabel("CCA Score")
    # plt.title("CCA Analysis Results")
    plt.show()


def scatter(result):
    """
    an object with two attributes: x_scores_ and y_scores_. Both x_scores_ and y_scores_ should be array-like objects of the same length. Each element in x_scores_ corresponds to an "X Score", and each element in y_scores_ corresponds to a "Y Score". The indices of the elements in the arrays correspond to the pairs of "X Scores" and "Y Scores"
    """

    # Visualize the result
    plt.figure()
    plt.plot(result.x_scores_, result.y_scores_, 'o')
    plt.xlabel('X Scores')
    plt.ylabel('Y Scores')
    plt.title('CCA Result')
    plt.show()


def simple_rrr_plot(result, params):
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    fig.suptitle('Coefficients of Reduced Rank Regression')
    
    # Find global min and max
    vmin = np.min(result)
    vmax = np.max(result)
    
    for i in range(params['cv']):
        im = axs[i].imshow(result[:, :, i], 
                           cmap='hot', aspect='auto', 
                           vmin=vmin, vmax=vmax)
        time_bin = preprocess['bin-size']
        duration = preprocess['stimulus-duration']
        timecourse = np.arange(0, duration+time_bin, time_bin) *1000
        axs[i].set_title(f'{int(timecourse[i])}-{int(timecourse[i+1])} ms')
        axs[i].set_xlabel('VISl')
        axs[i].set_ylabel('VISp')

    # Attach colorbar to the last plot
    fig.colorbar(im, ax=axs[i])

    return fig

def simple_rrr_plot_mean(result):
    
    # Calculate the mean of the results
    np.mean(result, axis=2)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(np.mean(result, axis=2), cmap='hot', aspect='auto')

    plt.colorbar(im, ax=ax)
    plt.xlabel('VISl')
    plt.ylabel('VISp')
    plt.title('Coefficients of Reduced Rank Regression')

    return fig


def raster_plot(spike_times, figsize=(8, 8), cmap=plt.cm.tab20, title='spike raster', cycle_colors=False):
    """
    imported from allensdk.brain_observatory.ecephys.visualization
    """

    fig, ax = plt.subplots(figsize=figsize)
    plotter = _VlPlotter(ax, num_objects=len(
        spike_times.keys().unique()), cmap=cmap, cycle_colors=cycle_colors)
    # aggregate is called on each column, so pass only one (eg the stimulus_presentation_id)
    # to plot each unit once
    spike_times[['stimulus_presentation_id', 'unit_id']
                ].groupby('unit_id').agg(plotter)

    ax.set_xlabel('time (s)', fontsize=16)
    ax.set_ylabel('unit', fontsize=16)
    ax.set_title(title, fontsize=20)

    plt.yticks([])
    plt.axis('tight')

    return fig
