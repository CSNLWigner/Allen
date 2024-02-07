# Matplotlib plot for the results of the cca analysis which has compared two brain areas by CCA and saved the results in the results folder.
# Save the plots in the figures folder.

import matplotlib.pyplot as plt
import numpy as np
from allensdk.brain_observatory.ecephys.visualization import _VlPlotter
import yaml

preprocess = yaml.safe_load(open('params.yaml'))['preprocess']

def simple_rrr_plot(result, params, ax=None):
    
    # Create a new figure and axes if not provided
    if ax is None:
        fig, axs = plt.subplots(1, 5, figsize=(15, 3))
        
    # Make supertitle
    fig.suptitle('Coefficients of Reduced Rank Regression')
    
    # Find global min and max
    vmin = np.min(result)
    vmax = np.max(result)
    
    # Plot the results
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

def simple_rrr_plot_mean(result, ax=None):
    
    # Calculate the mean of the results
    np.mean(result, axis=2)
    
    # Create a new figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Plot the mean of the results
    im = ax.imshow(np.mean(result, axis=2), cmap='hot', aspect='auto')

    plt.colorbar(im, ax=ax)
    plt.xlabel('VISl')
    plt.ylabel('VISp')
    plt.title('Coefficients of Reduced Rank Regression')

    return fig


def raster_plot(spike_times, figsize=(8, 8), cmap=plt.cm.tab20, title='spike raster', cycle_colors=False, ax=None):
    """
    imported from allensdk.brain_observatory.ecephys.visualization
    """

    # Create a new figure and axes if not provided
    if ax is None:
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

def cross_correlation_plot(cross_correlation, time_series=None, title='Cross-correlation', ax=None):
    """
    Plots the cross-correlation between two signals.

    Parameters:
    cross_correlation (array-like): A one-dimensional array-like object representing the cross-correlation values.
    time_series (array-like, optional): A one-dimensional array-like object representing the time series. If not provided, it will be generated using the length of cross_correlation.
    title (str, optional): The title of the plot. Default is 'Cross-correlation'.
    ax (matplotlib.axes.Axes, optional): The axes on which to plot. If not provided, a new figure and axes will be created.

    Returns:
    None
    """
    
    # If time_series is not provided, generate it
    if time_series is None:
        time_series = np.arange(len(cross_correlation))

    # Create a new figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots()
    
    # Plot the cross-correlation
    ax.plot(time_series, cross_correlation)
    ax.set_xlabel('Time lag')
    ax.set_ylabel('Cross-correlation')
    ax.set_title(title)

    # # Plot the cross-correlation
    # plt.plot(time_series, cross_correlation)
    # plt.xlabel('Time lag')
    # plt.ylabel('Cross-correlation')
    # plt.title(title)
    
    return fig