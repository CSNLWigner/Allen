# Matplotlib plot for the results of the cca analysis which has compared two brain areas by CCA and saved the results in the results folder.
# Save the plots in the figures folder.

import matplotlib.pyplot as plt
import numpy as np
from allensdk.brain_observatory.ecephys.visualization import _VlPlotter
import yaml

from utils.utils import iterate_dimension

preprocess = yaml.safe_load(open('params.yaml'))['preprocess']

def simple_rrr_plot(result, ax=None) -> plt.Figure:
    
    # Add a third dimension if not present
    if len(result.shape) == 2:
        result = result[:,:,np.newaxis]
    
    # Create a new figure and axes if not provided
    if ax is None:
        fig, axs = plt.subplots(1, result.shape[2], figsize=(15, 3))
    
    # If axs is not subscriptable, make it subscriptable
    try:
        axs[0]
    except TypeError:
        axs = [axs]
    
    # Make supertitle
    fig.suptitle('Coefficients of Reduced Rank Regression')
    
    # Find global min and max
    vmin = np.min(result)
    vmax = np.max(result)
    
    # Plot the results
    for t, coefficients_t in iterate_dimension(result, 2):
        im = axs[t].imshow(coefficients_t,
                           cmap='hot', aspect='auto', 
                           vmin=vmin, vmax=vmax)
        time_bin = preprocess['bin-size']
        duration = preprocess['stimulus-duration']
        timecourse = np.arange(0, duration+time_bin, time_bin) *1000
        axs[t].set_title(f'{int(timecourse[t])}-{int(timecourse[t+1])} ms')
        axs[t].set_xlabel('VISl')
        axs[t].set_ylabel('VISp')

    # Attach colorbar to the last plot
    fig.colorbar(im, ax=axs[t])

    return fig

def simple_rrr_plot_mean(result, ax=None) -> plt.Figure:
    
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


def raster_plot(spike_times, figsize=(8, 8), cmap=plt.cm.tab20, title='spike raster', cycle_colors=False, ax=None) -> plt.Figure:
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

def cross_correlation_plot(cross_correlation, time_series=None, title='Cross-correlation', ax=None) -> plt.Figure:
    """
    Plots the cross-correlation between two signals.

    Parameters:
    cross_correlation (array-like): A one-dimensional array-like object representing the cross-correlation values.
    time_series (array-like, optional): A one-dimensional array-like object representing the time series. If not provided, it will be generated using the length of cross_correlation.
    title (str, optional): The title of the plot. Default is 'Cross-correlation'.
    ax (matplotlib.axes.Axes, optional): The axes on which to plot. If not provided, a new figure and axes will be created.

    Returns:
    matplotlib.figure.Figure: The figure object containing the plot.
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

def cross_time_correlation_coefficients_plot(coeffs, timecourse=None, title='Cross-time-correlation', ax=None) -> plt.Figure:
    """
    Plots the cross-correlation between two signals.

    Parameters:
    coeffs (array-like): A three-dimensional array-like object representing the cross-correlation coefficients.
    timecourse (array-like, optional): A one-dimensional array-like object representing the time series. If not provided, it will be generated using the length of coeffs.
    title (str, optional): The title of the plot. Default is 'Cross-time-correlation'.
    ax (matplotlib.axes.Axes, optional): The axes on which to plot. If not provided, a new figure and axes will be created.

    Returns:
    matplotlib.pyplot.Figure: The figure containing the plot.
    """
    
    # If timecourse is not provided, generate it
    if timecourse is None:
        time_bin = preprocess['bin-size']
        duration = preprocess['stimulus-duration']
        timecourse = np.arange(0, duration+time_bin, time_bin)

    # Create a new figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots()

    # Plot the results
    im = ax.imshow(coeffs,
                   cmap='hot', aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('LM (s)')
    ax.set_ylabel('V1 (s)')
    
    # Set xticklabels and yticklabels corresponding to some values of the timecourse
    ax.set_xticks(np.arange(0, len(timecourse), 10))
    ax.set_xticklabels(timecourse[::10])
    ax.set_yticks(np.arange(0, len(timecourse), 10))
    ax.set_yticklabels(timecourse[::10])

    # Attach colorbar to the last plot
    fig.colorbar(im, ax=ax)

    return fig