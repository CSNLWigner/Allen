# Matplotlib plot for the results of the cca analysis which has compared two brain areas by CCA and saved the results in the results folder.
# Save the plots in the figures folder.

import matplotlib.pyplot as plt
import numpy as np
from allensdk.brain_observatory.ecephys.visualization import _VlPlotter
import yaml
from utils.data_io import save_pickle

from utils.utils import get_time, iterate_dimension

from scipy.stats import sem

preprocess = yaml.safe_load(open('params.yaml'))['preprocess']


def simple_mean_SEM_time_plot(ax, mean, ylabel, title=None, SEM=None, SEM_multiplier=2, time_series=None, color=None, xlabel=None, alpha=0.2, linewidth=None, xticks=None, xticklabels=None, yticks=None, yticklabels=None, label=None, xlim=None, ylim=None) -> plt.Figure:
    """
    Plots the mean and standard error of the mean of the results as a function of time.

    Parameters:
    mean (array-like): A one-dimensional array-like object representing the mean of the results.
    SEM (array-like): A one-dimensional array-like object representing the standard error of the mean of the results.
    title (str, optional): The title of the plot. Default is 'Mean and SEM of the results'.
    time_series (array-like, optional): A one-dimensional array-like object representing the time series. If not provided, it will be generated using the params.yaml file.
    ax (matplotlib.axes.Axes, optional): The axes on which to plot. If not provided, a new figure and axes will be created.

    Returns:
    matplotlib.figure.Figure: The figure object containing the plot.
    """
    
    # Set default values
    if time_series is None:
        duration = preprocess['stimulus-duration']
        time_step = preprocess['step-size']
        time_series = np.arange(0, duration+time_step, time_step).round(3)
    
    if xlabel is None:
        xlabel = 'Time (s)'
    
    # Plot the mean and standard error of the mean of the results as a function of time
    if SEM is not None:
        ax.fill_between(time_series,
                        mean-SEM*SEM_multiplier,
                        mean+SEM*SEM_multiplier,
                        alpha=alpha, color=color)
    cax=ax.plot(time_series, mean, color=color, linewidth=linewidth, label=label)
    
    # Set the lims
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Set the x-axis and y-axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Set the title of the plot
    ax.set_title(title)
    
    # Set xticks and yticks
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticks is not None:
        ax.set_yticks(yticks)
    
    return cax

def simple_rrr_plot(result, axs=None) -> plt.Figure:
    
    # Add a third dimension if not present
    if len(result.shape) == 2:
        result = result[:,:,np.newaxis]
    
    # Create a new figure and axes if not provided
    if axs is None:
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
        time_series = np.arange(0, duration+time_bin, time_bin) *1000
        axs[t].set_title(f'{int(time_series[t])}-{int(time_series[t+1])} ms')
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

def cross_time_correlation_coefficients_plot(coeffs, time_series=None, first_dim_label=None, second_dim_label=None, title='Cross-time-correlation', ax=None) -> plt.Figure:
    """
    Plots the cross-correlation between two signals. Colors range from blue (negative) to red (positive), with white representing zero.

    Parameters:
    coeffs (array-like): A three-dimensional array-like object representing the cross-correlation coefficients.
    time_series (array-like, optional): A one-dimensional array-like object representing the time series. If not provided, it will be generated using the length of coeffs.
    title (str, optional): The title of the plot. Default is 'Cross-time-correlation'.
    ax (matplotlib.axes.Axes, optional): The axes on which to plot. If not provided, a new figure and axes will be created.

    Returns:
    matplotlib.pyplot.Figure: The figure containing the plot.
    """
    
    # Reverse the second dimension of the coefficients
    coeffs = np.flip(coeffs, axis=1)
    
    # If time_series is not provided, generate it
    if time_series is None:
        time_step = preprocess['step-size']
        duration = preprocess['stimulus-duration']
        time_series = np.arange(0, duration, time_step).round(3)
    
    # Create a custom color palette ranging from blue (negative) to red (positive), with white representing zero
    cmap = 'bwr'
    
    # Create a new figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots()

    # Plot the results using the custom color palette
    im = ax.imshow(coeffs, cmap=cmap, aspect='auto')
    ax.set_title(title)
    ax.set_xlabel(first_dim_label)
    ax.set_ylabel(second_dim_label)
    
    # Set xticklabels and yticklabels corresponding to some values of the time_series
    ax.set_xticks(np.arange(0, len(time_series), 4))
    ax.set_xticklabels(time_series[::4])
    
    # Plot a reverse time series on the y-axis
    ax.set_yticks(np.arange(0, len(time_series), 4))
    ax.set_yticklabels(time_series[::4][::-1])

    # Attach colorbar to the last plot
    fig.colorbar(im, ax=ax)

    return fig

def rrr_rank_plot(scores, title='RRR test scores (r2)', time_series=None, ax=None) -> plt.Figure:
    """
    Plots the RRR test scores as a function of rank and time.

    Parameters:
    scores (array-like): A two-dimensional array-like object representing the scores.
    rank (array-like): A one-dimensional array-like object representing the rank.
    title (str, optional): The title of the plot. Default is 'Activity Estimation Error'.
    time_series (array-like, optional): A one-dimensional array-like object representing the time series. If not provided, it will be generated using the params.yaml file.
    ax (matplotlib.axes.Axes, optional): The axes on which to plot. If not provided, a new figure and axes will be created.

    Returns:
    matplotlib.figure.Figure: The figure object containing the plot.
    """
    
    # Set default values
    if time_series is None:
        duration = preprocess['stimulus-duration']
        time_step = preprocess['step-size']
        time_series = np.arange(0, duration, time_step).round(3)
    time_step = 2
    
    # Create a new figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots()

    # Plot the errors as a function of rank
    im = ax.imshow(scores, aspect='auto')
    ax.set_xlabel('Time (ms)')
    ax.set_xticks(np.arange(0, len(time_series), time_step))
    ax.set_xticklabels(time_series[::time_step])
    ax.set_ylabel('Rank')
    ax.set_title(title)
    
    # Add colorbar
    fig.colorbar(im, ax=ax)

    return fig

# Define a function that iterates through the time dimennsion of the rrr-rank-scores and plots the scores for each time in a separate plot
def rrr_rank_plot_over_time(scores, title='RRR test scores', time_series=None, fig=None, axs=None, label=None, log=False) -> plt.Figure:
    """
    Plots the RRR test scores as a function of rank and time.

    Parameters:
    scores (array-like): A two-dimensional array-like object representing the scores.
    rank (array-like): A one-dimensional array-like object representing the rank.
    title (str, optional): The title of the plot. Default is 'Activity Estimation Error'.
    time_series (array-like, optional): A one-dimensional array-like object representing the time series. If not provided, it will be generated using the params.yaml file.
    ax (matplotlib.axes.Axes, optional): The axes on which to plot. If not provided, a new figure and axes will be created.

    Returns:
    matplotlib.figure.Figure: The figure object containing the plot.
    """
    
    # Set default values
    if time_series is None:
        duration = preprocess['stimulus-duration']
        time_step = preprocess['step-size']
        time_bin = preprocess['bin-size']
        time_series = np.arange(0, duration+time_step, time_step).round(3)
    
    # Create a new figure and axes if not provided
    if axs is None:
        fig, axs = plt.subplots(1, scores.shape[1], figsize=(15, 3))
    
    # Create suptitle
    if fig is not None:
        fig.suptitle(title)

    # Iterate through the time dimension of the scores
    for t, scores_t in iterate_dimension(scores, 1):
        
        # Calculate optimal rank for the current time
        optimal_rank = np.argmax(scores_t)+1
        
        # Set the time range for the current time
        from_time, to_time = time_series[t].round(3), (time_series[t]+time_bin).round(3)
        
        # Print optimal rank for the current time
        if log:
            print(f'Optimal rank for {from_time}-{to_time} ms: {optimal_rank}')
        
        # Save the optimal rank for the current time
        save_pickle(optimal_rank, f'optimal-rank-{from_time}-{to_time}ms', path='results')
        
        # Plot the scores for the current time
        axs[t].plot(scores_t, label=label)
        axs[t].set_title(f'{from_time}-{to_time} ms')
        axs[t].set_xlabel('Rank')
    
    # Set the y-label for the first plot
    axs[0].set_ylabel('Test score (r2)')
    
    return fig

# Define a function that iterates through the time dimennsion of the rrr-test-scores (shape (cv, time)) and plots the scores for each time in a separate plot
def score_plot_by_time(scores, title=None, time_series=None, ax=None, label='', color=None) -> plt.Figure:
    """
    Plots the RRR test scores as a function of rank and time.

    Parameters:
    scores (array-like): A two-dimensional array-like object representing the scores. Shape (cv, time)
    rank (array-like): A one-dimensional array-like object representing the rank.
    title (str, optional): The title of the plot. Default is 'Activity Estimation Error'.
    time_series (array-like, optional): A one-dimensional array-like object representing the time series. If not provided, it will be generated using the params.yaml file.
    ax (matplotlib.axes.Axes, optional): The axes on which to plot. If not provided, a new figure and axes will be created.

    Returns:
    matplotlib.figure.Figure: The figure object containing the plot.
    """
    
    # If color is not provided, generate it from the default color cycle. If color is None, the color will be determined by the axes.
    if color is None:
        color = next(ax._get_lines.prop_cycler)['color']
        
    
    # Calculate tshe mean and standard error of the mean of the scores
    mean_scores = np.mean(scores, axis=0)
    sem_scores = sem(scores, axis=0)
    
    # Calculate the maximum value of the scores
    max_score = np.max(mean_scores+sem_scores)
    
    # Set default values
    if time_series is None:
        duration = preprocess['stimulus-duration'] # 0.250
        time_step = preprocess['step-size'] # 0.050
        half_time_step = time_step/2
        time_series = np.arange(0+half_time_step, duration+half_time_step, time_step).round(3)
    
    # Create a new figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots()
        return_fig = True
    
    # Create title
    if title is not None:
        ax.set_title(title)

    # Plot the mean and standard error of the mean of the scores as a function of time
    ax.plot(range(len(mean_scores)), mean_scores, label=label, color=color)
    ax.fill_between(range(len(mean_scores)), mean_scores-sem_scores, mean_scores+sem_scores, alpha=0.1, color=color)
    
    ax.set_xlabel('Time (s)')
    ax.set_xticks(np.arange(0, len(time_series), 1))
    ax.set_xticklabels(time_series[::1])
    ax.set_ylabel('Test score (r2)')
    # ax.set_xlim(1, len(mean_scores))
    ax.set_ylim(0, max_score)
    
    return


def cv_rank_time_plot(results, title=None, ax=None, max=None, xlabel=None, ylabel=None, xticks=None, yticks=None):
    '''
    Plot the results of the cross-validation and rank.

    Parameters:
    - results: numpy array, the results of the cross-validation and rank
    - title: str, the title of the plot
    - ax: matplotlib Axes object, the axes to plot on (optional)
    - max: int, the maximum value for the colorbar (optional)
    - xlabel: str, the label for the x-axis (optional)
    - ylabel: str, the label for the y-axis (optional)
    - xticks: list, the tick labels for the x-axis (optional)
    - yticks: list, the tick labels for the y-axis (optional)

    Returns:
    - fig: matplotlib Figure object, the figure containing the plot (optional)
    - im: matplotlib Image object, the image representing the plot

    If ax is None, a new figure is created. The plot is displayed using a colormap
    with the 'viridis' color map. The colorbar is added to the plot. If fig is not None,
    the figure is returned. If fig is None, the image is returned.
    
    Note:
    Since the first dimension corresponds to the Y-axis, and the second dimension corresponds to the X-axis, the results is transposed before plotting.
    '''

    # If ax is None, create a new figure
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = None

    # Plot the results
    im = ax.imshow(results.T, cmap='viridis', vmin=0, vmax=max)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(len(xticks)))
    ax.set_xticklabels(xticks)
    ax.set_yticks(range(len(yticks)))
    ax.set_yticklabels(yticks)

    if fig is not None:
        fig.colorbar(im, ax=ax)
        return fig
    
    if fig is None:
        return im

def score_time(mean, sem, title=None, xlabel='Time', ylabel='R^2', time_series=None):
    
    # Set default values
    T = len(mean)

    # If time_series is not provided, generate it
    if time_series is None:
        duration = preprocess['stimulus-duration']
        time_step = preprocess['step-size']
        time_series = np.arange(0, duration+time_step, time_step).round(3)

    # Init the figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Plot the result
    ax.plot(mean)
    ax.fill_between(range(T), np.array(mean) - np.array(sem), np.array(mean) + np.array(sem), alpha=0.2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('R^2')
    ax.set_title(title)
    ax.set_xticks(np.arange(0, len(time_series), 5))
    ax.set_xticklabels(time_series[::5])
    
    return fig

def crosstime_RRR(ax, matrix, predictor, target, timeseries):
    """
    Plot a cross-timepoint correlation matrix.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes on which to plot the matrix.
    - matrix (numpy.ndarray): The correlation matrix to be plotted.
    - predictor (str): The label for the predictor variable.
    - target (str): The label for the target variable.
    - timeseries (numpy.ndarray): The array of timepoints.

    Returns:
    - cax (matplotlib.image.AxesImage): The plotted image of the matrix.
    """

    # The diagonal of the matrix should be nan
    # np.fill_diagonal(matrix, np.nan)
    
    # Reverse the rows of the matrix
    matrix = matrix[::-1]

    # tick frequency
    tick_frequency = 5
    # Plot the matrix. colormap do not use white color. Make the resolution higher.
    cax = ax.imshow(matrix, cmap='terrain', interpolation='bilinear', 
                    extent=[0, timeseries[-1], 0, timeseries[-1]])
    
    # black line from 0;0 to the max;max
    ax.plot([0, timeseries[-1]], [0, timeseries[-1]],
            color='black', linewidth=1)
    
    # Set the ticks and labels
    ax.set_xticks(timeseries[::tick_frequency])
    ax.set_yticks(timeseries[::tick_frequency])
    ax.set_xlabel(f"Timepoints of {target}")
    ax.set_ylabel(f"Timepoints of {predictor}")

    return cax

def rrr_time_slice(ax, results, predictor_time, timepoints=None, colors=None, ylim=(None, None)):
    
    # TODO: Plot self prediction
    
    if type(colors) == tuple:
        color_TD, color_BU = colors
    else:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        color_TD = colors[0]
        color_BU = colors[1]

    if type(timepoints) is None:
        timepoints = np.arange(len(results['top-down']['mean']))
        raise Warning('Timepoints are not provided. Using the length of the results instead.')

    # Plot the results
    cax_V1 = simple_mean_SEM_time_plot(ax, results['V1']['mean'], 'R^2', title='V1', SEM=results['V1']['sem'], time_series=timepoints, color='blue', label='V1', xlim=(0, timepoints[-1]), ylim=ylim, alpha=0.05, linewidth=0.5)
    cax_LM = simple_mean_SEM_time_plot(ax, results['LM']['mean'], 'R^2', title='LM', SEM=results['LM']['sem'], time_series=timepoints, color='red', label='LM', xlim=(0, timepoints[-1]), ylim=ylim, alpha=0.05, linewidth=0.5)
    cax_TD = simple_mean_SEM_time_plot(ax, results['top-down']['mean'], 'R^2', title='Top-down', SEM=results['top-down']['sem'], time_series=timepoints, color=color_TD, label='Top-down', xlim=(0, timepoints[-1]), ylim=ylim, linewidth=2)
    cax_BU = simple_mean_SEM_time_plot(ax, results['bottom-up']['mean'], 'R^2', title='Bottom-up', SEM=results['bottom-up']['sem'], time_series=timepoints, color=color_BU, label='Bottom-up', xlim=(0, timepoints[-1]), ylim=ylim, linewidth=2)
    ax.legend()
    
    # Make a vertical line at the predictor time
    ax.axvline(x=predictor_time, color='k', linestyle='--')
    ax.set_xticks([0, predictor_time, timepoints[-1]])
    
    return cax_TD, cax_BU, cax_V1, cax_LM