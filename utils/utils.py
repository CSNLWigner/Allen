# utils/utils.py

"""
This submodule contains utility tools for various tasks.

**Functions**:

- calculate_accuracy(Y_data, prediction) -> float: Calculate the accuracy based on the similarity between Y_data and the prediction.
- MSE(target, prediction) -> float: Calculate the Mean Squared Error based on the mean squared error between Y_data and the prediction.
- iterate_dimension(arr, dim) -> None: Iterate through a specific dimension of a numpy array.
- normality_test(full_activity, dim=2) -> None: Determine if the data has normal distribution.
- get_time(time_bin, bin_size=preprocess['step-size'], digits=3) -> float: Get the time in seconds based on the time bin and step size.
- shift_with_nans(arr, shift, axis=2, constant=np.nan) -> np.ndarray: Shift the elements of a numpy array along a specified axis by padding with NaNs.
- printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r", onComplete='delete') -> None: Print iterations progress.
- ProgressBarManager -> class: A class to manage multiple progress bars.
- options_and_arguments() -> None: A simple example of using options and arguments in a Python script.
- dfs(node, graph, visited, component) -> None: Function to find the connected components in a graph using DFS.
- elements_to_dfs(input: iter) -> defaultdict: Convert elements to a dictionary of dataframes.
- dfs_to_graph(column_to_dfs: defaultdict) -> defaultdict: Convert a dictionary of dataframes to a graph.
- createGraph(input: Iterable) -> defaultdict: Create a graph based on the input data.
- iterate_common_elements(lists) -> None: Generator function to yield merged lists with common elements.
- mergeDataframes(dataframes: pd.DataFrame) -> list: Merge a list of DataFrames into a list of merged DataFrames.
- get_args(argv) -> tuple: Get the options and arguments from the command line.
"""

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.stats import shapiro

preprocess = yaml.safe_load(open('params.yaml'))['preprocess']

def calculate_accuracy(Y_data, prediction):
    """
    Calculate the accuracy based on the similarity between Y_data and the prediction.

    Args:
        Y_data (np.ndarray): The actual data.
        prediction (np.ndarray): The predicted data.

    Returns:
        accuracy (float): The accuracy score.
    """
    similarity = np.mean(np.equal(Y_data, prediction))
    accuracy = similarity 
    return accuracy


def MSE(target, prediction):
    """
    Calculate the Mean Squared Error based on the mean squared error between target and prediction.

    Args:
        target (np.ndarray): The actual data.
        prediction (np.ndarray): The predicted data.

    Returns:
        similarity (float): The similarity score.
    """
    mse = np.mean((target - prediction) ** 2)
    similarity = 1 / (1 + mse)
    return similarity

def iterate_dimension(arr, dim):
    """
    Iterate through a specific dimension of a numpy array.
    
    (Similar to the built-in np.ndenumerate() function, but with the ability to specify higher dimensions.)

    Args:
        arr (np.ndarray): The input array.
        dim (int): The dimension to iterate through.

    Yields:
        Tuple[int, np.ndarray]: The counter and the sliced array along the specified dimension.

    Example:
    ```
    # Make a simple matrix with 3 dimensions
    arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

    # Iterate through the second dimension
    for counter, sliced in iterate_dimension(arr, 1):
        print(counter, sliced)
    ```
    """
    for i in range(arr.shape[dim]):
        yield i, arr.take(i, axis=dim)

def normality_test(full_activity, dim=2):
    """
    Perform the Shapiro-Wilk test to determine if the data has a normal distribution.

    Args:
        full_activity (np.ndarray): The input data.
        dim (int, optional): The dimension along which to perform the test. Default is 2.

    """
    for counter, activity_slice in iterate_dimension(full_activity, dim=dim):
        # Normality test
        stat, p = shapiro(activity_slice)
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        # Interpret
        alpha = 0.05
        if p > alpha:
            print(f'{counter}: Activity looks Gaussian (fail to reject H0)')
        else:
            print(f'{counter}: Activity does not look Gaussian (reject H0)')

def get_time(time_bin, bin_size=preprocess['step-size'], digits=3):
    """
    Get the time in seconds based on the time bin and step size.

    Args:
        time_bin (float): The time bin.
        bin_size (float, optional): The step size. Default is the value specified in the 'preprocess' parameter.
        digits (int, optional): The number of decimal places to round the result to. Default is 3.

    Returns:
        rounded_value (float): The time in seconds.
    """
    return round(time_bin*bin_size, digits)

def shift_with_nans(arr, shift, axis=2, constant=np.nan):
    """
    Shift the elements of a numpy array along a specified axis by padding with NaNs.

    Args:
        arr (np.ndarray): The input array.
        shift (int): The number of positions to shift the elements. Positive values shift to the right, negative values shift to the left.
        axis (int, optional): The axis along which to shift the elements. Default is 2.
        constant (int, optional): The value to use for padding. Default is np.nan.

    Returns:
        return (np.ndarray): The shifted array.

    Example:
    ```
    # Create a 2D array
    arr = np.array([[1, 2, 3], [4, 5, 6]])

    # Shift the elements by 1 position to the right along axis 1
    shifted_arr = shift_with_nans(arr, 1, axis=1)
    print(shifted_arr)
    # Output: [[nan 1 2]
    #          [nan 4 5]]

    # Shift the elements by 2 positions to the left along axis 0
    shifted_arr = shift_with_nans(arr, -2, axis=0)
    print(shifted_arr)
    # Output: [[nan nan nan]
    #          [nan nan nan]
    #          [1 2 3]
    #          [4 5 6]]
    ```
    """
    padding = [(0, 0) for _ in range(arr.ndim)]
    if shift > 0:
        padding[axis] = (shift, 0)
    else:
        padding[axis] = (0, -shift)
    arr = np.pad(arr, padding, mode='constant', constant_values=constant)
    slices = [slice(None) if i != axis else slice(None, -shift) if shift > 0 else slice(-shift, None) for i in range(arr.ndim)]
    return arr[tuple(slices)]


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r", onComplete='delete'):
    """
    Print iterations progress.

    Args:
        iteration (int): Current iteration.
        total (int): Total iterations.
        prefix (str, optional): Prefix string. Defaults to ''.
        suffix (str, optional): Suffix string. Defaults to ''.
        decimals (int, optional): Number of decimals in percent complete. Defaults to 1.
        length (int, optional): Character length of the progress bar. Defaults to 100.
        fill (str, optional): Bar fill character. Defaults to '█'.
        printEnd (str, optional): End character. Defaults to "\r".
        onComplete (str, optional): Action to perform when the progress is complete. 
            Options: 'delete' (clear the progress bar), 'newline' (print a newline character). Defaults to 'delete'.

    Example:
    ```
    printProgressBar(0, 10)
    for i in range(10):
        printProgressBar(i + 1, 10)
        time.sleep(0.1)
    ```
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    
    # Complete
    if iteration == total: 
        
        if onComplete == 'delete':
            print(' ' * len(prefix) + ' ' * len(suffix) + ' ' * len(percent) + ' ' * length + '      ', end=printEnd)
        
        if onComplete == 'newline':
            print()


class ProgressBarManager:
    """
    A class to manage multiple progress bars.
    """

    def __init__(self):
        '''
        Initializes a ProgressBarManager object.

        Usage:
            manager = ProgressBarManager()

        Example:
        ```python
        manager = ProgressBarManager()

        # Simulate progress
        for i in range(10):
            manager.progress_bar('Download', i, 10)
            # time.sleep(5)
            for j in range(50):
                manager.progress_bar('Upload', j, 50)
                time.sleep(0.01)
        ```
        '''
        self.progress_bars = {}
        self.n_progress_bars = 0

    def print_progress_bars(self):
        # Clear the screen or move the cursor back to the top
        for id, bar in self.progress_bars.items():
            printProgressBar(bar['current'], bar['total'],
                             prefix=f'{id}:', length=50, printEnd='\n')

        for _ in range(self.n_progress_bars):
            print("\033[F", end='')

    def progress_bar(self, id, current, total):
        '''
        Updates the progress bar with the given id, current value, and total value.

        Args:
            id (str): The id of the progress bar.
            current (int): The current value of the progress bar.
            total (int): The total value of the progress bar.

        Example:
        ```python
        manager = ProgressBarManager()

        # Simulate progress
        for i in range(10):
            manager.progress_bar('Download', i, 10)
            # time.sleep(5)
            for j in range(50):
                manager.progress_bar('Upload', j, 50)
                time.sleep(0.01)
        ```
        '''
        # If not already initialized, initialize a new progress bar
        if id not in self.progress_bars:
            self.new_progress_bar(id, current, total)
        else:
            self.update_progress_bar(id, current)

        # If the progress bar is complete, delete it
        if current == total:
            self.delete_progress_bar(id)

        # Print all the progress bars
        self.print_progress_bars()

    def new_progress_bar(self, id, current, total):
        # Initialize a new progress bar with the given id and step
        self.progress_bars[id] = {'current': current, 'total': total}
        self.n_progress_bars += 1

    def update_progress_bar(self, id, current):
        # Update the progress bar with the given id
        self.progress_bars[id]['current'] = current

    def delete_progress_bar(self, id):
        # Delete the progress bar with the given id
        del self.progress_bars[id]
        self.n_progress_bars -= 1

global manager
manager = ProgressBarManager()

def options_and_arguments():
    """
    A simple example of using options and arguments in a Python script.

    Options:
        -h, --help: Show a help message and exit.
        -v, --verbose: Print verbose output.
        -o, --output <file>: Specify an output file.

    Arguments:
        input_file: The input file to process.

    Example:
        ```python script.py -v -o output.txt input_file.txt```
    """
    import argparse

    # Create the parser
    parser = argparse.ArgumentParser(description='A simple example of using options and arguments.')

    # Add options
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output.')
    parser.add_argument('-o', '--output', type=str, help='Specify an output file.')

    # Add arguments
    parser.add_argument('input_file', type=str, help='The input file to process.')

    # Parse the arguments
    args = parser.parse_args()

    # Print the arguments
    print('Input file:', args.input_file)
    print('Verbose:', args.verbose)
    print('Output file:', args.output)


def dfs(node, graph, visited, component):
    """
    Function to find the connected components in a graph using Depth-First Search (DFS).

        node (int): The starting node for the DFS traversal.
        graph (dict): The graph represented as an adjacency list.
        visited (list): A boolean array to keep track of visited nodes.
        component (list): A list to store the nodes in the connected component.

    """
    stack = [node]
    while stack:
        current = stack.pop()
        if not visited[current]:
            visited[current] = True
            component.append(current)
            for neighbor in graph[current]:
                if not visited[neighbor]:
                    stack.append(neighbor)


def elements_to_dfs(input: iter) -> defaultdict:
    """
    Convert elements to a dictionary of dataframes.

    Args:
        input (iter): An iterable containing the input data.

    Returns:
        defaultdict: A defaultdict representing the dictionary of dataframes.
    """
    column_to_dfs = defaultdict(set)
    for idx, df in enumerate(input):
        for column in df.columns:
            column_to_dfs[column].add(idx)
    return column_to_dfs


def dfs_to_graph(column_to_dfs: defaultdict) -> defaultdict:
    """
    Convert a dictionary of dataframes to a graph.

    Args:
        column_to_dfs (defaultdict): A defaultdict representing the dictionary of dataframes.

    Returns:
        defaultdict: A defaultdict representing the graph.
    """
    graph = defaultdict(list)
    for indices in column_to_dfs.values():
        indices = list(indices)
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                graph[indices[i]].append(indices[j])
                graph[indices[j]].append(indices[i])
    return graph


def createGraph(input: Iterable) -> defaultdict:
    """
    Create a graph based on the input data.

    Args:
        input (Iterable): An iterable containing the input data.

    Returns:
        defaultdict: A defaultdict representing the graph.
    """
    column_to_dfs = elements_to_dfs(input)
    graph = dfs_to_graph(column_to_dfs)
    return graph

def iterate_common_elements(lists):
    """
    Generator function to yield merged lists with common elements.
    Merge multiple lists into a single list by finding connected components in a graph.

    Args:
        lists (list): A list of lists to be merged.

    Yields:
        merged_set (list): A merged list containing unique elements from the input lists.

    """
    # Step 1: Create a graph
    graph = createGraph(lists)

    # Step 2: Find connected components
    visited = [False] * len(lists)

    for i in range(len(lists)):
        if not visited[i]:
            component = []
            dfs(i, graph, visited, component)
            # Step 3: Merge lists in each component
            merged_set = set()
            for idx in component:
                if not merged_set:
                    merged_set = set(lists[idx])
                else:
                    # yield (idx, lists[idx])
                    merged_set.update(lists[idx])
            yield list(merged_set)


def mergeDataframes(dataframes: pd.DataFrame) -> list:
    """
    Merge a list of DataFrames into a list of merged DataFrames.

    Args:
        dataframes (list): A list of pandas DataFrames to be merged.

    Returns:
        list: A list of merged pandas DataFrames.

    Example:
    ```
    df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df2 = pd.DataFrame({'B': [6, 7, 8], 'C': [9, 10, 11]})
    df3 = pd.DataFrame({'D': [12, 13, 14], 'E': [15, 16, 17]})
    df4 = pd.DataFrame({'C': [10, 11, 12], 'F': [18, 19, 20]})
    dataframes = [df1, df2, df3, df4]

    for df in mergeDataframes(dataframes):
        print(df)
    ```
    """
    # Create a graph
    graph = createGraph(dataframes)

    # Find connected components
    visited = [False] * len(dataframes)

    # Merge DataFrames in each component
    merged_graphs = []
    for i in range(len(dataframes)):
        if not visited[i]:
            component = []
            
            # Find connected components
            dfs(i, graph, visited, component)
            
            # Merge DataFrames in each component
            merged_df = pd.DataFrame()
            for idx in component:
                if merged_df.empty:
                    merged_df = dataframes[idx]
                else:
                    merged_df = pd.merge(merged_df, dataframes[idx], how='outer', on=list(
                        merged_df.columns.intersection(dataframes[idx].columns)) or None)
            merged_graphs.append(merged_df)
    return merged_graphs

def get_args(argv):
    """
    Get the options and arguments from the command line.

    Args:
        argv (list): The list of command line arguments.

    Returns:
        options (list): The list of options.
        args (list): The list of arguments.

    Example:
        opts, args = get_args(sys.argv)
    """

    import getopt
    import sys

    try:
        opts, args = getopt.getopt(argv, "ho:v", ["help", "output="])
    except getopt.GetoptError:
        print('Usage: script.py -o <output_file> -v')
        sys.exit(2)

    return opts, args