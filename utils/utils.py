
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
        float: The accuracy score.
    """
    similarity = np.mean(np.equal(Y_data, prediction))
    accuracy = similarity 
    return accuracy


def MSE(target, prediction):
    """
    Calculate the Mean Squared Error based on the mean squared error between Y_data and the prediction.

    Args:
        Y_data (np.ndarray): The actual data.
        prediction (np.ndarray): The predicted data.

    Returns:
        float: The similarity score.
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
        # Make a simple matrix with 3 dimensions
        arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

        # Iterate through the second dimension
        for counter, sliced in iterate_dimension(arr, 1):
            print(counter, sliced)

    Returns:
        None
    """
    for i in range(arr.shape[dim]):
        yield i, arr.take(i, axis=dim)


# Determine if the data has normal distribution
def normality_test(full_activity, dim=2):
    """
    Perform the Shapiro-Wilk test to determine if the data has a normal distribution.

    Args:
        full_activity (np.ndarray): The input data.

    Returns:
        None
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

def get_time(time_bin, bin_size=preprocess['step-size'], digits=3): return round(time_bin*bin_size, digits)
"""
```python
get_time = lambda time_bin, bin_size: round(time_bin*bin_size, 3)
def get_time(time_bin, bin_size): return round(time_bin*bin_size, 3)
```

The difference between the lines is that the first line defines a lambda function, while the second line defines a regular function.

A lambda function is an anonymous function that can be defined in a single line. It is typically used for simple, one-time operations. In this case, the lambda function get_time takes two arguments time_bin and bin_size, and returns the result of rounding time_bin * bin_size to 3 decimal places.

On the other hand, the regular function get_time is defined using the def keyword. It also takes two arguments time_bin and bin_size, and returns the result of rounding time_bin * bin_size to 3 decimal places.
"""


def shift_with_nans(arr, shift, axis=2, constant=np.nan):
    """
    Shifts the elements of a numpy array along a specified axis by padding with NaNs.

    Args:
        arr (np.ndarray): The input array.
        shift (int): The number of positions to shift the elements. Positive values shift to the right, negative values shift to the left.
        axis (int): The axis along which to shift the elements. Default is 2.
        constant (int): The value to use for padding. Default is np.nan.

    Returns:
        np.ndarray: The shifted array.

    Example:
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
    """
    padding = [(0, 0) for _ in range(arr.ndim)]
    if shift > 0:
        padding[axis] = (shift, 0)
    else:
        padding[axis] = (0, -shift)
    arr = np.pad(arr, padding, mode='constant', constant_values=constant)
    slices = [slice(None) if i != axis else slice(None, -shift) if shift > 0 else slice(-shift, None) for i in range(arr.ndim)]
    return arr[tuple(slices)]


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r", onComplete='delete'):
    """
    Call in a loop to create a terminal progress bar.

    Parameters:
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
        printProgressBar(0, 10)
        for i in range(10):
            printProgressBar(i + 1, 10)
            time.sleep(0.1)
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
        python script.py -v -o output.txt input_file.txt
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
    Function to find the connected components in a graph using DFS

    Parameters:
    - node: The starting node for the DFS traversal
    - graph: The graph represented as an adjacency list
    - visited: A boolean array to keep track of visited nodes
    - component: A list to store the nodes in the connected component

    Returns:
    None
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
    column_to_dfs = defaultdict(set)
    for idx, df in enumerate(input):
        for column in df.columns:
            column_to_dfs[column].add(idx)
    return column_to_dfs


def dfs_to_graph(column_to_dfs: defaultdict) -> defaultdict:
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
    Creates a graph based on the input data.

    Args:
        input (iter): An iterable containing the input data.

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
        list: A merged list containing unique elements from the input lists.

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
    Merges a list of DataFrames into a list of merged DataFrames.

    Args:
        dataframes (list): A list of pandas DataFrames to be merged.

    Returns:
        list: A list of merged pandas DataFrames.

    Example:
        df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        df2 = pd.DataFrame({'B': [6, 7, 8], 'C': [9, 10, 11]})
        df3 = pd.DataFrame({'D': [12, 13, 14], 'E': [15, 16, 17]})
        df4 = pd.DataFrame({'C': [10, 11, 12], 'F': [18, 19, 20]})
        dataframes = [df1, df2, df3, df4]

        for df in mergeDataframes(dataframes):
            print(df)
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

