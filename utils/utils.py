
from scipy.stats import shapiro
import numpy as np
import yaml

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