
import numpy as np


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
