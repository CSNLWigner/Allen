# Matplotlib plot for the results of the cca analysis which has compared two brain areas by CCA and saved the results in the results folder.
# Save the plots in the figures folder.

import matplotlib.pyplot as plt

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
