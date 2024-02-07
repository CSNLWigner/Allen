

import numpy as np


def calculate_time_lag(areaX, areaY):

    # Calculate the cross-correlation
    cross_correlation = np.correlate(
        areaX.mean(axis=(0, 1)),
        areaY.mean(axis=(0, 1)),
        'full')
    print(cross_correlation)

    # Find the time lag
    time_lag = np.argmax(cross_correlation)

    return cross_correlation, time_lag