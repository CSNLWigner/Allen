
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from scipy import stats

from utils.megaplot import megaplot

scaler = yaml.safe_load(open("params.yaml"))["crosstime"]['scaling-factor']

# Load the data
df = pd.read_csv('results/maxValues.csv')
df = df.set_index(['session', 'direction', 'slice', 'output layer', 'input layer'])

# Slice the data
df_first = df[df.index.get_level_values('slice') == 'first']
df_second = df[df.index.get_level_values('slice') == 'second']

# print('shape:', df_first.shape)


# def plot_indices(df: pd.DataFrame, ax: plt.Axes) -> plt.Axes:
#     noise_std = 0.2
#     nCol = df.shape[0]
#     xnoise = np.random.normal(scale=noise_std, size=nCol)
#     ynoise = np.random.normal(scale=noise_std, size=nCol)
#     df['x'] = df['x'] + xnoise
#     df['y'] = df['y'] + ynoise
#     im = df.plot.scatter(x='x', y='y', ax=ax)
#     return im


def plot_indices(df: pd.DataFrame, ax: plt.Axes) -> plt.Axes:
    # Adjust the calculation of x_min, x_max, y_min, and y_max to use the scaler
    x_min = int(df['x'].min() / scaler)
    x_max = int(df['x'].max() / scaler)
    y_min = int(df['y'].min() / scaler)
    y_max = int(df['y'].max() / scaler)
    
    # Create an empty 2D numpy array for the scaled dimensions
    table = np.empty((x_max - x_min + 1, y_max - y_min + 1))
    
    # Iterate through the rows of df and increase the corresponding scaled x and y values in the table
    for index, row in df.iterrows():
        scaled_x = int(row['x'] / scaler) - x_min
        scaled_y = int(row['y'] / scaler) - y_min
        table[scaled_x, scaled_y] += 1
    
    # Use imshow with adjusted extent to reflect the original x and y range, scaled for visualization
    im = ax.imshow(table, extent=[x_min * scaler, (x_max + 1) * scaler, y_min * scaler, (y_max + 1) * scaler], aspect='auto')
    return im

def plot_values(df: pd.DataFrame, ax: plt.Axes, name) -> plt.Axes:
    df_mean = df.groupby(name).mean()
    df_sem = df.groupby(name).sem()
    df_mean = df_mean.drop(columns=['x', 'y', 'output layer units', 'input layer units'])
    im = df_mean.plot(kind='bar', yerr=df_sem, ax=ax)
    return im


# Create the plot
plot = megaplot(ncols=3, nrows=2)
plot.row_names = ['First slice', 'Second slice']

# TODO: bottom-up and top-down separately.

# Plot the data
plot_indices(df_first, plot[0, 0])
plot_values(df_first, plot[0, 1], 'output layer')
plot_values(df_first, plot[0, 2], 'input layer')
plot_indices(df_second, plot[1, 0])
plot_values(df_second, plot[1, 1], 'output layer')
plot_values(df_second, plot[1, 2], 'input layer')

# Show the plot
plot.show()
