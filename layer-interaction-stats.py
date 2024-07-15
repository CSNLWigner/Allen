
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt

from utils.megaplot import megaplot

scaler = yaml.safe_load(open("params.yaml"))["crosstime"]['scaling-factor']

# Load the data
filenames = {
    '1': 'results/maxValues.csv',
    '2': 'results/maxValues-notUnderSampling.csv',
    '3': 'results/maxValues-firstUnderSampling.csv',
    '4': 'results/maxValues-adjR2.csv'
}
filename = filenames[input(filenames)]
print('Loading', filename)
df = pd.read_csv(filename)
# i = input('Enter the end of file name: ')
# i = '-' + i if i != '' else ''
# df = pd.read_csv(f'results/maxValues{i}.csv')
df = df.set_index(['session', 'direction', 'slice', 'output layer', 'input layer'])

# Get rid of the session 1087720624
df = df.drop([1087720624], level='session')

# Create a dictionary of rows, with one entry for each slice
rows = {}
rows['first'] = df[df.index.get_level_values('slice') == 'first']
rows['second'] = df[df.index.get_level_values('slice') == 'second']

# Add directional data to the rows dictionary
for rowName, rowData in rows.items():
    TD = rowData[rowData.index.get_level_values('direction') == 'LM-to-V1']
    BU = rowData[rowData.index.get_level_values('direction') == 'V1-to-LM']
    rows[rowName] = {
        'top-down': TD,
        'bottom-up': BU
    }

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
    table.fill(0)  # Initialize the table with zeros
    
    # Iterate through the rows of df and increase the corresponding scaled x and y values in the table
    for index, row in df.iterrows():
        if pd.isna(row['x']) or pd.isna(row['y']):
            continue  # Skip rows where 'x' or 'y' is NaN
        scaled_x = int(row['x'] / scaler) - x_min
        scaled_y = int(row['y'] / scaler) - y_min
        if row['x'] != 100 and row['y'] != 100:
            table[scaled_x, scaled_y] += 1
        else:
            print(f"Skipping row with x={row['x']} and y={row['y']}")
    
    # Flip the table along the x-axis to match the orientation of the plot
    table = table[::-1, :]
    
    # Use imshow with adjusted extent to reflect the original x and y range, scaled for visualization
    im = ax.imshow(table, extent=[x_min * scaler, (x_max + 1) * scaler, y_min * scaler, (y_max + 1) * scaler], aspect='auto')
    return im

def plot_values(df: pd.DataFrame, ax: plt.Axes, name) -> plt.Axes:
    df_mean = df.groupby(name).mean()
    df_sem = df.groupby(name).sem()
    df_mean = df_mean.drop(columns=['x', 'y', 'output layer units', 'input layer units'])
    im = df_mean.plot(kind='bar', yerr=df_sem, ax=ax)
    max_val = df_mean['max value'].max() + df_sem['max value'].max()
    ax.set_ylim(-0.01, max_val)
    # add a horizontal bar to the 0
    ax.axhline(0, color='black', linewidth=0.5)
    return im


# Create the plot
plot = megaplot(nrows=4, ncols=3)
plot.row_names = ['First slice top-down', 'First slice bottom-up', 'Second slice top-down', 'Second slice bottom-up']
plot.col_names = ['Indices', 'Output layer', 'Input layer']

# Plot the data
for i, rowName in enumerate(['first', 'second']):
    rowNumber = i*2
    rowNumber_TD = rowNumber
    rowNumber_BU = rowNumber + 1
    
    for direction, row in zip(['top-down', 'bottom-up'], [rowNumber_TD, rowNumber_BU]):
    
        # Plot the indices
        ax = plot[row, 0]
        im = plot_indices(rows[rowName][direction], ax)
        ax.set_xlabel('time target area')
        ax.set_ylabel('time source area')

        # Plot the values
        ax = plot[row, 1]
        im = plot_values(rows[rowName][direction], ax, 'output layer')
        ax.set_xlabel('Output layer')
        ax.set_ylabel('RRR')
        
        # Plot the values
        ax = plot[row, 2]
        im = plot_values(rows[rowName][direction], ax, 'input layer')
        ax.set_xlabel('Input layer')
        ax.set_ylabel('RRR')

# Show the plot
plot.show()
