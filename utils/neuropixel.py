# utils/neuropixel.py

"""
This submodule contains tools for working with Neuropixel data from the Allen Institute.

Functions:
- get_table(cache, session_id, table_name) -> DataFrame: Get a table from the cache.
- dict_from_dataframe(df, name) -> dict: Create a dictionary from a DataFrame.
- AllenTables(cache, session_id, layer_assignment=False) -> AllenTables: Create an AllenTables object.
- get_unit_channels(session, log_all_areas=False) -> DataFrame: Get the unit channels for the given session.
- makePSTH(spikes, startTimes, windowDur, binSize=0.001) -> tuple: Compute the Peri-Stimulus Time Histogram (PSTH).
- get_stimulus_presentations(session) -> DataFrame: Get the stimulus presentations for the given session.
- get_area_units(units: pd.DataFrame, area_of_interest) -> DataFrame: Get the units in a specific area of interest.
- get_area_change_responses(session, area_of_interest) -> np.ndarray: Get the change responses for the units in the area of interest.
- get_area_receptive_fields(spike_times, stimulus_presentations, area_of_interest) -> list: Get the receptive fields for the units in the area of interest.
- optotagging(opto_table, spike_times, area_of_interest) -> np.ndarray: Perform optotagging analysis on the units in the area of interest.
- get_response_magnitudes(opto_response) -> np.ndarray: Calculate the response magnitudes of optogenetic stimulation.
- get_average_unit_responses(units, spike_times, trial_start, duration=0.03, binSize=0.001) -> np.ndarray: Calculate the unit responses for each unit in the given units DataFrame.
- get_unit_responses(units, spike_times, trial_start, trial_end, stepSize=0.010, binSize=0.050, progressbar=True) -> np.ndarray: Calculate the unit responses for each trial and time bin.

Classes:
- AllenTables: A class that represents tables related to Allen Institute's Neuropixel data.

Info:
This module provides functions and a class for working with Neuropixel data from the Allen Institute. It includes functions for retrieving specific tables from the cache, creating dictionaries from DataFrames, and performing various analyses on the data. The AllenTables class represents tables related to Neuropixel data and provides methods for creating tables and columns, assigning cortical layer information to units, and retrieving dataframes based on keys. The module also includes functions for computing the Peri-Stimulus Time Histogram (PSTH), getting unit channels, stimulus presentations, units in a specific area of interest, change responses for units in an area of interest, receptive fields for units in an area of interest, performing optotagging analysis, calculating response magnitudes of optogenetic stimulation, and calculating unit responses for each unit in a given DataFrame.

Acronyms:
- VISp: Primary Visual Area (V1)
- VISpl: Posterolateral visual area
- VISli: Laterointermediate area (visually guided behav)
- VISl: Lateral visual area (V2, LM)
- VISal: Anteromedial visual area
- VISlla: Laterolateral anterior visual area
- VISrl: Rostrolateral visual area (visually guided behav)
- VISam: Anteromedial visual area
- VISpm: Posteromedial visual area (V4-V5, MT; PMC: associative area)
- VISm: Medial visual area (V6, "medial motion area", see: https://pages.ucsd.edu/~msereno/papers/V6Motion09.pdf)
- VISmma: Mediomedial anterior visual area
- VISmmp: Mediomedial posterior visual area

For more information, refer to the AllenSDK documentation: https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quickstart.html
"""


import numpy as np
import pandas as pd
from allensdk.brain_observatory.ecephys.visualization import raster_plot
from matplotlib import pyplot as plt

from utils.utils import mergeDataframes, printProgressBar




def get_table(cache, session_id, table_name):
    """
    Get a table from the cache.

    Args:
        cache (EcephysProjectCache): The cache object.
        session_id (int): The session identifier.
        table_name (str): The name of the table to retrieve.

    Returns:
        DataFrame: The table data.
    """
    return cache.get_session_data(session_id)[table_name]

def dict_from_dataframe(df: pd.DataFrame, name: str) -> dict:
    """
    Create a dictionary from a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to convert.
        name (str): The name to assign to each column in the dictionary.

    Returns:
        dict: A dictionary where the keys are the column names and the values are the assigned name.
    """
    return {col_name: name for col_name in df.columns}


jointColumns = {
    'session': ['behavior_session_id'],
    'behavior': ['ecephys_session_id'],
    'probes': ['ecephys_session_id'],
    'units': ['ecephys_session_id', 'ecephys_probe_id', 'ecephys_channel_id'],
    'channels': ['ecephys_session_id', 'ecephys_probe_id']
}

class AllenTables():
    """
    A class that represents tables related to Allen Institute's Neuropixel data.

    Attributes:
        cache (object): The cache object used to retrieve data.
        session_id (int): The ID of the session.

    Methods:
        make_tables: Creates a dictionary of tables.
        make_columns: Creates a dictionary of columns.
        layer_assignment: Assigns cortical layer information to the units based on the channels and units tables.
        __init__: Initializes the AllenTables object.
        __getitem__: Retrieves a dataframe based on the given key.
    """

    def make_tables(self):
        """
        Creates a dictionary of tables.

        Returns:
            dict: A dictionary of tables.
        """
        self.tables = {
            'session': self.session,
            # 'behavior': self.behavior,
            'probes': self.probes,
            'units': self.units,
            'channels': self.channels
        }
        return self.tables
    
    def make_columns(self):
        """
        Creates a dictionary of columns.

        Returns:
            dict: A dictionary of columns.
        """
        self.columns = {}

        for name, table in self.tables.items():
            # Get the columns from the dataframe
            new_columns = dict_from_dataframe(table, name)
            
            # Merge the new columns with the existing columns
            for key, value in new_columns.items():
                if key in self.columns:
                    # Key exists in dictionary
                    self.columns[key].append(value)
                else:
                    # Key does not exist in dictionary, add it
                    self.columns[key] = [value]
                
        return self.tables
    
    def layer_assignment(self):
        """
        Assigns cortical layer information to the units based on the channels and units tables.

        This method uses the cortical_layer_assignment function to assign the cortical layer information
        to the units in the tables. The assigned layer information is then appended to the 'layer' column
        in the columns list.

        Parameters:
            None

        Returns:
            None
        """
        # self.units = cortical_layer_assignment(self.tables.channels, self.tables.units)
        # self.columns.append('layer')
    
    def __init__(self, cache, session_id, layer_assignment=False):
        """
        Initializes the AllenTables object.
        
        Args:
            cache (object): The cache object used to retrieve data.
            session_id (int): The ID of the session.
            layer_assignment (bool, optional): Whether to assign cortical layers to the units. Defaults to False.
        
        Note:
            The elapsed time for this function is approximately 0.198 seconds.
        """
        self.cache = cache
        self.session_id = session_id
        
        session = cache.get_session_data(cache.get_session_table().index.values[session_id])
        
        # get the metadata tables
        self.session = pd.DataFrame(cache.get_session_table().iloc[session_id])
        self.session['ecephys_session_id'] = session_id
        # self.behavior = cache.get_behavior_session_table()[cache.get_behavior_session_table()['ecephys_session_id'] == session_id]
        self.probes = session.probes
        self.units = session.units
        self.channels = session.channels
        
        # Make the tables and columns utilities
        self.make_tables()
        self.make_columns()
        self.table_names = list(self.tables.keys())
        
        # Assign cortical layers to the units
        if layer_assignment:
            raise NotImplementedError("The layer_assignment method is not implemented yet.") # TODO: move this class to a new file, bcs the import of ccf files are too slow. Then just uncomment the lines in the layer_assignment function
            self.layer_assignment()
    
    def __getitem__(self, key: str)-> pd.DataFrame:
        """
        Retrieves a dataframe based on the given key.

        Args:
            key (str): The key to search for in the tables.

        Returns:
            pd.DataFrame: The dataframe containing the columns with the given key.
        """
        # Make a mask for the self.tables with the key
        key_mask = [name for name, table in self.tables.items() if key in table.columns]
        
        # If there are no tables with the key, return None
        if len(key_mask) == 0:
            print(f'No tables with the key: {key}')
            return None
        
        # Add the key to each list of the table if it is not already there
        commonColNames = jointColumns
        for tab in commonColNames:
            if key not in commonColNames[tab]:
                commonColNames[tab].append(key)
        
        # Get the columns with the key from each table
        subtables = [self.tables[name][commonColNames[name]].reset_index() for name in key_mask]
        
        # Merge the columns
        mergedColumns = mergeDataframes(subtables)
        
        if len(mergedColumns) == 1:
            return mergedColumns[0]
        else:
            # TODO: Do this whole procedure in __getitem__ twice (or more), to merge the distant columns. (In the first time it can be only a list with more than one merged subtables)
            raise ValueError('The mergedColumns is not a single dataframe')

def get_unit_channels(session, log_all_areas=False) -> pd.DataFrame:
    """
    Retrieve the unit channels for a given session.

    Args:
        session (Session): The session object.
        log_all_areas (bool, optional): Whether to log all brain areas recorded during the session. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the unit channels.
    """
    
    # Merging unit and channel dataframes will give us CCF coordinates for each unit
    units = session.get_units()
    channels = session.get_channels()
    unit_channels = units.merge(channels, left_on='peak_channel_id', right_index=True)

    # Which brain structures were recorded during this session
    brain_areas_recorded = unit_channels.value_counts('ecephys_structure_acronym')

    if log_all_areas:
        print(brain_areas_recorded, '\n', brain_areas_recorded)

    return unit_channels


def makePSTH(spikes, startTimes, windowDur, binSize=0.001):
    """
    Compute the Peri-Stimulus Time Histogram (PSTH).

    Args:
        spikes (array-like): Array of spike times.
        startTimes (array-like): Array of stimulus start times.
        windowDur (float): Duration of the time window to consider for the PSTH.
        binSize (float, optional): Size of the time bins for the histogram. Defaults to 0.001.

    Returns:
        tuple: A tuple containing the PSTH counts and the bin edges.
        tuple[NDArray[float64], NDArray[floating[Any]]]

    Notes:
        The PSTH is a histogram that represents the firing rate of neurons in response to a stimulus over time.
        It is computed by dividing the time window into bins and counting the number of spikes that fall into each bin.
    """
    bins = np.arange(0, windowDur+binSize, binSize)
    counts = np.zeros(bins.size-1)
    
    # enumerate through trials
    for i, start in enumerate(startTimes):
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start+windowDur)
        counts = counts + np.histogram(spikes[startInd:endInd]-start, bins)[0]

    counts = counts/startTimes.size
    return counts/binSize, bins


def get_stimulus_presentations(session):
    """
    Retrieve the stimulus presentations for a given session.

    Args:
        session (Session): The session object.

    Returns:
        stimulus_presentations (DataFrame): A DataFrame containing the stimulus presentations.
    """
    stimulus_presentations = session.stimulus_presentations
    return stimulus_presentations


def get_area_units(units: pd.DataFrame, area_of_interest) -> pd.DataFrame:
    """
    Retrieve the units in a specific area of interest.

    Args:
        units (pd.DataFrame): The DataFrame containing the units.
        area_of_interest (str or list[str]): The acronym of the area of interest.

    Returns:
        pd.DataFrame: A DataFrame containing the units in the specified area.
    """
    
    # Filter the units based on the area of interest
    good_unit_filter = ((units['snr'] > 1) &
                        (units['isi_violations'] < 1) &
                        (units['firing_rate'] > 0.1))
    good_units = units.loc[good_unit_filter]

    # Get the units in the area of interest
    if type(area_of_interest) == str:
        area_units = good_units[good_units['ecephys_structure_acronym'] == area_of_interest]
    elif type(area_of_interest) == list:
        area_units = good_units[good_units['ecephys_structure_acronym'].isin(area_of_interest)]
    else:
        raise ValueError('area_of_interest must be a string or a list of strings')

    return area_units


def get_area_change_responses(session, area_of_interest) -> np.ndarray:
    """
    Calculate the change responses for units in a specific area of interest.

    Args:
        session (Session): The session object.
        area_of_interest (str or list[str]): The acronym of the area of interest.

    Returns:
        np.ndarray: An array containing the change responses.
    """
    stimulus_presentations = get_stimulus_presentations(session)

    area_change_responses = []
    area_units = get_area_units(area_of_interest=area_of_interest)
    spike_times = session.spike_times
    time_before_change = 1
    duration = 2.5
    for iu, unit in area_units.iterrows():
        unit_spike_times = spike_times[iu]
        unit_change_response, bins = makePSTH(unit_spike_times,
                                                stimulus_presentations['start_time'].values - time_before_change,
                                                duration, binSize=0.01)
        area_change_responses.append(unit_change_response)
    area_change_responses = np.array(area_change_responses)

    return area_change_responses


def get_area_receptive_fields(spike_times, stimulus_presentations, area_of_interest) -> list:
    """
    Get the receptive fields for units in a specific area of interest by Gabors. (There are many trials, and only in the second block are gabors for the receptive fields.)

    Args:
        spike_times (dict): A dictionary mapping unit IDs to their corresponding spike times.
        stimulus_presentations (DataFrame): A DataFrame containing the stimulus presentations.
        area_of_interest (str or list[str]): The acronym of the area of interest.

    Returns:
        list: A list containing the receptive fields.
    """
    rf_stim_table = stimulus_presentations[stimulus_presentations['stimulus_name'].str.contains('gabor')]

    # position categories/ypes of gabor along azimuth
    xs = np.sort(rf_stim_table.position_x.unique())
    # positions of gabor along elevation
    ys = np.sort(rf_stim_table.position_y.unique())

    def find_rf(spikes, xs, ys) -> np.ndarray:
        unit_rf = np.zeros([ys.size, xs.size])
        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                stim_times = rf_stim_table[(rf_stim_table.position_x == x)
                                            & (rf_stim_table.position_y == y)]['start_time'].values
                unit_response, bins = makePSTH(spikes,
                                                stim_times + 0.01,
                                                0.2, binSize=0.001)
                unit_rf[iy, ix] = unit_response.mean()
        return unit_rf

    area_rfs = []
    area_units = get_area_units(area_of_interest=area_of_interest)
    for iu, unit in area_units.iterrows():
        unit_spike_times = spike_times[iu]
        unit_rf = find_rf(unit_spike_times, xs, ys)
        area_rfs.append(unit_rf)

    return area_rfs


def optotagging(opto_table, spike_times, area_of_interest) -> np.ndarray:
    """
    Perform optotagging analysis on units in a specific area of interest.

    Args:
        opto_table (DataFrame): A DataFrame containing the optotagging stimulus table.
        spike_times (dict): A dictionary mapping unit IDs to their corresponding spike times.
        area_of_interest (str or list[str]): The acronym of the area of interest.

    Returns:
        np.ndarray: An array containing the optogenetic responses.
    
    Notes:
    
    Since this is an SST mouse, we should see putative SST+ interneurons that are activated during our optotagging protocol. Let's load the optotagging stimulus table and plot PSTHs triggered on the laser onset. For more examples and useful info about optotagging, you can check out the Visual Coding Neuropixels Optagging notebook here (though note that not all the functionality in the visual coding SDK will work for this dataset).
    
    We use 2 different laser **waveforms**: a short square pulse that's **10 ms** long and a half-period cosine that's 1 second long.
    We drive each at three light **levels**, giving us 6 total conditions
    
    most units don't respond to the short laser pulse.
    Note that the activity occurring at the onset and offset of the laser is artifactual and should be excluded from analysis!

    ```
    opto_table = session.optotagging_table
    ```
    """
    print(opto_table.head())

    # Get the short pulses
    duration = opto_table.duration.min()
    
    # Get the high power trials
    level = opto_table.level.max()

    if 'VIS' not in area_of_interest:
        raise ValueError('To perform optotagging, the area of interest must be a visual area')

    cortical_units = get_area_units(area_of_interest=area_of_interest)

    opto_times = opto_table.loc[(opto_table['duration'] == duration) &
                                (opto_table['level'] == level)]['start_time'].values

    time_before = 0.01
    duration = 0.03
    binSize = 0.001
    opto_response = []
    unit_id = []
    for iu, unit in cortical_units.iterrows():
        unit_spike_times = spike_times[iu]
        unit_response, bins = makePSTH(unit_spike_times,
                                        opto_times - time_before, duration,
                                        binSize=binSize)

        opto_response.append(unit_response)
        unit_id.append(iu)

    opto_response = np.array(opto_response)

    return opto_response


def get_response_magnitudes(opto_response):
    """
    Calculate the response magnitudes of optogenetic stimulation.

    Args:
        opto_response (numpy.ndarray): Array containing the optogenetic response data.

    Returns:
        numpy.ndarray: Array of response magnitudes calculated for each trial.
    """
    baseline_window = slice(0, 9)  # baseline epoch
    response_window = slice(11, 18)  # laser epoch

    response_magnitudes = np.mean(opto_response[:, response_window], axis=1) \
                        - np.mean(opto_response[:, baseline_window], axis=1)

    return response_magnitudes


def get_average_unit_responses(units, spike_times, trial_start, duration=0.03, binSize=0.001):
    """
    Calculate the average unit responses for each unit in the given units DataFrame.

    Args:
        units (DataFrame): DataFrame containing information about the units.
        spike_times (list): List of spike times for each unit.
        trial_start (float): Start time of the trial.
        duration (float): Total duration of trial for PSTH in seconds. Default is 0.03.
        binSize (float): Bin size for PSTH in seconds. Default is 0.001.

    Returns:
        numpy.ndarray: Array containing the average unit responses, shape (units, duration/binSize)
    """
    response = []
    unit_id = []
    for iu, unit in units.iterrows():
        unit_spike_times = spike_times[iu]
        unit_response, bins = makePSTH(unit_spike_times,
                                        trial_start, duration,
                                        binSize=binSize)

        response.append(unit_response)
        unit_id.append(iu)

    return np.array(response)


def get_unit_responses(units, spike_times, trial_start, trial_end, stepSize=0.010, binSize=0.050, progressbar=True):
    """
    Calculate the unit responses for each trial and time bin.

    Args:
        units (DataFrame): A DataFrame containing unit data.
        spike_times (dict): A dictionary mapping unit IDs to their corresponding spike times.
        trial_start (array-like): An array-like object containing the start times of each trial.
        trial_end (array-like): An array-like object containing the end times of each trial.
        stepSize (float, optional): The size of the time step. Defaults to 0.010.
        binSize (float, optional): The size of the time bin. Defaults to 0.050.
        progressbar (bool, optional): Whether to display a progress bar. Defaults to True.

    Returns:
        tensor (ndarray): A 3-dimensional numpy array containing the unit responses for each trial and time bin.
    """
    n_unit = len(units)
    n_trial = len(trial_start)
    trial_length = np.mean(trial_end - trial_start)
    n_step = int(trial_length / stepSize)
    n_bin = int(trial_length / binSize)

    tensor = np.zeros((n_unit, n_trial, n_step))

    if progressbar:
        printProgressBar(0, n_unit, prefix='Units:', length=50)

    for i, unit_ID in enumerate([unit_ID for unit_ID, unit_data, in units.iterrows()]):
        # Get the spike times for the unit
        unit_spike_times = spike_times[unit_ID]

        # Loop through trials and time
        for j, (start, end) in enumerate(zip(trial_start, trial_end)):  # Trials
            for k, time in enumerate(np.arange(start, end, stepSize)): # Time
                
                # Check if k is out of range. (This can happen because of different floating point rounding in int() and np.arange() functions i guess.)
                if k == n_bin:
                    break

                # Find the bin indices
                bin_start_idx = np.searchsorted(unit_spike_times, time)
                bin_end_idx = np.searchsorted(unit_spike_times, time+binSize)

                # Get the spikes in the time bin
                spikes_in_timebin = unit_spike_times[bin_start_idx:bin_end_idx]

                # Count the number of spikes in the time bin
                tensor[i, j, k] = len(spikes_in_timebin)

        if progressbar:
            printProgressBar(i + 1, n_unit, prefix='Units:', length=50)

    # Count the number of spikes in the tensor
    count = np.count_nonzero(tensor)
    print('Spike count in the data:', count)

    return tensor


def rasterplot(session, times):
    """
    Generate a raster plot for a given session and times.

    Args:
        session (Session): The session object.
        times (DataFrame): The times DataFrame.

    Returns:
        None
    """
    first_drifting_grating_presentation_id = times['stimulus_presentation_id'].values[0]
    plot_times = times[times['stimulus_presentation_id'] == first_drifting_grating_presentation_id]

    fig = raster_plot(plot_times, title=f'spike raster for stimulus presentation {first_drifting_grating_presentation_id}')
    plt.show()

    # Print out this presentation also
    session.stimulus_presentations.loc[first_drifting_grating_presentation_id]


def stimulus_duration(session, stimulus_block):
    """
    Plot the histogram of stimulus durations for a given session and stimulus block.

    Args:
        session (Session): The session object.
        stimulus_block (int): The stimulus block.

    Returns:
        None
    """
    stimulus_presentations = session.stimulus_presentations[session.stimulus_presentations['active'] == True &
                                                            session.stimulus_presentations['stimulus_block'] == stimulus_block &
                                                            session.stimulus_presentations['omitted'] == False]
    stimulus_presentations['duration'].hist(bins=100)
    plt.xlabel('Flash Duration (s)')
    plt.ylabel('Count')
    plt.show()
