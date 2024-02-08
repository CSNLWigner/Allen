from allensdk.brain_observatory.ecephys.visualization import plot_mean_waveforms, plot_spike_counts, raster_plot
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
# import yaml

# params = yaml.safe_load(open('params.yaml'))['cache']

# cache = cache_allen(location=params['location'],
#                     force_download=params['force-download'])

# # get the metadata tables
# units_table = cache.get_unit_table()

# channels_table = cache.get_channel_table()

# probes_table = cache.get_probe_table()

# behavior_sessions_table = cache.get_behavior_session_table()

# ecephys_sessions_table = cache.get_ecephys_session_table()

"""
This dataset contains ephys recording sessions from 3 genotypes (C57BL6J, VIP-IRES-CrexAi32 and SST-IRES-CrexAi32). For each mouse, two recordings were made on consecutive days. One of these sessions used the image set that was familiar to the mouse from training. The other session used a novel image set containing two familiar images from training and six new images that the mouse had never seen.
"""

def get_unit_channels(session, log_all_areas=False) -> pd.DataFrame:

    # session.metadata

    "Merging unit and channel dataframes will give us CCF coordinates for each unit"
    units = session.get_units()
    channels = session.get_channels()
    unit_channels = units.merge(channels, left_on='peak_channel_id', right_index=True)

    "which brain structures were recorded during this session"
    brain_areas_recorded = unit_channels.value_counts('structure_acronym')
    
    if log_all_areas:
        print(brain_areas_recorded, '\n', brain_areas_recorded)

    return unit_channels


def makePSTH(spikes, startTimes, windowDur, binSize=0.001):
    """
    Convenience function to compute the Peri-Stimulus Time Histogram (PSTH).

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
    for i, start in enumerate(startTimes):  # enumerate through trials
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start+windowDur)
        counts = counts + np.histogram(spikes[startInd:endInd]-start, bins)[0]

    counts = counts/startTimes.size
    return counts/binSize, bins


"""
Acronyms:
Primary Visual Area (VISp), Posterolateral visual area (VISpl), Laterointermediate
area (VISli), Lateral visual area (VISl), Anteromedial visual area (VISal), Laterolateral anterior visual area (VISlla), Rostrolateral visual area
(VISrl), Anteromedial visual area (VISam), Posteromedial visual area (VISpm), Medial visual area (VISm), Mediomedial anterior visual area
(VISmma), Mediomedial posterior visual area (VISmmp). 

VISp:   V1
VISpl:  ?
VISli:  ? (visually guided behav)
VISl:   V2 (LM)?
VISal:  anteromedial cuneus
VISlla: ?
VISrl:  ?(visually guided behav)
VISam:  anteromedial cuneus
VISpm:  V4-V5(MT), PMC (associative area)
VISm:   V6 (medial motion area) https://pages.ucsd.edu/~msereno/papers/V6Motion09.pdf

other Allen experiment:
VISp     93
CA1      85
VISrl    58
VISl     56
VISam    49
VISal    43
SUB      41
CA3      33
DG       32
VISpm    17
LGv      16
LP        9
LGd       8
TH        4
ZI        4
CA2       3
POL       3
ProS      1
https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quickstart.html
"""



def get_stimulus_presentations(session):
    "We can get the times when the image changes occurred from the stimulus presentations table"

    stimulus_presentations = session.stimulus_presentations
        
    return stimulus_presentations
    

def get_area_units(session, area_of_interest) -> pd.DataFrame:
    """
    Retrieves the units in a specific area of interest.

    Args:
        session (str): The session identifier.
        area_of_interest (str): The acronym of the area of interest.

    Returns:
        pd.DataFrame: A DataFrame containing the units in the specified area.
    """
    
    # first let's sort our units by depth
    unit_channels = get_unit_channels(session)
    unit_channels = unit_channels.sort_values(
        'probe_vertical_position', ascending=False)

    # now we'll filter them
    good_unit_filter = ((unit_channels['snr'] > 1) &
                        (unit_channels['isi_violations'] < 1) & # If anyone spikes in the refracter period, they will be excluded
                        (unit_channels['firing_rate'] > 0.1)) # When they are fires too sparse, then they cannot involved in the time of the trial

    good_units = unit_channels.loc[good_unit_filter]
    
    area_units = good_units[good_units['structure_acronym']
                            == area_of_interest]
    
    return area_units


def get_area_change_responses(session, area_of_interest) -> np.ndarray:
    """
    Grab spike times and calculate the change response for 'good' units in V1. Note that how you filter units will depend on your analysis
    
    VISp:   V1
    VISpl:  ?
    VISli:  ? (visually guided behav)
    VISl:   ?
    VISal:  anteromedial cuneus
    VISlla: ?
    VISrl:  ?(visually guided behav)
    VISam:  anteromedial cuneus
    VISpm:  V4-V5(MT), PMC (associative area)
    VISm:   V6 (medial motion area) https://pages.ucsd.edu/~msereno/papers/V6Motion09.pdf
    """
    
    stimulus_presentations = get_stimulus_presentations(session)
    
    """
    'stimulus_block', 'image_name', 'duration', 'start_time', 'end_time',
    'start_frame', 'end_frame', 'is_change', 'is_image_novel', 'omitted',
    'flashes_since_change', 'trials_id', 'contrast', 'active',
    'is_sham_change', 'rewarded', 'stimulus_name', 'temporal_frequency',
    'position_y', 'position_x', 'spatial_frequency', 'stimulus_index',
    'color', 'orientation'
    
    See: Allen_change.py **stimulus_presentation_table** function
    """
    change_times = stimulus_presentations[stimulus_presentations['active'] &
                                            stimulus_presentations['is_change']]['start_time'].values  # CHANGE = the image identity changed for this stimulus presentation. ACTIVE + CHANGE = The mouse was rewarded for licking within the response window.

    # Here's where we loop through the units in our area of interest and compute their PSTHs
    area_change_responses = []
    area_units = get_area_units(area_of_interest=area_of_interest)
    spike_times = session.spike_times
    time_before_change = 1
    duration = 2.5
    for iu, unit in area_units.iterrows():
        unit_spike_times = spike_times[iu]
        unit_change_response, bins = makePSTH(unit_spike_times,
                                            change_times-time_before_change,
                                            duration, binSize=0.01)
        area_change_responses.append(unit_change_response)
    area_change_responses = np.array(area_change_responses)
    
    return area_change_responses
    

def get_area_receptive_fields(spike_times, stimulus_presentations, area_of_interest) -> list:
    """
    Get stimulus presentation data for the receptive field mapping stimulus (gabors).
    
    There are many trials, and only in the second block are gabors for the receptive fields.
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
                                            stim_times+0.01,
                                            0.2, binSize=0.001)
                unit_rf[iy, ix] = unit_response.mean() # along stimulus change
        return unit_rf


    area_rfs = []
    area_units = get_area_units(area_of_interest=area_of_interest)
    for iu, unit in area_units.iterrows():
        unit_spike_times = spike_times[iu]
        unit_rf = find_rf(unit_spike_times, xs, ys)
        area_rfs.append(unit_rf)
    
    # area_rfs = np.array(area_rfs)
    
    return area_rfs


def optotagging(opto_table, spike_times, area_of_interest) -> np.ndarray:
    """
    Since this is an SST mouse, we should see putative SST+ interneurons that are activated during our optotagging protocol. Let's load the optotagging stimulus table and plot PSTHs triggered on the laser onset. For more examples and useful info about optotagging, you can check out the Visual Coding Neuropixels Optagging notebook here (though note that not all the functionality in the visual coding SDK will work for this dataset).
    
    We use 2 different laser **waveforms**: a short square pulse that's **10 ms** long and a half-period cosine that's 1 second long.
    We drive each at three light **levels**, giving us 6 total conditions
    
    most units don't respond to the short laser pulse.
    Note that the activity occurring at the onset and offset of the laser is artifactual and should be excluded from analysis!

    opto_table = session.optotagging_table
    """

    print(opto_table.head())

    duration = opto_table.duration.min()  # get the short pulses
    level = opto_table.level.max()  # and the high power trials

    if 'VIS' not in area_of_interest: raise ValueError('To optotagging, the interested area must be a visual area')
    cortical_units = get_area_units(area_of_interest=area_of_interest)

    # Sort trials by duration and level
    opto_times = opto_table.loc[(opto_table['duration'] == duration) &
                                (opto_table['level'] == level)]['start_time'].values

    time_before = 0.01  # seconds to take before the laser start for PSTH
    duration = 0.03  # total duration of trial for PSTH in seconds
    binSize = 0.001  # 1ms bin size for PSTH
    opto_response = []
    unit_id = []
    for iu, unit in cortical_units.iterrows():
        unit_spike_times = spike_times[iu]
        unit_response, bins = makePSTH(unit_spike_times,
                                    opto_times-time_before, duration,
                                    binSize=binSize)

        opto_response.append(unit_response)
        unit_id.append(iu)

    opto_response = np.array(opto_response)

    return opto_response


def get_response_magnitudes(opto_response):
    """
    Calculate the response magnitudes of optogenetic stimulation.

    Parameters:
    opto_response (numpy.ndarray): Array containing the optogenetic response data.

    Returns:
    numpy.ndarray: Array of response magnitudes calculated for each trial.
    """ # docstring

    baseline_window = slice(0, 9)  # baseline epoch
    response_window = slice(11, 18)  # laser epoch

    response_magnitudes = np.mean(opto_response[:, response_window], axis=1) \
                        - np.mean(opto_response[:, baseline_window], axis=1)
    
    return response_magnitudes


def get_average_unit_responses(units, spike_times, trial_start, duration=0.03, binSize=0.001):
    """
    Calculate the unit responses for each unit in the given units DataFrame.

    Parameters:
    units (DataFrame): DataFrame containing information about the units.
    spike_times (list): List of spike times for each unit.
    trial_start (float): Start time of the trial.
    duration (float): Total duration of trial for PSTH in seconds. Default is 0.03.
    binSize (float): Bin size for PSTH in seconds. Default is 0.001.

    Returns:
    numpy.ndarray: Array containing the unit responses, shape (units, duration/binSize)
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


def get_unit_responses(units, spike_times, trial_start, duration=0.250, binSize=0.050):
    """
    Calculate the unit responses for each unit in the given units DataFrame.

    Parameters:
    units (DataFrame): DataFrame containing information about the units.
    spike_times (list): List of spike times for each unit.
    trial_start (float): Start time of the trial.
    duration (float): Total duration of trial for PSTH in seconds. Default is 0.03.
    binSize (float): Bin size for PSTH in seconds. Default is 0.001.

    Returns:
    numpy.ndarray: Array containing the unit responses, shape (feature, sample, time)
    """
    
    # def convert_to_tensor(spike_times, binSize, duration):
    n_unit = len(units)
    n_trial = len(trial_start)
    n_bin = int(duration / binSize)

    tensor = np.zeros((n_unit, n_trial, n_bin))

    for i, unit_ID in enumerate([unit_ID for unit_ID, unit_data, in units.iterrows()]):  # Units
        # print(unit_ID)
        unit_spike_times = spike_times[unit_ID]
        # print(type(unit_spike_times))
        # print(unit_spike_times.shape)
        for j, start in enumerate(trial_start):  # Trials
            for k, time in enumerate(np.arange(start, start + duration, binSize)): # Time
                
                # print(unit_spike_times, time)
                if k == n_bin: # This can happen because of different floating point rounding in int() and np.arange() functions i guess.
                    break # print('Warning: k is out of range')
                
                bin_start_idx = np.searchsorted(unit_spike_times, time)
                bin_end_idx = np.searchsorted(unit_spike_times, time+binSize)
                
                spikes_in_timebin = unit_spike_times[bin_start_idx:bin_end_idx]
                
                tensor[i, j, k] = len(spikes_in_timebin) # spike_count

    count = np.count_nonzero(tensor)
    print('Spike count in the data:', count)

    return tensor
    

def rasterplot(session, times):
    first_drifting_grating_presentation_id = times['stimulus_presentation_id'].values[0]
    plot_times = times[times['stimulus_presentation_id'] == first_drifting_grating_presentation_id]

    fig = raster_plot(plot_times, title=f'spike raster for stimulus presentation {first_drifting_grating_presentation_id}')
    plt.show()

    # also print out this presentation
    session.stimulus_presentations.loc[first_drifting_grating_presentation_id]
    
def stimulus_duration(session, stimulus_block):
    stimulus_presentations = session.stimulus_presentations[session.stimulus_presentations['active']==True &
                                                            session.stimulus_presentations['stimulus_block'] == stimulus_block &
                                                            session.stimulus_presentations['omitted']==False]
    stimulus_presentations['duration'].hist(bins=100)
    plt.xlabel('Flash Duration (s)')
    plt.ylabel('Count')
    plt.show()