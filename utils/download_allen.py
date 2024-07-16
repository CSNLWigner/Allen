import os
import shutil
from pathlib import Path

import allensdk
import yaml
from allensdk.brain_observatory.ecephys.ecephys_project_cache import \
    EcephysProjectCache

params = yaml.safe_load(open('params.yaml'))['cache']

def cacheData(location=""):
    """
    Caches data from the Allen Brain Observatory.

    Args:
        location (str): The location where the data will be cached. If not provided, the default location will be used.

    Returns:
        cache (EcephysProjectCache): The cache object containing the cached data.
    """
    
    # Location
    if location == "":
        location = params['location']
    print("Cache location:", params['location'])
    output_dir = Path(location)
    
    # Confirming your allensdk version
    print(f"Your allensdk version is: {allensdk.__version__}")

    # cache = EcephysProjectCache.from_warehouse(manifest=os.path.join(output_dir, "manifest.json"))
    cache = EcephysProjectCache(manifest=os.path.join(output_dir, "manifest.json"))
    
    # Print the manifest ids
    # print('all of the manifest files available for this dataset online:')
    # sessions = cache.get_session_table()
    # print(*sessions.index, sep='\n')
    
    return cache
    

def completeDownloadData(sessions, cache, output_dir, download_lfp=False):
    '''
    Downloads the complete dataset for analysis.

    Args:
        sessions (DataFrame): A DataFrame containing session information.
        cache (Cache): An object representing the cache.
        output_dir (str): The directory where the downloaded data will be stored.
        download_lfp (bool, optional): Whether to download LFP data files. Defaults to False.

    Notes:
        - This function downloads the complete dataset by iterating over each session in the provided DataFrame.
        - It checks for the presence of the complete file to ensure that the download is not interrupted due to an unreliable connection.
        - Make sure that you have enough space available in your cache directory before running this code.
          You'll need around 855 GB for the whole dataset, and 147 GB if you're not downloading the LFP data files.
    '''
    for session_id, row in sessions.iterrows():

        truncated_file = True
        directory = os.path.join(output_dir + '/session_' + str(session_id))

        while truncated_file:
            session = cache.get_session_data(session_id)
            try:
                print(session.specimen_name)
                truncated_file = False
            except OSError:
                shutil.rmtree(directory)
                print(" Truncated spikes file, re-downloading")

        print('Downloaded session ' + str(session_id))

        if download_lfp:
            for probe_id, probe in session.probes.iterrows():

                print(' ' + probe.description)
                truncated_lfp = True

                while truncated_lfp:
                    try:
                        lfp = session.get_lfp(probe_id)
                        truncated_lfp = False
                    except OSError:
                        fname = directory + '/probe_' + str(probe_id) + '_lfp.nwb'
                        os.remove(fname)
                        print("  Truncated LFP file, re-downloading")
                    except ValueError:
                        print("  LFP file not found.")
                        truncated_lfp = False
