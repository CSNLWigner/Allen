from pathlib import Path
import allensdk
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache

import yaml
params = yaml.safe_load(open('params.yaml'))['cache']

def cache_allen():
    
    print("cache location:", params['location'])
    
    output_dir = Path(params['location'])
    
    # Confirming your allensdk version
    print(f"Your allensdk version is: {allensdk.__version__}")

    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
        cache_dir=output_dir)
    
    # Print manifest status
    # print('all of the manifest files available for this dataset online:')
    # print(*cache.list_manifest_file_names(), sep='\n')
    available  = cache.latest_manifest_file()
    downloaded = cache.latest_downloaded_manifest_file()
    print('the most up-to-date manifest available: ', available )
    print('the most up-to-date manifest downloaded:', downloaded)
    if available != downloaded:
        get = input("Do you want to download the new manifest? (y/n) ")
        if get == "y":
            cache.load_latest_manifest()
            return cache
    elif params['force_download'] == True:
        cache.load_latest_manifest()
        return cache
    else:
        cache.current_manifest()
        return cache
    


# from urllib.parse import urljoin

# def get_behavior_session_url(ecephys_session_id: int) -> str:
#     '''Example:
#     >>> print(get_behavior_session_url(1052533639))'''
    
#     hostname = "https://visual-behavior-neuropixels-data.s3.us-west-2.amazonaws.com"
#     object_key = f"visual-behavior-neuropixels/ecephys_sessions/ecephys_session_{ecephys_session_id}.nwb"
#     return urljoin(hostname, object_key)

# from ruamel.yaml import YAML
# yaml = YAML(typ="safe")

