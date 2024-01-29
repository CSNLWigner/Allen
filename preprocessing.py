
from analyses.data_preprocessing import get_area_responses
from utils.download_allen import cache_allen
from utils.data_io import save_pickle
import yaml

params = yaml.safe_load(open('params.yaml'))['preprocess']

# An arbitrary session from the Allen Neuropixel dataset
session_id = 1064644573  # 1052533639
cache = cache_allen()
session = cache.get_ecephys_session(ecephys_session_id=session_id)

for area in params['areas']:
    area_responses = get_area_responses(session, area, session_block=params['stimulus-block'], log=True)
    save_pickle(area_responses, f'{params["stimulus-block"]}_block_{area}-responses', path='data/area-responses')