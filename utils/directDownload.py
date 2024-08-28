# utils/directDownload.py

"""
Utility tool for retrieving the download links for all sessions in a given manifest file.

Functions:
- retrieve_link(session_id): Retrieves the download link for the given session ID.
- get_download_links(manifest_path): Retrieves the download links for all sessions in the given manifest file.

"""

from allensdk.brain_observatory.ecephys.ecephys_project_api.rma_engine import \
    RmaEngine
from allensdk.brain_observatory.ecephys.ecephys_project_api.utilities import \
    build_and_execute
from allensdk.brain_observatory.ecephys.ecephys_project_cache import \
    EcephysProjectCache

rma_engine = RmaEngine(scheme="http", host="api.brain-map.org")

def retrieve_link(session_id):
    """
    Retrieves the download link for the given session ID.

    Parameters:
        session_id (int): The ID of the session.

    Returns:
        download_link (str): The download link for the session.
    """

    well_known_files = build_and_execute(
        (
            "criteria=model::WellKnownFile"
            ",rma::criteria,well_known_file_type[name$eq'EcephysNwb']"
            "[attachable_type$eq'EcephysSession']"
            r"[attachable_id$eq{{session_id}}]"
        ),
        engine=rma_engine.get_rma_tabular,
        session_id=session_id
    )

    return 'http://api.brain-map.org/' + well_known_files['download_link'].iloc[0]

def get_download_links(manifest_path):
    """
    Retrieves the download links for all sessions in the given manifest file.

    Parameters:
        manifest_path (str): The path to the manifest file.

    Returns:
        download_links (list): A list of download links for each session.
    """

    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

    sessions = cache.get_session_table()

    download_links = [retrieve_link(session_id)
                    for session_id in sessions.index.values]

    for session_id, link in zip(sessions.index.values, download_links):
        print(f"Session ID: {session_id}, Download Link: {link}")

    return download_links