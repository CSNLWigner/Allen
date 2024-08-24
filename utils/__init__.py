# __init__.py

"""
Tool-package for data analysis and visualization (plotting).

Main submodules
----------
debug : Debugging tool.
megaplot : This tool contains a class for creating and managing subplots in matplotlib.
plots : Plotting tool for all kinds of plots.

More submodules
----------
data_io : Data input/output tools.
download_allen: Download and initialize data from the Allen Brain Observatory.
layers: Tools for simple layer analysis.
directDownload: Direct download tools for retrieving the download links for all sessions in a given manifest file from the Allen Brain Observatory.
neuropixel: Tools for working with Neuropixel data from the Allen Institute.
ccf_volumes: Tools for assigning cortical layers to channels and units based on the Allen Brain Atlas Common Coordinate Framework (CCF) volumes.
utils: Utility tools for various tasks
feature_functions: Feature functions for data analysis.
"""

# Package metadata

__version__ = "1.0"
__author__ = "Zsombor Szabó"

# Import submodules

from . import debug
from . import megaplot
from . import plots

from . import data_io, download_allen, layers, directDownload, neuropixel, ccf_volumes, utils, feature_functions

__all__ = ['debug', 'megaplot', 'plots', 'data_io', 'download_allen', 'layers', 'directDownload', 'neuropixel', 'ccf_volumes', 'utils', 'feature_functions']
