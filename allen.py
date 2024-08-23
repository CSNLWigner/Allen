# allen.py

"""
Allen Project
============

Allen Project is a Python package for data analysis and visualization.

Subpackages
-------------

Allen Project includes subpackages:

- ``utils``: Utilities for data analysis and visualization.
- ``analyses``: Analyses for data analysis.

Utility tools
-------------

The following main utility tools are available (in the ``utils`` subpackage):

- ``debug``: Debugging tool.
- ``megaplot``: This tool contains a class for creating and managing subplots in matplotlib.
- ``plots``: Plotting tool for all kinds of plots.

Preprocessing tools
-------------------

- ``preprocessing``: Preprocessing tool for further data preprocessing after data initialization.
- ``behav-preprocessing``: Preprocessing tool for behavioral data.
- ``layer-assignment-analysis``: Analysis tool for layer assignment.

Experiments
-----------

The following two-step experiments are available (make extra attention to the file names, as they are not always the same as the experiment names):

- **layer-rank**: This experiment is about the rank of the layers in the RRR model. Corresponding codes are ``layer-rank-analysis`` and ``layer-rank-plot``.
- **layer-interaction**: This experiment is about the interaction between layers in a neural network. Corresponding codes are ``layer-interaction-analysis`` and ``layer-interaction-plot``.
- **rank-along-time**: This experiment is about the rank along time in the RRR model. Corresponding codes are ``rank-along-time-analysis`` and ``rank-along-time-plot``.
- **crosstime**: This experiment is about the cross-time RRR model. Corresponding codes are ``crosstime-analysis`` and ``crosstime-plot``.
- **rrr-time-slice**: This experiment is about the time slice of the RRR model. Corresponding codes are ``rrr-time-slice-analysis`` and ``rrr-time-slice-plot``.
- **lag-along-time**: This experiment is about the lag along time in the RRR model. Corresponding codes are ``lag-along-time-analysis`` and ``lags-along-time-plot``.
- **time-lag-search**: This experiment is about the time lag search in the RRR model. Corresponding codes are ``time-lag-search-analysis`` and ``time-lag-search-plot``.
- **cv-time-lag**: This experiment is about the cross-validation of the time lag in the RRR model. Corresponding codes are ``cv-time-lag-analysis`` and ``cv-time-lag-plot``.
- **time-lag**: This experiment is about the time lag in the RRR model. Corresponding codes are ``time_lag_analysis`` and ``time_lag_plot``.
- **cv-rank-time**: This experiment is about the cross-validation of the rank along time in the RRR model. Corresponding codes are ``cv-rank-time-analysis`` and ``cv-rank-time-plot``.
- **rrr-rank**: This experiment is about the rank of the RRR model. Corresponding codes are ``rrr-rank-analysis`` and ``rrr-rank-plot``.
- **rrr**: This experiment is about the RRR model. Corresponding codes are ``rrr_analysis`` and ``rrr_plot``.
- **pca**: This experiment is about the PCA model. Corresponding codes are ``pca-analysis`` and ``pca-plot``.
- **cca**: This experiment is about the CCA model. Corresponding codes are ``cca_analysis`` and ``cca_plot``.

The other experiment utilities are:

- ``histograms``: This utility contains functions for creating histograms.
- ``control-models``: This utility contains functions for creating control models.
- ``cv-rank-cross-time``: This utility contains functions for cross-validation of the rank along time.
- ``rrr-score-time-plot``: This utility contains functions for plotting the score along time in the RRR model.
- ``max-lags-along-time-plot``: This utility contains functions for plotting the maximum lags along time in the RRR model.
- ``crosstime-timeslice-plot``: This utility contains functions for plotting the cross-time time slice in the RRR model.
- ``save-slices``: This utility contains functions for saving slices in the crosstime RRR model.
- ``multiple-timeslices``: This utility contains functions for creating multiple time slices in the RRR model.
- ``multiple-timeslices-layers``: This utility contains functions for creating multiple time slices in the RRR model for each layer.
- ``layer-interaction-maxValues``: This utility contains functions for plotting the maximum values of the layer interaction in the RRR model.
- ``layer-interaction-stats``: This utility contains functions for plotting the statistics of the layer interaction in the RRR model.

"""
