# DVC framework

DVC is a version control system developed for data science and machine learning projects. It enhances reproducibility and parameter tuning by tracking data, code, and models. DVC is designed to work with Git repositories.

## Table Of Contents

- [DVC framework](#dvc-framework)
  - [Table Of Contents](#table-of-contents)
  - [Usage](#usage)
  - [Parameters](#parameters)
  - [Pipeline](#pipeline)
  - [Experiments](#experiments)

## Usage

There is a `.dvc` directory that stores metadata and configuration files. The `dvc` command is used to interact with the DVC framework. The following are some common commands:

- `dvc run -n <name> -d <dependencies> -o <outputs> <command>`: Runs a command and generates outputs.
- `dvc repro`: Reproduces the entire pipeline.
- `dvc repro -fs <stage>`: Forces the pipeline to run only a specific stage.

After running a command, the DVC framework generates a `dvc.lock` file that contains information about the pipeline. This file should be committed to the Git repository. (This file contains versioned and indexed information about the pipeline, parameters, codes, data and results.)

## Parameters

DVC uses parameters to manage the experiments. These are the hyperparameters that are used in the pipeline. They are stored in a `params.yaml` file.

Structure:

- `cache`: parameters for data caching on the file system
- `neuropixel`: historical parameters
- `load`: parameters for loading and initial preprocessing of a specific session and location of the data
- `preprocess`: parameters for preprocessing the data
- `cca`: parameters for the CCA analysis
- `rrr`: parameters for the RRR analysis
- `rrr-time-slice`: parameters for the `rrr-time-slice` analysis
- `crosstime`: parameters for the `crosstime` analysis
- `rrr-param-search`: parameters for the RRR parameter search: `rrr-param-search` analysis (This is a very important analysis for determining the optimal parameters for the RRR analysis. For further information, see the `rrr-param-search` analysis and [experiment history](<notion/Allen project d3cfe5aab8384495b58fba8a47eeadcc.md#multiple-session-param-search>))
- `layer-rank`: parameters for the `layer-rank` analysis
- `interaction-layers`: determines the interaction structure between V1 and LM

## Pipeline

The pipeline is a sequence of stages that are executed in order. Each stage is a separate step in the pipeline. The pipeline with the stages are defined in the `dvc.yaml` file. The pipeline is executed by running the `dvc repro` command.

## Experiments

Find the pipelines for the different experiments and the corresponding parameters in the pipelines folder or in the [experiment history](<notion/Allen project d3cfe5aab8384495b58fba8a47eeadcc.md>) documentation.