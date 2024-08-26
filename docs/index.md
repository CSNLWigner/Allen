# Welcome to Allen Project

This is the documentation for Allen Project.

## Introduction

Functional connectivity analysis in the visual hierarchy is the basement towards studying continuous video stimuli processing. In this project, we aim to analyze the functional connection between the first two stage of the mouse visual cortex, specifically the V1 and LM. The neural activity data is obtained from the [Visual Coding - Neuropixels dataset](https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html). I employed Reduced Rank Regression (RRR) to uncover the underlying relationships between these brain regions. This analysis will provide valuable insights into the neural mechanisms underlying visual processing.

The experiments are generated using the [DVC framework](DVC.md) and all the previous experiments are documented in the [experiment history](<notion/Allen project d3cfe5aab8384495b58fba8a47eeadcc.md>).

## Table Of Contents

The documentation follows the best practice for
project documentation as described by Daniele Procida
in the [Diátaxis documentation framework](https://diataxis.fr/)
and consists of four separate parts:

1. [Tutorials](tutorials.md)
2. [How-To Guides](how-to-guides.md)
3. [Reference](reference.md)
4. [Explanation](explanation.md)

Quickly find what you're looking for depending on
your use case by looking at the different pages.

## Project Overview

::: allen

## Getting Started

To get started, make sure you have Python 3.7 or higher installed on your machine. Then, clone the project repository from GitHub:

```sh
git clone https://github.com/CSNLWigner/Allen
```

Next, navigate to the project directory:

```sh
cd Allen_project
```

Install the required dependencies using pip:

```sh
pip3 install -r requirements.txt
```

Once the dependencies are installed, you can run the last experiment by the [DVC framework](DVC.md) using the following command:

```sh
dvc repro
```

This command will reproduce the entire pipeline and generate the results.

# Reproduce previous experiments

All the experiments are stored in the `pipelines` directory or in the [experiment history](<notion/Allen project d3cfe5aab8384495b58fba8a47eeadcc.md>) documentation. To reproduce the experiments we use the by the [DVC framework](DVC.md). To reproduce a specific experiment, follow these steps:

, follow these steps:

1. Copy the dvc file from the `pipelines` directory or from a specific experiment in the [experiment history](<notion/Allen project d3cfe5aab8384495b58fba8a47eeadcc.md>) to the `dvc.yaml` file in the root directory.
2. Copy the params file from the `pipelines` directory or from a specific experiment in the [experiment history](<notion/Allen project d3cfe5aab8384495b58fba8a47eeadcc.md>) to the `params.yaml` file in the root directory.
3. Run the following command:

```sh
   dvc repro
```

4. If there are plotting stage in the pipeline, the visual results can be found in the `figures` directory (see the `outs` in the last stage).
