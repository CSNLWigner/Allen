# Allen Project

Functional connection analysis of the V1 and LM in mouse.

## Introduction

Functional connectivity analysis in the visual hierarchy is the basement towards studying continuous video stimuli processing. In this project, we aim to analyze the functional connection between the first two stage of the mouse visual cortex, specifically the V1 and LM. The neural activity data is obtained from the [Visual Coding - Neuropixels dataset](https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html). I employed Reduced Rank Regression (RRR) to uncover the underlying relationships between these brain regions. This analysis will provide valuable insights into the neural mechanisms underlying visual processing. To learn more about the project and its documentation, visit [here](https://CSNLWigner.github.io/Allen/).

## Table of Contents

- [Allen Project](#allen-project)
  - [Introduction](#introduction)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Project Structure](#project-structure)
  - [Contributing](#contributing)
  - [Acknowledgments](#acknowledgments)

## Installation


1. Clone the repository:
    ```sh
    git clone https://github.com/CSNLWigner/Allen
    ```
2. Navigate to the project directory:
    ```sh
    cd Allen_project
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To run the analysis, use the following command:

```sh
dvc repro
```

## Project Structure

- `data/`: This directory contains all the raw and preprocessed data files controlled by DVC.
- `analysis/`: This directory contains the analysis-specific functions.
- `results/`: This directory contains the output of the analyses.
- `figures/`: This directory contains any figures or plots generated the analyses.
- `utils/`: This directory contains utility scripts that are used across multiple analysis scripts.
- `docs/`: This directory contains the documentation for the project.
- `notes/`: This directory contains useful notes made during the project.
- `fromm3/`: This directory contains results copied from the M3 project.
- `pipelines/`: This directory contains the different pipeline scripts to the analyses.

## Contributing

Contributions are welcome! Please read the contributing guidelines before making any changes.

## Acknowledgments

- [Allen Institute for Brain Science](https://alleninstitute.org/)
- [Contributors](https://github.com/CSNLWigner/Allen/graphs/contributors)
- Gergő Orbán
