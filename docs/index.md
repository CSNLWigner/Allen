# Welcome to Allen Project

This is the documentation for Allen Project.

## Introduction

Functional connectivity analysis in the visual hierarchy is the basement towards studying continuous video stimuli processing. In this project, we aim to analyze the functional connection between the first two stage of the mouse visual cortex, specifically the V1 and LM. The neural activity data is obtained from the [Visual Coding - Neuropixels dataset](https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html). I employed Reduced Rank Regression (RRR) to uncover the underlying relationships between these brain regions. This analysis will provide valuable insights into the neural mechanisms underlying visual processing.

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

Once the dependencies are installed, you can run the project by executing the main script:

```sh
dvc repro
```

Feel free to explore the different modules and customize the code to suit your needs. For more detailed instructions, refer to the project's documentation.
