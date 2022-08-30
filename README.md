# Automated-Screening

This python repo explores using various machine learning methods to perform semi-automated screening

## Installation

This python repo requires Python3 (run using version 3.10.6) and pip (22.2.2).

To install all package requirements run:

`pip install -r requirements.txt`

## Method Names

The options for method names are:
| Method Name  | Description                                                             |
| ------------ | ----------------------------------------------------------------------- |
| `control`    | Random order of model / control (used for `simulated-screening.py`)     |
| `simple`     | TF-IDF Embeddings                                                       |
| `scibert`    | Average last layer of scibert word embeddings for document embedding    |
| `bloom-350m` | Average last layer of bloom-350m word embeddings for document embedding |
| `bloom`      | Average last layer of bloom word embeddings for document embedding      |

## Calculate embeddings

The first step is to calculate embeddings for the testing data by running:

`python calculate-embeddings.py <METHOD_NAME>`

Pkl files will be saved inside the dataset folders representing a pandas dataframe of embeddings for that model.

## Simulate screening

This step involves simulating the title/abstract screening process:

`python simulated-screening.py <METHOD_NAME>`

CSV files will be saved in the root directory representing the stats of that model on a dataset.

## Stats analysis

This is a jupyter notebook to view stats graphs.
