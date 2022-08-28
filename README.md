# Automated-Screening

This repo explores using various machine learning methods to perform semi-automated screening

## Method Names

The options for method names are:
| Method Name  | Description                                                             |
| ------------ | ----------------------------------------------------------------------- |
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
