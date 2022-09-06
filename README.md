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
| `control`    | Random order of model / control    |
| `bow`     | BOW Embeddings                                                       |
| `tfidf`     | TF-IDF Embeddings                                                       |
| `doc2vec`     | Doc2Vec Embeddings                                                       |
| `scibert`    | Average last layer of scibert word embeddings for document embedding    |
| `scibert-average`    | Average 5 last layers of scibert word embeddings for document embedding    |
| `scibert-concat`    | Concatenated last layer of scibert word embeddings for document embedding    |
| `bloom-350m` | Average last layer of bloom-350m word embeddings for document embedding |
| `bloom-350m-average` | Average of 5 last layers of bloom-350m word embeddings for document embedding |
| `bloom-350m-concat` | Concatenated last layer of bloom-350m word embeddings for document embedding |

## Calculate stats

To calculate the stats using a simulated screening process run:

`python main.py <METHOD_NAME>`

If cuda is available it will be automatically used

## Stats analysis

This is a jupyter notebook to view/generate stats graphs.
