# ZnO MOV Failure Prediction Using Machine Learning

## Overview

This repository presents a reproducible demonstration of the
machine learning methodology developed during my Diploma Thesis
in Electrical and Computer Engineering at the Technical
University of Crete.

The research focuses on predicting the condition of Zinc Oxide
Metal Oxide Varistors (ZnO MOVs) using features extracted from
electrical waveform measurements.

The project also incorporates ATP/EMTP simulations to study
electrical transient behaviour.

## Technologies

- Python
- NumPy and pandas
- scikit-learn
- Jupyter Notebook
- ATP/EMTP

## Machine Learning Methodology

The original research workflow includes:

1. Waveform data processing
2. Feature extraction from electrical measurements
3. Dataset preparation
4. Logistic Regression
5. Support Vector Machine
6. Random Forest
7. Model evaluation and comparison

The classification task distinguishes between PASS and FAIL
conditions.

## Repository Structure

- `src/` - Reusable Python modules
- `notebooks/` - Demonstration notebooks
- `data/synthetic/` - Synthetic demonstration datasets
- `docs/` - Methodology and technical documentation
- `figures/` - Demonstration figures
- `tests/` - Automated tests

## Data Privacy

The original experimental datasets are confidential and are
not included in this repository.

Any demonstration datasets will be synthetically generated.
Synthetic results must not be interpreted as experimental
validation of the original research.

The original trained models and experimental outputs are
also excluded.

## Research Limitations

The original experimental evaluation included reused FAIL
sample identities across dataset partitions. Consequently,
the reported evaluation should not be interpreted as
fully independent generalisation testing.

The public implementation is intended to demonstrate
the methodology rather than reproduce confidential results.

## Status

Repository under development.

## Author

Iraklis Tsaousidis

Diploma Thesis - Electrical and Computer Engineering

Technical University of Crete
