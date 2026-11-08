# ZnO MOV Failure Prediction Using Machine Learning

**Electrical Engineering Research | Waveform Feature Engineering | Binary Classification**

A reproducible, privacy-conscious demonstration of a machine learning workflow inspired by my Diploma Thesis in Electrical and Computer Engineering at the **Technical University of Crete**.

The project explores the use of features extracted from electrical waveforms to classify the condition of **zinc-oxide metal-oxide varistors (ZnO MOVs)** as `PASS` or `FAIL`. The broader thesis also investigated electrical transient behaviour using **ATP/EMTP** simulations.

> **Scope:** This public repository contains an illustrative implementation using **synthetically generated waveforms only**. It does not include confidential experimental measurements, original trained models, or thesis evaluation artifacts. Demo metrics must not be interpreted as experimental validation.

## Engineering Context

ZnO MOVs are nonlinear components used in surge-protection applications. Evaluating their condition from electrical measurements motivates a workflow combining signal processing, engineered features, and supervised classification.

This repository demonstrates the software structure of that workflow. It does **not** establish a field-ready diagnostic system or claim performance on unseen real-world MOV measurements.

## System Architecture

See the [System Architecture Diagram](docs/ARCHITECTURE.md) for the complete machine learning pipeline and its components.

## Methodology

```text
Synthetic waveform generation
            |
            v
Waveform summary feature extraction
            |
            v
Stratified train/test split (75% / 25%)
            |
            v
Logistic Regression | RBF SVM | Random Forest
            |
            v
Accuracy, precision, recall, and F1-score
```

### Demonstration data

The generator creates **200 synthetic waveforms**, each represented by **16 voltage samples** and a `PASS` or `FAIL` label. The signals are simplified damped sinusoidal curves with random noise and class-dependent amplitude. They are designed for software demonstration, **not** as a physical or statistically validated model of MOV degradation.

### Feature engineering

The training script extracts eight summary features from each waveform:

- Peak value and minimum value
- Mean and standard deviation
- Root mean square (RMS)
- Signal energy
- Peak-to-peak amplitude
- Mean absolute value

### Models

Three classifiers are trained and compared using scikit-learn:

| Model | Implementation |
|---|---|
| Logistic Regression | Standardized features + logistic classifier |
| Support Vector Machine | Standardized features + RBF kernel |
| Random Forest | Tree ensemble classifier |

The example uses a **stratified 75/25 train/test split** with a fixed random seed for repeatability. The split evaluates performance on held-out *synthetic* samples from the same generator, not independent experimental measurements.

## Synthetic Demo Results

Results observed from the demonstration run with **50 synthetic test samples** (25 `PASS`, 25 `FAIL`):

| Model | Accuracy | Weighted F1-score |
|---|---:|---:|
| Logistic Regression | 0.92 | 0.92 |
| Support Vector Machine (RBF) | 0.92 | 0.92 |
| Random Forest | 0.92 | 0.92 |

**Interpretation:** These numbers illustrate the execution and comparison of the public code. They are **not thesis results**, and they do not support conclusions about real-world diagnostic accuracy.

## Technology Stack

**Python 3.12**, NumPy, pandas, scikit-learn, pytest, and JupyterLab. **ATP/EMTP** was part of the broader thesis research context; simulation models and original outputs are not distributed here.

## Quick Start

From the repository root:

```bash
python -m venv .venv
```

Activate the virtual environment using your operating system's preferred method, then run:

```bash
python -m pip install -r requirements.txt
python src/generate_synthetic_data.py
python src/train_models.py
python -m pytest tests -v
```

The generator writes `data/synthetic/waveforms.csv` locally. The CSV is excluded from version control and can be regenerated at any time. For additional instructions, see [Setup and Execution](docs/SETUP.md).

## Repository Structure

```text
ZnO-MOV-Failure-Prediction/
├── README.md
├── requirements.txt
├── src/
│   ├── generate_synthetic_data.py
│   └── train_models.py
├── tests/
│   └── test_synthetic_data.py
└── docs/
    └── SETUP.md
```

The generated `data/synthetic/waveforms.csv` file is local and intentionally not tracked by Git.

## Research Integrity, Privacy, and Limitations

- **Confidentiality:** Original experimental datasets, trained models, simulation artifacts, and detailed thesis outputs are not published.
- **Synthetic-only evidence:** The public signals are simplified examples and should not be treated as representative measured MOV responses.
- **Original evaluation limitation:** The thesis evaluation reused some `FAIL` sample identities across dataset partitions. Accordingly, those results should not be described as a fully independent generalization test.
- **No deployment claim:** This repository demonstrates a research-inspired workflow, not a certified surge-arrester diagnostic tool.

## Author

**Iraklis Tsaousidis**  
Diploma Thesis — Electrical and Computer Engineering  
**Technical University of Crete**

---

*This repository is an educational and portfolio-oriented demonstration of methodology; it is not a release of the confidential thesis dataset or its experimental findings.*
