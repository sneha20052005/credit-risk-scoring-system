# Credit Risk Scoring System

## Project Overview

This project develops a machine-learning system to estimate the likelihood of customers experiencing payment difficulties using customer application data.

The project includes data inspection, preprocessing, model training, evaluation, and a Streamlit dashboard.

## Dataset

- **Dataset:** Home Credit Default Risk
- **Target column:** `TARGET`
- `TARGET = 1`: Recorded payment difficulties
- `TARGET = 0`: No recorded payment difficulties

Dataset source: https://www.kaggle.com/competitions/home-credit-default-risk/data

The dataset is not included in this repository.

## Models

Two classification models were evaluated:

1. Logistic Regression — baseline model
2. LightGBM — tree-based model

## Current Results

| Model | ROC-AUC | KS Score |
|---|---:|---:|
| Logistic Regression | 0.6238 | 18.96 |
| LightGBM | 0.7608 | 39.22 |

LightGBM outperformed the Logistic Regression baseline on both reported metrics. The ROC-AUC result remains below the project's provisional target of 0.80.

## Dashboard Features

- Model performance metrics
- Customer risk distribution
- Low, medium, and high illustrative risk bands
- Sample customer predictions
- Feature importance visualization
- Preliminary average predicted-risk comparison across gender groups

## Technology Stack

- Python
- Pandas
- Scikit-learn
- LightGBM
- Matplotlib
- Jupyter Notebook
- Streamlit

## Running the Dashboard

Install the required packages:

```bash
python -m pip install streamlit pandas scikit-learn lightgbm matplotlib
```

From the project root, run:

```bash
python -m streamlit run app.py
```

Ensure the required output CSV files are available in the `outputs/` folder.

## Limitations

- The current LightGBM ROC-AUC is 0.7608.
- Risk-band thresholds are illustrative and are not validated lending cutoffs.
- Feature importance does not establish causation.
- Differences in predicted risk across gender groups require deeper fairness analysis.
- Model performance and fairness should be validated before any real lending use.
