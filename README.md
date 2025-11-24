# Utilizing the EnzyExtract Database and Environmental Conditions to Predict $k_{cat}$ and $K_M$ Values

**CHE1147 Final Project - Group 1:** Josh Goldman, Abiali Badani, Maxim Kirby

![Python](https://img.shields.io/badge/python-3.10-blue)
![Dependency Manager](https://img.shields.io/badge/dependency-uv-purple)
![Status](https://img.shields.io/badge/status-complete-green)
<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This repository contains the source code, datasets, and analysis for our study on predicting enzyme kinetic parameters ($k_{cat}$ and $K_M$) using the EnzyExtract database. By leveraging Machine Learning (XGBoost, MLP, and Linear Regression) and advanced feature engineering (ESM2 embeddings, Morgan Fingerprints), we aim to improve prediction accuracy by incorporating often-overlooked environmental data such as pH and temperature.



## Table of Contents
- [Project Overview](#project-overview)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Usage & Reproducibility](#usage--reproducibility)
- [Methodology](#methodology)
- [Acknowledgements](#acknowledgements)



## Project Overview

Model-guided engineering of enzymatic pathways is restricted by the scarcity of kinetic data. While the recent EnzyExtract database provides significantly more data points than traditional databases like BRENDA, it has yet to be fully utilized for predictive modeling.

This project develops a supervised ML pipeline to:
1.  **Parse and Clean** the EnzyExtract database.
2.  **Engineer Features** using structural data (ESM2 protein embeddings, Morgan Fingerprints) and environmental conditions (pH, Temp).
3.  **Benchmark Models** comparing Linear Regression, MLP, and XGBoost.
4.  **Interpret Results** using SHAP analysis to understand feature importance.

## Key Findings
*   **XGBoost** outperformed Linear and MLP models, achieving the lowest error and highest correlation ($R^2 \approx 0.64$).
*   **Advanced Featurization** (ESM2 + Morgan Fingerprints) significantly improved model performance compared to physicochemical features alone.
*   **Feature Importance** was determined by SHAP analysis revealed that sequence length and molecular weight are critical predictors. Environmental factors (pH/Temp) were utilized more heavily by non-linear models that capture complex interactions.

## Repository Structure

```text
├── Makefile             <- Automation commands (setup, run)
├── README.md            <- Project documentation
├── pyproject.toml       <- Project configuration and dependencies
├── uv.lock              <- Exact dependency lockfile for reproducibility
├── data
│   ├── raw              <- Original EnzyExtractDB parquet file
│   ├── interim          <- Cleaned data
│   └── processed        <- Featurized data and CV splits (folds 1-5)
├── models               <- Saved HPO configurations (.json)
├── notebooks            <- Analysis notebooks and figure generation
├── reports              <- Generated CSV summaries and Figures
└── src                  <- Source code for the project
    ├── config.py        <- Configuration variables
    ├── dataset.py       <- Data loading and splitting logic
    ├── features.py      <- Feature engineering (RDKit, ESM2, etc.)
    ├── hpo.py           <- Hyperparameter optimization scripts
    ├── main.py          <- Main entry point for the pipeline
    ├── modeling/        <- Training and inference logic
    ├── parse.py         <- Raw data cleaning and parsing
    └── metrics_plot.py  <- Plotting and utility scripts
```



## Installation & Setup
We use uv for fast, reliable, and cross-platform dependency management.

### 1. Clone Repo
```bash
git clone https://github.com/Galvanized-Heart/CHE1147_Final_Project.git
cd CHE1147_Final_Project
```

### 2. Install uv (if not installed)
```bash
# Install astral uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### 3. Install dependencies using Makefile command
```bash
# Full intial install environment
make setup
```



## Usage and Reproducibility
To run the entire pipeline for data parsing, feature engineering, training and results generation run:
```bash
make all
```



## Methodology

### Data Source
We utilized the EnzyExtract database, containing ~40,000 usable data points after cleaning. 
- **Inputs:** protein sequences, substrate SMILES, temperature, and pH. 
- **Targets:** $k_{cat}$ and $K_M$.

### Feature Engineering
- **Basic Features:** MW, logP, TPSA, H-bond donors/acceptors (RDKit), MW, sequence length, Instability Index, pI (BioPython).
- **Advanced Features:** Morgan Fingerprints (2048-bit, radius 2), ESM2 (8M parameter model) embeddings reduced to 320-dim vectors.

### Models
- **Baseline:** `LinearRegressor` (sklearn).
- **Neural Network:** `MLPRegressor` (sklearn) optimized via `BayesSearchCV` (sklearn).
- **XGBoost:** `XGBRegressor` (xgboost) optimized via `BayesSearchCV` (sklearn).



---
## Acknowledgements
This project was completed for CHE1147 at the University of Toronto by Josh Goldman, Abiali Badani and Maxim Kirby.
