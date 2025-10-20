# Heart Disease Analysis (CMSE 830 Project)

This repository contains an exploratory data analysis (EDA) project for the UCI Heart Disease datasets. The goal is to combine four location-specific datasets (Cleveland, Long Beach, Hungarian, Switzerland), clean and impute missing values (MICE for continuous variables and KNN for categorical variables), and provide visualizations and interactive exploration via a Streamlit app.

## Project Overview

This project performs the following main steps:

1. Load the four location-specific UCI heart disease data files and concatenate into a combined dataset.
2. Replace dataset-specific missing-value markers (e.g., `?`, `9`, `-9`) with proper NaNs.
3. Drop exact duplicates, label-encode the `location` column, and compute missingness statistics.
4. Impute missing continuous variables (`trestbps`, `chol`, `thalach`, `oldpeak`) using MICE (IterativeImputer) with RobustScaler.
5. Impute categorical variables (`fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`) using KNNImputer with StandardScaler.
6. Reassign appropriate dtypes and generate exploratory visualizations (correlation heatmaps, box plots, bar chart, scatter plot, and violin plot).

## Contents

- `heart_disease_Notebook.ipynb` — Jupyter notebook that walks through the full analysis, including data loading, cleaning, imputation, and visualizations; including their respective outputs along with the associated code.
- `heart_disease_Streamlit.py` — Streamlit application that reproduces key parts of the notebook in an interactive dashboard (filters in the sidebar, tabs for preprocessing, missingness, imputation analysis, and visualizations).
- `heart_disease_Datasets/` — expected location for the raw UCI dataset files: `cleveland.data`, `long_beach.data`, `hungarian.data`, `switzerland.data`.
- `requirements.txt` — Python package dependencies used by the project.

## Setup Instructions

1. Create and activate a Python virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Place the UCI dataset files under `heart_disease_Datasets/` with the expected filenames:

- `cleveland.data`
- `long_beach.data`
- `hungarian.data`
- `switzerland.data`

4. Launch the Streamlit app:

```bash
streamlit run heart_disease_Streamlit.py
```

The app will open in your default browser and provide interactive filters and visualizations.

## Notes and interpretation guidance

- Imputation increases the amount of usable data but may alter distributional properties (e.g., reduce apparent variance, shift medians, and strengthen correlations). Interpret post-imputation results with caution and, where possible, validate using sensitivity analyses.
- The notebook and Streamlit app use identical preprocessing/imputation logic; Streamlit caches results for responsiveness.

## Development

- Python 3.11+ recommended.
- Core libraries: streamlit, pandas, numpy, matplotlib, seaborn, scikit-learn, plotly, and statsmodels.
