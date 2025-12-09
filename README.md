# Heart Disease Analysis

**[Publically Available Web App (Streamlit) Link](https://sharm152-cmse830-fds-heart-disease-streamlit-oj5flv.streamlit.app)** of Standard Version (Logistic Regression + Random Forest)

## Project Overview

This project performs the following main steps:

1. **Data Loading & Preprocessing**: Load the four location-specific datasets (Cleveland, Long Beach, Hungarian, Switzerland) found from the UCI Machine Learning Repository and concatenate into a combined dataset.
2. **Handle Missing Values**: Replace dataset-specific missing-value markers (e.g., `?`, `9`, `-9`) with proper NaNs.
3. **Data Cleaning**: Drop exact duplicates, label-encode the `location` column, and compute missingness statistics.
4. **Imputation - Continuous Variables**: Impute missing continuous variables (`trestbps`, `chol`, `thalach`, `oldpeak`) using MICE (IterativeImputer) with RobustScaler.
5. **Imputation - Categorical Variables**: Impute categorical variables (`fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`) using KNNImputer with StandardScaler.
   . **Exploratory Data Analysis**: Generate interactive visualizations including correlation heatmaps, box plots, grouped bar charts, scatter plots, violin plots, categorical distributions, and parallel coordinates plots.
6. **Feature Engineering**: Apply Singular Value Decomposition (SVD) to reduce the 14-dimensional feature space to a user-defined number of principal components (1-14).
7. **Classification Modeling**: Train and evaluate two classification models (Logistic Regression and Random Forest) on the PCA-reduced dataset with interactive parameter controls.

### Performance Optimizations

- **Caching Strategy**: Expensive computations (SVD, correlations, model training) are cached using `@st.cache_data` to enable responsive parameter adjustments.

## Repository Contents

- `heart_disease_Notebook.ipynb` — Jupyter notebook that walks through the full analysis, including data loading, cleaning, imputation, and visualizations; including their respective outputs along with the associated code.
- `heart_disease_Streamlit.py` — Streamlit application that reproduces key parts of the notebook in an interactive dashboard with seven pages covering data exploration, preprocessing, imputation, feature engineering, and classification modeling (2 models: Logistic Regression and Random Forest).
- `heart_disease_Streamlit_XGBoost.py` — Extended Streamlit application with the same dashboard features plus an additional third classification model (XGBoost) for comprehensive model comparison.
- `heart_disease_Datasets` — Location for the raw UCI dataset files: `cleveland.data`, `long_beach.data`, `hungarian.data`, `switzerland.data`.
- `requirements.txt` — Python package dependencies used by the project.

## Application Versions

### Standard Version (`heart_disease_Streamlit.py`)

The standard version includes a complete analysis pipeline with two classification models:

- **Models**: Logistic Regression and Random Forest Classifier
- **Features**: All seven pages with interactive visualizations, feature engineering with SVD, and real-time model development/evaluation
- **Dependencies**: Core packages only (no XGBoost required)
- **Use Case**: Lightweight deployment, faster model training, ideal for learning and exploration

### Extended Version with XGBoost (`heart_disease_Streamlit_XGBoost.py`)

The extended version adds gradient boosting capabilities for comprehensive model comparison:

- **Models**: Logistic Regression, Random Forest Classifier, and XGBoost Classifier
- **Features**: Identical to standard version with additional XGBoost integration
- **Dependencies**: Requires XGBoost 2.0.0+ in addition to core packages
- **Use Case**: Production deployments, advanced model comparison, when gradient boosting performance is critical
- **Performance Note**: XGBoost training typically takes 2-3 minutes due to sequential tree building

### Comparison Notes

- **Key Differences (in `heart_disease_Streamlit_XGBoost.py`)**:
  - Three-column confusion matrix display (one per model) instead of two
  - Three-column classification reports for side-by-side model comparison
  - Three-stage progress bar during model training (33%, 66%, 100%)
  - Gradient boosting metrics (XGBoost) for enhanced prediction capability
- **Key Similarity**:
  - Both versions are identical in structure and functionality, differing only in the classification models used

## Web App (Streamlit) Usage

- **Publically Available Web App (Streamlit) Note**:
  - For ease of use, utiize the already publically available web app of *Standard Version (Logistic Regression + Random Forest)* using the link found at the top of this README.md right below the title "Heart Disease Analysis"
  - Those interested in using the web app locally and/or working with the *Extended Version (Logistic Regression + Random Forest + XGBoost)* should refer the **Setup (Locally) Instructions** section found below

1. **Explore Data**: Start with the "Data Background" page to understand the datasets and features
2. **Review Preprocessing**: Visit "Initial Preprocessing Steps" to see data cleaning, deduplication, and encoding
3. **Assess Imputation**: Check "Handling Missing Values" to understand imputation methods applied
4. **Analyze Correlations**: Use "Imputation Analysis" to verify data quality and correlation preservation
5. **Visualize Relationships**: Explore "Interactive Visualizations" to understand feature distributions and disease patterns
6. **Reduce Dimensionality**: Review "Feature Engineering" to understand SVD variance explained and optimal component selection
7. **Build & Evaluate Models**: Use "Classification Models" page to:
   - Adjust principal components, train-test split, and random state
   - View real-time confusion matrices for each model (2 in standard version, 3 in XGBoost version)
   - Review classification metrics (accuracy, precision, recall, F1-score)
   - Compare model performance side-by-side

### Interactive Filtering

At any point, use the sidebar controls to:

- Filter datasets by age range (affects all pages)
- Select specific geographic locations to include/exclude

## Setup (Locally) Instructions

1. Create and activate a Python virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Ensure the UCI dataset files are under `heart_disease_Datasets/` with the following filenames:

- `cleveland.data`
- `long_beach.data`
- `hungarian.data`
- `switzerland.data`

4. Launch the Streamlit app:

**Standard Version (Logistic Regression + Random Forest):**

```bash
streamlit run heart_disease_Streamlit.py
```

**Extended Version with XGBoost (Logistic Regression + Random Forest + XGBoost):**

```bash
streamlit run heart_disease_Streamlit_XGBoost.py
```

## Development

- **Python 3.11+** recommended
- **Core Libraries**: `streamlit` (1.50.0+), `pandas` (2.1.4+), `numpy` (1.26.4+), `matplotlib` (3.8.0+), `seaborn` (0.13.2+), `scikit-learn` (1.2.2+), `scipy` (1.11.0+), `plotly` (5.18.0+), `statsmodels` (0.14.0+), `xgboost` (2.0.0+)
