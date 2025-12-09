import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer # "IterativeImputer" is Dependent on This
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import io

st.set_page_config(page_title="Heart Disease EDA", layout="wide")
st.title("📈 Heart Disease Analysis")

### Helper functions for "load_and_prepare()"
# Function to inverse the scale transform of the data
def inverse_scale(data, scaler):
    return pd.DataFrame(scaler.inverse_transform(data), columns=data.columns, index=data.index)

# Function to calculate NaN stats in given data
def calc_nan_stats(data):
    nan_stats = pd.DataFrame(data.isna().sum(), columns=["NaN_Count"])
    nan_stats["NaN_Percent"] = np.round(nan_stats["NaN_Count"]/data.shape[0], 4)*100
    nan_stats["NaN_Percent"] = nan_stats["NaN_Percent"].astype(str) + "%"
    return nan_stats

@st.cache_data
def load_and_prepare():
    # Read datasets (expect files under heart_disease/ folder)
    df_cleveland = pd.read_csv("heart_disease_Datasets/cleveland.data", sep=',')
    df_long_beach = pd.read_csv("heart_disease_Datasets/long_beach.data", sep=',')
    df_hungarian = pd.read_csv("heart_disease_Datasets/hungarian.data", sep='\s+')
    df_switzerland = pd.read_csv("heart_disease_Datasets/switzerland.data", sep=',')

    # The original notebook assumes column names consistent with UCI Heart Disease dataset
    col_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "num"
    ]

    for df in [df_cleveland, df_long_beach, df_hungarian, df_switzerland]:
        df.columns = col_names

    # Replace missing value markers used in the notebook
    df_cleveland.replace("?", np.nan, inplace=True)
    df_cleveland["ca"] = df_cleveland["ca"].replace(9.0, np.nan)

    df_long_beach.replace("?", np.nan, inplace=True)
    df_long_beach["ca"] = df_long_beach["ca"].replace(9.0, np.nan)

    df_hungarian.replace(-9.0, np.nan, inplace=True)
    df_hungarian["ca"] = df_hungarian["ca"].replace(9.0, np.nan)

    df_switzerland.replace("?", np.nan, inplace=True)
    df_switzerland["ca"] = df_switzerland["ca"].replace(9.0, np.nan)

    # Add location column
    df_cleveland["location"] = "Cleveland"
    df_long_beach["location"] = "Long Beach"
    df_hungarian["location"] = "Hungary"
    df_switzerland["location"] = "Switzerland"

    # Combine the individual datasets
    df_combine = pd.concat([df_cleveland, df_long_beach, df_hungarian, df_switzerland])
    df_combine.reset_index(inplace=True)
    df_combine.drop("index", axis=1, inplace=True)

    # Combined dataset - initial info
    buffer = io.StringIO()
    df_combine.info(buf=buffer)
    buffer.seek(0)
    combine_init_info = buffer

    # Reset dtypes as notebook does
    df_combine = df_combine.astype({"age": float, "sex": int, "cp": int, "trestbps": float, "chol": float, 
                                    "fbs": float, "restecg": float, "thalach": float, "exang": float, "oldpeak": float, 
                                    "slope": float, "ca": float, "thal": float, "num": int, "location": str})
    
    # Combined dataset - new info and describe
    buffer = io.StringIO()
    df_combine.info(buf=buffer)
    buffer.seek(0)
    combine_new_info = buffer
    combine_new_describe = df_combine.describe()

    # Locate exact duplicates and then drop them
    dup_mask = df_combine.duplicated()
    df_no_dup = df_combine.drop_duplicates()

    # Encode location for df_combine and create DataFrame showing the mapping
    location_encode = pd.DataFrame(df_no_dup["location"])
    location_encode.rename(columns={"location":"location_str"}, inplace=True)
    # Utilize LabelEncoder() for the location variable
    le = LabelEncoder()
    df_no_dup["location"] = le.fit_transform(df_no_dup["location"])
    location_encode["location_encoded"] = df_no_dup["location"]
    location_encode.drop_duplicates(inplace=True)
    location_encode.reset_index(inplace=True)
    location_encode.drop("index", axis=1, inplace=True)
    
    # Prepare copies for MICE
    df_half_clean = df_no_dup.copy()
    df_before_MICE = df_half_clean.copy()
    # MICE for continuous columns
    continuous_cols = ["trestbps", "chol", "thalach", "oldpeak"]
    df_after_MICE = df_before_MICE[continuous_cols + ["age", "sex", "cp", "num", "location"]]
    # Scale features using RobustScaler
    rob_scaler = RobustScaler()
    X_rob_scaled = pd.DataFrame(rob_scaler.fit_transform(df_before_MICE), columns=df_before_MICE.columns, 
                                index=df_before_MICE.index)
    # Perform MICE imputation
    mice_imputer = IterativeImputer(random_state=42, max_iter=50)
    X_scaled_mice = pd.DataFrame(mice_imputer.fit_transform(X_rob_scaled), 
                                columns=X_rob_scaled.columns, index=X_rob_scaled.index)
    # Replace original missing (continuous) values with imputed values
    df_after_MICE.loc[X_rob_scaled.index, continuous_cols] = inverse_scale(data=X_scaled_mice, scaler=rob_scaler)
    # Final MICE step
    df_half_clean[continuous_cols] = df_after_MICE[continuous_cols]

    # Describe MICE and calculate NaN stats
    summary_before_MICE = df_before_MICE[continuous_cols].describe()
    summary_after_MICE = df_after_MICE[continuous_cols].describe()
    stats_MICE = calc_nan_stats(df_half_clean)

    # Prepare copies for KNN
    df_before_KNN = df_half_clean.copy()
    df_after_KNN = df_before_KNN.copy() # Utilize the entire dataset
    # KNN for categorical columns
    categorical_cols = ["fbs", "restecg", "exang", "slope", "ca", "thal"]
    # Scale features using StandardScaler
    std_scaler = StandardScaler()
    X_std_scaled = pd.DataFrame(std_scaler.fit_transform(df_before_KNN), columns=df_before_KNN.columns, 
                                index=df_before_KNN.index)
    # Perform KNN imputation
    knn_imputer = KNNImputer(n_neighbors=1)
    X_scaled_knn = pd.DataFrame(knn_imputer.fit_transform(X_std_scaled), 
                                columns=X_std_scaled.columns, index=X_std_scaled.index)
    # Replace original missing (categorical) values with imputed values
    df_after_KNN = inverse_scale(data=X_scaled_knn, scaler=std_scaler)
    # Final KNN step (dataset is now clean)
    df_clean = df_after_KNN.copy()

    # Describe KNN and calculate NaN stats
    summary_before_KNN = df_before_KNN[categorical_cols].describe()
    summary_after_KNN = df_after_KNN[categorical_cols].describe()
    stats_KNN = calc_nan_stats(df_clean)
    
    # Assign types similar to notebook
    df_clean = df_clean.astype({
        "age": float, "sex": int, "cp": int, "trestbps": float, "chol": float,
        "fbs": int, "restecg": int, "thalach": float, "exang": int, "oldpeak": float,
        "slope": int, "ca": int, "thal": int, "num": int, "location": int
    })



    return {
        "df_cleveland": df_cleveland,
        "df_long_beach": df_long_beach,
        "df_hungarian": df_hungarian,
        "df_switzerland": df_switzerland,
        "df_combine": df_combine,
        "combine_init_info": combine_init_info,
        "combine_new_info": combine_new_info,
        "combine_new_describe": combine_new_describe,
        "dup_mask": dup_mask,
        "df_no_dup": df_no_dup,
        "location_encode": location_encode,
        "continuous_cols": continuous_cols,
        "summary_before_MICE": summary_before_MICE,
        "summary_after_MICE": summary_after_MICE,
        "stats_MICE": stats_MICE,
        "df_half_clean": df_half_clean,
        "categorical_cols": categorical_cols,
        "summary_before_KNN": summary_before_KNN,
        "summary_after_KNN": summary_after_KNN,
        "stats_KNN": stats_KNN,
        "df_clean": df_clean
    }


artifacts = load_and_prepare()


# --- Slider to filter datasets by age ---
# Determine age bounds from the combined/no-dup dataset
try:
    _age_min = int(np.nanmin(artifacts['df_combine']['age']))
    _age_max = int(np.nanmax(artifacts['df_combine']['age']))
except Exception:
    # fallback if df_combine missing or malformed
    _age_min, _age_max = 0, 120

age_range = st.sidebar.slider("Filter by Age", min_value=_age_min, max_value=_age_max, value=(_age_min, _age_max))

# Apply filter to the main DataFrame artifacts so the rest of the app reflects selection.
def _apply_age_filter(df, col='age'):
    if col in df.columns:
        return df[(df[col] >= age_range[0]) & (df[col] <= age_range[1])]
    return df

# Filter common artifacts in-place (minimal and localized change)
if isinstance(artifacts, dict):
    for k in ['df_combine', 'df_no_dup', 'df_clean', 'df_cleveland', 'df_long_beach', 'df_hungarian', 'df_switzerland']:
        if k in artifacts and isinstance(artifacts[k], pd.DataFrame):
            try:
                artifacts[k] = _apply_age_filter(artifacts[k], col='age')
            except Exception:
                # keep original if something unexpected occurs
                pass
# End of minimal age slider addition


# --- Allow selecting (or de-selecting) locations to include ---
try:
    _loc_options = list(artifacts['df_combine']['location'].dropna().unique())
except Exception:
    _loc_options = []

selected_locations = st.sidebar.multiselect("Select (or Deselect) Locations to Include", options=_loc_options, 
                                            default=_loc_options)

# If user cleared selection, treat as 'all selected' to avoid empty displays
if not selected_locations:
    selected_locations = _loc_options

# Map selected string locations to encoded values if mapping exists
enc_selected = None
if 'location_encode' in artifacts and not artifacts['location_encode'].empty:
    try:
        enc_selected = artifacts['location_encode'][artifacts['location_encode']['location_str'].isin(selected_locations)]['location_encoded'].tolist()
    except Exception:
        enc_selected = None

def _apply_location_filter(df):
    if 'location' not in df.columns:
        return df
    # If location column is object/string, filter by strings
    try:
        if df['location'].dtype == object:
            return df[df['location'].isin(selected_locations)]
        # If integer encoded, use encoded values
        if np.issubdtype(df['location'].dtype, np.integer) and enc_selected is not None:
            return df[df['location'].isin(enc_selected)]
    except Exception:
        pass
    # Fallback: attempt to match by string representation
    try:
        return df[df['location'].astype(str).isin([str(s) for s in selected_locations])]
    except Exception:
        return df

# Apply location filter to the same artifact keys as before
if isinstance(artifacts, dict):
    for k in ['df_combine', 'df_no_dup', 'df_clean', 'df_cleveland', 'df_long_beach', 'df_hungarian', 'df_switzerland']:
        if k in artifacts and isinstance(artifacts[k], pd.DataFrame):
            try:
                artifacts[k] = _apply_location_filter(artifacts[k])
            except Exception:
                pass
# End of location-selection addition

page = st.sidebar.radio("Select Page", ["Data Background", "Initial Data Cleaning", "Handling Missing Values", "Imputation Analysis", 
                                        "Interactive Visuals (EDA)", "Feature Engineering", "Classification Models"])

if page == "Data Background":
    st.header("Data Background")

    tab1, tab2, tab3 = st.tabs(["Introduction & Variables", "Individual Datasets", "Combined Dataset"])

    with tab1:
        st.subheader("Introduction")
        st.markdown("""
        This project focuses on the various stages of heart disease as it combines four location-specific datasets (Cleveland, 
        Long Beach, Hungarian, Switzerland) found from the UCI Machine Learning Repository. Methodolgy involves the **imputation 
        of missing values** (MICE and KNN), **EDA via interactive visualizations**, **feature engineering through SVD**, and 
        **developing/evaluating classification models** (Logistic Regression and Random Forest). Specifically for models, users 
        can experiment with a different numbers of principal components, train-test splits, and random states to observe real-time 
        effects on model performance.
        
        **To analyze the Heart Disease Stages, our target variable will be `num` (described in-depth below).**
                    
        Notes Regarding Interactivity Throughout the Project:
        - You may filter the dataset by `age` using the slider on the left sidebar.
        - You may select (or deselect) the dataset by `location` using the multiselect on the left sidebar.
        """)

        st.subheader("Variables")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            1. `age` (Ratio): Age in years
            2. `sex` (Nominal):
                - 0: female
                - 1: male
            3. `cp` (Nominal): Chest pain type
                - 1: typical angina
                - 2: atypical angina
                - 3: non-anginal pain
                - 4: asymptomatic
            4. `trestbps` (Ratio): Resting blood pressure (in mm Hg on admission to the hospital)
            5. `chol` (Ratio): Serum cholestoral in mg/dl
            6. `fbs` (Nominal): Fasting blood sugar > 120 mg/dl
                - 0: false
                - 1: true
            7. `restecg` (Nominal): Resting electrocardiographic results
                - 0: normal
                - 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
                - 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
            """)
        with col2:
            st.markdown("""
            8. `thalach` (Ratio): Maximum heart rate achieved
            9. `exang` (Nominal): Exercise induced angina
                - 0: no
                - 1: yes
            10. `oldpeak` (Ratio): ST depression induced by exercise relative to rest
            11. `slope` (Ordinal): Slope of the peak exercise ST segment
                - 1: upsloping
                - 2: flat
                - 3: downsloping
            12. `ca` (Ratio): Number of major vessels (0-3) colored by fluoroscopy
            13. `thal` (Nominal): Results of a thallium stress test
                - 3: normal
                - 6: fixed defect
                - 7: reversable defect
            14. `num` (Ordinal): Diagnosis of heart disease (angiographic disease status)
                - 0: *no heart disease*
                - 1: *stage A* involves risk factors but no structural heart damage
                - 2: *stage B* includes structural damage without symptoms
                - 3: *stage C* involves structural damage and the presence of symptoms
                - 4: *stage D* is end-stage heart failure with severe symptoms that interfere with daily life
            """)

    with tab2:
        st.info("""
        Note: Manually added *location* column to each respective dataset prior to merging.
        """)

        st.subheader("Cleveland")
        st.dataframe(artifacts['df_cleveland'].head(10), height=388)
        st.subheader("Long Beach")
        st.dataframe(artifacts['df_long_beach'].head(10), height=388)
        st.subheader("Hungary")
        st.dataframe(artifacts['df_hungarian'].head(10), height=388)
        st.subheader("Switzerland")
        st.dataframe(artifacts['df_switzerland'].head(10), height=388)
    
    with tab3:
        st.subheader("Entirety of Combined Dataset (Raw Form)")
        st.dataframe(artifacts['df_combine'], height=500)

elif page == "Initial Preprocessing Steps":
    st.header("Initial Preprocessing Steps")
    tab1, tab2, tab3 = st.tabs(["Dataset Statistics", "Check Duplicates", "Location Encoding"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Initial Data Types")
            st.code(artifacts['combine_init_info'].getvalue(), language="text")
        with col2:
            st.subheader("Updated Data Types")
            st.code(artifacts['combine_new_info'].getvalue(), language="text")
        
        st.subheader("Updated Dataset Statistical Summary")
        st.dataframe(artifacts['combine_new_describe'], height=318)

    with tab2:
        dup_mask = artifacts["dup_mask"]
        df_with_dup = artifacts["df_combine"]

        st.subheader("Duplicate Rows")
        st.dataframe(df_with_dup[dup_mask])

        col1, col2 = st.columns(2)
        col1_no_dup = False
        with col1:
            try:
                st.subheader(f"Need to Remove Row at Index {df_with_dup[dup_mask].index[0]}")
                age_0 = df_with_dup[dup_mask].iloc[0, 0]
                cp_0 = df_with_dup[dup_mask].iloc[0, 2]
                trestbps_0 = df_with_dup[dup_mask].iloc[0, 3]
                st.dataframe(df_with_dup[(df_with_dup["age"] == age_0) & 
                                         (df_with_dup["cp"] == cp_0) & 
                                         (df_with_dup["trestbps"] == trestbps_0)])
            except IndexError:
                st.warning("No Duplicate Rows Found with Current Filters.")
                col1_no_dup = True
        with col2:
            try:
                st.subheader(f"Need to Remove Row at Index {df_with_dup[dup_mask].index[1]}")
                age_1 = df_with_dup[dup_mask].iloc[1, 0]
                cp_1 = df_with_dup[dup_mask].iloc[1, 2]
                trestbps_1 = df_with_dup[dup_mask].iloc[1, 3]
                st.dataframe(df_with_dup[(df_with_dup["age"] == age_1) & 
                                         (df_with_dup["cp"] == cp_1) & 
                                         (df_with_dup["trestbps"] == trestbps_1)])
            except IndexError:
                if col1_no_dup is False:
                    st.warning("Only ONE Duplicate Row Found with Current Filters.")

    with tab3:
        st.subheader("Location Encoding")
        col1, col2 = st.columns(2)
        with col1:
            buf = io.StringIO()
            artifacts['location_encode'].to_string(buf=buf)
            buf.seek(0)
            st.code(buf.getvalue(), language="text")
        with col2:
            st.info("""
            Using the encoding map to the left, we can see that the location column has been encoded with their
            corresponding values in the dataset view below.
            """)
        
        st.subheader("Dataset After Removing Duplicates and Encoding Location")
        st.dataframe(artifacts['df_no_dup'], height=500)

elif page == "Handling Missing Values":
    st.header("Handling Missing Values")
    tab1, tab2, tab3, tab4 = st.tabs(["Check Missing Values", "Imputation Part 1 (MICE)", 
                                      "Imputation Part 2 (KNN)", "Assign Data Types"])

    with tab1:
        st.subheader("Missingness Heatmap")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(artifacts['df_no_dup'].isna(), cmap="magma")
        st.pyplot(fig)

        st.info("""
        The heatmap above visualizes the presence of NaN values across all variables, with bright yellow representing
        missing data and black representing present data. The table below quantifies the count and percentage of
        missing values for each variable before any imputation methods are applied.
        """)

        st.subheader("Missingness Summary")
        nan_stats = pd.DataFrame(artifacts['df_no_dup'].isna().sum(), columns=["NaN_Count"]) 
        nan_stats["NaN_Percent"] = np.round(nan_stats["NaN_Count"]/artifacts['df_no_dup'].shape[0], 4)*100
        nan_stats["NaN_Percent"] = nan_stats["NaN_Percent"].astype(str) + "%"
        st.dataframe(nan_stats, height=562)

    with tab2:
        st.subheader("MICE (Imputation on Continuous Variables)")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Stats Before MICE")
            st.dataframe(artifacts["summary_before_MICE"])
        with col2:
            st.subheader("Stats After MICE")
            st.dataframe(artifacts["summary_after_MICE"])

        st.info("""
        The section above provides a statistical summary of the dataset before and after MICE imputation. The table
        below quantifies the count and percentage of missing values for each variable after applying MICE
        imputation.
                
        As you can see after MICE, the continuous columns (trestbps, chol, thalach, oldpeak) contain no missing values
        and preserve the distribution of the variables, maintaining stable summary statistics.
        """)
        st.dataframe(artifacts["stats_MICE"], height=562)
    
    with tab3:
        st.subheader("KNN (Imputation on Categorical Variables)")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Stats Before KNN")
            st.dataframe(artifacts["summary_before_KNN"])
        with col2:
            st.subheader("Stats After KNN")
            st.dataframe(artifacts["summary_after_KNN"])
        
        st.info("""
        The section above provides a statistical summary of the dataset before and after KNN imputation. The table
        below quantifies the count and percentage of missing values for each variable after applying KNN
        imputation.

        As you can see after KNN, the categorical columns (fbs, restecg, exang, slope, ca, thal) contain no missing
        values and preserve the distribution of the variables, maintaining stable summary statistics.
        """)
        st.dataframe(artifacts["stats_KNN"], height=562)

    with tab4:
        st.subheader("Assign Appropriate Data Types")
        col1, col2 = st.columns(2)
        with col1:
            buf = io.StringIO()
            artifacts['df_clean'].info(buf=buf)
            buf.seek(0)
            st.code(buf.getvalue(), language="text")
        with col2:
            st.info("""
            After completing the MICE and KNN imputation steps, we finalize the dataset by assigning appropriate data
            types; converting all continuous variables to *float64* and all categorical variables to *int64*.
            """)
        st.dataframe(artifacts['df_clean'], height=500)

elif page == "Imputation Analysis":
    st.header("Imputation Analysis")
    tab1, tab2 = st.tabs(["Correlation Comparison", "Outliers Check"])

    # Cached helper function for correlation computations
    @st.cache_data
    def compute_correlations():
        corr_before = np.round(artifacts['df_no_dup'].corr().values, 2)
        corr_after = np.round(artifacts['df_clean'].corr().values, 2)
        col_names = list(artifacts['df_clean'].columns)
        return corr_before, corr_after, col_names

    with tab1:
        st.subheader("Correlation Heatmaps (Before vs After MICE and KNN Imputation)")

        with st.spinner("Computing correlations..."):
            corr_before, corr_after, col_names = compute_correlations()
        
        fig = ff.create_annotated_heatmap(
            z=corr_before,
            x=col_names,
            y=col_names,
            colorscale="bluered"
        )
        fig.update_layout(title="Correlation Heatmap Before Imputation")
        st.plotly_chart(fig)

        st.info("""
        The after-imputation heatmap preserves the same overall pattern of strong positive (red) and strong negative
        (blue) pairwise relationships that appear in the before-imputation map, so the relative structure of the
        correlation matrix is maintained. Key high-correlation pairs remain high and negative pairs remain negative,
        with only small magnitude changes rather than sign flips, which shows the imputation process filled missing
        values without creating spurious associations.
        """)

        fig = ff.create_annotated_heatmap(
            z=corr_after,
            x=col_names,
            y=col_names,
            colorscale="bluered"
        )
        fig.update_layout(title="Correlation Heatmap After Imputation")
        st.plotly_chart(fig)

    with tab2:
        st.subheader("Box Plots (Before vs After MICE Imputation)")

        plot_df_before = artifacts['df_no_dup'][["age"] + artifacts['continuous_cols']].melt(var_name="variable", 
                                                                                             value_name="value")
        fig_box_before = px.box(plot_df_before, x="variable", y="value", title="Box Plot Before Imputation")
        fig_box_before.update_layout(xaxis_title="Continuous Variable", yaxis_title="Value", 
                                     yaxis=dict(range=[-25, None]))
        st.plotly_chart(fig_box_before)

        st.info("""
        The box plots for before and after MICE imputation show that the overall distributions (minimum and maximum
        values, median, and interquartiles) of the continuous variables (age, trestbps, chol, thalach, and oldpeak)
        remain consistent. This indicates that the imputation process preserved the original data’s statistical
        properties and variability without introducing bias or distortion.
        """)

        plot_df_after = artifacts['df_clean'][["age"] + artifacts['continuous_cols']].melt(var_name="variable", 
                                                                                           value_name="value")
        fig_box_after = px.box(plot_df_after, x="variable", y="value", title="Box Plot After Imputation")
        fig_box_after.update_layout(xaxis_title="Continuous Variable", yaxis_title="Value", 
                                    yaxis=dict(range=[-25, None]))
        st.plotly_chart(fig_box_after)

elif page == "Interactive Visualizations":
    st.header("Interactive Visualizations")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Heart Disease Stage by Location", "Age vs Maximum Heart Rate (thalach)", 
                                            "Heart Disease Stage vs Blood Flow to Heart (oldpeak)",
                                            "All Categorical Variables", "All Continuous Variables"])
    df_clean = artifacts['df_clean'].copy()
    with tab1:
        location_encode = artifacts['location_encode']
        # Unique num values
        num_categories = sorted(df_clean["num"].unique())
        # Count occurrences of each (num, location) pair
        counts = pd.crosstab(df_clean['num'], df_clean['location']).reindex(index=num_categories)
        # Re-assign city names to previously encoded "location" variable in "counts" DataFrame
        location_dict = location_encode.to_dict()
        counts.rename(columns=dict(zip(location_dict["location_encoded"].values(), 
                                       location_dict["location_str"].values())), 
                                       inplace=True)
        # Build grouped bar chart: x-axis = stages (num), separate bar trace for each source
        fig_bar_chart = go.Figure()
        for src_name in counts.columns:
            fig_bar_chart.add_trace(go.Bar(
                x=[str(i) for i in counts.index],
                y=counts[src_name].values,
                name=str(src_name),
                hovertemplate="Stage: %{x}<br>Source: " + str(src_name) + "<br>Count: %{y}<extra></extra>"
            ))
        fig_bar_chart.update_layout(
            title="Distribution of Heart Disease Stage by Location",
            xaxis_title="Heart Disease Stage",
            yaxis_title="Count",
            legend_title="Location",
        )
        st.plotly_chart(fig_bar_chart)

        st.info("""
        This grouped bar chart shows the counts of patients at each heart-disease stage (num) separated by dataset
        source (location). Each cluster on the x-axis corresponds to a disease stage (0–4), and within each cluster
        separate bars compare how many records from each origin (Cleveland, Long Beach, Hungary, Switzerland) fall into
        that stage. Cleveland and Hungary datasets show a very high proportion of no heart disease cases, while Long
        Beach and Switzerland datasets are more evenly distributed across stages.
        """)

    with tab2:
        fig_scatter = px.scatter(df_clean, x="age", y="thalach", color="num", trendline="ols", 
                                 color_continuous_scale="turbo", 
                                 labels={"age":"Age", "thalach":"Max Heart Rate", "num":"Disease Stage"}, 
                                 title="Scatter Plot of Age vs Maximum Heart Rate (thalach)")
        st.plotly_chart(fig_scatter)

        st.info("""
        This scatter plot displays individual patients' ages on the x-axis versus their maximum heart rate achieved
        (thalach) on the y-axis, with points colored by heart disease stage (num). An OLS trendline summarizes the
        overall relationship between age and max heart rate, while the color layering shows how disease stages are
        distributed across that relationship. You can observe how maximum heart rate tends to decrease with age, and
        how the clusters of each heart disease stage are non-partitioned.
        """)

    with tab3:
        fig_violin = px.violin(df_clean, x="num", y="oldpeak", box=True, points="all", 
                               labels={"num":"Heart Disease Stage", "oldpeak":"Blood Flow to Heart"}, 
                               title="Violin Plot of Heart Disease Stage vs Blood Flow to Heart " \
                               "During Physical Exertion (oldpeak)")
        st.plotly_chart(fig_violin)

        st.info("""
        This violin plot shows the distribution of blood flow to heart during physical exertion (oldpeak) across heart
        disease stages (num), with each violin representing a stage. Hovering over each stage's violin enables the user
        to view its respective minimum and maximum values, median, and interquartiles. As the heart disease stage increases,
        the distribution of oldpeak values shifts higher, indicating that patients with more severe heart disease
        experience greater ST depression during physical exertion (oldpeak).
        """)
    
    with tab4:
        # Define categorical variables
        categorical_vars = {"sex": "Gender", "cp": "Chest Pain Type",
                            "fbs": "Fasting Blood Sugar > 120", "restecg": "Resting ECG Results",
                            "exang": "Exercise Induced Angina", "slope": "ST Segment Slope",
                            "ca": "Number of Major Vessels","thal": "Thallium Stress Test Results"}

        # Create subplots with 2 rows and 4 columns
        fig_categorical = make_subplots(rows=2, cols=4,
                                        subplot_titles=[f"{k.upper()}<br>{v}" for k, v in categorical_vars.items()],
                                        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                                               [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}, {"type": "bar"}]])

        # Color palette
        colors = px.colors.qualitative.Set2
        # Add bar charts for each categorical variable
        col_idx = 1
        for row_num in [1, 2]:
            for col_num in [1, 2, 3, 4]:
                if col_idx <= len(categorical_vars):
                    var_name = list(categorical_vars.keys())[col_idx - 1]
                    # Calculate value counts
                    value_counts = df_clean[var_name].value_counts().sort_index()
                    # Add trace
                    fig_categorical.add_trace(go.Bar(x=[f"{x}" for x in value_counts.index], y=value_counts.values,
                                                     marker_color=colors[(col_idx - 1) % len(colors)], showlegend=False,
                                                     hovertemplate=f"<b>{var_name.upper()}</b><br>Category: %{{x}}<br>Count: %{{y}}<extra></extra>"),
                                                     row=row_num, col=col_num)
                    col_idx += 1

        # Update layout
        fig_categorical.update_layout(title_text="Distribution of All Categorical Variables in Dataset", height=600)

        # Update y-axis labels
        fig_categorical.update_yaxes(title_text="Count", row=1, col=1)
        fig_categorical.update_yaxes(title_text="Count", row=2, col=1)

        st.plotly_chart(fig_categorical)

        st.info("""
        This multi-panel visualization displays the frequency distribution of all eight categorical variables in the
        dataset across a 2x4 grid. Each subplot is a bar chart showing the count of patients in each category for
        variables like gender (sex), chest pain type (cp), fasting blood sugar (fbs), resting ECG results (restecg),
        exercise-induced angina (exang), ST segment slope (slope), number of major vessels (ca), and thallium stress
        test results (thal). This provides a comprehensive overview of how patients are distributed across all categorical
        features, which assists with identifying potential class imbalances.
        """)

    with tab5:
        fig_parallel = px.parallel_coordinates(df_clean, 
                                       dimensions=["age", "trestbps", "chol", "thalach", "oldpeak"],
                                       color="num",
                                       color_continuous_scale="turbo",
                                       labels={"age":"Age", "trestbps":"Resting BP", "chol":"Cholesterol",
                                               "thalach":"Max Heart Rate", "oldpeak":"Oldpeak", "num":"Disease Stage"},
                                       title="Parallel Coordinates Plot of All Continuous Variables in Dataset")

        # Add margins to prevent axis labels from being cut off in Streamlit
        fig_parallel.update_layout(margin=dict(l=25))

        st.plotly_chart(fig_parallel)

        st.info("""
        This parallel coordinates plot visualizes relationships across five continuous variables (age, resting blood
        pressure, cholesterol, maximum heart rate, and oldpeak) simultaneously. Each line represents a patient, colored
        by their heart disease stage (num). You can interact with the plot by dragging axis ranges to filter the data
        and identify patterns, such as observing which combinations of values are associated with specific disease
        stages. The color gradient (turbo) makes it easier to track patients as their values change across dimensions,
        revealing multivariate correlations and clusters within the heart disease progression.
        """)

elif page == "Feature Engineering":
    st.header("Feature Engineering")
    tab1, tab2 = st.tabs(["Singular Value Decomposition (SVD) Plots", "Explained Variances & Analysis Summary"])

    # Cached helper function for SVD computation
    @st.cache_data
    def compute_svd_fe():
        X = artifacts['df_clean'].drop(columns=["num"])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        U, s, Vt = np.linalg.svd(X_scaled, full_matrices=True)
        eigenvalues = s**2 / (len(X_scaled) - 1)
        explained_variance_ratio = eigenvalues / eigenvalues.sum()
        cumulative_variance = np.cumsum(explained_variance_ratio)
        return s, eigenvalues, explained_variance_ratio, cumulative_variance
    
    s, eigenvalues, explained_variance_ratio, cumulative_variance = compute_svd_fe()
    
    with tab1:
        # Create the 2x2 subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Singular values scree plot
        axes[0, 0].plot(range(1, len(s)+1), s, 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Component', fontsize=11)
        axes[0, 0].set_ylabel('Singular Value', fontsize=11)
        axes[0, 0].set_title('Scree Plot: Singular Values', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xticks(range(1, len(s)+1))
        
        # Eigenvalues scree plot
        axes[0, 1].plot(range(1, len(eigenvalues)+1), eigenvalues, 'o-', linewidth=2, markersize=8)
        axes[0, 1].axhline(y=1, color='red', linestyle='--', linewidth=2, label='Kaiser Criterion (λ=1)')
        axes[0, 1].set_xlabel('Component', fontsize=11)
        axes[0, 1].set_ylabel('Eigenvalue (λ)', fontsize=11)
        axes[0, 1].set_title('Scree Plot: Eigenvalues', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xticks(range(1, len(eigenvalues)+1))
        axes[0, 1].legend()
        
        # Individual variance explained
        axes[1, 0].plot(range(1, len(explained_variance_ratio)+1), explained_variance_ratio, 'o-', 
                        linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Component', fontsize=11)
        axes[1, 0].set_ylabel('Explained Variance Ratio', fontsize=11)
        axes[1, 0].set_title('Individual Variance Explained by Component', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylim(-0.005, max(explained_variance_ratio) * 1.1)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xticks(range(1, len(s)+1))
        
        # Cumulative variance plot
        axes[1, 1].plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'o-', 
                        linewidth=2, markersize=8, label='Cumulative Variance')
        axes[1, 1].axhline(y=0.50, color='red', linestyle='--', linewidth=2, label='50% Threshold')
        axes[1, 1].axhline(y=0.85, color='green', linestyle='--', linewidth=2, label='85% Threshold')
        axes[1, 1].set_xlabel('Component', fontsize=11)
        axes[1, 1].set_ylabel('Explained Variance Ratio', fontsize=11)
        axes[1, 1].set_title('Cumulative Explained Variance', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylim(-0.005, 1.05)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xticks(range(1, len(s)+1))
        axes[1, 1].legend(loc='lower right')
        
        # plt.tight_layout()
        st.pyplot(fig)
        
        st.info("""
        These four plots provide a comprehensive view of the SVD and variance explained by each Principal Component (PC):
        
        - **Singular Values Scree Plot (top-left)**: Shows the singular values from the SVD decomposition, which declines 
          as the component index increases, indicating diminishing variance.
        - **Eigenvalues Scree Plot (top-right)**: Displays eigenvalues with the Kaiser criterion (λ=1) threshold. Components 
          with eigenvalues > 1 are typically retained.
        - **Individual Variance Explained (bottom-left)**: Demonstrates the fraction of total variance explained by each 
          component individually, showing a steep decline after the first few components.
        - **Cumulative Variance (bottom-right)**: Illustrates how variance accumulates as more components are included, 
          with reference lines at 50% and 85% thresholds for dimensionality reduction decisions.
        """)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Component-wise Explained Variance")
            # Create a DataFrame for display
            individual_variance_df = pd.DataFrame({'Component': [f'PC{i+1}' for i in range(len(explained_variance_ratio))],
                                                   'Individual Variance': [f'{v:.4f}' for v in explained_variance_ratio],
                                                   'Percentage': [f'{v*100:.2f}%' for v in explained_variance_ratio]})
            st.dataframe(individual_variance_df, hide_index=True, height=528)
        with col2:
            st.subheader("Cumulative Explained Variance")
            # Create a DataFrame for display
            cumulative_variance_df = pd.DataFrame({'Components': [f'PC1-PC{i+1}' for i in range(len(cumulative_variance))],
                                                   'Cumulative Variance': [f'{v:.4f}' for v in cumulative_variance],
                                                   'Percentage': [f'{v*100:.2f}%' for v in cumulative_variance]})
            st.dataframe(cumulative_variance_df, hide_index=True, height=528)
        
        st.markdown("---")

        st.subheader("Analysis Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="PCs with Eigenvalues (λ) > 1", value=sum(1 for val in eigenvalues if val > 1.0))
        with col2:
            st.metric(label="PC1-PC4 Cumulative Variance", value=f"{cumulative_variance[3]*100:.2f}%")
        with col3:
            st.metric(label="PC1-PC10 Cumulative Variance", value=f"{cumulative_variance[9]*100:.2f}%")
        
        st.info("""
        - **Kaiser Criterion**: The first 4 principal components have eigenvalues > 1, making them candidates for retention 
          according to the Kaiser criterion.
        - **Elbow Method**: A clear elbow point is observed at PC4, suggesting that including components beyond this point 
          provides diminishing returns in terms of explained variance.
        - **Cumulative Variance**: The first 4 components combined explain more than 50% of the variance in the dataset, and 
          the first 10 components combined explain more than 85% of the variance in the dataset.
        - **Conclusion**: Even though 4 principal components only preserve just over half the total variance, 4 PCs is supported 
          by the Kaiser criterion and the elbow method. Plus, effective dimensionality reduction is achieved as the original 
          14-dimensional feature space reduces to 4 dimensions.
        """)

elif page == "Classification Models":
    st.header("Classification Models")
    
    st.info("""
    Adjust the parameters below to see how they affect the classification models (Logistic Regression and Random Forest) in real-time.
            
    - **Number of Principal Components**: Select between 1 and 14 components (default: 4)
    - **Train-Test Split (Test Size)**: Select between 0.01 and 0.99 (default: 0.25 or 25%)
    - **Random State**: Enter any non-negative integer for reproducibility (default: 42)
    """)

    # User-controlled inputs for Principal Components, Train-Test Split, and Random State
    col1, col2, col3 = st.columns(3)
    with col1:
        n_components = st.slider("Number of Principal Components", min_value=1, max_value=14, value=4)
    with col2:
        test_size = st.slider("Train-Test Split (Test Size)", min_value=0.01, max_value=0.99, value=0.25, step=0.01)
    with col3:
        random_state = st.number_input("Random State", min_value=0, value=42, step=1) 
    
     # Create tabs for the three sections
    tab1, tab2, tab3 = st.tabs(["User-defined Parameters", "Development (Confusion Matrices)", "Evaluation (Classification Reports)"])
    
    # Cached helper function for SVD computation
    @st.cache_data
    def compute_svd_pca():
        X = artifacts['df_clean'].drop(columns=["num"])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        U, s, Vt = np.linalg.svd(X_scaled, full_matrices=True)
        eigenvalues = s**2 / (len(X_scaled) - 1)
        explained_variance_ratio = eigenvalues / eigenvalues.sum()
        cumulative_variance = np.cumsum(explained_variance_ratio)
        return X_scaled, Vt, cumulative_variance
    
    # Cached helper function for model training
    @st.cache_data
    def train_models(X_train, X_test, y_train, y_test, _random_state):        
        # Train Logistic Regression
        log_reg = LogisticRegression(random_state=_random_state)
        log_reg.fit(X_train, y_train)
        y_pred_lr = log_reg.predict(X_test)
        cm_lr = confusion_matrix(y_test, y_pred_lr)
        
        # Train Random Forest
        rf = RandomForestClassifier(random_state=_random_state)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        
        return (y_pred_lr, cm_lr), (y_pred_rf, cm_rf)
    
    # Get cached SVD and PCA
    X_scaled, Vt, cumulative_variance = compute_svd_pca()
    
    # User-defined principal components to create PCA dataset
    V = Vt[:n_components, :].T
    X_pca = X_scaled @ V
    y = artifacts['df_clean']["num"].copy()
    
    # User-defined split of data into training and testing sets
    X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=test_size, 
                                                                random_state=int(random_state), stratify=y)
    
    with tab1:
        st.subheader("PCA Configuration")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Original Dimensions", value=f"{X_scaled.shape[0]} × {X_scaled.shape[1]}")
        with col2:
            st.metric(label="Reduced Dimensions", value=f"{X_pca.shape[0]} × {X_pca.shape[1]}")
        with col3:
            st.metric(label="Variance Retained", value=f"{cumulative_variance[n_components-1]*100:.2f}%")
        
        st.markdown("---")
        
        st.subheader("Train-Test Set Details")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Training Samples", value=f"{X_train_pca.shape[0]}")
        with col2:
            st.metric(label="Testing Samples", value=f"{X_test_pca.shape[0]}")
    
    with tab2:
        st.subheader("Model Development")
        
        # Train all models with caching
        (y_pred_lr, cm_lr), (y_pred_rf, cm_rf) = train_models(X_train_pca, X_test_pca, y_train, y_test, int(random_state))
        
        # Display confusion matrices in columns
        col1, col2 = st.columns(2)
        
        with col1:
            fig_lr, ax_lr = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=np.sort(y.unique()),
                        yticklabels=np.sort(y.unique()),
                        cbar_kws={'label': 'Count'}, ax=ax_lr)
            ax_lr.set_xlabel('Predicted Heart Disease Stage', fontsize=11, fontweight='bold')
            ax_lr.set_ylabel('True Heart Disease Stage', fontsize=11, fontweight='bold')
            ax_lr.set_title(f"Logistic Regression Confusion Matrix (n_components={n_components})",
                            fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_lr)
        
        with col2:
            fig_rf, ax_rf = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', 
                        xticklabels=np.sort(y.unique()),
                        yticklabels=np.sort(y.unique()),
                        cbar_kws={'label': 'Count'}, ax=ax_rf)
            ax_rf.set_xlabel('Predicted Heart Disease Stage', fontsize=11, fontweight='bold')
            ax_rf.set_ylabel('True Heart Disease Stage', fontsize=11, fontweight='bold')
            ax_rf.set_title(f"Random Forest Classifier Confusion Matrix (n_components={n_components})",
                            fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_rf)
    
    with tab3:
        st.subheader("Model Evaluation")
        
        # Get predictions again (from cache)
        (y_pred_lr, _), (y_pred_rf, _) = train_models(X_train_pca, X_test_pca, y_train, y_test, int(random_state))
        
        # Generate classification reports
        report_lr = classification_report(y_test, y_pred_lr, zero_division=0.0,
                                          target_names=[f'Stage {i}' for i in np.sort(y.unique())],
                                          output_dict=False)
        report_rf = classification_report(y_test, y_pred_rf, zero_division=0.0,
                                          target_names=[f'Stage {i}' for i in np.sort(y.unique())],
                                          output_dict=False)
        
        # Display reports in columns
        col1, col2 = st.columns(2)
        with col1:
            st.code(f"Logistic Regression ({n_components} PCs)\n\n{report_lr}", language="text")
        with col2:
            st.code(f"Random Forest Classifier ({n_components} PCs)\n\n{report_rf}", language="text")
