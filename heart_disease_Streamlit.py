import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
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
    df_cleveland = pd.read_csv("heart_disease/cleveland.data", sep=",")
    df_long_beach = pd.read_csv("heart_disease/long_beach.data", sep=",")
    df_hungarian = pd.read_csv("heart_disease/hungarian.data", delim_whitespace=True)
    df_switzerland = pd.read_csv("heart_disease/switzerland.data", sep=",")

    # The original notebook assumes column names consistent with UCI Heart Disease dataset
    col_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "num"
    ]

    for df in [df_cleveland, df_long_beach, df_hungarian, df_switzerland]:
        df.columns = col_names

    # Replace missing value markers used in the notebook
    df_cleveland.replace("?", np.nan, inplace=True)
    df_cleveland["ca"].replace(9.0, np.nan, inplace=True)

    df_long_beach.replace("?", np.nan, inplace=True)
    df_long_beach["ca"].replace(9.0, np.nan, inplace=True)

    df_hungarian.replace(-9.0, np.nan, inplace=True)
    df_hungarian["ca"].replace(9.0, np.nan, inplace=True)

    df_switzerland.replace("?", np.nan, inplace=True)
    df_switzerland["ca"].replace(9.0, np.nan, inplace=True)

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


# --- Minimal addition: age-range slider to filter datasets by age ---
# Determine age bounds from the combined/no-dup dataset (ignore NaNs)
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


# --- Minimal addition: allow selecting (or de-selecting) locations to include ---
try:
    _loc_options = list(artifacts['df_combine']['location'].dropna().unique())
except Exception:
    _loc_options = []

selected_locations = st.sidebar.multiselect("Select (or Unselect) Locations to Include", options=_loc_options, 
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


# Sidebar pages as requested
page = st.sidebar.radio("Select Page", [
    "Data Background",
    "Initial Preprocessing Steps",
    "Handling Missing Values",
    "Imputation Analysis",
    "Interactive Visualizations"
])

if page == "Data Background":
    st.header("Data Background")

    tab1, tab2, tab3 = st.tabs(["Introduction", "Individual Datasets", "Combined Dataset"])

    with tab1:
        st.subheader("Variables")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            0. `age` (Ratio): Age in years
            1. `sex` (Nominal): 1: male, 0: female
            2. `cp` (Nominal): Chest pain type (1-4)
            3. `trestbps` (Ratio): Resting blood pressure
            4. `chol` (Ratio): Serum cholesterol
            5. `fbs` (Nominal): Fasting blood sugar > 120 mg/dl
            6. `restecg` (Nominal): Resting ECG results (0-2)
            """)
        with col2:
            st.markdown("""
            7. `thalach` (Ratio): Maximum heart rate achieved
            8. `exang` (Nominal): Exercise induced angina (1/0)
            9. `oldpeak` (Ratio): ST depression induced by exercise relative to rest
            10. `slope` (Ordinal): Slope of the peak exercise ST segment (1-3)
            11. `ca` (Ratio): Number of major vessels (0-3) colored by fluoroscopy
            12. `thal` (Nominal): Thallium stress test result (3,6,7)
            13. `num` (Ordinal): Diagnosis of heart disease (0-4)
            """)

    with tab2:
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
        st.write("Dataset Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.code(artifacts['combine_init_info'].getvalue(), language="text")
        with col2:
            st.code(artifacts['combine_new_info'].getvalue(), language="text")
        st.dataframe(artifacts['combine_new_describe'], height=318)

    with tab2:
        dup_mask = artifacts["dup_mask"]
        df_with_dup = artifacts["df_combine"]

        st.subheader("Duplicate Rows")
        st.dataframe(df_with_dup[dup_mask])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Need to Remove Row at Index 490")
            df_with_dup[(df_with_dup["age"] == 58.0) & 
                        (df_with_dup["cp"] == 3) & 
                        (df_with_dup["trestbps"] == 150.0)]
        with col2:
            st.subheader("Need to Remove Row at Index 666")
            df_with_dup[(df_with_dup["age"] == 49.0) & 
                        (df_with_dup["cp"] == 2) & 
                        (df_with_dup["trestbps"] == 110.0)]

    with tab3:
        st.subheader("Location Encoding")
        buf = io.StringIO()
        artifacts['location_encode'].to_string(buf=buf)
        buf.seek(0)
        st.code(buf.getvalue(), language="text")
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
        st.subheader("Missingness Summary")
        nan_stats = pd.DataFrame(artifacts['df_no_dup'].isna().sum(), columns=["NaN_Count"]) 
        nan_stats["NaN_Percent"] = np.round(nan_stats["NaN_Count"]/artifacts['df_no_dup'].shape[0], 4)*100
        nan_stats["NaN_Percent"] = nan_stats["NaN_Percent"].astype(str) + "%"
        st.dataframe(nan_stats, height=562)

    with tab2:
        col1, col2 = st.columns(2)
        st.subheader("MICE (Imputation on Continuous Variables)")
        with col1:
            st.dataframe(artifacts["summary_before_MICE"])
        with col2:
            st.dataframe(artifacts["summary_after_MICE"])
        st.dataframe(artifacts["stats_MICE"], height=562)
    
    with tab3:
        col1, col2 = st.columns(2)
        st.subheader("KNN (Imputation on Categorical Variables)")
        with col1:
            st.dataframe(artifacts["summary_before_KNN"])
        with col2:
            st.dataframe(artifacts["summary_after_KNN"])
        st.dataframe(artifacts["stats_KNN"], height=562)

    with tab4:
        st.subheader("Assign Appropriate Data Types")
        buf = io.StringIO()
        artifacts['df_clean'].info(buf=buf)
        buf.seek(0)
        st.code(buf.getvalue(), language="text")
        st.dataframe(artifacts['df_clean'], height=500)

elif page == "Imputation Analysis":
    st.header("Imputation Analysis")
    tab1, tab2 = st.tabs(["Correlation Comparison", "Outliers Check"])

    with tab1:
        st.subheader("Correlation Heatmaps (Before vs After MICE and KNN Imputation)")

        col_names = list(artifacts['df_no_dup'].columns)
        corr_matrix_before = np.round(artifacts['df_no_dup'].corr().values, 2)
        fig = ff.create_annotated_heatmap(
            z=corr_matrix_before,
            x=col_names,
            y=col_names,
            colorscale="bluered"
        )
        fig.update_layout(title="Correlation Heatmap Before Imputation")
        st.plotly_chart(fig, use_container_width=True)

        col_names = list(artifacts['df_clean'].columns)
        corr_matrix_before = np.round(artifacts['df_clean'].corr().values, 2)
        fig = ff.create_annotated_heatmap(
            z=corr_matrix_before,
            x=col_names,
            y=col_names,
            colorscale="bluered"
        )
        fig.update_layout(title="Correlation Heatmap After Imputation")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Box Plots (Before vs After MICE and KNN Imputation)")

        plot_df_before = artifacts['df_no_dup'][["age"] + artifacts['continuous_cols']].melt(var_name="variable", 
                                                                                             value_name="value")
        fig_box_before = px.box(plot_df_before, x="variable", y="value", title="Box Plot Before Imputation")
        st.plotly_chart(fig_box_before, use_container_width=True)

        plot_df_after = artifacts['df_clean'][["age"] + artifacts['continuous_cols']].melt(var_name="variable", 
                                                                                           value_name="value")
        fig_box_after = px.box(plot_df_after, x="variable", y="value", title="Box Plot After Imputation")
        st.plotly_chart(fig_box_after, use_container_width=True)

elif page == "Interactive Visualizations":
    st.header("Interactive Visualizations")
    tab1, tab2, tab3 = st.tabs(["Heart Disease Stage by Location", "Age vs Maximum Heart Rate (thalach)", 
                                "Heart Disease Stage vs Blood Flow to Heart (oldpeak)"])
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
        st.plotly_chart(fig_bar_chart, use_container_width=True)

    with tab2:
        fig_scatter = px.scatter(df_clean, x="age", y="thalach", color="num", trendline="ols", 
                                 color_continuous_scale="turbo", 
                                 labels={"age":"Age", "thalach":"Max Heart Rate", "num":"Disease Stage"}, 
                                 title="Scatter Plot of Age vs Maximum Heart Rate (thalach)")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab3:
        fig_violin = px.violin(df_clean, x="num", y="oldpeak", box=True, points="all", 
                               labels={"num":"Heart Disease Stage", "oldpeak":"Blood Flow to Heart"}, 
                               title="Violin Plot of Heart Disease Stage vs Blood Flow to Heart During Physical Exertion (oldpeak)")
        st.plotly_chart(fig_violin, use_container_width=True)

