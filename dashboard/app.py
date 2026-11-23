
from pathlib import Path
from datetime import datetime, time

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px


# ==========================================================
# 1. PROJECT PATHS AND PAGE CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

st.set_page_config(
    page_title="EV Charging Demand Prediction",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ EV Charging Demand Prediction")
st.caption(
    "Demand Analysis | Machine Learning | Time-Series Forecasting"
)


# ==========================================================
# 2. FEATURE CONFIGURATION
# ==========================================================

FEATURE_COLUMNS = [
    "Location",
    "ChargerType",
    "ChargerCompany",
    "Hour",
    "Day",
    "Month",
    "DayOfWeek",
    "DayName",
    "IsWeekend",
    "TimeOfDay",
    "Location_PastSessionCount",
    "Location_PastAvgDemand",
    "Location_HasPastHistory",
    "ChargerType_PastSessionCount",
    "ChargerType_PastAvgDemand",
    "ChargerType_HasPastHistory",
    "Company_PastSessionCount",
    "Company_PastAvgDemand",
    "Company_HasPastHistory"
]


# ==========================================================
# 3. LOAD DATA AND MODELS
# ==========================================================

@st.cache_data
def load_data():

    data_path = DATA_DIR / "charging_ml_dataset.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    data = pd.read_csv(data_path)

    required_columns = [
        "StartDatetime",
        "Location",
        "ChargerType",
        "ChargerCompany",
        "Demand"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: {missing_columns}"
        )

    data["StartDatetime"] = pd.to_datetime(
        data["StartDatetime"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["StartDatetime", "Demand"]
    )

    data["Demand"] = pd.to_numeric(
        data["Demand"],
        errors="coerce"
    )

    data = data.dropna(subset=["Demand"])

    data = data.sort_values(
        "StartDatetime"
    ).reset_index(drop=True)

    return data


@st.cache_resource
def load_models():

    regression_path = (
        MODEL_DIR / "best_regression_model.pkl"
    )

    classification_path = (
        MODEL_DIR / "best_classification_model.pkl"
    )

    preprocessor_path = (
        MODEL_DIR / "preprocessor.pkl"
    )

    threshold_path = (
        MODEL_DIR / "demand_threshold.pkl"
    )

    required_files = [
        regression_path,
        classification_path,
        preprocessor_path,
        threshold_path
    ]

    for file_path in required_files:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required model file not found: {file_path}"
            )

    regression_model = joblib.load(
        regression_path
    )

    classification_model = joblib.load(
        classification_path
    )

    preprocessor = joblib.load(
        preprocessor_path
    )

    threshold = joblib.load(
        threshold_path
    )

    return (
        regression_model,
        classification_model,
        preprocessor,
        threshold
    )


try:

    df = load_data()

    (
        regression_model,
        classification_model,
        preprocessor,
        threshold
    ) = load_models()

except Exception as error:

    st.error(
        f"Unable to load project files: {error}"
    )

    st.info(
        "Check the dataset, trained models and preprocessing "
        "files in their respective project folders."
    )

    st.stop()


# ==========================================================
# 4. HISTORICAL FEATURE CALCULATION
# ==========================================================

def get_prediction_features(
    location,
    charger_type,
    company,
    prediction_datetime
):
    """
    Create the 19 model input features.

    Only charging sessions that started before the selected
    prediction datetime are used for historical statistics.
    """

    prediction_datetime = pd.Timestamp(
        prediction_datetime
    )

    # Exclude all sessions at or after prediction time.
    past_data = df[
        df["StartDatetime"] < prediction_datetime
    ].copy()

    # Historical records for selected categories.
    location_rows = past_data[
        past_data["Location"] == location
    ]

    charger_rows = past_data[
        past_data["ChargerType"] == charger_type
    ]

    company_rows = past_data[
        past_data["ChargerCompany"] == company
    ]

    # Date and time features.
    hour = prediction_datetime.hour

    day = prediction_datetime.day

    month = prediction_datetime.month

    day_of_week = prediction_datetime.weekday()

    day_name = prediction_datetime.strftime("%A")

    is_weekend = (
        1 if day_of_week >= 5 else 0
    )

    # Time-of-day categories must match feature engineering.
    if 0 <= hour <= 5:

        time_of_day = "Night"

    elif 6 <= hour <= 11:

        time_of_day = "Morning"

    elif 12 <= hour <= 16:

        time_of_day = "Afternoon"

    elif 17 <= hour <= 20:

        time_of_day = "Evening"

    else:

        time_of_day = "Late Evening"

    # Calculate historical count and average demand.
    def calculate_history(rows):

        if rows.empty:

            return 0, 0.0, 0

        count = len(rows)

        average_demand = float(
            rows["Demand"].mean()
        )

        has_history = 1

        return (
            count,
            average_demand,
            has_history
        )

    (
        location_count,
        location_avg,
        location_history
    ) = calculate_history(location_rows)

    (
        charger_count,
        charger_avg,
        charger_history
    ) = calculate_history(charger_rows)

    (
        company_count,
        company_avg,
        company_history
    ) = calculate_history(company_rows)

    # Build model input.
    input_data = {

        "Location": location,

        "ChargerType": charger_type,

        "ChargerCompany": company,

        "Hour": hour,

        "Day": day,

        "Month": month,

        "DayOfWeek": day_of_week,

        "DayName": day_name,

        "IsWeekend": is_weekend,

        "TimeOfDay": time_of_day,

        "Location_PastSessionCount": location_count,

        "Location_PastAvgDemand": location_avg,

        "Location_HasPastHistory": location_history,

        "ChargerType_PastSessionCount": charger_count,

        "ChargerType_PastAvgDemand": charger_avg,

        "ChargerType_HasPastHistory": charger_history,

        "Company_PastSessionCount": company_count,

        "Company_PastAvgDemand": company_avg,

        "Company_HasPastHistory": company_history
    }

    input_df = pd.DataFrame(
        [input_data],
        columns=FEATURE_COLUMNS
    )

    # Display history information for verification.
    history_info = {
        "Location History Records": location_count,
        "Charger Type History Records": charger_count,
        "Company History Records": company_count
    }

    return input_df, history_info


# ==========================================================
# 5. COMMON PREDICTION INPUTS
# ==========================================================

def prediction_inputs(key_prefix):

    locations = sorted(
        df["Location"].dropna().unique().tolist()
    )

    charger_types = sorted(
        df["ChargerType"].dropna().unique().tolist()
    )

    companies = sorted(
        df["ChargerCompany"].dropna().unique().tolist()
    )

    col1, col2 = st.columns(2)

    with col1:

        location = st.selectbox(
            "Location",
            locations,
            key=f"{key_prefix}_location"
        )

        charger_type = st.selectbox(
            "Charger Type",
            charger_types,
            key=f"{key_prefix}_charger"
        )

        company = st.selectbox(
            "Charger Company",
            companies,
            key=f"{key_prefix}_company"
        )

    with col2:

        prediction_date = st.date_input(
            "Prediction Date",
            value=datetime.now().date(),
            key=f"{key_prefix}_date"
        )

        prediction_time = st.time_input(
            "Prediction Time",
            value=time(18, 0),
            key=f"{key_prefix}_time"
        )

    prediction_datetime = datetime.combine(
        prediction_date,
        prediction_time
    )

    return (
        location,
        charger_type,
        company,
        prediction_datetime
    )


# ==========================================================
# 6. PREPROCESS PREDICTION INPUT
# ==========================================================

def preprocess_prediction(input_df):

    if input_df.columns.tolist() != FEATURE_COLUMNS:

        raise ValueError(
            "Input feature order does not match model features."
        )

    processed_input = preprocessor.transform(
        input_df
    )

    expected_features = len(
        preprocessor.get_feature_names_out()
    )

    if processed_input.shape[1] != expected_features:

        raise ValueError(
            "Processed feature dimension mismatch."
        )

    return processed_input


# ==========================================================
# 7. SIDEBAR NAVIGATION
# ==========================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "Demand Analysis",
        "Demand Prediction",
        "Demand Classification",
        "Time-Series Forecasting",
        "Model Comparison",
        "Explainability"
    ]
)


# ==========================================================
# 8. PROJECT OVERVIEW
# ==========================================================

if page == "Project Overview":

    st.header("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Sessions",
        f"{len(df):,}"
    )

    col2.metric(
        "Locations",
        df["Location"].nunique()
    )

    col3.metric(
        "Charger Types",
        df["ChargerType"].nunique()
    )

    col4.metric(
        "Average Demand",
        f"{df['Demand'].mean():.2f}"
    )

    st.subheader("Project Objective")

    st.write(
        """
        The objective of this project is to analyze EV charging
        usage patterns and predict charging demand using
        machine learning and time-series forecasting.

        The system includes regression, classification,
        forecasting and model explainability.
        """
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Dataset Shape**")

        st.write(df.shape)

    with col2:

        st.write("**Missing Values**")

        st.write(
            int(df.isnull().sum().sum())
        )

    st.subheader("Dataset Date Range")

    st.write(
        f"From {df['StartDatetime'].min()} "
        f"to {df['StartDatetime'].max()}"
    )


# ==========================================================
# 9. DEMAND ANALYSIS
# ==========================================================

elif page == "Demand Analysis":

    st.header("EV Charging Demand Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Demand by Location")

        location_demand = (
            df.groupby("Location")["Demand"]
            .mean()
            .reset_index()
            .sort_values(
                "Demand",
                ascending=False
            )
        )

        fig = px.bar(
            location_demand,
            x="Location",
            y="Demand",
            title="Average Demand by Location"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("Demand by Charger Type")

        charger_demand = (
            df.groupby("ChargerType")["Demand"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            charger_demand,
            x="ChargerType",
            y="Demand",
            title="Average Demand by Charger Type"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Demand Distribution")

    fig = px.histogram(
        df,
        x="Demand",
        nbins=40,
        title="Distribution of Charging Demand"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Demand by Hour")

    hourly = (
        df.groupby("Hour")["Demand"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        hourly,
        x="Hour",
        y="Demand",
        markers=True,
        title="Average Demand by Hour"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Demand by Time of Day")

    time_demand = (
        df.groupby("TimeOfDay")["Demand"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        time_demand,
        x="TimeOfDay",
        y="Demand",
        title="Average Demand by Time of Day"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================================
# 10. REGRESSION DEMAND PREDICTION
# ==========================================================

elif page == "Demand Prediction":

    st.header("Predict EV Charging Demand")

    st.write(
        """
        Enter the information available when a charging
        session starts. Historical features are calculated
        using sessions before the selected date and time.
        """
    )

    (
        location,
        charger_type,
        company,
        prediction_datetime
    ) = prediction_inputs("regression")

    st.info(
        f"Selected prediction time: {prediction_datetime}"
    )

    if st.button(
        "Predict Demand",
        type="primary"
    ):

        try:

            input_df, history_info = (
                get_prediction_features(
                    location,
                    charger_type,
                    company,
                    prediction_datetime
                )
            )

            processed_input = (
                preprocess_prediction(input_df)
            )

            prediction = regression_model.predict(
                processed_input
            )[0]

            st.success(
                "Demand prediction completed!"
            )

            st.metric(
                "Predicted Demand",
                f"{prediction:.2f}"
            )

            st.subheader(
                "Historical Feature Verification"
            )

            st.write(
                "Only sessions before the selected "
                "prediction time were used."
            )

            st.json(history_info)

            with st.expander(
                "View model input features"
            ):

                st.dataframe(
                    input_df,
                    use_container_width=True
                )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# ==========================================================
# 11. HIGH / NORMAL DEMAND CLASSIFICATION
# ==========================================================

elif page == "Demand Classification":

    st.header(
        "High / Normal Demand Classification"
    )

    st.write(
        f"Training median threshold: {float(threshold):.2f}"
    )

    st.write(
        """
        The classifier predicts whether the charging
        demand belongs to the High Demand or Normal
        Demand category.
        """
    )

    (
        location,
        charger_type,
        company,
        prediction_datetime
    ) = prediction_inputs("classification")

    st.info(
        f"Selected prediction time: {prediction_datetime}"
    )

    if st.button(
        "Classify Demand",
        type="primary"
    ):

        try:

            input_df, history_info = (
                get_prediction_features(
                    location,
                    charger_type,
                    company,
                    prediction_datetime
                )
            )

            processed_input = (
                preprocess_prediction(input_df)
            )

            prediction = classification_model.predict(
                processed_input
            )[0]

            st.subheader(
                "Classification Result"
            )

            if hasattr(
                classification_model,
                "predict_proba"
            ):

                probabilities = (
                    classification_model.predict_proba(
                        processed_input
                    )[0]
                )

                classes = list(
                    classification_model.classes_
                )

                high_class_index = classes.index(1)

                probability = probabilities[
                    high_class_index
                ]

                st.metric(
                    "Estimated High-Demand Probability",
                    f"{probability * 100:.2f}%"
                )

            if prediction == 1:

                st.warning(
                    "Predicted Category: HIGH DEMAND"
                )

            else:

                st.success(
                    "Predicted Category: NORMAL DEMAND"
                )

            st.subheader(
                "Historical Feature Verification"
            )

            st.write(
                "Only sessions before the selected "
                "prediction time were used."
            )

            st.json(history_info)

            with st.expander(
                "View model input features"
            ):

                st.dataframe(
                    input_df,
                    use_container_width=True
                )

        except Exception as error:

            st.error(
                f"Classification failed: {error}"
            )


# ==========================================================
# 12. TIME SERIES FORECASTING
# ==========================================================

elif page == "Time-Series Forecasting":

    st.header(
        "Time-Series Demand Forecasting"
    )

    hourly_path = (
        DATA_DIR / "demand_hourly.csv"
    )

    daily_path = (
        DATA_DIR / "demand_daily.csv"
    )

    forecast_path = (
        REPORT_DIR
        / "time_series"
        / "forecast_predictions.csv"
    )

    if hourly_path.exists():

        hourly_df = pd.read_csv(
            hourly_path
        )

        st.subheader("Hourly Demand")

        st.dataframe(
            hourly_df.tail(10),
            use_container_width=True
        )

        if "TotalDemand" in hourly_df.columns:

            fig = px.line(
                hourly_df,
                y="TotalDemand",
                title="Hourly Demand Trend"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "TotalDemand column not found in hourly dataset."
            )

    else:

        st.info(
            "Hourly demand dataset was not found."
        )

    if daily_path.exists():

        daily_df = pd.read_csv(
            daily_path
        )

        st.subheader("Daily Demand")

        st.dataframe(
            daily_df.tail(10),
            use_container_width=True
        )

        if "TotalDemand" in daily_df.columns:

            fig = px.line(
                daily_df,
                y="TotalDemand",
                title="Daily Demand Trend"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.info(
            "Daily demand dataset was not found."
        )

    if forecast_path.exists():

        forecast_df = pd.read_csv(
            forecast_path
        )

        st.subheader("Forecast Predictions")

        st.dataframe(
            forecast_df.head(20),
            use_container_width=True
        )

        numeric_columns = (
            forecast_df.select_dtypes(
                include="number"
            ).columns.tolist()
        )

        if numeric_columns:

            fig = px.line(
                forecast_df,
                y=numeric_columns,
                title="Forecast Comparison"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.info(
            "Forecast prediction file was not found."
        )


# ==========================================================
# 13. MODEL COMPARISON
# ==========================================================

elif page == "Model Comparison":

    st.header(
        "Machine Learning Model Comparison"
    )

    regression_validation_path = (
        REPORT_DIR / "regression_validation_results.csv"
    )

    regression_test_path = (
        REPORT_DIR / "regression_model_results.csv"
    )

    classification_validation_path = (
        REPORT_DIR / "classification_validation_results.csv"
    )

    classification_test_path = (
        REPORT_DIR / "classification_test_results.csv"
    )

    # Regression validation.
    if regression_validation_path.exists():

        st.subheader(
            "Regression Validation Results"
        )

        regression_validation_df = pd.read_csv(
            regression_validation_path
        )

        st.dataframe(
            regression_validation_df,
            use_container_width=True
        )

        if "RMSE" in regression_validation_df.columns:

            fig = px.bar(
                regression_validation_df,
                x="Model",
                y="RMSE",
                title="Regression Validation RMSE Comparison"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.info(
            "Regression validation results not found."
        )

    # Final regression test.
    if regression_test_path.exists():

        st.subheader(
            "Final Regression Test Results"
        )

        regression_test_df = pd.read_csv(
            regression_test_path
        )

        st.dataframe(
            regression_test_df,
            use_container_width=True
        )

    else:

        st.info(
            "Regression test results not found."
        )

    # Classification validation.
    if classification_validation_path.exists():

        st.subheader(
            "Classification Validation Results"
        )

        classification_validation_df = pd.read_csv(
            classification_validation_path
        )

        st.dataframe(
            classification_validation_df,
            use_container_width=True
        )

        if "F1_Score" in classification_validation_df.columns:

            fig = px.bar(
                classification_validation_df,
                x="Model",
                y="F1_Score",
                title="Classification Validation F1 Comparison"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.info(
            "Classification validation results not found."
        )

    # Final classification test.
    if classification_test_path.exists():

        st.subheader(
            "Final Classification Test Results"
        )

        classification_test_df = pd.read_csv(
            classification_test_path
        )

        st.dataframe(
            classification_test_df,
            use_container_width=True
        )

    else:

        st.info(
            "Classification test results not found."
        )


# ==========================================================
# 14. MODEL EXPLAINABILITY
# ==========================================================

elif page == "Explainability":

    st.header(
        "Model Explainability"
    )

    explain_dir = (
        REPORT_DIR / "explainability"
    )

    files = {
        "Feature Importance": "feature_importance.csv",
        "SHAP Importance": "shap_importance.csv",
        "LIME Explanation": "lime_explanation.csv"
    }

    for title, filename in files.items():

        file_path = explain_dir / filename

        st.subheader(title)

        if file_path.exists():

            result_df = pd.read_csv(
                file_path
            )

            st.dataframe(
                result_df.head(20),
                use_container_width=True
            )

        else:

            st.info(
                f"{filename} was not found."
            )

    st.subheader(
        "Explainability Charts"
    )

    chart_dir = (
        explain_dir / "charts"
    )

    chart_files = [
        "feature_importance.png",
        "shap_summary.png",
        "shap_feature_importance.png",
        "lime_explanation.png"
    ]

    for chart_name in chart_files:

        chart_path = chart_dir / chart_name

        if chart_path.exists():

            st.write(
                f"**{chart_name.replace('.png', '').replace('_', ' ').title()}**"
            )

            st.image(
                str(chart_path),
                use_container_width=True
            )

    lime_html = (
        explain_dir / "lime_explanation.html"
    )

    if lime_html.exists():

        with st.expander(
            "LIME HTML Explanation File"
        ):

            st.write(
                f"File location: {lime_html}"
            )

    st.caption(
        "Feature importance, SHAP and LIME explain model "
        "behaviour. They do not establish real-world causation."
    )


# ==========================================================
# 15. FOOTER
# ==========================================================

st.sidebar.divider()

st.sidebar.caption(
    "EV Charging Demand Prediction Project"
)

st.sidebar.caption(
    "Machine Learning & Data Analytics"
)

st.sidebar.caption(
    "Developed using Python and Streamlit"
)
