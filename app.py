from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "WA_Fn-UseC_-Telco-Customer-Churn-cleaned.csv"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
PIPELINE_FILE = ARTIFACTS_DIR / "telco_churn_logistic_pipeline.joblib"
METADATA_FILE = ARTIFACTS_DIR / "telco_churn_logistic_pipeline_metadata.json"

st.set_page_config(
    page_title="Telco Churn Prediction Dashboard",
    page_icon="📉",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    return df


@st.cache_data
def load_metadata() -> dict:
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_pipeline():
    return joblib.load(PIPELINE_FILE)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    working_df = df.copy()
    working_df["service_count"] = (
        (working_df["PhoneService"] == "Yes").astype(int)
        + (working_df["MultipleLines"] == "Yes").astype(int)
        + (working_df["InternetService"] != "No").astype(int)
        + (working_df["OnlineSecurity"] == "Yes").astype(int)
        + (working_df["OnlineBackup"] == "Yes").astype(int)
        + (working_df["DeviceProtection"] == "Yes").astype(int)
        + (working_df["TechSupport"] == "Yes").astype(int)
        + (working_df["StreamingTV"] == "Yes").astype(int)
        + (working_df["StreamingMovies"] == "Yes").astype(int)
    )
    working_df["avg_monthly_value_from_total"] = (
        working_df["TotalCharges"] / working_df["tenure"].replace(0, 1)
    ).round(2)
    working_df["is_new_customer"] = (working_df["tenure"] <= 12).astype(int)
    working_df["has_long_term_contract"] = working_df["Contract"].isin(["One year", "Two year"]).astype(int)
    return working_df


def prepare_model_input(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    feature_columns = metadata["categorical_features"] + metadata["numeric_features"]
    return df[feature_columns].copy()


def risk_label(probability: float, threshold: float) -> str:
    medium_cutoff = max(0.3, threshold * 0.7)
    if probability >= threshold:
        return "High"
    if probability >= medium_cutoff:
        return "Medium"
    return "Low"


def score_dataframe(df: pd.DataFrame, pipeline, metadata: dict, threshold: float) -> pd.DataFrame:
    featured_df = add_engineered_features(df)
    model_input = prepare_model_input(featured_df, metadata)
    probabilities = pipeline.predict_proba(model_input)[:, 1]

    scored_df = featured_df.copy()
    scored_df["predicted_churn_probability"] = probabilities
    scored_df["predicted_label"] = np.where(scored_df["predicted_churn_probability"] >= threshold, "Yes", "No")
    scored_df["risk_level"] = scored_df["predicted_churn_probability"].apply(lambda p: risk_label(p, threshold))
    scored_df["risk_bucket"] = pd.cut(
        scored_df["predicted_churn_probability"],
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        include_lowest=True,
        labels=["0.00-0.20", "0.21-0.40", "0.41-0.60", "0.61-0.80", "0.81-1.00"],
    )
    return scored_df


def build_single_customer_input(reference_df: pd.DataFrame) -> dict:
    st.subheader("Customer details")

    categorical_columns = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]
    numeric_columns = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

    input_data: dict[str, object] = {"customerID": "Manual Entry"}

    col1, col2 = st.columns(2)

    for idx, column in enumerate(categorical_columns):
        options = sorted(reference_df[column].dropna().astype(str).unique().tolist())
        default_value = reference_df[column].mode().iloc[0]
        target_col = col1 if idx % 2 == 0 else col2
        with target_col:
            default_index = options.index(str(default_value)) if str(default_value) in options else 0
            input_data[column] = st.selectbox(column, options=options, index=default_index, key=f"single_{column}")

    numeric_ranges = {
        "SeniorCitizen": (0, 1, int(reference_df["SeniorCitizen"].mode().iloc[0])),
        "tenure": (0, int(reference_df["tenure"].max()), int(reference_df["tenure"].median())),
        "MonthlyCharges": (
            float(reference_df["MonthlyCharges"].min()),
            float(reference_df["MonthlyCharges"].max()),
            float(round(reference_df["MonthlyCharges"].median(), 2)),
        ),
        "TotalCharges": (
            0.0,
            float(reference_df["TotalCharges"].max()),
            float(round(reference_df["TotalCharges"].median(), 2)),
        ),
    }

    st.subheader("Account values")
    num_col1, num_col2 = st.columns(2)

    for idx, column in enumerate(numeric_columns):
        min_value, max_value, default_value = numeric_ranges[column]
        target_col = num_col1 if idx % 2 == 0 else num_col2
        with target_col:
            if column in {"SeniorCitizen", "tenure"}:
                input_data[column] = st.number_input(
                    column,
                    min_value=int(min_value),
                    max_value=int(max_value),
                    value=int(default_value),
                    step=1,
                    key=f"single_{column}",
                )
            else:
                input_data[column] = st.number_input(
                    column,
                    min_value=float(min_value),
                    max_value=float(max_value),
                    value=float(default_value),
                    step=0.1,
                    key=f"single_{column}",
                )

    return input_data


def show_overview(scored_df: pd.DataFrame, threshold: float) -> None:
    st.title("Telco Churn Prediction Dashboard")
    st.write(
        "A business-friendly dashboard for identifying customers most likely to churn and prioritizing retention action."
    )

    high_risk_count = int((scored_df["predicted_churn_probability"] >= threshold).sum())
    churn_rate = (scored_df["Churn"] == "Yes").mean()
    avg_risk = scored_df["predicted_churn_probability"].mean()

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Customers scored", f"{len(scored_df):,}")
    metric_col2.metric("Actual churn rate", f"{churn_rate:.1%}")
    metric_col3.metric("Average predicted risk", f"{avg_risk:.1%}")
    metric_col4.metric("High-risk customers", f"{high_risk_count:,}")

    st.subheader("Business takeaway")
    st.write(
        "Use this dashboard to estimate churn risk, identify the highest-risk customers, and decide which accounts should be prioritized for retention outreach."
    )


def show_single_prediction(reference_df: pd.DataFrame, pipeline, metadata: dict, threshold: float) -> None:
    st.title("Predict One Customer")
    st.write("Enter customer information and generate a churn risk prediction.")

    with st.form("single_prediction_form"):
        input_data = build_single_customer_input(reference_df)
        submitted = st.form_submit_button("Predict churn risk")

    if not submitted:
        return

    single_df = pd.DataFrame([input_data])
    scored_single_df = score_dataframe(single_df, pipeline, metadata, threshold)
    result = scored_single_df.iloc[0]

    prob = float(result["predicted_churn_probability"])
    label = result["predicted_label"]
    risk = result["risk_level"]

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Predicted churn probability", f"{prob:.1%}")
    metric_col2.metric("Predicted class", label)
    metric_col3.metric("Risk level", risk)

    reasons = []
    if result["Contract"] == "Month-to-month":
        reasons.append("month-to-month contract")
    if float(result["tenure"]) <= 12:
        reasons.append("short tenure")
    if float(result["MonthlyCharges"]) >= float(reference_df["MonthlyCharges"].median()):
        reasons.append("relatively high monthly charges")
    if result["InternetService"] == "Fiber optic":
        reasons.append("fiber optic service")

    st.subheader("Business interpretation")
    if reasons:
        st.write(f"This customer looks **{risk.lower()} risk** mainly because of: {', '.join(reasons)}.")
    else:
        st.write(
            "This prediction does not show the strongest high-risk profile in the dataset, so the customer appears more stable than the riskiest groups."
        )

    display_columns = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "InternetService",
        "Contract",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "service_count",
        "predicted_churn_probability",
        "predicted_label",
        "risk_level",
    ]
    st.dataframe(scored_single_df[display_columns], use_container_width=True)


def show_batch_predictions(scored_df: pd.DataFrame, threshold: float) -> None:
    st.title("Batch Customer Scoring")
    st.write("Review predicted churn risk across the existing customer base and identify the accounts that need attention first.")

    minimum_probability = st.slider(
        "Minimum churn probability to display",
        min_value=0.0,
        max_value=1.0,
        value=float(threshold),
        step=0.01,
    )
    risk_filter = st.multiselect("Risk levels", options=["High", "Medium", "Low"], default=["High", "Medium", "Low"])

    filtered_df = scored_df[
        (scored_df["predicted_churn_probability"] >= minimum_probability)
        & (scored_df["risk_level"].isin(risk_filter))
    ].copy()

    filtered_df = filtered_df.sort_values(by="predicted_churn_probability", ascending=False)

    top_n = st.slider("Customers to show", min_value=10, max_value=200, value=25, step=5)
    preview_df = filtered_df.head(top_n)

    st.subheader("Highest-risk customers")
    st.dataframe(
        preview_df[
            [
                "customerID",
                "predicted_churn_probability",
                "predicted_label",
                "risk_level",
                "Contract",
                "tenure",
                "InternetService",
                "PaymentMethod",
                "MonthlyCharges",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download scored customer list",
        data=csv_data,
        file_name="telco_churn_scored_customers.csv",
        mime="text/csv",
    )


def show_risk_summary(scored_df: pd.DataFrame, threshold: float) -> None:
    st.title("Risk Summary")
    st.write("A simple presentation view showing where risk is concentrated across the customer base.")

    bucket_counts = scored_df["risk_bucket"].value_counts().sort_index()
    level_counts = scored_df["risk_level"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Predicted probability bands")
        st.bar_chart(bucket_counts)

    with chart_col2:
        st.subheader("Customers by risk level")
        st.bar_chart(level_counts)

    st.subheader("High-risk concentration by segment")
    segment_choice = st.selectbox(
        "Choose a business segment",
        options=["Contract", "InternetService", "PaymentMethod"],
    )

    high_risk_segment_df = (
        scored_df.assign(is_high_risk=scored_df["predicted_churn_probability"] >= threshold)
        .groupby(segment_choice)["is_high_risk"]
        .mean()
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
    )
    st.bar_chart(high_risk_segment_df)

    st.dataframe(
        high_risk_segment_df.rename("high_risk_percentage").reset_index(),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    if not DATA_FILE.exists() or not PIPELINE_FILE.exists() or not METADATA_FILE.exists():
        st.error("Required data or model files are missing. Make sure the cleaned CSV and saved artifacts exist in the project.")
        return

    reference_df = load_data()
    metadata = load_metadata()
    pipeline = load_pipeline()

    st.sidebar.title("Dashboard controls")
    threshold = st.sidebar.slider(
        "High-risk threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.01,
    )

    page = st.sidebar.radio(
        "Go to",
        options=["Overview", "Predict One Customer", "Batch Customer Scoring", "Risk Summary"],
    )

    scored_df = score_dataframe(reference_df, pipeline, metadata, threshold)

    if page == "Overview":
        show_overview(scored_df, threshold)
    elif page == "Predict One Customer":
        show_single_prediction(reference_df, pipeline, metadata, threshold)
    elif page == "Batch Customer Scoring":
        show_batch_predictions(scored_df, threshold)
    else:
        show_risk_summary(scored_df, threshold)


if __name__ == "__main__":
    main()
