from pathlib import Path

import pandas as pd
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_telco_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)
    return df


def document_data_source(df: pd.DataFrame, file_path: Path) -> None:
    print("=" * 80)
    print("STEP 2 AND STEP 3: DATA COLLECTION, UNDERSTANDING, AND INITIAL EXPLORATION")
    print("=" * 80)
    print(f"Data source: {file_path.name}")
    print(f"Full path: {file_path}")
    print("Data format: Structured tabular CSV")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print()


def inspect_structure(df: pd.DataFrame) -> None:
    print("Columns:")
    for column in df.columns:
        print(f"- {column}")
    print()

    print("Data types:")
    print(df.dtypes)
    print()

    print("First 5 rows:")
    print(df.head())
    print()

    print("Dataset info:")
    df.info()
    print()

    print("Missing values per column:")
    print(df.isnull().sum())
    print()

    print("Unique values in target column 'Churn':")
    print(df["Churn"].value_counts(dropna=False))
    print()


def inspect_target_balance(df: pd.DataFrame) -> None:
    churn_counts = df["Churn"].value_counts(dropna=False)
    churn_percentages = df["Churn"].value_counts(normalize=True, dropna=False).mul(100).round(2)

    print("Target balance and class distribution:")
    target_balance = pd.DataFrame({
        "count": churn_counts,
        "percentage": churn_percentages,
    })
    print(target_balance)
    print()


def inspect_blank_strings(df: pd.DataFrame) -> None:
    object_columns = df.select_dtypes(include="object").columns
    blank_counts = {}

    for column in object_columns:
        blank_counts[column] = df[column].astype(str).str.strip().eq("").sum()

    blank_series = pd.Series(blank_counts).sort_values(ascending=False)
    print("Blank string counts in object columns:")
    print(blank_series)
    print()


def inspect_categorical_columns(df: pd.DataFrame) -> None:
    categorical_columns = df.select_dtypes(include="object").columns.tolist()
    print("Categorical column cardinality:")

    cardinality = pd.DataFrame({
        "unique_values": df[categorical_columns].nunique(dropna=False)
    }).sort_values(by="unique_values", ascending=False)
    print(cardinality)
    print()

    for column in categorical_columns:
        print(f"Top values for {column}:")
        print(df[column].value_counts(dropna=False).head(10))
        print()


def inspect_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    working_df = df.copy()
    working_df["TotalCharges"] = pd.to_numeric(working_df["TotalCharges"], errors="coerce")
    numeric_columns = working_df.select_dtypes(include=np.number).columns.tolist()

    print("Numeric summary statistics:")
    print(working_df[numeric_columns].describe().T)
    print()

    return working_df


def inspect_outliers(df: pd.DataFrame) -> None:
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    outlier_summary = {}

    for column in numeric_columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column].count()
        outlier_summary[column] = {
            "outlier_count": outlier_count,
            "outlier_percentage": round((outlier_count / len(df)) * 100, 2),
        }

    print("Outlier screening using IQR:")
    print(pd.DataFrame(outlier_summary).T)
    print()


def inspect_churn_by_segments(df: pd.DataFrame) -> None:
    segment_columns = [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "Partner",
        "Dependents",
        "PaperlessBilling",
        "SeniorCitizen",
    ]

    print("Churn rate by key segments:")
    for column in segment_columns:
        churn_by_segment = (
            df.groupby(column)["Churn"]
            .value_counts(normalize=True)
            .rename("proportion")
            .mul(100)
            .round(2)
            .reset_index()
        )
        churn_yes = churn_by_segment[churn_by_segment["Churn"] == "Yes"]
        print(f"{column}:")
        print(churn_yes[[column, "proportion"]].sort_values(by="proportion", ascending=False))
        print()


def main() -> None:
    df = load_telco_data(DATA_FILE)
    document_data_source(df, DATA_FILE)
    inspect_structure(df)
    inspect_target_balance(df)
    inspect_blank_strings(df)
    inspect_categorical_columns(df)
    numeric_df = inspect_numeric_columns(df)
    inspect_outliers(numeric_df)
    inspect_churn_by_segments(df)


if __name__ == "__main__":
    main()
