
import pandas as pd
from pathlib import Path

# 1. PROJECT PATHS


project_path = Path(__file__).resolve().parent.parent

raw_folder = project_path / "data" / "raw"
interim_folder = project_path / "data" / "interim"

interim_folder.mkdir(parents=True, exist_ok=True)

file_path = raw_folder / "ChargingRecords.csv"


# 2. LOAD ORIGINAL DATASET


df = pd.read_csv(file_path, low_memory=False)

print("Original Dataset Shape:", df.shape)


# 3. REMOVE DUPLICATE RECORDS


duplicate_count = df.duplicated().sum()

print("\nDuplicate Records Found:", duplicate_count)

df = df.drop_duplicates()

print("Duplicates Removed:", duplicate_count)
print("Dataset Shape After Removing Duplicates:", df.shape)


# 4. CONVERT DATETIME COLUMNS


datetime_columns = ["StartDatetime", "EndDatetime"]

for column in datetime_columns:

    df[column] = pd.to_datetime(
        df[column],
        errors="coerce"
    )

    print(
        f"\nInvalid dates in {column}:",
        df[column].isnull().sum()
    )


# 5. CHECK INVALID VALUES


print("\n DATA VALIDATION ")

print(
    "Zero or negative Duration:",
    (df["Duration"] <= 0).sum()
)

print(
    "Zero or negative Demand:",
    (df["Demand"] <= 0).sum()
)

print(
    "EndDatetime before StartDatetime:",
    (df["EndDatetime"] < df["StartDatetime"]).sum()
)


# 6. CHECK MISSING VALUES


print("\n MISSING VALUES ")

print(df.isnull().sum())


# 7. SAVE CLEANED DATASET

output_path = interim_folder / "charging_cleaned.csv"

df.to_csv(output_path, index=False)


# 8. FINAL SUMMARY


print("\n CLEANING SUMMARY ")

print("Original Records:", 72856)
print("Cleaned Records:", len(df))
print("Total Columns:", len(df.columns))

print("\nSaved Dataset:", output_path)

print("\nData Cleaning Completed Successfully!")
