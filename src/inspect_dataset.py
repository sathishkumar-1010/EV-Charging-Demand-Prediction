
import pandas as pd
from pathlib import Path

# Get project folder location
project_path = Path(__file__).resolve().parent.parent

# Locate raw dataset folder
raw_folder = project_path / "data" / "raw"

# Find CSV files
csv_files = list(raw_folder.glob("*.csv"))

if not csv_files:
    print("No CSV file found!")
    raise SystemExit

if len(csv_files) > 1:
    print("Multiple CSV files found:")
    for file in csv_files:
        print(file.name)
    raise SystemExit

# Load dataset
file_path = csv_files[0]
df = pd.read_csv(file_path, low_memory=False)

print("\n========== DATASET INFORMATION ==========")
print("File Name:", file_path.name)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n========== FIRST 5 RECORDS ==========")
print(df.head())

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE RECORDS ==========")
print("Duplicates:", df.duplicated().sum())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe(include="all").T)

print("\nDataset inspection completed!")
