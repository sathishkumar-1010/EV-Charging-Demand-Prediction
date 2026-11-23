
import pandas as pd
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ==========================================
# STEP 1: LOAD TRAINING AND TESTING DATA
# ==========================================

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTraining target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)


# ==========================================
# STEP 2: VERIFY INPUT DATA
# ==========================================

if list(X_train.columns) != list(X_test.columns):
    raise ValueError(
        "Training and testing feature columns do not match!"
    )

if len(X_train) != len(y_train):
    raise ValueError(
        "Training features and targets have different lengths!"
    )

if len(X_test) != len(y_test):
    raise ValueError(
        "Testing features and targets have different lengths!"
    )

print("\nInput data verification successful!")


# ==========================================
# STEP 3: IDENTIFY FEATURE TYPES
# ==========================================

categorical_features = [
    "Location",
    "DayName",
    "TimeOfDay"
]

numerical_features = [
    column for column in X_train.columns
    if column not in categorical_features
]

print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


# ==========================================
# STEP 4: CREATE PREPROCESSING PIPELINES
# ==========================================

# Categorical preprocessing
categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

# Numerical preprocessing
numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ==========================================
# STEP 5: COMBINE BOTH PIPELINES
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        )
    ]
)


# ==========================================
# STEP 6: FIT AND TRANSFORM DATA
# ==========================================

# Fit only on training data
X_train_processed = preprocessor.fit_transform(
    X_train
)

# Transform testing data using the same fitted pipeline
X_test_processed = preprocessor.transform(
    X_test
)

print("\nPreprocessing completed successfully!")


# ==========================================
# STEP 7: VERIFY PROCESSED DATA
# ==========================================

print("\nOriginal training shape:", X_train.shape)
print("Processed training shape:", X_train_processed.shape)

print("\nOriginal testing shape:", X_test.shape)
print("Processed testing shape:", X_test_processed.shape)

print(
    "\nNumber of processed features:",
    X_train_processed.shape[1]
)

if X_train_processed.shape[1] != X_test_processed.shape[1]:
    raise ValueError(
        "Processed training and testing feature counts do not match!"
    )

print("\nProcessed data verification successful!")


# ==========================================
# STEP 8: SAVE PREPROCESSOR AND DATA
# ==========================================

output_folder = "data/processed"
models_folder = "models"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(models_folder, exist_ok=True)

# Save fitted preprocessor
joblib.dump(
    preprocessor,
    os.path.join(models_folder, "preprocessor.pkl")
)

# Save processed training features
joblib.dump(
    X_train_processed,
    os.path.join(output_folder, "X_train_processed.pkl")
)

# Save processed testing features
joblib.dump(
    X_test_processed,
    os.path.join(output_folder, "X_test_processed.pkl")
)


# ==========================================
# STEP 9: VERIFY SAVED FILES
# ==========================================

loaded_preprocessor = joblib.load(
    os.path.join(models_folder, "preprocessor.pkl")
)

loaded_train = joblib.load(
    os.path.join(output_folder, "X_train_processed.pkl")
)

loaded_test = joblib.load(
    os.path.join(output_folder, "X_test_processed.pkl")
)

print("\nSaved files loaded and verified successfully!")

print("Loaded training shape:", loaded_train.shape)
print("Loaded testing shape:", loaded_test.shape)

print("\nPREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
