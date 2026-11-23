
import pandas as pd
import joblib
import os


# ==========================================
# STEP 1: LOAD TRAINING AND TESTING TARGETS
# ==========================================

y_train = pd.read_csv(
    "data/processed/y_train.csv"
)["Demand"]

y_test = pd.read_csv(
    "data/processed/y_test.csv"
)["Demand"]

print("Training target records:", len(y_train))
print("Testing target records:", len(y_test))


# ==========================================
# STEP 2: VERIFY TARGET DATA
# ==========================================

if y_train.isnull().any() or y_test.isnull().any():
    raise ValueError(
        "Missing values found in target data!"
    )

if len(y_train) == 0 or len(y_test) == 0:
    raise ValueError(
        "Training or testing target dataset is empty!"
    )

print("\nTarget data verification successful!")


# ==========================================
# STEP 3: REGRESSION TARGET
# ==========================================

# Regression predicts actual numerical demand.

y_train_reg = y_train.copy()
y_test_reg = y_test.copy()

print("\nREGRESSION TARGET")
print("Training shape:", y_train_reg.shape)
print("Testing shape:", y_test_reg.shape)


# ==========================================
# STEP 4: CLASSIFICATION TARGET
# ==========================================

# Calculate threshold using training data only.
# Never calculate the threshold using test data.

demand_threshold = y_train.median()

print("\nHigh Demand threshold:", demand_threshold)

# High Demand = 1
# Normal Demand = 0

y_train_clf = (
    y_train >= demand_threshold
).astype(int)

y_test_clf = (
    y_test >= demand_threshold
).astype(int)


# ==========================================
# STEP 5: DISPLAY CLASS DISTRIBUTION
# ==========================================

print("\nTRAINING CLASS DISTRIBUTION")

print(
    y_train_clf.value_counts()
    .reindex([0, 1], fill_value=0)
    .rename(
        index={
            0: "Normal Demand",
            1: "High Demand"
        }
    )
)

print("\nTESTING CLASS DISTRIBUTION")

print(
    y_test_clf.value_counts()
    .reindex([0, 1], fill_value=0)
    .rename(
        index={
            0: "Normal Demand",
            1: "High Demand"
        }
    )
)


# ==========================================
# STEP 6: SAVE TARGETS
# ==========================================

output_folder = "data/processed"
models_folder = "models"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(models_folder, exist_ok=True)


# Regression targets

y_train_reg.to_frame(
    name="Demand"
).to_csv(
    f"{output_folder}/y_train_reg.csv",
    index=False
)

y_test_reg.to_frame(
    name="Demand"
).to_csv(
    f"{output_folder}/y_test_reg.csv",
    index=False
)


# Classification targets

y_train_clf.to_frame(
    name="DemandCategory"
).to_csv(
    f"{output_folder}/y_train_clf.csv",
    index=False
)

y_test_clf.to_frame(
    name="DemandCategory"
).to_csv(
    f"{output_folder}/y_test_clf.csv",
    index=False
)


# ==========================================
# STEP 7: SAVE CLASSIFICATION THRESHOLD
# ==========================================

joblib.dump(
    demand_threshold,
    f"{models_folder}/demand_threshold.pkl"
)


# ==========================================
# STEP 8: VERIFY SAVED FILES
# ==========================================

loaded_threshold = joblib.load(
    f"{models_folder}/demand_threshold.pkl"
)

if loaded_threshold != demand_threshold:
    raise ValueError(
        "Saved classification threshold verification failed!"
    )

print("\nSaved target files successfully!")

print("Regression training target:", len(y_train_reg))
print("Regression testing target:", len(y_test_reg))

print("Classification training target:", len(y_train_clf))
print("Classification testing target:", len(y_test_clf))

print("Classification threshold:", loaded_threshold)

print("\nTARGET PREPARATION COMPLETED SUCCESSFULLY!")
