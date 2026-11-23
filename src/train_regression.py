
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# ==========================================
# STEP 1: LOAD PROCESSED DATA
# ==========================================

X_train = joblib.load(
    "data/processed/X_train_processed.pkl"
)

X_test = joblib.load(
    "data/processed/X_test_processed.pkl"
)

y_train = pd.read_csv(
    "data/processed/y_train_reg.csv"
)["Demand"]

y_test = pd.read_csv(
    "data/processed/y_test_reg.csv"
)["Demand"]

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

if len(X_train) != len(y_train):
    raise ValueError("Training feature and target lengths do not match!")

if len(X_test) != len(y_test):
    raise ValueError("Testing feature and target lengths do not match!")


# ==========================================
# STEP 2: CHRONOLOGICAL VALIDATION SPLIT
# ==========================================

# Use the first 80% of training data for fitting
# and the final 20% for validation.

split_index = int(len(X_train) * 0.80)

X_fit = X_train[:split_index]
X_validation = X_train[split_index:]

y_fit = y_train.iloc[:split_index]
y_validation = y_train.iloc[split_index:]

print("\nVALIDATION SPLIT")
print("Model fitting records:", len(X_fit))
print("Validation records:", len(X_validation))


# ==========================================
# STEP 3: DEFINE REGRESSION MODELS
# ==========================================

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=15,
        min_samples_leaf=5,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror"
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=10,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
}


# ==========================================
# STEP 4: TRAIN AND EVALUATE ON VALIDATION
# ==========================================

validation_results = []

best_model_name = None
best_rmse = float("inf")

best_validation_model = None

for name, model in models.items():

    print(f"\nTraining {name}...")

    # Train using fitting data only
    model.fit(X_fit, y_fit)

    # Predict validation demand
    predictions = model.predict(X_validation)

    mae = mean_absolute_error(
        y_validation, predictions
    )

    mse = mean_squared_error(
        y_validation, predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_validation, predictions
    )

    print("Validation MAE:", round(mae, 4))
    print("Validation MSE:", round(mse, 4))
    print("Validation RMSE:", round(rmse, 4))
    print("Validation R2:", round(r2, 4))

    validation_results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    # Select best model using validation RMSE
    if rmse < best_rmse:

        best_rmse = rmse
        best_model_name = name
        best_validation_model = model


# ==========================================
# STEP 5: SAVE VALIDATION RESULTS
# ==========================================

os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

validation_df = pd.DataFrame(validation_results)

validation_df = validation_df.sort_values(
    by="RMSE"
)

validation_df.to_csv(
    "reports/regression_validation_results.csv",
    index=False
)

print("\n========== REGRESSION VALIDATION RESULTS ==========")
print(validation_df.to_string(index=False))

print("\nSelected Model:", best_model_name)
print("Validation RMSE:", round(best_rmse, 4))


# ==========================================
# STEP 6: RETRAIN BEST MODEL ON FULL TRAINING DATA
# ==========================================

print("\nRetraining selected model on full training data...")

final_model = models[best_model_name]

final_model.fit(
    X_train,
    y_train
)


# ==========================================
# STEP 7: FINAL TEST EVALUATION
# ==========================================

print("\nEvaluating selected model on unseen test data...")

test_predictions = final_model.predict(X_test)

test_mae = mean_absolute_error(
    y_test, test_predictions
)

test_mse = mean_squared_error(
    y_test, test_predictions
)

test_rmse = np.sqrt(test_mse)

test_r2 = r2_score(
    y_test, test_predictions
)

test_results = pd.DataFrame([
    {
        "Model": best_model_name,
        "MAE": test_mae,
        "MSE": test_mse,
        "RMSE": test_rmse,
        "R2 Score": test_r2
    }
])

print("\n========== FINAL REGRESSION TEST RESULTS ==========")
print(test_results.to_string(index=False))


# ==========================================
# STEP 8: SAVE FINAL MODEL AND RESULTS
# ==========================================

joblib.dump(
    final_model,
    "models/best_regression_model.pkl"
)

test_results.to_csv(
    "reports/regression_model_results.csv",
    index=False
)

# Save test predictions for further analysis
prediction_results = pd.DataFrame({
    "Actual_Demand": y_test.values,
    "Predicted_Demand": test_predictions
})

prediction_results.to_csv(
    "reports/regression_test_predictions.csv",
    index=False
)


# ==========================================
# STEP 9: FINAL SUMMARY
# ==========================================

print("\nBest Regression Model:", best_model_name)

print("Validation RMSE:", round(best_rmse, 4))
print("Final Test MAE:", round(test_mae, 4))
print("Final Test RMSE:", round(test_rmse, 4))
print("Final Test R2 Score:", round(test_r2, 4))

print("\nAll regression training steps completed successfully!")
