
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ==========================================
# STEP 1: LOAD DATA
# ==========================================

print("Loading classification datasets...")

X_train = joblib.load(
    "data/processed/X_train_processed.pkl"
)

X_test = joblib.load(
    "data/processed/X_test_processed.pkl"
)

y_train = pd.read_csv(
    "data/processed/y_train_clf.csv"
)["DemandCategory"]

y_test = pd.read_csv(
    "data/processed/y_test_clf.csv"
)["DemandCategory"]

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)


# ==========================================
# STEP 2: VERIFY DATA
# ==========================================

if len(X_train) != len(y_train):
    raise ValueError(
        "Training feature and target lengths do not match!"
    )

if len(X_test) != len(y_test):
    raise ValueError(
        "Testing feature and target lengths do not match!"
    )

if set(y_train.unique()) != {0, 1}:
    raise ValueError(
        "Training target must contain both classes 0 and 1!"
    )

print("\nDataset verification successful!")


# ==========================================
# STEP 3: CHRONOLOGICAL VALIDATION SPLIT
# ==========================================

print("\nSplitting training data into training and validation...")

split_index = int(len(X_train) * 0.80)

X_fit = X_train[:split_index]
X_val = X_train[split_index:]

y_fit = y_train.iloc[:split_index]
y_val = y_train.iloc[split_index:]

print("Model fitting records:", X_fit.shape)
print("Validation records:", X_val.shape)

print("\nFitting class distribution:")
print(y_fit.value_counts())

print("\nValidation class distribution:")
print(y_val.value_counts())


# ==========================================
# STEP 4: DEFINE CLASSIFICATION MODELS
# ==========================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=15,
        min_samples_leaf=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    ),

    "SVM": LinearSVC(
        max_iter=5000,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        n_jobs=-1,
        eval_metric="logloss",
        random_state=42
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=10,
        verbosity=-1,
        n_jobs=-1,
        random_state=42
    )
}


# ==========================================
# STEP 5: TRAIN AND EVALUATE ON VALIDATION
# ==========================================

results = []

print("\nStarting classification model training...")

for name, model in models.items():

    print("\n----------------------------------")
    print("Training:", name)
    print("----------------------------------")

    model.fit(X_fit, y_fit)

    predictions = model.predict(X_val)

    # Obtain continuous scores for ROC-AUC
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_val)[:, 1]
    else:
        scores = model.decision_function(X_val)

    accuracy = accuracy_score(y_val, predictions)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(y_val, scores)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc
    })

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))


# ==========================================
# STEP 6: SAVE VALIDATION RESULTS
# ==========================================

os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1_Score",
    ascending=False
)

results_df.to_csv(
    "reports/classification_validation_results.csv",
    index=False
)

print("\n======================================")
print("CLASSIFICATION VALIDATION RESULTS")
print("======================================")

print(results_df.to_string(index=False))


# ==========================================
# STEP 7: SELECT BEST MODEL
# ==========================================

best_model_name = results_df.iloc[0]["Model"]

print("\nBest model selected:", best_model_name)

best_model = models[best_model_name]


# ==========================================
# STEP 8: RETRAIN ON FULL TRAINING DATA
# ==========================================

print("\nRetraining best model on full training data...")

best_model.fit(
    X_train,
    y_train
)


# ==========================================
# STEP 9: FINAL TEST EVALUATION
# ==========================================

print("\nEvaluating selected model on unseen test data...")

test_predictions = best_model.predict(X_test)

if hasattr(best_model, "predict_proba"):
    test_scores = best_model.predict_proba(X_test)[:, 1]
else:
    test_scores = best_model.decision_function(X_test)

test_metrics = {
    "Model": best_model_name,

    "Accuracy": accuracy_score(
        y_test,
        test_predictions
    ),

    "Precision": precision_score(
        y_test,
        test_predictions,
        zero_division=0
    ),

    "Recall": recall_score(
        y_test,
        test_predictions,
        zero_division=0
    ),

    "F1_Score": f1_score(
        y_test,
        test_predictions,
        zero_division=0
    ),

    "ROC_AUC": roc_auc_score(
        y_test,
        test_scores
    )
}


# ==========================================
# STEP 10: DISPLAY FINAL TEST RESULTS
# ==========================================

print("\n======================================")
print("FINAL CLASSIFICATION TEST RESULTS")
print("======================================")

for metric, value in test_metrics.items():

    if metric == "Model":
        print(f"{metric}: {value}")
    else:
        print(f"{metric}: {value:.4f}")


# ==========================================
# STEP 11: SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_model,
    "models/best_classification_model.pkl"
)


# ==========================================
# STEP 12: SAVE FINAL TEST RESULTS
# ==========================================

pd.DataFrame([test_metrics]).to_csv(
    "reports/classification_test_results.csv",
    index=False
)

prediction_results = pd.DataFrame({
    "Actual_DemandCategory": y_test.values,
    "Predicted_DemandCategory": test_predictions,
    "Prediction_Score": test_scores
})

prediction_results.to_csv(
    "reports/classification_test_predictions.csv",
    index=False
)


# ==========================================
# STEP 13: FINAL SUMMARY
# ==========================================

print("\nBest Classification Model:", best_model_name)

print("Final Test Accuracy:", round(test_metrics["Accuracy"], 4))
print("Final Test Precision:", round(test_metrics["Precision"], 4))
print("Final Test Recall:", round(test_metrics["Recall"], 4))
print("Final Test F1 Score:", round(test_metrics["F1_Score"], 4))
print("Final Test ROC-AUC:", round(test_metrics["ROC_AUC"], 4))

print("\nBest classification model saved successfully!")

print("\nCLASSIFICATION TRAINING COMPLETED SUCCESSFULLY!")
