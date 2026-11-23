
import os
import joblib
import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse


# ==========================================
# STEP 1: CREATE OUTPUT FOLDERS
# ==========================================

OUTPUT_DIR = "reports/explainability"
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

print("Starting Model Explainability...")


# ==========================================
# STEP 2: LOAD TRAINED REGRESSION MODEL
# ==========================================

print("\nLoading trained regression model...")

model = joblib.load(
    "models/best_regression_model.pkl"
)

print("Regression model loaded successfully!")
print("Model type:", type(model).__name__)


# ==========================================
# STEP 3: LOAD PREPROCESSOR AND DATASETS
# ==========================================

print("\nLoading processed datasets...")

preprocessor = joblib.load(
    "models/preprocessor.pkl"
)

X_train = joblib.load(
    "data/processed/X_train_processed.pkl"
)

X_test = joblib.load(
    "data/processed/X_test_processed.pkl"
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# ==========================================
# STEP 4: CONVERT SPARSE MATRICES
# ==========================================

if sparse.issparse(X_train):
    X_train = X_train.toarray()

if sparse.issparse(X_test):
    X_test = X_test.toarray()

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)

# Select test records for explainability
X_sample = X_test[:500]

print("Explainability sample shape:", X_sample.shape)


# ==========================================
# STEP 5: VERIFY FEATURE DIMENSIONS
# ==========================================

if X_train.shape[1] != X_test.shape[1]:
    raise ValueError(
        "Training and testing feature dimensions do not match!"
    )

if X_train.shape[1] != model.n_features_in_:
    raise ValueError(
        "Model input feature count does not match processed data!"
    )

print("\nFeature dimension verification successful!")


# ==========================================
# STEP 6: EXTRACT FEATURE NAMES
# ==========================================

print("\nExtracting feature names...")

feature_names = preprocessor.get_feature_names_out()

feature_names = [
    name.replace("__", "_")
    for name in feature_names
]

if len(feature_names) != X_train.shape[1]:
    raise ValueError(
        "Feature names and processed feature dimensions do not match!"
    )

print("Total features:", len(feature_names))


# ==========================================
# STEP 7: FEATURE IMPORTANCE
# ==========================================

print("\nCalculating Feature Importance...")

if not hasattr(model, "feature_importances_"):
    raise AttributeError(
        "The selected regression model does not provide "
        "feature_importances_."
    )

importance_values = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_df.to_csv(
    os.path.join(OUTPUT_DIR, "feature_importance.csv"),
    index=False
)

print("\nTOP 15 IMPORTANT FEATURES")

print(
    importance_df.head(15).to_string(index=False)
)


# ==========================================
# STEP 8: FEATURE IMPORTANCE CHART
# ==========================================

top_features = importance_df.head(15)

plt.figure(figsize=(11, 7))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.title("Top 15 Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "feature_importance.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nFeature importance chart saved!")


# ==========================================
# STEP 9: SHAP ANALYSIS
# ==========================================

print("\nStarting SHAP analysis...")

explainer = shap.TreeExplainer(model)

shap_explanation = explainer(
    X_sample
)

print("SHAP values calculated successfully!")


# ==========================================
# STEP 10: SHAP SUMMARY PLOT
# ==========================================

print("\nGenerating SHAP summary plot...")

shap.plots.beeswarm(
    shap_explanation,
    max_display=15,
    show=False
)

plt.gcf().set_size_inches(11, 7)
plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "shap_summary.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("SHAP summary chart saved!")


# ==========================================
# STEP 11: SHAP FEATURE IMPORTANCE PLOT
# ==========================================

print("\nGenerating SHAP feature importance plot...")

shap.plots.bar(
    shap_explanation,
    max_display=15,
    show=False
)

plt.gcf().set_size_inches(11, 7)
plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "shap_feature_importance.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("SHAP feature importance chart saved!")


# ==========================================
# STEP 12: SAVE SHAP IMPORTANCE VALUES
# ==========================================

mean_shap_values = np.abs(
    shap_explanation.values
).mean(axis=0)

shap_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": mean_shap_values
})

shap_importance_df = shap_importance_df.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False
)

shap_importance_df.to_csv(
    os.path.join(OUTPUT_DIR, "shap_importance.csv"),
    index=False
)

print("\nTOP 15 SHAP FEATURES")

print(
    shap_importance_df.head(15).to_string(index=False)
)


# ==========================================
# STEP 13: LIME EXPLAINABILITY
# ==========================================

print("\nStarting LIME analysis...")

lime_explainer = lime.lime_tabular.LimeTabularExplainer(

    training_data=X_train[:5000],

    feature_names=feature_names,

    mode="regression",

    discretize_continuous=True,

    random_state=42

)

# Explain the first test record
sample_index = 0

sample_record = X_test[sample_index]

lime_explanation = lime_explainer.explain_instance(

    data_row=sample_record,

    predict_fn=model.predict,

    num_features=15

)


# ==========================================
# STEP 14: DISPLAY LIME RESULTS
# ==========================================

predicted_demand = model.predict(
    sample_record.reshape(1, -1)
)[0]

print("\nLIME EXPLANATION")

print(
    "Predicted Demand:",
    round(predicted_demand, 4)
)

print("\nFeatures influencing this prediction:")

for feature, weight in lime_explanation.as_list():

    print(f"{feature}: {weight:.4f}")


# ==========================================
# STEP 15: SAVE LIME RESULTS
# ==========================================

lime_explanation.save_to_file(
    os.path.join(OUTPUT_DIR, "lime_explanation.html")
)

lime_results = pd.DataFrame(
    lime_explanation.as_list(),
    columns=[
        "Feature_Condition",
        "LIME_Weight"
    ]
)

lime_results.to_csv(
    os.path.join(OUTPUT_DIR, "lime_explanation.csv"),
    index=False
)


# ==========================================
# STEP 16: SAVE LIME CHART
# ==========================================

fig = lime_explanation.as_pyplot_figure()

fig.tight_layout()

fig.savefig(
    os.path.join(CHART_DIR, "lime_explanation.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\nLIME explanation saved successfully!")


# ==========================================
# STEP 17: COMPLETION
# ==========================================

print("\n======================================")
print("MODEL EXPLAINABILITY COMPLETED!")
print("======================================")

print("\nGenerated files:")

print("1. feature_importance.csv")
print("2. shap_importance.csv")
print("3. lime_explanation.csv")
print("4. lime_explanation.html")

print("\nGenerated charts:")

print("1. feature_importance.png")
print("2. shap_summary.png")
print("3. shap_feature_importance.png")
print("4. lime_explanation.png")

print("\nAll explainability analyses completed successfully!")
