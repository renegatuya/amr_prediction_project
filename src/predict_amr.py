import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import os

# Load dummy data
df = pd.read_csv("../outputs/synthetic_git_amr_data.csv")

# Relevant columns for training
X = df[["cases", "amr_abundance"]]  # Features (predictor variables)
y = (df["amr_abundance"] > 0.5).astype(int)  # Label: 1 if AMR abundance > 0.5, else 0 (binary classification)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# start models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "XGBoost": XGBClassifier()
}
results = []

# Train and evaluate performance of each model
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC

    # Calculate metrics
    accuracy = model.score(X_test, y_test)
    auc = roc_auc_score(y_test, y_pred_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Confusion Matrix": cm,
        "Classification Report": report
    })

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"ROC Curve for {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    os.makedirs("../figures", exist_ok=True)
    plt.savefig(f"../figures/roc_curve_{name}.png")
    plt.close()

# Compare results
results_df = pd.DataFrame(results)
print(results_df[["Model", "Accuracy", "AUC"]])
results_df.to_csv("../outputs/model_comparison.csv", index=False)

# Save confusion matrices as figures
for result in results:
    cm = result["Confusion Matrix"]
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No AMR", "AMR"], yticklabels=["No AMR", "AMR"])
    plt.title(f"Confusion Matrix for {result['Model']}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(f"../figures/confusion_matrix_{result['Model']}.png")
    plt.close()

# classification reports for reference
for result in results:
    model_name = result["Model"]
    report = result["Classification Report"]
    with open(f"../figures/classification_report_{model_name}.txt", "w") as f:
        f.write(report)

print("Model comparisons, ROC curves, confusion matrices, and classification reports have been saved to the figures directory.")

