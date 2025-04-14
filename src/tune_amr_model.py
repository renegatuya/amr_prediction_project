import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Paths
data_path = "../outputs/synthetic_git_amr_data.csv"
figures_dir = "../figures"
os.makedirs(figures_dir, exist_ok=True)

# Load data
df = pd.read_csv(data_path)

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=["country", "district", "pathogen"], drop_first=True)

# Features and target
X = df_encoded.drop(columns=["week", "amr_abundance"])
joblib.dump(X.columns, "model_features.pkl")
y = df_encoded["amr_abundance"]

# Save feature names
joblib.dump(X.columns, 'model_features.pkl')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
# Save the trained model
joblib.dump(model, 'best_amr_model.pkl')
#Save the model
import joblib


# ---------------------------
# Quick Random Forest Tuning
# ---------------------------
rf = RandomForestRegressor(random_state=42)
param_dist_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}
rf_random = RandomizedSearchCV(rf, param_distributions=param_dist_rf, n_iter=10, cv=3, n_jobs=-1, random_state=42)
rf_random.fit(X_train, y_train)

best_rf = rf_random.best_estimator_
y_pred_rf = best_rf.predict(X_test)

# ---------------------------
# Quick SVR Tuning
# ---------------------------
svr = SVR()
param_dist_svr = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}
svr_random = RandomizedSearchCV(svr, param_distributions=param_dist_svr, n_iter=5, cv=3, n_jobs=-1, random_state=42)
svr_random.fit(X_train, y_train)

best_svr = svr_random.best_estimator_
y_pred_svr = best_svr.predict(X_test)

# ---------------------------
# Compare Models
# ---------------------------
def evaluate_model(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name} MSE: {mse:.2f}, R²: {r2:.2f}")
    return {"model": name, "MSE": mse, "R2": r2}

results = [
    evaluate_model("Random Forest", y_test, y_pred_rf),
    evaluate_model("SVR", y_test, y_pred_svr)
]

results_df = pd.DataFrame(results)

# Plot results
plt.figure(figsize=(8, 4))
sns.barplot(data=results_df, x="model", y="R2", palette="viridis")
plt.title("Model Comparison (R² Score)")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "model_comparison_r2.png"))
plt.close()

plt.figure(figsize=(8, 4))
sns.barplot(data=results_df, x="model", y="MSE", palette="magma")
plt.title("Model Comparison (MSE)")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "model_comparison_mse.png"))
plt.close()

print("Model tuning complete. Visuals saved in 'figures/' directory.")

import joblib

# After you've finished tuning and identifying the best model, save it
best_model = rf_random.best_estimator_  # This assumes you're using Random Forest
joblib.dump(best_model, 'best_amr_model.pkl')

import os
import joblib

# Ensure the 'models' directory exists
if not os.path.exists('models'):
    os.makedirs('models')

# Save the best model
joblib.dump(best_model, 'best_amr_model.pkl')

