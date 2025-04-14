import joblib
import pandas as pd

# Load the trained model
model = joblib.load('best_amr_model.pkl')

# Load new dummy made data for prediction
new_data = pd.read_csv('/Users/ndore/Academics/PhD/courses/python_1/assignements/amr_prediction_project/outputs/synthetic_git_amr_data.csv')

# One-hot encode categorical features
new_data_encoded = pd.get_dummies(new_data, columns=["country", "district", "pathogen"], drop_first=True)
X_new = new_data_encoded.drop(columns=["week", "amr_abundance"])

# Predict using the trained model
predictions = model.predict(X_new)

# Print predictions
print(predictions)

