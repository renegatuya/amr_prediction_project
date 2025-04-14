
import pandas as pd
import joblib

# Load model and model features
model = joblib.load("best_amr_model.pkl")
model_features = joblib.load("model_features.pkl")

# Load and process new data
df = pd.read_csv("/Users/ndore/Academics/PhD/courses/python_1/assignements/amr_prediction_project/outputs/synthetic_git_amr_data.csv")

# Categorical variables to numeric values using one-hot encoding
df_encoded = pd.get_dummies(df, columns=["country", "district", "pathogen"], drop_first=True)

#Matching the data:  Ensure that the new data has the same columns as the model's training data
X_new = df_encoded.reindex(columns=model_features, fill_value=0)

# Predictions based on the new data
predictions = model.predict(X_new)

# Print the predictions
print(predictions)

