import pandas as pd
import joblib

# Load the saved model
model = joblib.load('models/amr_predictor.pkl')

# Load new data (replace 'new_data.csv' with the actual data path)
new_data = pd.read_csv('/Users/ndore/Academics/PhD/courses/python_1/assignements/amr_prediction_project/outputs/synthetic_git_amr_data.csv')

# Preprocess the new data (apply any necessary transformations that were applied to the training data)
# Ensure the columns in the new data match the columns used during model training

# Make predictions on the new data
predictions = model.predict(new_data)

# Save the predictions to a CSV file
new_data['predictions'] = predictions
new_data.to_csv('predictions.csv', index=False)

print("Predictions for new data saved to 'predictions.csv'")

