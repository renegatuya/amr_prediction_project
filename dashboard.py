import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Load model and figures
model = joblib.load('best_amr_model.pkl')
model_comparison_img = Image.open('/Users/ndore/Academics/PhD/courses/python_1/assignements/amr_prediction_project/outputs/model_comparison.png')

# Streamlit Dashboard
st.title("AMR Prediction Dashboard")

# Display model comparison figure
st.subheader("Model Comparison")
st.image(model_comparison_img, caption='Model Comparison')

#  Upload CSV for predictions
uploaded_file = st.file_uploader("Choose a CSV file for prediction", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data preview:", df.head())

    #  model predictions
    predictions = model.predict(df) 
    st.subheader("Predictions")
    st.write(predictions)

    # Plot a simple graph with matplotlib
    st.subheader("Predictions Over Time")
    plt.figure(figsize=(10, 6))
    plt.plot(predictions)
    plt.title("Predictions")
    st.pyplot()

