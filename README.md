AMR Prediction Project
Predicting antimicrobial resistance (AMR) trends of GIT infections in East African Great Lakes region countries,
using synthetic/dummy climate, and genomic data from 


Structure
amr_prediction_project/ ├── data/ #  synthetic data in xls
├── figures/ # Output plots comparing models 
├── dashboard.py # Streamlit dashboard for predictions 
├── generate_data.py # Script to generate synthetic dataset 
├── tune_amr_model.py # Model training & tuning 
├── predict_amr.py # Predicting AMR with trained model 
├── predict_on_unknown.py # Predict on new data(also synthetic)
├── best_amr_model.pkl # Saved trained model 
├── model_features.pkl # Model features 
├── amr_predictions.csv # Predictions made by the model 
└── README.md # documentation

Author: Rene Ndoyi, PhD Student
Student Number: 251217
