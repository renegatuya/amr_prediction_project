AMR Prediction Project
Predicitng antimicrobial resistance (AMR) trends of GIT infections data,
using synthetic/dummy climate, and genomic data from East African Great Lakes region countries.

Structure
amr_prediction_project/ ├── data/ #  synthetic data in xls
├── figures/ # Output plots comparing models 
├── dashboard.py # Streamlit dashboard for predictions 
├── generate_data.py # Script to generate synthetic dataset 
├── tune_amr_model.py # Model training & hyperparameter tuning 
├── predict_amr.py # Predict AMR with trained model 
├── predict_on_unknown.py # Predict on new, unseen data (also sysntehtic)
├── best_amr_model.pkl # Saved trained model 
├── model_features.pkl # Model feature names 
├── amr_predictions.csv # Predictions made by the model 
└── README.md # documentation

Author: Rene Ndoyi, PhD Student
Student Number: 
