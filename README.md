# Academic Performance Prediction

This project predicts academic performance of students using data related to mental health, lifestyle, and behavioral patterns.

## Project Files

- `academic_performance_prediction.py` – Final script with preprocessing, model training, and evaluation
- `mental_health_analysis.csv` – Dataset used (only if sharing is permitted)
- `final_academic_model.pkl` – Trained machine learning model (Stacking Ensemble)
- `final_predictions.csv` – Predictions from the model on the test set
- `evaluation_metrics.png` – Visual evaluation including confusion matrix and ROC curve

## Dataset Features

The dataset includes features such as:

- Stress scores (survey-based and wearable-based)
- Sleep duration
- Screen time
- Social media usage
- Exercise habits
- Support system rating
- Gender

The target variable is academic performance, classified as Excellent or Not Excellent.

## Machine Learning Approach

- Data cleaning and feature engineering
- Categorical encoding
- Feature scaling using StandardScaler
- Handling class imbalance using SMOTE
- Models used in stacking ensemble: Random Forest, XGBoost, and Support Vector Machine
- Meta-model: XGBoost

## Performance

- Accuracy: 82.38 percent
- ROC-AUC Score: 0.89
- Precision, Recall, F1-score also evaluated

## Requirements

Install the necessary dependencies using the command:

