import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, auc
)
from imblearn.over_sampling import SMOTE
import pickle

# Load dataset
df = pd.read_csv("mental_health_analysis.csv")
df.drop(columns=['User_ID'], inplace=True)

# Encode categorical variables
df['Gender'] = df['Gender'].map({'M': 0, 'F': 1})
df['Support_System'] = df['Support_System'].map({'Poor': 0, 'Moderate': 1, 'Strong': 2})
df['Academic_Performance'] = df['Academic_Performance'].map({
    'Excellent': 1, 'Good': 0, 'Average': 0, 'Poor': 0
})

# Drop rows with missing values
df.dropna(inplace=True)

# Feature engineering
df['Combined_Stress'] = df['Survey_Stress_Score'] + df['Wearable_Stress_Score']
df['Productivity_Ratio'] = df['Exercise_Hours'] / (df['Social_Media_Hours'] + 1)
df['Sleep_Quality'] = df['Sleep_Hours'] / (df['Screen_Time_Hours'] + 1)

# Features and target
X = df.drop(columns=['Academic_Performance'])
y = df['Academic_Performance']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SMOTE for balancing
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)

# Base models
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
svm = SVC(probability=True, kernel='rbf', C=10, gamma=0.1, random_state=42)

# Stacking classifier
stacking_model = StackingClassifier(
    estimators=[
        ('xgb', xgb),
        ('rf', rf),
        ('svm', svm)
    ],
    final_estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    passthrough=True
)

# Train the model
stacking_model.fit(X_train, y_train)

# Predictions
y_pred = stacking_model.predict(X_test)
y_proba = stacking_model.predict_proba(X_test)[:, 1]

# Evaluation metrics
print("Final Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# Save model
with open("final_academic_model.pkl", "wb") as f:
    pickle.dump(stacking_model, f)

# Save predictions
pred_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Probability': y_proba
})
pred_df.to_csv("final_predictions.csv", index=False)

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Excellent", "Excellent"], yticklabels=["Not Excellent", "Excellent"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# ROC Curve Plot
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.show()

# Feature Importance Plot (XGBoost)
xgb_model = stacking_model.named_estimators_['xgb']
importances = xgb_model.feature_importances_
feature_names = df.drop(columns=['Academic_Performance']).columns

plt.figure(figsize=(8, 6))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance (XGBoost)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()
