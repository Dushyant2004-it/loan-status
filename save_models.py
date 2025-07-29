import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# Load the data
df = pd.read_csv('loan_approval_dataset.csv')

# Clean column names
df.columns = [col.strip() for col in df.columns]

# Prepare features
X = df.drop(['loan_id', 'loan_status'], axis=1)
y = df['loan_status'].map({' Approved': 1, ' Rejected': 0})

# Convert categorical variables to numeric
X = pd.get_dummies(X, columns=['education', 'self_employed'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Train Logistic Regression model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Save models and scaler
joblib.dump(rf_model, 'models/random_forest_model.pkl')
joblib.dump(lr_model, 'models/logistic_regression_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Save feature names
joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')

print("Models saved successfully!")
