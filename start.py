# ==========================================
# Diabetes Prediction & Analytics Project
# Author: Hero
# ==========================================

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ==============================
# 1. Load Dataset
# ==============================
df = pd.read_csv("diabetes.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ==============================
# 2. Data Preprocessing
# ==============================
# Columns where 0 is invalid
invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in invalid_cols:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].median(), inplace=True)

# Features & Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 3. Exploratory Data Analysis (EDA)
# ==============================
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

plt.figure(figsize=(8,6))
sns.histplot(data=df, x="Age", hue="Outcome", kde=True)
plt.title("Age Distribution vs Diabetes")
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x="Outcome", y="Glucose", data=df)
plt.title("Glucose Levels by Outcome")
plt.show()

# ==============================
# 4. Train Models
# ==============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    print(f"\n{name}")
    print("Accuracy:", acc)
    print("AUC:", auc)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    results[name] = acc

# ==============================
# 5. Save Best Model
# ==============================
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
joblib.dump(best_model, "best_diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print(f"\nBest Model Saved: {best_model_name}")
