# ==========================================
# 🚀 ADVANCED FRAUD DETECTION TRAINING SCRIPT
# ==========================================
# Copy and Paste this ENTIRE script into a Google Colab cell.
# Make sure you have uploaded 'creditcard.csv' to the Colab runtime first.

# 1. INSTALL DEPENDENCIES
import os
os.system('pip install xgboost imbalanced-learn joblib scikit-learn pandas numpy matplotlib seaborn')

# 2. IMPORTS
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb

print("✅ Libraries Imported Successfully")

# 3. LOAD & PREPROCESS DATA
print("⏳ Loading Dataset...")
try:
    df = pd.read_csv('creditcard.csv')
except FileNotFoundError:
    print("❌ ERROR: 'creditcard.csv' not found. Please upload it to Colab files!")
    exit()

print(f"   Original Shape: {df.shape}")

# Scale 'Amount' using RobustScaler (better for outliers)
scaler = RobustScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

# Drop 'Time' (not useful) and original 'Amount'
df.drop(['Time', 'Amount'], axis=1, inplace=True)

# Define X and y
X = df.drop('Class', axis=1)
y = df['Class']

# Ensure Column Order matches API: [V1, V2, ... V28, scaled_amount]
# This is crucial for the API to work correctly!
cols = [f'V{i}' for i in range(1, 29)] + ['scaled_amount']
X = X[cols]

print("✅ Preprocessing Complete")
print(f"   Feature Columns: {X.columns.tolist()}")

# 4. SPLIT DATA
# Stratified split to maintain fraud ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"   Training Set: {X_train.shape}")
print(f"   Test Set: {X_test.shape}")

# 5. HANDLE IMBALANCE (SMOTE)
print("⏳ Applying SMOTE (Synthetic Minority Over-sampling)...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"   Resampled Training Shape: {X_train_res.shape}")

# 6. DEFINE ADVANCED MODELS
print("⏳ Initializing Ensemble Models...")

# Model 1: XGBoost (Gradient Boosting)
clf_xgb = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method='hist', # Faster
    random_state=42,
    n_jobs=-1
)

# Model 2: Random Forest (Bagging - stabilizes variance)
clf_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# Model 3: Logistic Regression (Baseline - good for linear boundaries)
# clf_lr = LogisticRegression(solver='liblinear', random_state=42) # Optional, can skip for speed

# --- VOTING CLASSIFIER (ENSEMBLE) ---
# Combines predictions from both models. 
# Soft voting means it averages the probabilities.
ensemble_model = VotingClassifier(
    estimators=[
        ('xgb', clf_xgb), 
        ('rf', clf_rf)
    ],
    voting='soft',
    n_jobs=-1
)

# 7. TRAIN
print("\n🚀 Starting Training (This may take 2-5 minutes)...")
ensemble_model.fit(X_train_res, y_train_res)
print("✅ Training Complete!")

# 8. EVALUATION
print("\n--- 📊 MODEL EVALUATION ---")
y_pred = ensemble_model.predict(X_test)
y_prob = ensemble_model.predict_proba(X_test)[:, 1]

# Report
print(classification_report(y_test, y_pred))

# AUPRC
auprc = average_precision_score(y_test, y_prob)
print(f"🌟 AUPRC Score: {auprc:.4f} (Target > 0.85)")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix\nAUPRC: {auprc:.3f}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# 9. SAVE ARTIFACTS
print("\n💾 Saving Model and Scaler...")
joblib.dump(ensemble_model, 'ensemble_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Files Saved: 'ensemble_model.pkl', 'scaler.pkl'")
print("👉 Please verify the AUPRC score is high, then download these two files!")
print("👉 Place them in your local 'src/model/' folder.")
