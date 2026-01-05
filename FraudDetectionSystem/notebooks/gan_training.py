# ==========================================
# 🚀 GAN-ENHANCED TRAINING SCRIPT (CTGAN)
# ==========================================
# This script uses Generative AI (GANs) to create REALISTIC fake fraud data.
# It usually results in higher precision than SMOTE.
# ⚠️ CTGAN is computationally heavy! Enable GPU in Colab (Runtime > Change runtime type > T4 GPU).

# 1. INSTALL
import os
print("⏳ Installing CTGAN and dependencies (this takes a moment)...")
os.system('pip install ctgan sdv xgboost imbalanced-learn joblib scikit-learn pandas numpy matplotlib seaborn')

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
from imblearn.over_sampling import SMOTE # Fallback
import xgboost as xgb

# Try importing CTGAN
try:
    from ctgan import CTGAN
    GAN_AVAILABLE = True
    print("✅ CTGAN Library Loaded")
except ImportError:
    print("⚠️ CTGAN install failed. Using SMOTE fallback.")
    GAN_AVAILABLE = False

# 2. LOAD & PREPROCESS
print("⏳ Loading Data...")
df = pd.read_csv('creditcard.csv')

# Scale Amount
scaler = RobustScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df.drop(['Time', 'Amount'], axis=1, inplace=True)
cols = [f'V{i}' for i in range(1, 29)] + ['scaled_amount']
X = df[cols]
y = df['Class']

# 3. SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"   Train Size: {X_train.shape[0]}")

# 4. TRAIN GAN (GENERATIVE AI)
if GAN_AVAILABLE:
    print("\n🔮 Initializing CTGAN (Generative Adversarial Network)...")
    
    # Isolate Fraud Cases for Training
    train_data = pd.concat([X_train, y_train], axis=1)
    fraud_data = train_data[train_data['Class'] == 1]
    
    print(f"   Training GAN on {len(fraud_data)} real fraud examples...")
    
    # Initialize CTGAN
    # epochs=200 is a good balance for speed/quality on Colab GPU
    ctgan = CTGAN(epochs=200, verbose=True) 
    ctgan.fit(fraud_data)
    
    # Generate 1000 new realistic fraud cases
    print("   Generating 1000 synthetic fraud transactions...")
    synthetic_fraud = ctgan.sample(1000)
    
    # Combine with original training data
    X_train_res = pd.concat([X_train, synthetic_fraud.drop('Class', axis=1)])
    y_train_res = pd.concat([y_train, synthetic_fraud['Class']])
    
    print(f"✅ Data Augmentation Complete. New Training Size: {X_train_res.shape[0]}")
    
else:
    # Fallback to SMOTE if GAN fails to install
    print("⚠️ Using SMOTE fallback...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# 5. TRAIN ENSEMBLE MODEL
print("\n🚀 Training Ensemble Model on GEN-AI Data...")

clf_xgb = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, tree_method='hist', n_jobs=-1
)

clf_rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1)

model = VotingClassifier(
    estimators=[('xgb', clf_xgb), ('rf', clf_rf)],
    voting='soft', n_jobs=-1
)

model.fit(X_train_res, y_train_res)

# 6. EVALUATE
print("\n--- 📊 GAN-ENHANCED MODEL REPORT ---")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
auprc = average_precision_score(y_test, y_prob)
print(f"🌟 AUPRC Score: {auprc:.4f}")

# Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.title(f'GAN-Trained Model\nAUPRC: {auprc:.3f}')
plt.show()

# 7. SAVE
print("\n💾 Saving GAN-Enhanced Model...")
joblib.dump(model, 'gan_ensemble_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Saved as 'gan_ensemble_model.pkl'")
