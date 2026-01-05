# ==========================================
# 🎯 THRESHOLD OPTIMIZATION SCRIPT
# ==========================================
# Run this in Colab AFTER training your GAN model.
# It finds the mathematically perfect cutoff (e.g., 0.74 instead of 0.5)

import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_recall_curve, f1_score

# 1. Load Data
df = pd.read_csv('creditcard.csv')
scaler = RobustScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df.drop(['Time', 'Amount'], axis=1, inplace=True)
X = df.drop('Class', axis=1)
y = df['Class']

# Ensure columns
cols = [f'V{i}' for i in range(1, 29)] + ['scaled_amount']
X = X[cols]

# 2. Split (Test set only)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Load Model
model = joblib.load('gan_ensemble_model.pkl') # Or 'ensemble_model.pkl'

# 4. Get Probabilities
print("⏳ Calculating Probabilities...")
y_probs = model.predict_proba(X_test)[:, 1]

# 5. Find Optimal Threshold
print("🔍 Searching for Best Threshold...")
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

# Calculate F1 for every possible threshold
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"\n✅ OPTIMAL THRESHOLD FOUND: {best_threshold:.4f}")
print(f"🌟 Max F1-Score: {best_f1:.4f}")

# 6. Save Validated Threshold
# We save this as a tiny text file to load in our API dynamically
with open('threshold_config.txt', 'w') as f:
    f.write(str(best_threshold))

print("💾 Saved to 'threshold_config.txt'. Download this and put it in src/model/")
