# ==========================================
# 📊 ENSEMBLE MODEL EVALUATION SCRIPT
# ==========================================
# Run this AFTER you have trained usage 'advanced_colab_training.py' 
# OR if you just uploaded 'ensemble_model.pkl' to Colab.

import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score

# 1. Load Data
print("⏳ Loading Data...")
try:
    df = pd.read_csv('creditcard.csv')
except:
    print("❌ ERROR: Please upload 'creditcard.csv'")
    exit()

# 2. Preprocessing (Must match training exactly)
scaler = RobustScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df.drop(['Time', 'Amount'], axis=1, inplace=True)

X = df.drop('Class', axis=1)
y = df['Class']

# Ensure correct column order
cols = [f'V{i}' for i in range(1, 29)] + ['scaled_amount']
X = X[cols]

# 3. Create Test Set (Same random state as training to ensure we test on unseen data)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✅ Data Ready. Test Set Size: {X_test.shape[0]}")

# 4. Load Model
print("⏳ Loading Model...")
try:
    model = joblib.load('ensemble_model.pkl')
    print("✅ Model Loaded Successfully")
except:
    print("❌ ERROR: 'ensemble_model.pkl' not found. Train it first!")
    exit()

# 5. Evaluate
print("\n🚀 Running Predictions...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 6. Metrics
print("\n--- 📝 CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred))

auprc = average_precision_score(y_test, y_prob)
print(f"\n🌟 AUPRC Score: {auprc:.4f}")

# 7. Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title(f'Ensemble Evaluation\nAUPRC: {auprc:.3f}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
