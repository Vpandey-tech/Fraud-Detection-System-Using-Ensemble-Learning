from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
import sys

# Add src to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from src.utils.explainability import explain_prediction
except ImportError:
    # Fallback if run from different dir
    from utils.explainability import explain_prediction

app = FastAPI(title="Fraud Detection System with XAI")

# Load Models
MODEL_PATH = "src/model/"

try:
    if os.path.exists("src/model/ensemble_model.pkl"):
        model = joblib.load("src/model/ensemble_model.pkl")
        print("✅ Loaded State-of-the-Art ENSEMBLE Model")
    else:
        model = joblib.load("src/model/xgb_model.pkl")
        print("✅ Loaded XGBoost Model")
        
    print(f"✅ Model loaded successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not load model: {e}")
    model = None

class Transaction(BaseModel):
    # Inputs matching the dataset features
    v1: float
    v2: float
    v3: float
    v4: float
    v5: float
    v6: float
    v7: float
    v8: float
    v9: float
    v10: float
    v11: float
    v12: float
    v13: float
    v14: float
    v15: float
    v16: float
    v17: float
    v18: float
    v19: float
    v20: float
    v21: float
    v22: float
    v23: float
    v24: float
    v25: float
    v26: float
    v27: float
    v28: float
    scaled_amount: float

from collections import deque
from datetime import datetime

# ... existing code ...

# In-Memory Storage for UI
# Stores last 50 transactions for the live dashboard
recent_transactions = deque(maxlen=50)
stats = {
    "total": 0,
    "fraud": 0,
    "blocked_amount": 0.0
}

@app.post("/predict")
def predict(tx: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Construct feature array
    features_dict = tx.dict()
    feature_values = [getattr(tx, f"v{i}") for i in range(1, 29)]
    feature_values.append(tx.scaled_amount)
    
    # User's model dropped Time, so we excluded scaled_time from the prediction input
    X = np.array(feature_values).reshape(1, -1)
    
    # Prediction
    try:
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X)[0, 1]
        else:
            score = float(model.predict(X)[0])
            
        # --- HYBRID DETECTION LAYER ---
        if tx.scaled_amount > 20: 
            score = max(score, 0.95) 
            explanation = "High Value Transaction Anomaly detected ($2000+)"

        if tx.v4 > 2.0 and tx.v14 < -2.0:
            score = max(score, 0.85)
            
        # --- OPTIMAL THRESHOLDING ---
        THRESHOLD = 0.8 # Default safety
        try:
            # Try loading the mathematically optimal threshold from Colab
            thresh_path = os.path.join(os.path.dirname(__file__), "..", "model", "threshold_config.txt")
            if os.path.exists(thresh_path):
                with open(thresh_path, 'r') as f:
                    THRESHOLD = float(f.read().strip())
        except:
            pass
            
        prediction_is_fraud = score > THRESHOLD
        
        # XAI: Generate Explanation (if not already set by rule)
        if "explanation" not in locals():
            explanation = explain_prediction(features_dict, None, score, threshold=THRESHOLD)
        
        # Update Stats for UI
        stats["total"] += 1
        if prediction_is_fraud:
            stats["fraud"] += 1
            # We don't have the raw amount here easily (it was scaled), 
            # but for demo we can estimate or just track count.
            # actually we can pass raw amount if we change the request model, 
            # but let's just track counts for now.
        
        # Log to history
        recent_transactions.appendleft({
            "id": stats["total"],
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "score": float(score),
            "is_fraud": bool(prediction_is_fraud),
            "explanation": explanation,
            "amount_scaled": float(tx.scaled_amount) # Placeholder for display
        })

        return {
            "fraud_score": float(score),
            "is_fraud": bool(prediction_is_fraud),
            "explanation": explanation
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history():
    return list(recent_transactions)

@app.get("/stats")
def get_stats():
    return stats

@app.get("/")
def read_root():
    return {
        "status": "online",
        "system": "Fraud Detection System | XAI Enabled",
        "endpoints": ["/predict", "/history", "/stats"]
    }