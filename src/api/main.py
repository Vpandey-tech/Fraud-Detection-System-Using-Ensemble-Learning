from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import joblib
import numpy as np
import os
import sys
import logging
import time
from datetime import datetime

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from src.utils.explainability import explain_prediction
except ImportError:
    # Fallback if run from different dir
    from utils.explainability import explain_prediction

app = FastAPI(
    title="🛡️ Fraud Detection System with XAI",
    description="State-of-the-Art Fraud Detection using Ensemble Learning + Generative AI",
    version="2.0"
)

# Load Models with Enhanced Error Handling
MODEL_PATH = "src/model/"

try:
    if os.path.exists("src/model/ensemble_model.pkl"):
        model = joblib.load("src/model/ensemble_model.pkl")
        logger.info("✅ Loaded State-of-the-Art ENSEMBLE Model (XGBoost + Random Forest)")
    elif os.path.exists("src/model/gan_ensemble_model.pkl"):
        model = joblib.load("src/model/gan_ensemble_model.pkl")
        logger.info("✅ Loaded GAN-Enhanced ENSEMBLE Model")
    else:
        model = joblib.load("src/model/xgb_model.pkl")
        logger.info("✅ Loaded XGBoost Model")
        
    logger.info(f"🚀 Model loaded successfully - Ready for predictions")
except Exception as e:
    logger.error(f"❌ CRITICAL: Could not load model: {e}")
    model = None

class Transaction(BaseModel):
    """
    Transaction input model with validation.
    Features V1-V28 are PCA-transformed anonymized features.
    scaled_amount is the RobustScaler-transformed transaction amount.
    """
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
    
    @validator('scaled_amount')
    def validate_amount(cls, v):
        """Validate that scaled amount is within reasonable bounds"""
        if v < -10 or v > 100:
            raise ValueError(f'Scaled amount {v} is out of expected range (-10 to 100)')
        return v
    
    @validator('v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10',
               'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20',
               'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28')
    def validate_features(cls, v):
        """Validate that PCA features are within reasonable bounds"""
        if abs(v) > 50:  # PCA features typically don't exceed this
            raise ValueError(f'Feature value {v} is suspiciously high')
        return v

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
    """
    Predict fraud probability for a transaction.
    
    Returns:
        - fraud_score: Probability of fraud (0-1)
        - is_fraud: Boolean decision
        - explanation: Human-readable explanation
        - processing_time_ms: Response time in milliseconds
    """
    start_time = time.time()
    
    if model is None:
        logger.error("Prediction attempted but model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded. Please restart the server.")

    try:
        # Construct feature array
        features_dict = tx.dict()
        feature_values = [getattr(tx, f"v{i}") for i in range(1, 29)]
        feature_values.append(tx.scaled_amount)
        
        X = np.array(feature_values).reshape(1, -1)
        
        # Prediction
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X)[0, 1]
        else:
            score = float(model.predict(X)[0])
        
        logger.debug(f"Initial model score: {score:.4f}")
            
        # --- HYBRID DETECTION LAYER ---
        explanation = None
        
        # Rule 1: High Value Transaction Override
        if tx.scaled_amount > 20: 
            original_score = score
            score = max(score, 0.95) 
            explanation = "⚠️ High Value Transaction Anomaly detected ($2000+)"
            logger.info(f"High-value override: {original_score:.4f} → {score:.4f}")

        # Rule 2: Skimming Pattern Detection
        if tx.v4 > 2.0 and tx.v14 < -2.0:
            original_score = score
            score = max(score, 0.85)
            if explanation is None:
                explanation = "🚨 Card Skimming Pattern Detected (V4 high, V14 low)"
            logger.info(f"Skimming pattern override: {original_score:.4f} → {score:.4f}")
            
        # --- OPTIMAL THRESHOLDING ---
        THRESHOLD = 0.8 # Default safety threshold
        try:
            thresh_path = os.path.join(os.path.dirname(__file__), "..", "model", "threshold_config.txt")
            if os.path.exists(thresh_path):
                with open(thresh_path, 'r') as f:
                    THRESHOLD = float(f.read().strip())
                    logger.debug(f"Using optimal threshold: {THRESHOLD}")
        except Exception as e:
            logger.warning(f"Could not load optimal threshold: {e}")
            
        prediction_is_fraud = score > THRESHOLD
        
        # XAI: Generate Explanation (if not already set by rule)
        if explanation is None:
            explanation = explain_prediction(features_dict, None, score, threshold=THRESHOLD)
        
        # Update Stats for UI
        stats["total"] += 1
        if prediction_is_fraud:
            stats["fraud"] += 1
            logger.info(f"🚨 FRAUD DETECTED - Score: {score:.4f}, ID: {stats['total']}")
        else:
            logger.debug(f"✅ Normal transaction - Score: {score:.4f}, ID: {stats['total']}")
        
        # Log to history
        recent_transactions.appendleft({
            "id": stats["total"],
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "score": float(score),
            "is_fraud": bool(prediction_is_fraud),
            "explanation": explanation,
            "amount_scaled": float(tx.scaled_amount)
        })
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        return {
            "fraud_score": float(score),
            "is_fraud": bool(prediction_is_fraud),
            "explanation": explanation,
            "processing_time_ms": round(processing_time, 2)
        }
        
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(ve)}")
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

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