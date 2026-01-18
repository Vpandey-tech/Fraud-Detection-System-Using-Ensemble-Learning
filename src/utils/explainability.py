
import numpy as np

def explain_prediction(features, feature_names, model_prediction, threshold=0.5):
    """
    Generates a human-readable explanation for the fraud prediction.
    This is a lightweight rule-based explainer that mimics SHAP/LIME for speed.
    """
    explanation = []
    
    # Feature map based on PCA components (Simplified interpretation)
    # In real PCA, V1-V28 are abstract, but we can infer some behavior for the demo.
    feature_map = {
        'V1': 'User Profile Affinity',
        'V3': 'Transaction Location',
        'V4': 'Device Consistency',
        'V12': 'Spending Habit',
        'V14': 'time',
        'scaled_amount': 'Transaction Amount',
        'scaled_time': 'Time of Day'
    }

    # If it is fraud (score > threshold)
    if model_prediction > threshold:
        explanation.append("Suspicious activity detected.")
        
        # Check specific triggers (simulated logic for V-features)
        # In a real scenario, we would use shap_values[0] index sorting
        # specific logic for demo purposes:
        
        if features.get('scaled_amount', 0) > 2.0:
            explanation.append(f"- Unusually high transaction amount.")
            
        if features.get('v4', 0) > 1.5:
            explanation.append(f"- Irregular device usage pattern detected.")
        elif features.get('v12', 0) < -1.5:
             explanation.append(f"- Transaction deviates from normal spending habits.")
        elif features.get('v14', 0) < -2.0:
             explanation.append(f"- Transaction location high risk.")
             
        if not explanation[1:]: # Fallback if no specific rule triggered
             explanation.append("- Complex pattern match with known fraud signatures.")

    else:
        explanation.append("Transaction appears normal.")

    return " ".join(explanation)
