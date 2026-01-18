
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import random
import numpy as np
from datetime import datetime

# Configure Page
st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Light Theme & Modern Styling with Visible Sidebar
st.markdown("""
<style>
    /* Global Theme - Light Mode */
    .stApp {
        background-color: #ffffff;
        color: #333333;
    }
    
    /* SIDEBAR STYLING - Fix Black Background Issue */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f4f8 0%, #e1e8ed 100%) !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #f0f4f8 0%, #e1e8ed 100%) !important;
    }
    
    /* Sidebar Text - Ensure High Contrast */
    [data-testid="stSidebar"] * {
        color: #1a1a1a !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #333333 !important;
    }
    
    /* Sidebar Radio Buttons */
    [data-testid="stSidebar"] label {
        color: #2c3e50 !important;
        font-weight: 500 !important;
    }
    
    /* Sidebar Divider */
    [data-testid="stSidebar"] hr {
        border-color: #cbd5e0 !important;
        opacity: 0.6;
    }
    
    /* Stats Cards */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] {
        color: #555 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    div[data-testid="stMetricValue"] {
        color: #1a1a1a !important;
        font-weight: 700 !important;
    }
    
    /* Headers - Improved Visibility */
    h1 { 
        color: #1a202c !important; 
        font-family: 'Helvetica Neue', sans-serif; 
        font-weight: 700 !important;
    }
    h2, h3 { 
        color: #2d3748 !important;
        font-weight: 600 !important;
    }
    
    /* Main Content Text */
    .stMarkdown, p, span, div {
        color: #333333 !important;
    }
    
    /* Custom Badge Styles for Feed */
    .feed-item {
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 4px solid #ddd;
        background: #f8f9fa;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: 'Courier New', monospace;
        font-size: 0.95rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .feed-fraud {
        background-color: #fff5f5;
        border-left-color: #dc3545;
        color: #c0392b !important;
    }
    .feed-fraud strong {
        color: #a71d2a !important;
    }
    .feed-normal {
        background-color: #f1fcf5;
        border-left-color: #28a745;
        color: #27ae60 !important;
    }
    .feed-normal strong {
        color: #1e8449 !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background-color: #4299e1;
        color: white !important;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #3182ce;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Form Submit Button */
    .stButton > button[kind="primary"] {
        background-color: #48bb78 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #38a169 !important;
    }
    
    /* Info/Warning/Error Boxes - Better Contrast */
    .stAlert {
        color: #1a1a1a !important;
    }
    
    /* Slider Labels */
    .stSlider label {
        color: #2d3748 !important;
        font-weight: 500 !important;
    }
    
    /* Number Input Labels */
    .stNumberInput label {
        color: #2d3748 !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000"

# --- Sidebar Controls ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/security-checked.png", width=60)
    st.title("FraudGuard AI")
    st.caption("Admin Dashboard")
    st.divider()
    
    mode = st.radio("Mode", ["🛡️ Live Monitor", "🕵️ Manual Check", "📊 Analytics"])
    
    # Simulation Control
    # Using a Toggle Switch for easier control
    st.divider()
    st.subheader("⚙️ Simulation")
    sim_active = st.toggle("Activate Traffic Generator", value=False)
    
    speed = st.select_slider("Speed", options=["Slow", "Normal", "Fast", "Real-time"], value="Normal")
    delay_map = {"Slow": 1.5, "Normal": 0.5, "Fast": 0.2, "Real-time": 0.05}

# --- FUNCTIONS ---
def generate_random_tx():
    """Generates random transaction data"""
    data = {f"v{i}": np.random.normal(0, 1) for i in range(1, 29)}
    amount = np.random.exponential(100)
    data['scaled_amount'] = (amount - 88.0) / 100.0
    
    # Random fraud patterns
    if random.random() < 0.15: # 15% fraud chance
        data['v4'] += 4.5 
        data['v14'] -= 4.0
        data['v11'] += 3.0
    
    return data, amount

# --- PAGE: LIVE MONITOR ---
if mode == "🛡️ Live Monitor":
    col_header, col_refresh = st.columns([4, 1])
    col_header.title("Live Transaction Feed")
    if col_refresh.button("🔄 Refresh"):
        st.rerun()
    
    col_kpi, col_feed = st.columns([1, 2])
    
    # Placeholder containers
    kpi_container = col_kpi.container()
    feed_container = col_feed.container()
    
    # Simulation Logic loop
    if sim_active:
        # Generate and Post
        tx_data, amount = generate_random_tx()
        try:
            requests.post(f"{API_URL}/predict", json=tx_data, timeout=1) # 1s timeout
        except Exception as e:
            # st.toast(f"API Error: {e}", icon="⚠️") # Silence frequent errors
            pass
    
    # Fetch Data (Active Fetching)
    try:
        # 2s timeout for fetching data to prevent UI freeze
        stats = requests.get(f"{API_URL}/stats", timeout=2).json()
        history = requests.get(f"{API_URL}/history", timeout=2).json()
        
        # 1. RENDER KPIs
        with kpi_container:
            st.subheader("System Metrics")
            c1, c2 = st.columns(2)
            c1.metric("Total Processed", stats['total'])
            c2.metric("Fraud Intercepted", stats['fraud'])
            c1.metric("Active Model", "Ensemble (Voting)")
            
            # Gauge Chart
            latest_score = history[0]['score'] if history else 0
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = latest_score * 100,
                title = {'text': "Current Risk Level"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#dc3545" if latest_score > 0.8 else "#28a745"},
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 80}
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)

        # 2. RENDER FEED (Like Terminal)
        with feed_container:
            st.subheader("Incoming Stream")
            
            if not history:
                st.info("Waiting for traffic...")
            else:
                # Show last 10 transactions formatted nicely
                for tx in history[:10]: # Top 10 recent
                    status_class = "feed-fraud" if tx['is_fraud'] else "feed-normal"
                    icon = "🚨" if tx['is_fraud'] else "✅"
                    score_pct = tx['score'] * 100
                    
                    html_content = f"""
                    <div class="feed-item {status_class}">
                        <div>
                            <strong>{icon} {tx['timestamp']}</strong><br>
                            <span style="font-size:0.8em">ID: #{tx['id']} | Amt: ${abs(tx['amount_scaled']*100 + 88):.2f}</span>
                        </div>
                        <div style="text-align:right">
                            <strong>{score_pct:.1f}% Risk</strong><br>
                            <span style="font-size:0.8em">{tx['explanation'] if tx['is_fraud'] else 'Verified Legit'}</span>
                        </div>
                    </div>
                    """
                    st.markdown(html_content, unsafe_allow_html=True)

    except Exception as e:
        # If API is down, show friendly error instead of crash
        st.warning("Backend API is unreachable. Ensure 'uvicorn' is running in Terminal 1.")
        # Stop simulation to prevent error loop
        if sim_active:
            st.error("Stopping simulation due to connection failure.")

    # Rerun logic
    if sim_active:
        time.sleep(delay_map[speed])
        st.rerun()
    # ELSE: Do NOT rerun automatically. This fixes the "constant loading" issue.
    # The user can click the "Refresh Feed" button we added at the top if they want to check for new data.


# --- PAGE: MANUAL CHECK ---
elif mode == "🕵️ Manual Check":
    st.title("Manual Inspector")
    st.markdown("Test specific transaction parameters against the model.")
    
    with st.form("manual_check"):
        c1, c2 = st.columns(2)
        amount = c1.number_input("Amount ($)", value=250.00)
        v4 = c2.slider("V4 (Device Score)", -5.0, 5.0, 0.0, help="Positive values (> 2.0) increase fraud risk")
        
        c3, c4 = st.columns(2)
        v14 = c3.slider("V14 (Location Score)", -5.0, 5.0, 0.0, help="Negative values (< -2.0) increase fraud risk")
        v11 = c4.slider("V11 (Behavior Score)", -5.0, 5.0, 0.0, help="Positive values (> 2.0) increase fraud risk")
        
        st.caption("Tip: To simulate card skimming fraud, set V4 > 2.0 and V14 < -2.0")
        
        submitted = st.form_submit_button("Analyze Transaction", type="primary")
        
        if submitted:
            # Construct Payload
            # Fill others with zeros
            data = {f"v{i}": 0.0 for i in range(1, 29)}
            data['v4'] = v4
            data['v14'] = v14
            data['v11'] = v11
            data['scaled_amount'] = (amount - 88.0) / 100.0
            
            try:
                res = requests.post(f"{API_URL}/predict", json=data).json()
                
                # Result
                if res['is_fraud']:
                    st.error(f"## ⛔ BLOCKED\n**Risk Score:** {res['fraud_score']*100:.2f}%\n\n**Reason:** {res['explanation']}")
                else:
                    st.success(f"## ✅ APPROVED\n**Risk Score:** {res['fraud_score']*100:.2f}%\n\nTransaction is safe.")
                    
            except Exception as e:
                st.error(f"Error: {e}")

# --- PAGE: Analytics ---
elif mode == "📊 Analytics":
    st.title("Model Performance & Improvement")
    
    st.markdown("### 📈 Current Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Precision", "39.0%", help="Ratio of true frauds to all flagged")
    m2.metric("Recall", "89.0%", help="Ratio of caught frauds to all actual frauds")
    m3.metric("AUPRC", "0.852", "Excellent")
    
    st.markdown("### 💡 How to Improve?")
    st.info("""
    **1. Reduce False Positives (Improve Precision):**
    - The current model is 'Recall-Oriented' (safety first).
    - To balance it, we can increase the **Prediction Threshold**.
    - Currently set to **0.8** in the API. Raising to **0.9** would block fewer 'Normal' transactions but might miss subtle fraud.
    
    **2. Better Data Simulation:**
    - The current simulation injects 'heuristic' fraud patterns (V4 high, V14 low).
    - Training the model on **GAN-generated** data (Generative Adversarial Networks) instead of SMOTE would improve realism.
    """)
