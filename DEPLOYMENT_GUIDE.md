# 🚀 Deployment Guide: Fraud Detection System

Since your project has two parts (Frontend & Backend), "perfect" deployment requires a specific strategy. You cannot just upload everything to one place because they run as separate services.

## Choose Your Strategy

### Option A: Local Demo (Best for Presentations)
**Status:** ✅ Ready
**Pros:** Lowest latency, no internet required, free.
**Cons:** Only works on your computer.

1.  Simply run `run_app.bat`.
2.  Open your browser to the Streamlit URL.
3.  Done.

---

### Option B: Cloud Deployment (Best for Sharing Links)
**Status:** 🛠️ Requires Setup
**Pros:** Accessible by anyone via URL.
**Cons:** Two-step process, potential cold-start delays on free tiers.

Because your Streamlit frontend talks to a FastAPI backend, you must deploy them separately.

#### Phase 1: Deploy the Backend (FastAPI)
*We will use Render (free tier).*

1.  **Push your code to GitHub** (if you haven't already).
2.  Go to [Render Dashboard](https://dashboard.render.com/) -> New -> **Web Service**.
3.  Connect your GitHub repo.
4.  **Settings:**
    *   **Name:** `fraud-backend` (or similar)
    *   **Runtime:** Python 3
    *   **Build Command:** `pip install -r requirements.txt`
    *   **Start Command:** `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
5.  Click **Create Web Service**.
6.  Wait for it to go live. You will get a URL like: `https://fraud-backend.onrender.com`.
7.  **Copy this URL.**

#### Phase 2: Configure the Frontend
*Now we need to tell Streamlit to look at the *Cloud* backend, not *Localhost*.*

1.  Open `FraudDetectionSystem/app.py`.
2.  Find this line (approx line 173):
    ```python
    API_URL = "http://localhost:8000"
    ```
3.  Change it to look for a secret, or fallback to localhost:
    ```python
    import os
    # Priority: Streamlit Secrets > Environment Variable > Localhost
    if "API_URL" in st.secrets:
        API_URL = st.secrets["API_URL"]
    else:
        API_URL = "http://localhost:8000"
    ```

#### Phase 3: Deploy the Frontend (Streamlit)
*We will use Streamlit Community Cloud.*

1.  Go to [Streamlit Cloud](https://share.streamlit.io/).
2.  Connect your GitHub repo.
3.  Select `app.py` as the main file.
4.  **CRITICAL STEP:** Click "Advanced Settings" (or "Secrets").
    *   Add the following secret:
        ```toml
        API_URL = "https://fraud-backend.onrender.com"
        ```
        *(Replace with your actual Render URL from Phase 1)*
5.  Click **Deploy**.

## Summary
*   **Local:** Backend runs on `localhost:8000`. Frontend connects to `localhost`.
*   **Cloud:** Backend runs on `Render`. Frontend (Streamlit Cloud) has a Secret variable pointing to Render.

This setup allows you to run locally AND in the cloud using the exact same code!
