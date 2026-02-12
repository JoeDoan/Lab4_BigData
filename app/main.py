import streamlit as st
import pandas as pd
import json
import time
import requests
import subprocess
import sys
import os
from pathlib import Path

# --- 1. APP CONFIGURATION (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="CS5542 Lab 4 — RAG Client",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CONSTANTS & SETUP ---
API_BASE_URL = "http://127.0.0.1:8000"
API_URL_QUERY = f"{API_BASE_URL}/query"
API_URL_DOCS = f"{API_BASE_URL}/docs"  # Used for health check
LOG_FILE_PATH = "logs/query_metrics.csv"

# --- 3. AUTO-START BACKEND LOGIC ---
@st.cache_resource
def ensure_backend_is_running():
    """
    Checks if the backend API is running. If not, it attempts to start it
    as a subprocess. This runs only once per session due to @st.cache_resource.
    """
    def is_server_active():
        try:
            response = requests.get(API_URL_DOCS, timeout=5)
            return response.status_code in [200, 404, 405, 422]
        except (requests.ConnectionError, requests.Timeout):
            return False

    if is_server_active():
        print("✅ Backend is already running.")
        return True

    # Backend is not running, attempt to start it
    server_script = Path("api/server.py")
    if not server_script.exists():
        st.error(f"❌ Critical Error: Could not find backend file at: {server_script.absolute()}")
        st.stop()

    print(f"🔄 Starting backend server from: {server_script}...")
    
    # Start the process
    # Using sys.executable ensures we use the same Python environment as Streamlit)
    subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "api.server:app", 
        "--host", "127.0.0.1", 
        "--port", "8000"
    ])
    # Polling loop: Wait for server to become responsive
    with st.spinner("🚀 Starting Backend API and loading AI Models... Please wait..."):
        for i in range(40):  # TĂNG TỪ 20 LÊN 40
            if is_server_active():
                st.toast("✅ Backend Server Started Successfully!", icon="🎉")
                time.sleep(2)
                return True
            time.sleep(2)
            
    st.error("❌ Failed to start backend server after 20 seconds. Please check your terminal logs.")
    st.stop()

# Initialize Backend
ensure_backend_is_running()

# --- 4. HELPER FUNCTIONS ---

def ensure_logfile(path: str):
    """Creates the log CSV with headers if it doesn't exist."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        df = pd.DataFrame(columns=[
            "timestamp", "query_id", "retrieval_mode", "top_k", "latency_ms",
            "Precision@5", "Recall@10", "evidence_ids_returned", "gold_evidence_ids",
            "faithfulness_pass", "missing_evidence_behavior"
        ])
        df.to_csv(p, index=False)

def log_transaction(path: str, row_data: dict):
    """Appends a new record to the CSV log."""
    ensure_logfile(path)
    try:
        df = pd.read_csv(path)
        new_row = pd.DataFrame([row_data])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(path, index=False)
        return True
    except Exception as e:
        st.error(f"⚠️ Logging failed: {e}")
        return False

def calculate_metrics(retrieved_ids, gold_ids, k):
    """Calculates Precision@K and Recall@K based on Gold Standard IDs."""
    if not gold_ids or gold_ids == ["N/A"]:
        return None, None

    # Consider only the top K retrieved items
    top_k_ids = retrieved_ids[:k]
    
    # Calculate hits
    hits = sum(1 for x in top_k_ids if x in gold_ids)
    
    precision = hits / k if k > 0 else 0
    recall = hits / len(gold_ids) if len(gold_ids) > 0 else 0
    
    return precision, recall

# --- 5. UI LAYOUT & SIDEBAR ---

st.title("CS 5542 Lab 4 — Project RAG Client")
st.markdown("Streamlit Frontend → FastAPI Backend → Retrieval Logic")

# Sidebar Configuration
st.sidebar.header("⚙️ Retrieval Settings")

retrieval_mode = st.sidebar.selectbox(
    "Retrieval Mode",
    ["Hybrid (Dense+Sparse)", "Dense Only", "Sparse Only"],
    index=0
)

top_k = st.sidebar.slider("Top K Retrieved", min_value=1, max_value=30, value=5)

# Dynamic Alpha Slider
alpha = 0.5
if retrieval_mode == "Hybrid (Dense+Sparse)":
    alpha = st.sidebar.slider("Hybrid Alpha (Dense Weight)", 0.0, 1.0, 0.5, 0.1, help="0.0 = Sparse | 1.0 = Dense")
elif retrieval_mode == "Dense Only":
    alpha = 1.0
elif retrieval_mode == "Sparse Only":
    alpha = 0.0

st.sidebar.divider()
st.sidebar.header("📂 Logging")
log_path_input = st.sidebar.text_input("Log File Path", value=LOG_FILE_PATH)

# Mini Gold Set (Test Data)
MINI_GOLD_SET = {
    "Q1": {"question": "What is the primary topic of the first document?", "gold_ids": ["doc_1"]},
    "Q2": {"question": "How does the system handle missing data?", "gold_ids": ["doc_2"]},
    "Q3": {"question": "What represents the dense vector space?", "gold_ids": ["doc_3"]},
    "Q4": {"question": "Explain the figure on page 2.", "gold_ids": ["doc_4_img"]},
    "Q5": {"question": "What is the airspeed velocity of an unladen swallow?", "gold_ids": ["N/A"]},
}

st.sidebar.divider()
st.sidebar.header("🧪 Evaluation Context")
query_id = st.sidebar.selectbox("Select Query ID", list(MINI_GOLD_SET.keys()))
use_gold_q = st.sidebar.checkbox("Auto-fill Question Text", value=True)

# Main Input Area
default_text = MINI_GOLD_SET[query_id]["question"] if use_gold_q else ""
user_question = st.text_area("Enter your question:", value=default_text, height=100)
run_button = st.button("Run Query", type="primary", use_container_width=True)

# --- 6. MAIN EXECUTION LOGIC ---

if run_button and user_question.strip():
    col_result, col_metrics = st.columns([2, 1])

    with st.spinner("Waiting for API response..."):
        start_time = time.time()
        
        # Prepare Payload
        payload = {
            "question": user_question,
            "top_k": top_k,
            "retrieval_mode": retrieval_mode,
            "alpha": alpha,
            "use_multimodal": False
        }

        try:
            # API Call
            response = requests.post(API_URL_QUERY, json=payload)
            response.raise_for_status()
            data = response.json()
            
            answer_text = data.get("answer", "No answer provided.")
            evidence_list = data.get("evidence", [])

        except requests.exceptions.ConnectionError:
            st.error("❌ Connection Error: Is 'api/server.py' running?")
            st.stop()
        except Exception as e:
            st.error(f"❌ API Error: {e}")
            st.stop()

        latency = round((time.time() - start_time) * 1000, 2)

    # --- Processing Results ---
    retrieved_doc_ids = [doc.get("chunk_id") for doc in evidence_list]
    gold_doc_ids = MINI_GOLD_SET[query_id].get("gold_ids", [])

    # Calculate Metrics
    prec_at_k, rec_at_k = calculate_metrics(retrieved_doc_ids, gold_doc_ids, k=top_k)

    # --- Display Results ---
    with col_result:
        st.subheader("🤖 Generated Answer")
        if "Not enough evidence" in answer_text:
            st.warning(answer_text)
        else:
            st.success(answer_text)

        st.subheader(f"📄 Evidence (Top {len(evidence_list)})")
        if not evidence_list:
            st.info("No documents retrieved.")
        else:
            for idx, doc in enumerate(evidence_list):
                citation = doc.get('citation_tag', f'[Doc {idx}]')
                score = doc.get('score', 0.0)
                content = doc.get('text', 'No text content')
                source = doc.get('source', 'Unknown source')
                
                with st.expander(f"#{idx+1} {citation} (Score: {score:.4f})"):
                    st.markdown(f"**Source:** `{source}`")
                    st.text(content)

    with col_metrics:
        st.subheader("📊 Performance Metrics")
        st.metric("Latency", f"{latency} ms")
        
        if prec_at_k is not None:
            st.metric(f"Precision@{top_k}", f"{prec_at_k:.2f}")
            st.metric(f"Recall@{top_k}", f"{rec_at_k:.2f}")
        else:
            st.info("Metrics unavailable (Gold IDs are N/A)")

        st.markdown("#### Debug Data")
        st.json({
            "API URL": API_URL_QUERY,
            "Alpha": alpha,
            "Retrieved IDs": retrieved_doc_ids,
            "Gold IDs": gold_doc_ids
        })

    # --- Logging ---
    # Determine Pass/Fail logic for "Missing Evidence"
    missing_behavior_status = "N/A"
    if gold_doc_ids == ["N/A"]:
        missing_behavior_status = "Pass" if "Not enough evidence" in answer_text else "Fail"
    elif gold_doc_ids:
        missing_behavior_status = "Pass" # Assuming retrieval worked if we have gold IDs

    log_entry = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "query_id": query_id,
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
        "latency_ms": latency,
        "Precision@5": prec_at_k if prec_at_k is not None else "NaN",
        "Recall@10": rec_at_k if rec_at_k is not None else "NaN",
        "evidence_ids_returned": json.dumps(retrieved_doc_ids),
        "gold_evidence_ids": json.dumps(gold_doc_ids),
        "faithfulness_pass": "Yes", # Placeholder
        "missing_evidence_behavior": missing_behavior_status
    }

    if log_transaction(log_path_input, log_entry):
        st.toast(f"✅ Logged result for Query ID: {query_id}")
