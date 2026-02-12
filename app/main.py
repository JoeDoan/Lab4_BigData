import streamlit as st
import pandas as pd
import json
import time
import requests
from pathlib import Path

# --- CONFIGURATION ---
API_URL = "http://localhost:8000/query"  # Address of your FastAPI server
MISSING_EVIDENCE_MSG = "Not enough evidence in the retrieved context."
LOG_FILE_DEFAULT = "logs/query_metrics.csv"

st.set_page_config(page_title="CS5542 Lab 4 — RAG Client", layout="wide")
st.title("CS 5542 Lab 4 — Project RAG Client")
st.caption("Streamlit Frontend → FastAPI Backend → Retrieval Logic")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Retrieval Settings")

# 1. Retrieval Mode
retrieval_mode = st.sidebar.selectbox(
    "Retrieval Mode",
    ["Hybrid (Dense+Sparse)", "Dense Only", "Sparse Only"],
    index=0
)

# 2. Hyperparameters
top_k = st.sidebar.slider("Top K", min_value=1, max_value=30, value=5, step=1)

# 3. Hybrid Alpha (Dense Weight)
# We calculate alpha here to send to the API
alpha = 0.5
if retrieval_mode == "Hybrid (Dense+Sparse)":
    alpha = st.sidebar.slider(
        "Hybrid Alpha (Dense Weight)", 0.0, 1.0, 0.5, 0.1)
    st.sidebar.caption("0.0 = Pure Sparse | 1.0 = Pure Dense")
elif retrieval_mode == "Dense Only":
    alpha = 1.0
elif retrieval_mode == "Sparse Only":
    alpha = 0.0

st.sidebar.header("Logging")
log_path = st.sidebar.text_input("Log File Path", value=LOG_FILE_DEFAULT)

# --- MINI GOLD SET (Reflects your Lab 4 Requirements) ---
MINI_GOLD = {
    "Q1": {
        "question": "What is the primary topic of the first document?",
        "gold_evidence_ids": ["doc_1"]
    },
    "Q2": {
        "question": "How does the system handle missing data?",
        "gold_evidence_ids": ["doc_2"]
    },
    "Q3": {
        "question": "What represents the dense vector space?",
        "gold_evidence_ids": ["doc_3"]
    },
    "Q4": {
        "question": "Explain the figure on page 2.",
        "gold_evidence_ids": ["doc_4_img"]
    },
    "Q5": {
        "question": "What is the airspeed velocity of an unladen swallow?",
        "gold_evidence_ids": ["N/A"]
    },
}

st.sidebar.header("Evaluation")
query_id = st.sidebar.selectbox(
    "Query ID (for logging)", list(MINI_GOLD.keys()))
use_gold_question = st.sidebar.checkbox("Use Gold Question Text", value=True)

# Main query input
default_q = MINI_GOLD[query_id]["question"] if use_gold_question else ""
question = st.text_area("Enter your question", value=default_q, height=100)
run_btn = st.button("Run Query", type="primary")

colA, colB = st.columns([2, 1])

# --- HELPER FUNCTIONS ---


def ensure_logfile(path: str):
    """Creates log file with headers if it doesn't exist."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        df = pd.DataFrame(columns=[
            "timestamp", "query_id", "retrieval_mode", "top_k", "latency_ms",
            "Precision@5", "Recall@10", "evidence_ids_returned", "gold_evidence_ids",
            "faithfulness_pass", "missing_evidence_behavior"
        ])
        df.to_csv(p, index=False)


def log_row(path: str, row: dict):
    """Appends a new row to the CSV log."""
    ensure_logfile(path)
    try:
        df = pd.read_csv(path)
        new_df = pd.DataFrame([row])
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(path, index=False)
        return True
    except Exception as e:
        st.error(f"Error logging: {e}")
        return False


def calculate_metrics(retrieved_ids, gold_ids, k=5):
    """Calculates Precision@K and Recall@K."""
    if not gold_ids or gold_ids == ["N/A"]:
        return None, None

    top_k_ids = retrieved_ids[:k]
    hits_k = sum(1 for x in top_k_ids if x in gold_ids)
    p_k = hits_k / k if k > 0 else 0
    r_k = hits_k / len(gold_ids) if len(gold_ids) > 0 else 0
    return p_k, r_k

# --- MAIN APP LOGIC ---


if run_btn and question.strip():
    # =========================================================
    # THE KEY CHANGE: CALLING THE API INSTEAD OF LOCAL FUNCTIONS
    # =========================================================
    with st.spinner("Connecting to RAG API..."):
        t0 = time.time()

        # 1. Prepare Payload
        payload = {
            "question": question,
            "top_k": top_k,
            "retrieval_mode": retrieval_mode,
            "alpha": alpha,  # Sending the hybrid weight
            "use_multimodal": False
        }

        try:
            # 2. Send Request
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()  # Check for HTTP errors (404, 500)

            # 3. Parse Response
            api_data = response.json()
            answer = api_data["answer"]
            evidence = api_data["evidence"]

        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to the API. Is 'api/server.py' running?")
            st.stop()
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
            st.stop()

        t1 = time.time()
        latency_ms = round((t1 - t0) * 1000, 2)
    # =========================================================

    # 4. Metrics & Logging (Client Side)
    retrieved_ids = [e["chunk_id"] for e in evidence]
    gold_ids = MINI_GOLD[query_id].get("gold_evidence_ids", [])

    p5, r10 = calculate_metrics(retrieved_ids, gold_ids, k=top_k)

    # 5. Display UI
    with colA:
        st.subheader("🤖 Answer")
        if "Not enough evidence" in answer:
            st.warning(answer)
        else:
            st.success(answer)

        st.subheader(f"📄 Retrieved Evidence (Top {len(evidence)})")
        if not evidence:
            st.info("No evidence found.")
        else:
            for i, doc in enumerate(evidence):
                citation = doc.get('citation_tag', f'[Doc {i}]')
                score = doc.get('score', 0.0)
                with st.expander(f"#{i+1} {citation} (Score: {score:.4f})"):
                    st.markdown(f"**Source:** {doc.get('source', 'Unknown')}")
                    st.text(doc.get('text', ''))

    with colB:
        st.subheader("📊 Metrics")
        st.metric("Latency (Client)", f"{latency_ms} ms")

        if p5 is not None:
            st.metric(f"Precision@{top_k}", f"{p5:.2f}")
            st.metric(f"Recall@{top_k}", f"{r10:.2f}")
        else:
            st.info("Metrics N/A (Gold IDs missing or N/A)")

        st.markdown("### Debug Info")
        st.json({
            "API URL": API_URL,
            "Mode": retrieval_mode,
            "Alpha": alpha,
            "Retrieved IDs": retrieved_ids,
        })

    # 6. Logging
    faithfulness_pass = "Yes"  # Placeholder

    missing_behavior_pass = "N/A"
    if gold_ids == ["N/A"]:
        missing_behavior_pass = "Pass" if "Not enough evidence" in answer else "Fail"
    elif gold_ids:
        missing_behavior_pass = "Pass"

    log_entry = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "query_id": query_id,
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
        "latency_ms": latency_ms,
        "Precision@5": p5 if p5 is not None else "NaN",
        "Recall@10": r10 if r10 is not None else "NaN",
        "evidence_ids_returned": json.dumps(retrieved_ids),
        "gold_evidence_ids": json.dumps(gold_ids),
        "faithfulness_pass": faithfulness_pass,
        "missing_evidence_behavior": missing_behavior_pass
    }

    if log_row(log_path, log_entry):
        st.toast(f"✅ Logged query {query_id}")
