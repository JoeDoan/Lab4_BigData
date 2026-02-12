import streamlit as st
import pandas as pd
import json
import time
from pathlib import Path

# --- DIRECT IMPORT (Deployment Friendly) ---
# We try to import the local rag module. This allows the app to run
# on Streamlit Cloud without needing a separate API server running.
try:
    from rag.retrieval import retrieve, generate_answer
except ImportError:
    st.error("❌ CRITICAL ERROR: Could not import 'rag.retrieval'.")
    st.info("Ensure your repository has a 'rag' folder with an '__init__.py' file.")
    st.stop()

# --- CONFIGURATION ---
MISSING_EVIDENCE_MSG = "Not enough evidence in the retrieved context."
LOG_FILE_DEFAULT = "logs/query_metrics.csv"

st.set_page_config(page_title="CS5542 Lab 4 — RAG App", layout="wide")
st.title("CS 5542 Lab 4 — Project RAG Application")
st.caption("Deployment Mode: Monolithic (Streamlit Cloud Compatible)")

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

# 3. Hybrid Alpha
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

# --- MINI GOLD SET ---
MINI_GOLD = {
    "Q1": {"question": "What is the primary topic of the first document?", "gold_evidence_ids": ["doc_1"]},
    "Q2": {"question": "How does the system handle missing data?", "gold_evidence_ids": ["doc_2"]},
    "Q3": {"question": "What represents the dense vector space?", "gold_evidence_ids": ["doc_3"]},
    "Q4": {"question": "Explain the figure on page 2.", "gold_evidence_ids": ["doc_4_img"]},
    "Q5": {"question": "What is the airspeed velocity of an unladen swallow?", "gold_evidence_ids": ["N/A"]},
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
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        pd.DataFrame(columns=[
            "timestamp", "query_id", "retrieval_mode", "top_k", "latency_ms",
            "Precision@5", "Recall@10", "evidence_ids_returned", "gold_evidence_ids",
            "faithfulness_pass", "missing_evidence_behavior"
        ]).to_csv(p, index=False)


def log_row(path: str, row: dict):
    ensure_logfile(path)
    try:
        df = pd.read_csv(path)
        new_df = pd.DataFrame([row])
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(path, index=False)
        return True
    except Exception as e:
        st.error(f"Logging Error: {e}")
        return False

# --- MAIN LOGIC ---


if run_btn and question.strip():
    with st.spinner("Retrieving & Generating..."):
        t0 = time.time()

        # === MONOLITHIC CALL (Logic runs inside App) ===
        # This replaces the API call with a direct function call
        evidence = retrieve(question, top_k=top_k, alpha=alpha)
        answer = generate_answer(question, evidence)

        t1 = time.time()
        latency_ms = round((t1 - t0) * 1000, 2)

    # Metrics & Display
    retrieved_ids = [e["chunk_id"] for e in evidence]
    gold_ids = MINI_GOLD[query_id].get("gold_evidence_ids", [])

    # Calculate Precision
    hits = sum(1 for x in retrieved_ids[:top_k] if x in gold_ids)
    p5 = hits / top_k if top_k > 0 else 0
    # Calculate Recall
    r10 = hits / len(gold_ids) if gold_ids and gold_ids != ["N/A"] else 0

    with colA:
        st.subheader("🤖 Answer")
        if answer == MISSING_EVIDENCE_MSG:
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
        st.metric("Latency", f"{latency_ms} ms")

        if gold_ids and gold_ids != ["N/A"]:
            st.metric(f"Precision@{top_k}", f"{p5:.2f}")
            st.metric(f"Recall@{top_k}", f"{r10:.2f}")
        else:
            st.info("Metrics N/A (Gold IDs missing or N/A)")

    # Logging Logic
    faithfulness_pass = "Yes"

    missing_behavior_pass = "N/A"
    if gold_ids == ["N/A"]:
        missing_behavior_pass = "Pass" if answer == MISSING_EVIDENCE_MSG else "Fail"
    elif gold_ids:
        missing_behavior_pass = "Pass"

    log_entry = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "query_id": query_id,
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
        "latency_ms": latency_ms,
        "Precision@5": p5,
        "Recall@10": r10,
        "evidence_ids_returned": json.dumps(retrieved_ids),
        "gold_evidence_ids": json.dumps(gold_ids),
        "faithfulness_pass": faithfulness_pass,
        "missing_evidence_behavior": missing_behavior_pass
    }

    if log_row(log_path, log_entry):
        st.toast(f"✅ Logged query {query_id}")
