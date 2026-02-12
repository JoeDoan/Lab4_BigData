# api/server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

# Import your actual RAG logic
# Ensure your project root is in PYTHONPATH or this is run from root
try:
    from rag.retrieval import retrieve, generate_answer
except ImportError:
    print("WARNING: Could not import rag.retrieval. Make sure you run this from the project root.")

app = FastAPI(title="CS5542 Lab 4 RAG Backend")

# Define the data expected from the UI


class QueryIn(BaseModel):
    question: str
    top_k: int = 5
    retrieval_mode: str = "hybrid"  # options: hybrid, dense, sparse
    alpha: float = 0.5              # Hybrid weight
    use_multimodal: bool = False

# Define the response structure


class QueryOut(BaseModel):
    answer: str
    evidence: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    failure_flag: bool


@app.post("/query", response_model=QueryOut)
def query_endpoint(q: QueryIn):
    try:
        # 1. Retrieve Evidence
        # Map the mode/alpha to your retrieve function
        current_alpha = q.alpha
        if q.retrieval_mode == "Dense Only":
            current_alpha = 1.0
        elif q.retrieval_mode == "Sparse Only":
            current_alpha = 0.0

        # Call the logic from rag/retrieval.py
        evidence = retrieve(q.question, top_k=q.top_k, alpha=current_alpha)

        # 2. Generate Answer
        answer = generate_answer(q.question, evidence)

        # 3. Check for Failure (Missing Evidence)
        fail_flag = False
        if "Not enough evidence" in answer:
            fail_flag = True

        return {
            "answer": answer,
            "evidence": evidence,
            "metrics": {
                "top_k": q.top_k,
                "mode": q.retrieval_mode,
                "alpha": current_alpha
            },
            "failure_flag": fail_flag
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
