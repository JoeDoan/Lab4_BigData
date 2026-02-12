from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import sys
from pathlib import Path
import traceback # Thêm thư viện này để truy vết lỗi

# 1. FIX ĐƯỜNG DẪN 
root_path = Path(__file__).parent.parent.absolute()
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

# 2. BỎ BẪY TRY...EXCEPT Ở ĐÂY
# Để nếu lỗi thư viện (thiếu langchain, chromadb, v.v.), nó sẽ báo đỏ ngay lập tức!
from rag.retrieval import retrieve, generate_answer

app = FastAPI(title="CS5542 Lab 4 RAG Backend")

class QueryIn(BaseModel):
    question: str
    top_k: int = 5
    retrieval_mode: str = "hybrid"  
    alpha: float = 0.5              
    use_multimodal: bool = False

class QueryOut(BaseModel):
    answer: str
    evidence: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    failure_flag: bool

@app.post("/query", response_model=QueryOut)
def query_endpoint(q: QueryIn):
    try:
        current_alpha = q.alpha
        if q.retrieval_mode == "Dense Only":
            current_alpha = 1.0
        elif q.retrieval_mode == "Sparse Only":
            current_alpha = 0.0

        print(f"Bắt đầu truy vấn RAG cho câu hỏi: {q.question}")
        
        # Gọi RAG Logic
        evidence = retrieve(q.question, top_k=q.top_k, alpha=current_alpha)
        answer = generate_answer(q.question, evidence)

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
        # 3. NẾU LỖI LLM HOẶC DB, IN CHI TIẾT RA TERMINAL
        print("❌ LỖI NGHIÊM TRỌNG BÊN TRONG HÀM QUERY:")
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
