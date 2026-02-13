import json
import os
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import google.generativeai as genai

GEMINI_API_KEY = "AIzaSyDLsqbLpPRAmzZML99WKBFsiSsopNJfaIU"
genai.configure(api_key=GEMINI_API_KEY)
llm_model = genai.GenerativeModel('gemini-3-flash-preview')
# --- 1. SETUP & DATA LOADING ---
print("🚀 Initializing RAG Retrieval System...")

def load_documents_from_json(filename="page_chunks.json"):
    """
    Loads document chunks from a JSON file located in the project root.
    """
    # 1. Determine the path to the JSON file
    # This assumes the structure is: project_root/rag/retrieval.py
    # So we go up two levels to find project_root/page_chunks.json
    base_path = Path(__file__).parent.parent
    file_path = base_path / filename

    # Fallback: check current directory if not found in root
    if not file_path.exists():
        file_path = Path(__file__).parent / filename

    # 2. Check if file exists
    if not file_path.exists():
        print(f"⚠️ WARNING: Could not find '{filename}'. Using dummy data instead.")
        return []

    print(f"📂 Loading knowledge base from: {file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        # 3. Map JSON fields to our internal format
        processed_docs = []
        for item in raw_data:
            processed_docs.append({
                # Use 'chunk_id' as the unique identifier
                "doc_id": item.get("chunk_id", "unknown_id"),
                # Use 'doc_id' (filename) as the source
                "source": item.get("doc_id", "unknown_source"),
                # The actual content
                "text": item.get("text", "")
            })
        return processed_docs
        
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return []

# Load the data
documents = load_documents_from_json("page_chunks.json")

# Handle case where file is missing or empty
if not documents:
    documents = [
        {"doc_id": "dummy_1", "source": "System", "text": "Error: No data loaded. Please check page_chunks.json."}
    ]

# Extract lists for indexing
corpus_texts = [d["text"] for d in documents]
doc_ids = [d["doc_id"] for d in documents]
sources = [d["source"] for d in documents]

print(f"✅ Successfully loaded {len(documents)} document chunks.")

# --- 2. INDEXING (Runs once on startup) ---

# A) DENSE INDEX (SentenceTransformers)
print("⏳ Generating dense embeddings (this may take a moment)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# NOTE: convert_to_tensor=False creates NumPy arrays. 
# This is crucial for Streamlit Cloud (Free Tier) to avoid running out of RAM.
document_embeddings = embedding_model.encode(corpus_texts, convert_to_tensor=False)

# B) SPARSE INDEX (BM25)
# Simple tokenization by splitting on whitespace
tokenized_corpus = [doc.lower().split(" ") for doc in corpus_texts]
bm25 = BM25Okapi(tokenized_corpus)

# C) RERANKER MODEL
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

print("✅ Indexing complete. System is ready.")


# --- 3. HELPER FUNCTIONS ---

def normalize_scores(scores):
    """Normalizes a list of scores to a 0-1 range for hybrid fusion."""
    if len(scores) == 0:
        return scores
    
    # Convert list to numpy array if it isn't one
    scores = np.array(scores)
    
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    # Avoid division by zero if all scores are identical
    if max_score == min_score:
        return np.ones_like(scores)
        
    return (scores - min_score) / (max_score - min_score)

def hybrid_retrieve(question: str, top_k: int = 5, alpha: float = 0.5):
    """
    Performs Hybrid Search (Dense + Sparse) followed by Reranking.
    
    Args:
        question: The user's query.
        top_k: Number of final results to return.
        alpha: Weight for Dense Search (0.0 = Sparse Only, 1.0 = Dense Only).
    """
    # 1. Dense Retrieval (Vector Search)
    # Convert query to vector
    query_emb = embedding_model.encode(question, convert_to_tensor=False)
    # Calculate Cosine Similarity
    dense_scores = cosine_similarity([query_emb], document_embeddings)[0]

    # 2. Sparse Retrieval (Keyword Search)
    tokenized_query = question.lower().split(" ")
    sparse_scores = bm25.get_scores(tokenized_query)

    # 3. Score Fusion (Hybrid)
    norm_dense = normalize_scores(dense_scores)
    norm_sparse = normalize_scores(sparse_scores)
    
    # Combined score formula
    hybrid_scores = (alpha * norm_dense) + ((1 - alpha) * norm_sparse)

    # 4. Pre-Selection
    # Get top N candidates (retrieve 2x top_k to give the reranker more options)
    candidate_indices = np.argsort(-hybrid_scores)[:top_k * 2]

    # 5. Reranking (Cross-Encoder)
    # Create pairs of [Query, Document Text]
    rerank_pairs = [[question, corpus_texts[i]] for i in candidate_indices]
    
    if len(rerank_pairs) == 0:
        return []

    rerank_scores = reranker.predict(rerank_pairs)

    # Sort final results by Reranker score (descending)
    reranked_results = sorted(
        zip(candidate_indices, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    # 6. Format Output
    evidence = []
    for idx, score in reranked_results:
        evidence.append({
            "chunk_id": doc_ids[idx],
            "source": sources[idx],
            "score": float(score),
            "citation_tag": f"[{doc_ids[idx]}]", # Used for generation reference
            "text": corpus_texts[idx]
        })

    return evidence

# --- 4. EXPORTED API FUNCTIONS ---

def retrieve(question: str, top_k: int = 5, alpha: float = 0.5):
    """
    Public function called by the API to get documents.
    """
    return hybrid_retrieve(question, top_k=top_k, alpha=alpha)

def generate_answer(question: str, evidence: list):

    if not evidence:
        return "I'm sorry, I couldn't find enough evidence in the provided documents to answer your question."


    context_text = ""
    for i, doc in enumerate(evidence):
        context_text += f"--- Document {i+1} (Source: {doc['citation_tag']}) ---\n"
        context_text += f"{doc['text']}\n\n"


    prompt = f"""
    You are an intelligent assistant for a Big Data Analytics project. 
    Use the following retrieved documents to answer the user's question.

    RULES:
    1. Answer ONLY based on the context provided below. Do not use outside knowledge.
    2. If the answer is not in the context, state "Not enough evidence in the retrieved context."
    3. Cite your sources using the format [Source: Document Name] when you use specific information.
    4. Keep the answer concise and professional.

    CONTEXT:
    {context_text}

    USER QUESTION: 
    {question}

    YOUR ANSWER:
    """


    try:
        response = llm_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error generating answer with Gemini: {str(e)}"
