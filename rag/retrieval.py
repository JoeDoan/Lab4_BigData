import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# --- 1. SETUP & INDEXING (Runs once when module is imported) ---
print("Initializing Hybrid Index...")

# TODO: Replace this with your actual document loading logic
# Ideally, load this from a JSON file or database
documents = [
    {"doc_id": "doc_1", "source": "Policy Manual",
        "text": "The primary topic of this document is enterprise knowledge graphs."},
    {"doc_id": "doc_2", "source": "IT Guide",
        "text": "If data is missing, the system will return a specific error code."},
    {"doc_id": "doc_3", "source": "AI Textbook",
        "text": "Dense vector spaces represent semantic meaning of text."},
    {"doc_id": "doc_4_img", "source": "Figure 2",
        "text": "This figure shows the architecture of the transformer model."},
    # Add more of your project documents here
]

corpus_texts = [d["text"] for d in documents]
doc_ids = [d["doc_id"] for d in documents]
sources = [d["source"] for d in documents]

# A) DENSE INDEX (SentenceTransformers)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
document_embeddings = embedding_model.encode(
    corpus_texts, convert_to_tensor=True)
document_embeddings = document_embeddings.cpu().numpy()

# B) SPARSE INDEX (BM25)
tokenized_corpus = [doc.split(" ") for doc in corpus_texts]
bm25 = BM25Okapi(tokenized_corpus)

# C) RERANKER
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

print("✅ Indexes built.")

# --- 2. HELPER FUNCTIONS ---


def normalize_scores(scores):
    """Normalize scores to 0-1 range for hybrid fusion."""
    if len(scores) == 0:
        return scores
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score == min_score:
        return np.ones_like(scores)
    return (scores - min_score) / (max_score - min_score)


def hybrid_retrieve(question: str, top_k: int = 5, alpha: float = 0.5):
    """
    Performs Hybrid Search (Dense + Sparse) with Reranking.
    alpha: 1.0 = Pure Dense, 0.0 = Pure Sparse
    """
    # 1. Dense Retrieval
    query_emb = embedding_model.encode(
        question, convert_to_tensor=True).cpu().numpy()
    dense_scores = cosine_similarity([query_emb], document_embeddings)[0]

    # 2. Sparse Retrieval
    tokenized_query = question.split(" ")
    sparse_scores = bm25.get_scores(tokenized_query)

    # 3. Fuse Scores
    norm_dense = normalize_scores(dense_scores)
    norm_sparse = normalize_scores(sparse_scores)
    hybrid_scores = (alpha * norm_dense) + ((1 - alpha) * norm_sparse)

    # 4. Pre-Selection (Top 2*k)
    candidate_indices = np.argsort(-hybrid_scores)[:top_k * 2]

    # 5. Reranking
    rerank_pairs = [[question, corpus_texts[i]] for i in candidate_indices]
    rerank_scores = reranker.predict(rerank_pairs)

    # Sort by Reranker scores
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
            "citation_tag": f"[{doc_ids[idx]}]",
            "text": corpus_texts[idx]
        })

    return evidence

# --- 3. EXPORTED FUNCTIONS ---


def retrieve(question: str, top_k: int = 5, alpha: float = 0.5):
    """Wrapper function called by the API."""
    return hybrid_retrieve(question, top_k=top_k, alpha=alpha)


def generate_answer(question: str, evidence: list):
    """Generates an answer based on evidence."""
    if not evidence:
        return "Not enough evidence in the retrieved context."

    # STUB: Replace with real LLM call
    top_doc = evidence[0]
    return (
        f"Based on {top_doc['citation_tag']}, the text says: '{top_doc['text']}'. "
        "(This is a stub answer from rag/retrieval.py)"
    )
