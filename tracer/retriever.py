"""
AgentLens — Real FAISS Vector Retriever
Replaces the fake keyword-based retriever in rag_agent.py
Uses sentence-transformers for embeddings + FAISS for similarity search
"""
import os
import sys
import json
import pickle
import numpy as np
sys.path.insert(0, "/home/prince-mittal/agentlens")

from typing import List, Dict, Optional
from dataclasses import dataclass

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache")
INDEX_PATH = os.path.join(CACHE_DIR, "faiss.index")
DOCS_PATH  = os.path.join(CACHE_DIR, "docs.pkl")
os.makedirs(CACHE_DIR, exist_ok=True)


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class RetrievedDoc:
    doc_id: str
    title: str
    content: str
    score: float       # cosine similarity score


# ─── Embedder (shared with clustering) ───────────────────────────────────────

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        print("  Loading embedding model...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def embed(texts: List[str]) -> np.ndarray:
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return np.array(embeddings, dtype="float32")


# ─── FAISS Index ─────────────────────────────────────────────────────────────

class FAISSRetriever:
    def __init__(self):
        self.index = None
        self.documents = []   # list of dicts: {id, title, content}
        self.embeddings = None

    def build(self, documents: List[Dict]):
        """Build FAISS index from documents."""
        import faiss

        self.documents = documents
        texts = [f"{d['title']}. {d['content']}" for d in documents]

        print(f"  Building FAISS index for {len(documents)} documents...")
        self.embeddings = embed(texts)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product = cosine on normalized vecs

        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

        # Save to cache
        faiss.write_index(self.index, INDEX_PATH)
        with open(DOCS_PATH, "wb") as f:
            pickle.dump(self.documents, f)

        print(f"  Index built and saved to {INDEX_PATH}")

    def load(self):
        """Load existing FAISS index from cache."""
        import faiss
        if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(DOCS_PATH, "rb") as f:
                self.documents = pickle.load(f)
            print(f"  Loaded FAISS index ({len(self.documents)} docs)")
            return True
        return False

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedDoc]:
        """Retrieve top-k most similar documents for a query."""
        if self.index is None:
            raise ValueError("Index not built. Call build() or load() first.")

        import faiss

        query_embedding = embed([query])
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = self.documents[idx]
            results.append(RetrievedDoc(
                doc_id=doc["id"],
                title=doc["title"],
                content=doc["content"],
                score=round(float(score), 4),
            ))

        return results


# ─── Singleton retriever ──────────────────────────────────────────────────────

_retriever: Optional[FAISSRetriever] = None

def get_retriever(documents: Optional[List[Dict]] = None) -> FAISSRetriever:
    """Get or build the global retriever instance."""
    global _retriever
    if _retriever is not None:
        return _retriever

    _retriever = FAISSRetriever()

    # Try loading from cache first
    if _retriever.load():
        return _retriever

    # Build from provided documents
    if documents:
        _retriever.build(documents)
        return _retriever

    raise ValueError("No cached index found and no documents provided.")


if __name__ == "__main__":
    # Test the retriever standalone
    TEST_DOCS = [
        {"id": "d1", "title": "AgentLens Overview",
         "content": "AgentLens is an observability platform for LLM agents built by Prince Mittal in 2024."},
        {"id": "d2", "title": "Tracer Module",
         "content": "The Tracer module records latency, tokens, cost, and steps using SQLite."},
        {"id": "d3", "title": "Evaluation Engine",
         "content": "The Evaluation Engine runs rule-based and LLM-judge metrics on every run."},
        {"id": "d4", "title": "Regression Detection",
         "content": "Regression detection compares metric distributions across agent versions."},
        {"id": "d5", "title": "Failure Clustering",
         "content": "Failure clustering uses HDBSCAN and sentence-transformers to group failed runs."},
        {"id": "d6", "title": "Dashboard",
         "content": "The dashboard is built with FastAPI and Jinja2. It shows runs, charts, clusters, and regression reports."},
        {"id": "d7", "title": "Supported Models",
         "content": "AgentLens supports Groq-hosted models including qwen/qwen3.8-27b for agents and judges."},
        {"id": "d8", "title": "GitHub Actions CI",
         "content": "AgentLens has a GitHub Actions CI pipeline that runs evals on every push and fails if scores drop below 0.7."},
    ]

    print("Building FAISS retriever...")
    retriever = FAISSRetriever()
    retriever.build(TEST_DOCS)

    print("\nRunning test queries:")
    queries = [
        "How does the tracer work?",
        "What is regression detection?",
        "Who built AgentLens?",
        "What models are supported?",
        "How does CI work?",
        "What is the population of Mars?",  # out-of-domain query
    ]

    for q in queries:
        print(f"\nQuery: '{q}'")
        results = retriever.retrieve(q, top_k=2)
        for r in results:
            print(f"  [{r.score:.4f}] {r.title}: {r.content[:80]}...")
