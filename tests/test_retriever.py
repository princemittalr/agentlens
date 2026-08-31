"""
Unit tests for tracer/retriever.py
Tests FAISS index build, load, and retrieval quality.
"""
import sys
import os
sys.path.insert(0, "/home/prince-mittal/agentlens")

import pytest
import numpy as np
from tracer.retriever import FAISSRetriever, embed


# ─── Sample documents ────────────────────────────────────────────────────────

SAMPLE_DOCS = [
    {"id": "d1", "title": "Python Programming",
     "content": "Python is a high-level programming language known for its simplicity."},
    {"id": "d2", "title": "Machine Learning",
     "content": "Machine learning is a subset of AI that learns patterns from data."},
    {"id": "d3", "title": "Web Development",
     "content": "Web development involves building websites using HTML, CSS, and JavaScript."},
    {"id": "d4", "title": "Database Systems",
     "content": "Databases store and organize data using SQL or NoSQL approaches."},
    {"id": "d5", "title": "Cloud Computing",
     "content": "Cloud computing provides on-demand computing resources over the internet."},
]


# ─── embed() ─────────────────────────────────────────────────────────────────

class TestEmbed:
    def test_returns_numpy_array(self):
        result = embed(["Hello world"])
        assert isinstance(result, np.ndarray)

    def test_correct_shape(self):
        result = embed(["Hello world"])
        assert result.shape[0] == 1
        assert result.shape[1] == 384  # all-MiniLM-L6-v2 dim

    def test_multiple_texts(self):
        result = embed(["Hello", "World", "Test"])
        assert result.shape[0] == 3

    def test_different_texts_different_embeddings(self):
        a = embed(["Python programming"])
        b = embed(["Cloud computing"])
        assert not np.allclose(a, b)

    def test_same_text_same_embedding(self):
        a = embed(["Python programming"])
        b = embed(["Python programming"])
        assert np.allclose(a, b, atol=1e-5)


# ─── FAISSRetriever.build() ───────────────────────────────────────────────────

class TestFAISSRetrieverBuild:
    def test_builds_successfully(self):
        retriever = FAISSRetriever()
        retriever.build(SAMPLE_DOCS)
        assert retriever.index is not None
        assert len(retriever.documents) == 5

    def test_index_has_correct_count(self):
        import faiss
        retriever = FAISSRetriever()
        retriever.build(SAMPLE_DOCS)
        assert retriever.index.ntotal == 5

    def test_documents_stored(self):
        retriever = FAISSRetriever()
        retriever.build(SAMPLE_DOCS)
        ids = [d["id"] for d in retriever.documents]
        assert "d1" in ids
        assert "d5" in ids


# ─── FAISSRetriever.retrieve() ───────────────────────────────────────────────

class TestFAISSRetrieverRetrieve:
    @pytest.fixture(autouse=True)
    def build_retriever(self):
        self.retriever = FAISSRetriever()
        self.retriever.build(SAMPLE_DOCS)

    def test_returns_correct_count(self):
        results = self.retriever.retrieve("programming language", top_k=2)
        assert len(results) == 2

    def test_top_result_is_relevant(self):
        results = self.retriever.retrieve("Python programming language", top_k=3)
        assert results[0].title == "Python Programming"

    def test_ml_query_retrieves_ml_doc(self):
        results = self.retriever.retrieve("machine learning AI patterns", top_k=2)
        assert results[0].title == "Machine Learning"

    def test_scores_between_0_and_1(self):
        results = self.retriever.retrieve("web development HTML", top_k=3)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_scores_descending(self):
        results = self.retriever.retrieve("database SQL NoSQL", top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_out_of_domain_query_low_score(self):
        results = self.retriever.retrieve("population of Mars planets", top_k=1)
        assert results[0].score < 0.3

    def test_relevant_query_high_score(self):
        results = self.retriever.retrieve("Python programming language", top_k=1)
        assert results[0].score > 0.5

    def test_returns_doc_fields(self):
        results = self.retriever.retrieve("cloud internet computing", top_k=1)
        r = results[0]
        assert hasattr(r, "doc_id")
        assert hasattr(r, "title")
        assert hasattr(r, "content")
        assert hasattr(r, "score")

    def test_top_k_respected(self):
        results = self.retriever.retrieve("programming", top_k=1)
        assert len(results) == 1

        results = self.retriever.retrieve("programming", top_k=4)
        assert len(results) == 4
