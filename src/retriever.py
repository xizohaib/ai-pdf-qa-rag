import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HybridRetriever:
    """Hybrid keyword + semantic retrieval with reciprocal-rank fusion."""
    SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, chunks: list[dict], rrf_constant: int = 60):
        if not chunks:
            raise ValueError("At least one chunk is required.")
        self.chunks = chunks
        self.rrf_constant = int(rrf_constant)
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=100_000
        )
        self.tfidf_matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])
        from sentence_transformers import SentenceTransformer
        self.semantic_model = SentenceTransformer(self.SEMANTIC_MODEL)
        self.semantic_matrix = self.semantic_model.encode(
            [c["text"] for c in chunks], convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=len(chunks) > 30
        ).astype("float32")

    def search(self, question: str, top_k: int = 5) -> list[dict]:
        if not question or not question.strip():
            return []
        k = max(1, int(top_k))
        lexical_scores = cosine_similarity(
            self.vectorizer.transform([question.strip()]), self.tfidf_matrix
        ).ravel()
        semantic_query = self.semantic_model.encode(
            [question.strip()], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")[0]
        semantic_scores = self.semantic_matrix @ semantic_query

        # Stable descending ranks; RRF combines rankings rather than incomparable raw scores.
        lexical_order = np.argsort(-lexical_scores)
        semantic_order = np.argsort(-semantic_scores)
        lexical_rank = np.empty(len(self.chunks), dtype=int)
        semantic_rank = np.empty(len(self.chunks), dtype=int)
        lexical_rank[lexical_order] = np.arange(1, len(self.chunks) + 1)
        semantic_rank[semantic_order] = np.arange(1, len(self.chunks) + 1)

        fused = (1.0 / (self.rrf_constant + lexical_rank)) + (
            1.0 / (self.rrf_constant + semantic_rank)
        )
        candidate_count = min(len(self.chunks), max(k, 1))
        ids = np.argsort(-fused)[:candidate_count]
        results = []
        for idx in ids:
            item = dict(self.chunks[int(idx)])
            item.update({
                "score": float(fused[idx]),
                "lexical_score": float(lexical_scores[idx]),
                "semantic_score": float(semantic_scores[idx]),
                "lexical_rank": int(lexical_rank[idx]),
                "semantic_rank": int(semantic_rank[idx])
            })
            results.append(item)
        return results

def rerank_passages(question: str, passages: list[dict],
                    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> list[dict]:
    """Optionally rerank retrieved passages with a cross-encoder."""
    if not passages:
        return []
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_name, max_length=512)
    pairs = [(question, item["text"]) for item in passages]
    scores = np.asarray(model.predict(pairs, show_progress_bar=False)).reshape(-1)
    ranked = []
    for item, score in zip(passages, scores):
        updated = dict(item)
        updated["rerank_score"] = float(score)
        # Preserve original fusion score for diagnostics; use rerank score for final ordering.
        updated["score"] = float(score)
        ranked.append(updated)
    return sorted(ranked, key=lambda x: x["rerank_score"], reverse=True)
