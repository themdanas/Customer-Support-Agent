from __future__ import annotations
import pickle
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class PolicyChunk:
    policy_id: str
    text: str
    source_doc: str
    similarity_score: float

class PolicyVectorStore:
    def __init__(self):
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.chunks: list[dict] = []

    def build(self, chunks: list[dict]) -> None:
        #chunks: list of {policy_id, text, source_doc} dicts
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])

    def query(self, text:str, k:int = 3) -> list[PolicyChunk]:
        if self.vectorizer is None:
            raise RuntimeError("Vector store is empty = call build( or load() first)")

        query_vec = self.vectorizer.transform([text])
        scores = cosine_similarity(query_vec, self.matrix) [0]
        top_k_idx = scores.argsort()[::-1][:k]
        return [
            PolicyChunk(
                policy_id=self.chunks[i]["policy_id"],
                text=self.chunks[i]["text"],
                source_doc = self.chunks[i]["source_doc"],
                similarity_score=float(scores[i]),
            )
            for i in top_k_idx
        ]

    def save(self, path:str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "vectorizer": self.vectorizer, "matrix": self.matrix}, f)

    @classmethod
    def load(cls, path:str) -> "PolicyVectorStore":
        store = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        store.chunks = data["chunks"]
        store.vectorizer = data["vectorizer"]
        store.matrix = data["matrix"]
        return store

    

