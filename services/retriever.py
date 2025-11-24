import faiss
import numpy as np
import re
from typing import List, Dict
from services.embedder import Embedder

class Retriever:
    """
    FAISS-based Retriever with numeric + table boosting for precise queries.
    """

    def __init__(self, chunks: List[Dict]):
        if not chunks:
            raise ValueError("Chunks list is empty.")

        self.embedder = Embedder()
        for chunk in chunks:
            # append numeric values to text for better matching
            nums = re.findall(r'\d+\.?\d*', chunk.get("text", ""))
            chunk_text = chunk.get("text", "") + " " + " ".join(nums)
            chunk["embedding"] = self.embedder.embed_chunk(
                text=chunk_text,
                image_bytes=chunk.get("image_bytes"),
                chunk_type=chunk.get("chunk_type", "text")
            )

        embeddings = np.vstack([c["embedding"] for c in chunks]).astype("float32")
        self.index = faiss.IndexFlatIP(self.embedder.dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        self.metadata = chunks
        print(f"✅ FAISS index ready with {len(chunks)} chunks.")

    # -------------------------------
    # Query preprocessing
    # -------------------------------
    def _preprocess_query(self, query: str) -> str:
        query = query.lower()
        query = re.sub(r'\bfoot\b|\bfeet\b', 'ft', query)
        query = re.sub(r'\bcelsius\b', 'degC', query)
        query = re.sub(r'\bfahrenheit\b', 'degF', query)
        query = re.sub(r'\bpercent\b', 'percent', query)
        query = re.sub(r'\s+', ' ', query)
        return query.strip()

    # -------------------------------
    # Retrieve top-K chunks with numeric boost
    # -------------------------------
    def retrieve(self, query: str, top_k: int = 5, similarity_threshold: float = 0.1) -> List[Dict]:
        query_clean = self._preprocess_query(query)
        query_emb = self.embedder.embed_chunk(text=query_clean, image_bytes=None, chunk_type="text")
        query_emb = query_emb.reshape(1, -1).astype("float32")
        faiss.normalize_L2(query_emb)

        sims, idxs = self.index.search(query_emb, min(len(self.metadata), top_k * 5))

        candidates = []
        # extract numeric query terms
        query_nums = re.findall(r'\d+\.?\d*', query_clean)

        for sim, idx in zip(sims[0], idxs[0]):
            if idx >= len(self.metadata):
                continue
            chunk = self.metadata[idx]
            # numeric boosting
            chunk_nums = re.findall(r'\d+\.?\d*', chunk.get("text", ""))
            numeric_match = any(qn.replace(',', '') in [cn.replace(',', '') for cn in chunk_nums] for qn in query_nums)
            boost = 0.15 if numeric_match else 0
            if chunk.get("chunk_type") == "table":
                boost += 0.1  # table boost
            score = sim + boost
            if score < similarity_threshold:
                continue
            candidates.append({**chunk, "score": float(score)})

        # sort by final score
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # relax page uniqueness for tables
        selected, seen_pages = [], set()
        for c in candidates:
            c_pages = set(c.get("pages", []))
            if c.get("chunk_type") != "table" and c_pages & seen_pages:
                continue
            selected.append(c)
            seen_pages.update(c_pages)
            if len(selected) >= top_k:
                break

        return selected
