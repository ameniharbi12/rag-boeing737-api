import numpy as np
import faiss
import pickle
import re
from typing import List, Dict
import os
from google.genai import Client


class Embedder:
    """
    Multimodal embedder using Google's models/embedding-001.
    Supports text, tables, and images.
    """

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found.")

        self.client = Client(api_key=api_key)

        # embedding-001 → 768-dimensional universal multimodal embeddings
        self.model_name = "models/embedding-001"
        self.dim = 768

        print(f"✅ Multimodal Embedder initialized: {self.model_name}, dim={self.dim}")

        self.index: faiss.IndexFlatIP = None
        self.metadata: List[Dict] = []


    # -------------------------------
    # TEXT NORMALIZATION
    # -------------------------------
    def _preprocess_text(self, text: str) -> str:
        if not text:
            return ""

        text = text.replace("\n", " ").strip()
        text = re.sub(r"(\d),(\d)", r"\1\2", text)

        # Unit normalization (consistent matching)
        replaces = [
            (r'(\d+)\s?°C', r'\1 degC'),
            (r'(\d+)\s?°F', r'\1 degF'),
            (r'\bfeet\b|\bfoot\b', 'ft'),
            (r'\bmeters\b|\bmetres\b', 'm'),
            (r'\bknots\b|\bkt\b', 'kt'),
            (r'(\d+)\s?%', r'\1 percent'),
            (r'(\d+)\s?kg\b', r'\1 kg'),
            (r'(\d+)\s?lb\b', r'\1 lb'),
        ]
        for pat, repl in replaces:
            text = re.sub(pat, repl, text)

        return re.sub(r'\s+', ' ', text)


    # -------------------------------
    # MULTIMODAL EMBEDDING
    # -------------------------------
    def embed_chunk(self, text: str, image_bytes: bytes, chunk_type: str) -> np.ndarray:
        """
        Create a Gemini multimodal embedding.
        text: OCR or extracted text
        image_bytes: raw PNG/JPG bytes, or None
        chunk_type: "text", "table", "image"
        """

        contents = []
        clean_text = self._preprocess_text(text or "")

        # attach text if present
        if clean_text:
            contents.append({"text": clean_text})

        # attach image if present
        if image_bytes:
            contents.append({"image": image_bytes})

        if not contents:
            contents.append({"text": ""})

        # Call Gemini embeddings API
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=contents
        )

        emb = np.array(result.embeddings[0].values, dtype="float32")

        # Boost weighting depending on chunk type
        if chunk_type == "table":
            emb *= 3.5
        elif chunk_type == "image":
            emb *= 2.0

        return emb


    # -------------------------------
    # BATCH EMBEDDING
    # -------------------------------
    def embed_batch(self, chunks: List[Dict]) -> np.ndarray:
        embeddings = []

        for c in chunks:
            emb = self.embed_chunk(
                text=c.get("text", ""),
                image_bytes=c.get("image_bytes"),  
                chunk_type=c.get("chunk_type", "text")
            )
            embeddings.append(emb)

        return np.vstack(embeddings)


    # -------------------------------
    # FAISS INDEXING
    # -------------------------------
    def build_faiss_index(self, chunks: List[Dict]):
        if not chunks:
            raise ValueError("Chunks list is empty.")

        embeddings = self.embed_batch(chunks)

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)

        # metadata for retrieval (preserves image_bytes)
        self.metadata = [
            {
                "chunk_id": c["chunk_id"],
                "pages": c["pages"],
                "text": c.get("text", ""),
                "chunk_type": c.get("chunk_type", "text"),
                "image_bytes": c.get("image_bytes")
            }
            for c in chunks
        ]

        print(f"✅ FAISS index created with {len(chunks)} multimodal chunks.")
        return self.index, embeddings


    # -------------------------------
    # SAVE / LOAD INDEX
    # -------------------------------
    def save_index(self, index_path="faiss.index", chunks_path="chunks.pkl"):
        if self.index is None:
            raise ValueError("No FAISS index to save.")

        faiss.write_index(self.index, index_path)

        with open(chunks_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"💾 Saved FAISS index → {index_path}")
        print(f"💾 Saved metadata → {chunks_path}")

    def load_index(self, index_path="faiss.index", chunks_path="chunks.pkl"):
        self.index = faiss.read_index(index_path)

        with open(chunks_path, "rb") as f:
            self.metadata = pickle.load(f)

        print(f"📂 Loaded FAISS index & metadata: {len(self.metadata)} chunks.")
