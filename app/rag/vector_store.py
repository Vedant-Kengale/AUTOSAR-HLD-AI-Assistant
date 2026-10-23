"""
Vector Store — wraps FAISS for local embedding storage and retrieval,
with a native NumPy cosine similarity fallback if FAISS is unavailable.
Each project gets its own index saved to vector_store/<project_id>/.
"""

from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    _FAISS_AVAILABLE = False

from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL, RETRIEVAL_TOP_K, VECTOR_STORE_DIR


class VectorStore:
    """
    Local vector store with FAISS backend and NumPy fallback.

    Files on disk (inside *store_path*):
        index.faiss     — FAISS IndexFlatIP index (if FAISS available)
        embeddings.npy  — NumPy array of normalized vectors (fallback)
        metadata.pkl    — list of chunk metadata dicts
        config.json     — store configuration
    """

    INDEX_FILE      = "index.faiss"
    EMBEDDINGS_FILE = "embeddings.npy"
    META_FILE       = "metadata.pkl"
    CONFIG_FILE     = "config.json"

    def __init__(self, project_id: str, model_name: str = EMBEDDING_MODEL):
        self.project_id  = project_id
        self.model_name  = model_name
        self.store_path  = VECTOR_STORE_DIR / project_id
        self.store_path.mkdir(parents=True, exist_ok=True)

        self._model: SentenceTransformer | None = None
        self._index = None
        self._embeddings: np.ndarray | None = None
        self._metadata: List[Dict[str, Any]]    = []

        self._load_if_exists()

    # ── Embedding model (lazy) ─────────────────────────────────────────────────
    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # ── Persistence ────────────────────────────────────────────────────────────
    def _load_if_exists(self):
        meta_path = self.store_path / self.META_FILE
        if not meta_path.exists():
            return

        with open(meta_path, "rb") as f:
            self._metadata = pickle.load(f)

        idx_path = self.store_path / self.INDEX_FILE
        if _FAISS_AVAILABLE and idx_path.exists():
            try:
                self._index = faiss.read_index(str(idx_path))
            except Exception:
                self._index = None

        emb_path = self.store_path / self.EMBEDDINGS_FILE
        if emb_path.exists():
            try:
                self._embeddings = np.load(str(emb_path))
            except Exception:
                self._embeddings = None

    def _save(self):
        if _FAISS_AVAILABLE and self._index is not None:
            faiss.write_index(self._index, str(self.store_path / self.INDEX_FILE))

        if self._embeddings is not None:
            np.save(str(self.store_path / self.EMBEDDINGS_FILE), self._embeddings)

        with open(self.store_path / self.META_FILE, "wb") as f:
            pickle.dump(self._metadata, f)

        cfg = {
            "model_name": self.model_name,
            "n_vectors": len(self._metadata),
            "backend": "faiss" if (_FAISS_AVAILABLE and self._index is not None) else "numpy",
        }
        (self.store_path / self.CONFIG_FILE).write_text(json.dumps(cfg, indent=2))

    # ── Build / Add ────────────────────────────────────────────────────────────
    def add_chunks(self, chunks: List[Dict[str, Any]], show_progress: bool = True):
        """Embed *chunks* and add to the FAISS index or NumPy fallback."""
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # cosine via inner-product
        )
        embeddings = np.array(embeddings, dtype="float32")

        if _FAISS_AVAILABLE:
            if self._index is None:
                dim = embeddings.shape[1]
                self._index = faiss.IndexFlatIP(dim)
            self._index.add(embeddings)

        if self._embeddings is None:
            self._embeddings = embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])

        self._metadata.extend(chunks)
        self._save()

    def reset(self):
        """Clear the index (e.g. re-process the same project)."""
        self._index      = None
        self._embeddings = None
        self._metadata   = []
        for f in [self.INDEX_FILE, self.EMBEDDINGS_FILE, self.META_FILE, self.CONFIG_FILE]:
            p = self.store_path / f
            if p.exists():
                p.unlink()

    # ── Search ─────────────────────────────────────────────────────────────────
    def search(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> List[Dict[str, Any]]:
        """
        Returns top-k chunks most similar to *query*.
        Each result has an extra "score" key (cosine similarity, 0-1).
        """
        n_total = self.chunk_count
        if n_total == 0:
            return []

        q_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        k = min(top_k, n_total)

        if _FAISS_AVAILABLE and self._index is not None and self._index.ntotal > 0:
            scores, indices = self._index.search(q_emb, k)
            selected_scores = scores[0]
            selected_indices = indices[0]
        elif self._embeddings is not None and len(self._embeddings) > 0:
            # Cosine similarity on normalized vectors is dot product
            sims = np.dot(self._embeddings, q_emb.T).flatten()
            selected_indices = np.argsort(sims)[::-1][:k]
            selected_scores = sims[selected_indices]
        else:
            return []

        results = []
        for score, idx in zip(selected_scores, selected_indices):
            if idx < 0 or idx >= len(self._metadata):
                continue
            chunk = dict(self._metadata[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    # ── Info ───────────────────────────────────────────────────────────────────
    @property
    def is_ready(self) -> bool:
        if _FAISS_AVAILABLE and self._index is not None:
            return self._index.ntotal > 0
        return self._embeddings is not None and len(self._embeddings) > 0

    @property
    def chunk_count(self) -> int:
        return len(self._metadata)

    @property
    def document_names(self) -> List[str]:
        return list({c["doc_name"] for c in self._metadata})

    def get_stats(self) -> Dict[str, Any]:
        backend = "FAISS Local" if (_FAISS_AVAILABLE and self._index is not None) else "NumPy Cosine"
        return {
            "project_id":   self.project_id,
            "model":        self.model_name,
            "n_chunks":     self.chunk_count,
            "documents":    self.document_names,
            "is_ready":     self.is_ready,
            "backend":      backend,
        }
