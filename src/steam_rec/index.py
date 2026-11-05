"""FAISS index over the final fused game embeddings."""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    faiss.normalize_L2(x)
    return x


class GameIndex:
    """Exact inner product search on unit vectors, which equals cosine similarity."""

    def __init__(self, game_vecs: np.ndarray, app_ids: list[int], user_vecs: np.ndarray | None = None):
        if len(app_ids) != len(game_vecs):
            raise ValueError("app_ids and game_vecs must have the same length")
        self.game_vecs = _normalize(game_vecs)
        self.user_vecs = _normalize(user_vecs) if user_vecs is not None else None
        self.app_ids = list(app_ids)
        self.row_of = {app_id: i for i, app_id in enumerate(self.app_ids)}
        self.index = faiss.IndexFlatIP(self.game_vecs.shape[1])
        self.index.add(self.game_vecs)

    @property
    def n_users(self) -> int:
        return 0 if self.user_vecs is None else len(self.user_vecs)

    def _search(self, query: np.ndarray, k: int, exclude: int | None = None):
        extra = 1 if exclude is not None else 0
        scores, rows = self.index.search(query.reshape(1, -1), min(k + extra, len(self.app_ids)))
        results = [
            {"app_id": self.app_ids[r], "score": float(s)}
            for s, r in zip(scores[0], rows[0])
            if r >= 0 and r != exclude
        ]
        return results[:k]

    def recommend_for_user(self, user_node_id: int, k: int = 10):
        if self.user_vecs is None or not 0 <= user_node_id < self.n_users:
            raise KeyError(user_node_id)
        return self._search(self.user_vecs[user_node_id], k)

    def similar_games(self, app_id: int, k: int = 10):
        row = self.row_of[app_id]
        return self._search(self.game_vecs[row], k, exclude=row)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "game_embeddings.npy", self.game_vecs)
        if self.user_vecs is not None:
            np.save(directory / "user_embeddings.npy", self.user_vecs)
        (directory / "app_ids.json").write_text(json.dumps(self.app_ids))

    @classmethod
    def load(cls, directory: str | Path) -> "GameIndex":
        directory = Path(directory)
        users = directory / "user_embeddings.npy"
        return cls(
            np.load(directory / "game_embeddings.npy"),
            json.loads((directory / "app_ids.json").read_text()),
            np.load(users) if users.exists() else None,
        )
