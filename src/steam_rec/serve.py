"""FastAPI service for recommendations. Run with: uvicorn steam_rec.serve:app"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

from steam_rec.index import GameIndex

ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts"))


def create_app(artifact_dir: Path = ARTIFACT_DIR) -> FastAPI:
    app = FastAPI(title="Steam Games Recommender", version="1.0.0")
    index = GameIndex.load(artifact_dir)
    titles_path = artifact_dir / "titles.json"
    titles = {int(k): v for k, v in json.loads(titles_path.read_text()).items()} if titles_path.exists() else {}

    def with_titles(results):
        return [{**r, "title": titles.get(r["app_id"])} for r in results]

    @app.get("/health")
    def health():
        return {"status": "ok", "games": len(index.app_ids), "users": index.n_users}

    @app.get("/recommend/{user_node_id}")
    def recommend(user_node_id: int, k: int = Query(10, ge=1, le=100)):
        try:
            return {"user": user_node_id, "results": with_titles(index.recommend_for_user(user_node_id, k))}
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown user")

    @app.get("/similar/{app_id}")
    def similar(app_id: int, k: int = Query(10, ge=1, le=100)):
        try:
            return {"app_id": app_id, "results": with_titles(index.similar_games(app_id, k))}
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown game")

    return app


app = create_app() if (ARTIFACT_DIR / "game_embeddings.npy").exists() else None
