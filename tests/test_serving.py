import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from steam_rec.index import GameIndex
from steam_rec.serve import create_app

APP_IDS = [10, 20, 30, 40]


@pytest.fixture
def index():
    games = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    users = np.array([[0, 0, 1], [1, 0, 0]], dtype=np.float32)
    return GameIndex(games, APP_IDS, users)


@pytest.fixture
def client(index, tmp_path):
    index.save(tmp_path)
    (tmp_path / "titles.json").write_text(json.dumps({"10": "Fallout 4", "20": "Fallout: New Vegas"}))
    return TestClient(create_app(tmp_path))


def test_similar_games_excludes_itself(index):
    results = index.similar_games(10, k=2)
    assert [r["app_id"] for r in results] == [20, 30]


def test_user_recommendations_rank_by_cosine(index):
    assert index.recommend_for_user(0, k=1)[0]["app_id"] == 40
    assert [r["app_id"] for r in index.recommend_for_user(1, k=2)] == [10, 20]


def test_unknown_ids_raise(index):
    with pytest.raises(KeyError):
        index.recommend_for_user(99)
    with pytest.raises(KeyError):
        index.similar_games(999)


def test_save_and_load_round_trip(index, tmp_path):
    index.save(tmp_path)
    loaded = GameIndex.load(tmp_path)
    assert loaded.app_ids == APP_IDS
    assert loaded.similar_games(10, k=1) == index.similar_games(10, k=1)


def test_api_health(client):
    assert client.get("/health").json() == {"status": "ok", "games": 4, "users": 2}


def test_api_similar_adds_titles(client):
    body = client.get("/similar/10", params={"k": 1}).json()
    assert body["results"][0]["app_id"] == 20
    assert body["results"][0]["title"] == "Fallout: New Vegas"


def test_api_unknown_user_is_404(client):
    assert client.get("/recommend/99").status_code == 404


def test_api_rejects_bad_k(client):
    assert client.get("/recommend/0", params={"k": 0}).status_code == 422
