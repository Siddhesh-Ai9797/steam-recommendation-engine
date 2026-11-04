"""Filters the Kaggle Steam data to active users and saves the graph.

Dataset: https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from steam_rec.data import build_graph, filter_active_users

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", required=True, help="Folder with recommendations.csv and games.csv")
parser.add_argument("--out-dir", default="work")
parser.add_argument("--min-reviews", type=int, default=50)
args = parser.parse_args()

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)

recs = pd.read_csv(Path(args.data_dir) / "recommendations.csv")
edges, maps = filter_active_users(recs, args.min_reviews)
graph = build_graph(edges)
torch.save(graph, out / "graph.pt")

node_to_game = maps.node_to_game
(out / "app_ids.json").write_text(json.dumps([int(node_to_game[i]) for i in range(len(node_to_game))]))
games = pd.read_csv(Path(args.data_dir) / "games.csv").set_index("app_id")["title"]
(out / "titles.json").write_text(json.dumps({int(a): t for a, t in games.items() if a in maps.game_to_node}))

print(f"users {graph['user'].num_nodes:,}  games {graph['game'].num_nodes:,}  edges {len(edges):,}")
