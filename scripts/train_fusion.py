"""Trains the two tower fusion model and exports the FAISS serving artifacts."""
import argparse
import json
import shutil
from pathlib import Path

import torch

from steam_rec.index import GameIndex
from steam_rec.metrics import random_recall_baseline, recall_at_k
from steam_rec.models import fuse_inputs
from steam_rec.training import train_fusion

parser = argparse.ArgumentParser()
parser.add_argument("--work-dir", default="work")
parser.add_argument("--artifact-dir", default="artifacts")
parser.add_argument("--final-eval-size", type=int, default=20000)
args = parser.parse_args()

work = Path(args.work_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"
gnn = torch.load(work / "gnn_embeddings.pt", weights_only=True)
pairs = torch.load(work / "pairs.pt", weights_only=True)
fused = fuse_inputs(gnn["game"], torch.load(work / "text_embeddings.pt", weights_only=True), torch.load(work / "image_embeddings.pt", weights_only=True)).to(device)
user_gnn = gnn["user"].to(device)

game_tower, user_tower, info = train_fusion(fused, user_gnn, pairs["train"].to(device), pairs["val"].to(device))

with torch.no_grad():
    game_vecs, user_vecs = game_tower(fused), user_tower(user_gnn)

torch.manual_seed(123)
idx = torch.randperm(pairs["val"].size(1))[: args.final_eval_size]
val = pairs["val"][:, idx].to(device)
recall = recall_at_k(user_vecs, game_vecs, val[0], val[1], k=10)
baseline = random_recall_baseline(game_vecs.size(0))
info.update(final_recall_at_10=recall, random_baseline=baseline, lift_over_random=recall / baseline)
print(json.dumps(info, indent=2))

app_ids = json.loads((work / "app_ids.json").read_text())
GameIndex(game_vecs.cpu().numpy(), app_ids, user_vecs.cpu().numpy()).save(args.artifact_dir)
shutil.copy(work / "titles.json", Path(args.artifact_dir) / "titles.json")
(Path(args.artifact_dir) / "metrics.json").write_text(json.dumps(info, indent=2))
