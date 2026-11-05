"""Trains the GraphSAGE candidate generator on one device and saves node embeddings."""
import argparse
import json
from pathlib import Path

import torch

from steam_rec.training import train_gnn

parser = argparse.ArgumentParser()
parser.add_argument("--work-dir", default="work")
parser.add_argument("--epochs", type=int, default=100)
args = parser.parse_args()

work = Path(args.work_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"
graph = torch.load(work / "graph.pt", weights_only=False)

model, (train, val, test), metrics = train_gnn(graph, epochs=args.epochs, device=device)
print(json.dumps(metrics, indent=2))

with torch.no_grad():
    out = model.embed(train.edge_index_dict)
torch.save({"user": out["user"].cpu(), "game": out["game"].cpu()}, work / "gnn_embeddings.pt")
torch.save({
    "train": train["user", "reviews", "game"].edge_index.cpu(),
    "val": val["user", "reviews", "game"].edge_label_index[:, val["user", "reviews", "game"].edge_label == 1].cpu(),
}, work / "pairs.pt")
(work / "gnn_metrics.json").write_text(json.dumps(metrics, indent=2))
