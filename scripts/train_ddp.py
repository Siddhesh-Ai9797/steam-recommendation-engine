"""Distributed GNN training with PyTorch DDP.

Every rank holds the full graph for message passing and trains on its own
slice of the supervision edges. DDP averages gradients across ranks.

    torchrun --nproc_per_node=2 scripts/train_ddp.py --work-dir work
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from steam_rec.models import LinkPredictor
from steam_rec.training import link_auc, link_loss, split_edges

parser = argparse.ArgumentParser()
parser.add_argument("--work-dir", default="work")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.01)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
dist.init_process_group("nccl" if use_cuda else "gloo")
rank, world = dist.get_rank(), dist.get_world_size()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
if use_cuda:
    torch.cuda.set_device(device)

torch.manual_seed(0)
graph = torch.load(Path(args.work_dir) / "graph.pt", weights_only=False)
train, val, test = (s.to(device) for s in split_edges(graph))

n_edges = train["user", "reviews", "game"].edge_label.numel()
shard = slice(rank * n_edges // world, (rank + 1) * n_edges // world)

model = LinkPredictor(graph["user"].num_nodes, graph["game"].num_nodes, graph.metadata()).to(device)
# Lazy SAGEConv layers need one forward pass before DDP wraps the model.
with torch.no_grad():
    model.embed(train.edge_index_dict)
ddp_model = DDP(model, device_ids=[local_rank] if use_cuda else None)
optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.lr)

start = time.time()
for epoch in range(1, args.epochs + 1):
    ddp_model.train()
    optimizer.zero_grad()
    loss = link_loss(ddp_model, train, shard)
    loss.backward()
    optimizer.step()
    if rank == 0 and (epoch == 1 or epoch % 10 == 0):
        print(f"epoch {epoch:4d} loss {loss.item():.4f}")

if rank == 0:
    metrics = {
        "world_size": world,
        "train_seconds": round(time.time() - start, 2),
        "val_auc": link_auc(model, val),
        "test_auc": link_auc(model, test),
    }
    print(json.dumps(metrics, indent=2))
    (Path(args.work_dir) / f"ddp_metrics_{world}.json").write_text(json.dumps(metrics, indent=2))
dist.destroy_process_group()
