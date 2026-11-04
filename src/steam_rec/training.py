"""Training loops shared by the single GPU and DDP scripts."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.transforms import RandomLinkSplit

from steam_rec.data import EDGE, REV_EDGE
from steam_rec.metrics import recall_at_k
from steam_rec.models import FusionLayer, LinkPredictor, UserProjection, in_batch_contrastive_loss


def split_edges(data, val: float = 0.1, test: float = 0.05):
    return RandomLinkSplit(
        num_val=val, num_test=test, is_undirected=False,
        edge_types=EDGE, rev_edge_types=REV_EDGE, neg_sampling_ratio=1.0,
    )(data)


def link_loss(model, split, edge_slice: slice | None = None) -> torch.Tensor:
    """`model` may be a LinkPredictor or a DDP wrapper around one."""
    label_index = split[EDGE].edge_label_index
    labels = split[EDGE].edge_label
    if edge_slice is not None:
        label_index, labels = label_index[:, edge_slice], labels[edge_slice]
    return F.binary_cross_entropy_with_logits(model(split.edge_index_dict, label_index), labels)


@torch.no_grad()
def link_auc(model: LinkPredictor, split) -> float:
    model.eval()
    scores = model(split.edge_index_dict, split[EDGE].edge_label_index).cpu().numpy()
    return float(roc_auc_score(split[EDGE].edge_label.cpu().numpy(), scores))


def train_gnn(data, epochs: int = 100, lr: float = 0.01, dim: int = 64, device: str = "cpu", log_every: int = 10):
    train, val, test = split_edges(data)
    train, val, test = train.to(device), val.to(device), test.to(device)
    model = LinkPredictor(data["user"].num_nodes, data["game"].num_nodes, data.metadata(), dim).to(device)
    with torch.no_grad():
        model.embed(train.edge_index_dict)  # initializes the lazy SAGEConv weights
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = link_loss(model, train)
        loss.backward()
        optimizer.step()
        if log_every and (epoch == 1 or epoch % log_every == 0):
            print(f"epoch {epoch:4d} loss {loss.item():.4f}")
    metrics = {"val_auc": link_auc(model, val), "test_auc": link_auc(model, test)}
    return model, (train, val, test), metrics


def train_fusion(
    fused_input: torch.Tensor,
    user_gnn: torch.Tensor,
    train_pairs: torch.Tensor,
    val_pairs: torch.Tensor,
    epochs: int = 300,
    batch_size: int = 1024,
    lr: float = 1e-3,
    temperature: float = 0.1,
    check_every: int = 20,
    patience: int = 3,
    eval_size: int = 2000,
):
    """Trains both towers with in batch negatives and early stops on validation Recall@10."""
    device = fused_input.device
    game_tower = FusionLayer(fused_input.size(1)).to(device)
    user_tower = UserProjection(user_gnn.size(1)).to(device)
    optimizer = torch.optim.Adam([*game_tower.parameters(), *user_tower.parameters()], lr=lr, weight_decay=1e-5)

    generator = torch.Generator().manual_seed(42)
    eval_idx = torch.randperm(val_pairs.size(1), generator=generator)[:eval_size]
    eval_users, eval_games = val_pairs[0, eval_idx], val_pairs[1, eval_idx]

    best = {"recall_at_10": -1.0, "epoch": 0, "game_state": None, "user_state": None}
    strikes = 0
    n = train_pairs.size(1)
    for epoch in range(1, epochs + 1):
        game_tower.train()
        user_tower.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            if idx.numel() < 2:
                continue
            optimizer.zero_grad()
            users, games = train_pairs[0, idx], train_pairs[1, idx]
            loss = in_batch_contrastive_loss(user_tower(user_gnn[users]), game_tower(fused_input[games]), temperature)
            loss.backward()
            optimizer.step()

        if epoch % check_every == 0 or epoch == epochs:
            game_tower.eval()
            user_tower.eval()
            with torch.no_grad():
                recall = recall_at_k(user_tower(user_gnn), game_tower(fused_input), eval_users, eval_games, k=10)
            print(f"epoch {epoch:4d} recall@10 {recall:.4f}")
            if recall > best["recall_at_10"]:
                best.update(
                    recall_at_10=recall, epoch=epoch,
                    game_state={k: v.clone() for k, v in game_tower.state_dict().items()},
                    user_state={k: v.clone() for k, v in user_tower.state_dict().items()},
                )
                strikes = 0
            else:
                strikes += 1
                if strikes >= patience:
                    break

    game_tower.load_state_dict(best["game_state"])
    user_tower.load_state_dict(best["user_state"])
    return game_tower.eval(), user_tower.eval(), {"best_recall_at_10": best["recall_at_10"], "best_epoch": best["epoch"]}
