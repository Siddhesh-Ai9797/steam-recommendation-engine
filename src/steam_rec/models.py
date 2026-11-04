"""Model parts: GNN encoder, multimodal fusion tower and user tower."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero


class GNNEncoder(nn.Module):
    """Two layer GraphSAGE encoder, turned heterogeneous with `to_hetero`."""

    def __init__(self, hidden_channels: int = 64, out_channels: int = 64):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class LinkPredictor(nn.Module):
    """Learned user and game embeddings passed through the hetero GNN."""

    def __init__(self, n_users: int, n_games: int, metadata, dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.game_emb = nn.Embedding(n_games, dim)
        self.encoder = to_hetero(GNNEncoder(dim, dim), metadata, aggr="sum")

    def embed(self, edge_index_dict):
        x = {"user": self.user_emb.weight, "game": self.game_emb.weight}
        return self.encoder(x, edge_index_dict)

    def forward(self, edge_index_dict, edge_label_index):
        """Scores (user, game) pairs. Going through forward keeps DDP gradient sync intact."""
        return score_edges(self.embed(edge_index_dict), edge_label_index)


def score_edges(out: dict, edge_label_index: torch.Tensor) -> torch.Tensor:
    return (out["user"][edge_label_index[0]] * out["game"][edge_label_index[1]]).sum(dim=-1)


class FusionLayer(nn.Module):
    """Game tower: fuses GNN, sentence transformer and CLIP vectors into one embedding."""

    def __init__(self, input_dim: int = 960, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.net(x)


class UserProjection(nn.Module):
    """User tower: projects GNN user vectors into the fused space."""

    def __init__(self, input_dim: int = 64, output_dim: int = 128):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def fuse_inputs(gnn: torch.Tensor, text: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """L2 normalizes each modality so no single one dominates, then concatenates."""
    return torch.cat([F.normalize(gnn, dim=1), F.normalize(text, dim=1), F.normalize(image, dim=1)], dim=1)


def in_batch_contrastive_loss(user_vecs: torch.Tensor, game_vecs: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Every other game in the batch is a negative. This fixed the collapse seen with one negative per pair."""
    users = F.normalize(user_vecs, dim=-1)
    games = F.normalize(game_vecs, dim=-1)
    logits = users @ games.T / temperature
    labels = torch.arange(users.size(0), device=users.device)
    return F.cross_entropy(logits, labels)
