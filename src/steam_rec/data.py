"""Loading the Kaggle Steam dataset and building the user to game graph."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from torch_geometric.data import HeteroData

EDGE = ("user", "reviews", "game")
REV_EDGE = ("game", "rev_reviews", "user")


@dataclass
class IdMaps:
    """Maps raw Steam ids to contiguous node ids and back."""

    user_to_node: dict
    game_to_node: dict

    @property
    def node_to_game(self) -> dict:
        return {node: app_id for app_id, node in self.game_to_node.items()}


def filter_active_users(recommendations: pd.DataFrame, min_reviews: int = 50) -> tuple[pd.DataFrame, IdMaps]:
    """Keeps users with at least `min_reviews` reviews and assigns node ids."""
    counts = recommendations["user_id"].map(recommendations["user_id"].value_counts())
    filtered = recommendations[counts >= min_reviews].copy()

    user_to_node = {uid: i for i, uid in enumerate(filtered["user_id"].unique())}
    game_to_node = {gid: i for i, gid in enumerate(filtered["app_id"].unique())}

    filtered["user_node_id"] = filtered["user_id"].map(user_to_node)
    filtered["game_node_id"] = filtered["app_id"].map(game_to_node)
    return filtered.reset_index(drop=True), IdMaps(user_to_node, game_to_node)


def build_graph(edges: pd.DataFrame) -> HeteroData:
    """Builds a bipartite graph with recommendation and normalized playtime as edge features."""
    data = HeteroData()
    data["user"].num_nodes = int(edges["user_node_id"].max()) + 1
    data["game"].num_nodes = int(edges["game_node_id"].max()) + 1

    users = torch.tensor(edges["user_node_id"].values, dtype=torch.long)
    games = torch.tensor(edges["game_node_id"].values, dtype=torch.long)

    recommended = torch.tensor(edges["is_recommended"].astype(int).values, dtype=torch.float).unsqueeze(1)
    hours = torch.tensor(edges["hours"].values, dtype=torch.float)
    hours = ((hours - hours.mean()) / (hours.std() + 1e-8)).unsqueeze(1)
    edge_attr = torch.cat([recommended, hours], dim=1)

    data[EDGE].edge_index = torch.stack([users, games])
    data[EDGE].edge_attr = edge_attr
    data[REV_EDGE].edge_index = torch.stack([games, users])
    data[REV_EDGE].edge_attr = edge_attr
    return data
