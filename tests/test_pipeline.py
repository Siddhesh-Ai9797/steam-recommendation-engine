import torch

from steam_rec.data import EDGE, REV_EDGE, build_graph, filter_active_users
from steam_rec.metrics import random_recall_baseline, recall_at_k
from steam_rec.models import FusionLayer, UserProjection, fuse_inputs, in_batch_contrastive_loss
from steam_rec.training import train_fusion, train_gnn


def test_filter_drops_inactive_users(interactions):
    edges, maps = filter_active_users(interactions, min_reviews=10)
    assert edges["user_id"].nunique() == 40
    assert set(maps.user_to_node.values()) == set(range(40))
    assert edges["game_node_id"].max() == len(maps.game_to_node) - 1


def test_graph_has_both_directions_and_edge_features(interactions):
    edges, _ = filter_active_users(interactions, min_reviews=10)
    graph = build_graph(edges)
    assert graph[EDGE].edge_index.shape == (2, len(edges))
    assert torch.equal(graph[REV_EDGE].edge_index, graph[EDGE].edge_index.flip(0))
    assert graph[EDGE].edge_attr.shape == (len(edges), 2)


def test_gnn_trains_and_beats_chance(interactions):
    edges, _ = filter_active_users(interactions, min_reviews=10)
    _, _, metrics = train_gnn(build_graph(edges), epochs=60, log_every=0)
    assert 0.0 <= metrics["val_auc"] <= 1.0
    assert metrics["val_auc"] > 0.5


def test_fused_input_is_balanced_across_modalities():
    fused = fuse_inputs(torch.randn(5, 64) * 100, torch.randn(5, 384), torch.randn(5, 512) * 0.01)
    assert fused.shape == (5, 960)
    for part in (fused[:, :64], fused[:, 64:448], fused[:, 448:]):
        assert torch.allclose(part.norm(dim=1), torch.ones(5), atol=1e-5)


def test_towers_output_shared_space():
    assert FusionLayer()(torch.randn(3, 960)).shape == (3, 128)
    assert UserProjection()(torch.randn(3, 64)).shape == (3, 128)


def test_contrastive_loss_prefers_matching_pairs():
    users = torch.eye(4)
    assert in_batch_contrastive_loss(users, users) < in_batch_contrastive_loss(users, users.roll(1, dims=0))


def test_recall_at_k_counts_hits():
    games = torch.eye(4)
    users = torch.eye(4)
    assert recall_at_k(users, games, torch.tensor([0, 1]), torch.tensor([0, 1]), k=1) == 1.0
    assert recall_at_k(users, games, torch.tensor([0, 1]), torch.tensor([2, 3]), k=1) == 0.0
    assert random_recall_baseline(1000, 10) == 0.01


def test_fusion_training_returns_best_checkpoint():
    n_users, n_games = 20, 15
    pairs = torch.stack([torch.randint(0, n_users, (300,)), torch.randint(0, n_games, (300,))])
    _, _, info = train_fusion(
        torch.randn(n_games, 960), torch.randn(n_users, 64), pairs, pairs[:, :50],
        epochs=4, batch_size=64, check_every=2, eval_size=50,
    )
    assert info["best_epoch"] in (2, 4)
    assert 0.0 <= info["best_recall_at_10"] <= 1.0
