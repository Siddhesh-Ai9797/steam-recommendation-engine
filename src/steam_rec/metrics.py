"""Ranking metrics."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def recall_at_k(user_vecs: torch.Tensor, game_vecs: torch.Tensor, users: torch.Tensor, true_games: torch.Tensor, k: int = 10) -> float:
    """Share of held out (user, game) pairs where the true game is in the user's top k."""
    user_norm = F.normalize(user_vecs[users], dim=-1)
    game_norm = F.normalize(game_vecs, dim=-1)
    top_k = (user_norm @ game_norm.T).topk(k, dim=-1).indices
    hits = (top_k == true_games.unsqueeze(1)).any(dim=1)
    return hits.float().mean().item()


def random_recall_baseline(n_games: int, k: int = 10) -> float:
    return k / n_games
