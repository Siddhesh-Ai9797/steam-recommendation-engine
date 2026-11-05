import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture
def interactions():
    """Synthetic review log with two taste groups so link prediction has real signal.

    Users 0 to 19 review games 0 to 14, users 20 to 39 review games 15 to 29.
    Users 100 to 104 are below the activity threshold.
    """
    rng = np.random.default_rng(0)
    rows = []
    for user in range(40):
        pool = np.arange(15) if user < 20 else np.arange(15, 30)
        for game in rng.choice(pool, size=12, replace=False):
            rows.append((user, 1000 + int(game), bool(rng.random() > 0.2), float(rng.random() * 50)))
    for user in range(100, 105):
        rows.append((user, 1000, True, 1.0))
    return pd.DataFrame(rows, columns=["user_id", "app_id", "is_recommended", "hours"])


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)
