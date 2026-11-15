"""Fixed replay-constrained residual update for an existing structured head."""
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "A_research_round08"))
from A_research_round08.models import StructuredModel, TRAIN_SHA, ex, np


def balanced_weights(y):
    counts = np.bincount(y, minlength=6)
    present = int(np.count_nonzero(counts))
    return len(y) / (present * counts[y])


def anchor_update(parent, x, y, replay_x, replay_y):
    """Update only coefficients; labels belong to the caller's released training set."""
    if y.shape != (len(x),) or replay_y.shape != (len(replay_x),):
        raise ValueError("label count mismatch")
    if any(np.any((labels < 0) | (labels >= 6)) or labels.dtype.kind not in "iu" for labels in (y, replay_y)):
        raise ValueError("expected six-class integer labels")
    if set(replay_y.tolist()) != set(range(6)):
        raise ValueError("fixed recipe requires historical replay covering six classes")
    updated = copy.deepcopy(parent)
    if not len(y):
        return updated
    design = parent.design(x)
    source = parent.design(replay_x)
    weights = balanced_weights(y)
    source_weights = balanced_weights(replay_y)
    gram = design.T @ (design * weights[:, None]) + np.eye(design.shape[1])
    gram += len(y) / len(replay_y) * source.T @ (source * source_weights[:, None])
    residual = np.eye(6)[y] - design @ parent.coef
    delta = np.linalg.solve(gram, design.T @ (residual * weights[:, None]))
    updated.coef = parent.coef + delta
    if not np.isfinite(updated.coef).all():
        raise ValueError("nonfinite updated checkpoint")
    return updated
