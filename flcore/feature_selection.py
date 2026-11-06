import numpy as np
from scipy.stats import rankdata
from . import utils as U

def model_importance(W_local_first, W_start_first):
    # PyTorch Linear stores [hidden, input], so features are columns.
    A = np.asarray(W_local_first, dtype=np.float32)
    B = np.asarray(W_start_first, dtype=np.float32)
    return np.sum((A - B) ** 2, axis=0)

def marginal_entropy(X, B=10):
    X = np.asarray(X, dtype=np.float32); d = X.shape[1]; H = np.zeros(d)
    for j in range(d):
        col = X[:, j]
        if col.std() < 1e-8: continue
        lo, hi = col.min(), col.max()
        if hi - lo < 1e-8: continue
        bins = np.linspace(lo, hi, B + 1)
        p, _ = np.histogram(col, bins=bins)
        p = p / p.sum(); p = p[p > 0]
        H[j] = -np.sum(p * np.log(p))
    return H

def correlation_redundancy(X, max_samples=512, seed=0):
    X = np.asarray(X, dtype=np.float32)
    if X.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], max_samples, replace=False); X = X[idx]
    d = X.shape[1]
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(C, 0.0)
    return np.mean(np.abs(C), axis=1)

def percentile_rank(v):
    v = np.asarray(v, dtype=np.float64)
    if len(v) <= 1: return np.zeros_like(v)
    return (rankdata(v, method="average") - 1.0) / (len(v) - 1.0)

def build_shared_mask(client_stats, sizes, rho, w=(0.60, 0.25, 0.15)):
    """client_stats: list of (G,H,R); sizes: per-client n_i. Returns mask (bool, d)."""
    a = np.asarray(sizes, dtype=np.float64); a = a / a.sum()
    G = np.stack([percentile_rank(s[0]) for s in client_stats])
    H = np.stack([percentile_rank(s[1]) for s in client_stats])
    R = np.stack([percentile_rank(s[2]) for s in client_stats])
    S = (a[:, None] * (w[0]*G + w[1]*H - w[2]*R)).sum(axis=0)
    d = len(S); k = int(np.clip(round(rho * d), 1, d))
    chosen = np.argsort(S, kind="stable")[-k:]
    mask = np.zeros(d, dtype=bool); mask[chosen] = True
    return mask

def setup_bytes(n_clients, d):
    # 3 float32 vectors per client + 1-bit mask broadcast
    return n_clients * 3 * d * U.FLOAT + n_clients * int(np.ceil(d / 8))
