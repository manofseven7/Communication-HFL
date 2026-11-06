import numpy as np
from . import utils as U

def _flat_dict(state):
    keys, vecs, shapes = [], [], []
    for k in state:
        a = np.asarray(state[k], dtype=np.float32).reshape(-1)
        keys.append(k); vecs.append(a); shapes.append(state[k].shape)
    return keys, vecs, shapes

def _unflat_dict(keys, vecs, shapes, template):
    out = {}
    for k, v, s in zip(keys, vecs, shapes):
        out[k] = v.reshape(s).astype(np.float32)
    return out

# ---------------- truncated SVD per 2-D-ish layer (proposed) ----------------
def svd_compress(state, k):
    """payload = list of (U,S,V,shape) for ndim>=2 layers, raw vector otherwise.
    Returns (payload, bytes). Transmits factors only when cheaper than dense."""
    payload, nbytes = {}, 0
    for name, w in state.items():
        w = np.asarray(w, dtype=np.float32)
        if w.ndim >= 2:
            m = w.reshape(w.shape[0], -1)
            U_, S_, Vt = np.linalg.svd(m, full_matrices=False)
            kk = min(k, len(S_))
            dense = m.size * U.FLOAT
            fac = (U_.shape[0]*kk + kk + kk*Vt.shape[1]) * U.FLOAT
            if fac < dense:
                payload[name] = ("svd", U_[:, :kk].copy(), S_[:kk].copy(),
                                 Vt[:kk, :].copy(), w.shape)
                nbytes += int(fac); continue
        payload[name] = ("dense", w.copy()); nbytes += w.size * U.FLOAT
    return payload, nbytes

def svd_decompress(payload):
    out = {}
    for name, item in payload.items():
        if item[0] == "svd":
            _, U_, S_, Vt, sh = item
            out[name] = (U_ @ np.diag(S_) @ Vt).reshape(sh).astype(np.float32)
        elif item[0] == "svd_rows":
            _, U_, S_, Vt, sh, rows, transposed = item
            compact = U_ @ np.diag(S_) @ Vt
            matrix_shape = (sh[1], sh[0]) if transposed else (sh[0], int(np.prod(sh[1:])))
            full = np.zeros(matrix_shape, dtype=np.float32)
            full[np.asarray(rows, dtype=np.int64)] = compact
            out[name] = (full.T if transposed else full).reshape(sh)
        else:
            out[name] = item[1].astype(np.float32)
    return out

def svd_compress_compact_rows(state, k, first_key, mask):
    """SVD compression with compact active rows for the input layer.

    Row indices are part of the already-shared mask and are therefore not counted
    in the per-round payload. Other tensors use the ordinary dense-or-SVD rule.
    """
    payload, nbytes = {}, 0
    rows = np.flatnonzero(np.asarray(mask, dtype=bool))
    for name, w in state.items():
        w = np.asarray(w, dtype=np.float32)
        if name == first_key and w.ndim >= 2:
            transposed = (w.ndim == 2 and w.shape[1] == len(mask))
            full_matrix = w.T if transposed else w.reshape(w.shape[0], -1)
            if full_matrix.shape[0] != len(mask):
                raise ValueError("shared feature mask does not match the input-layer dimension")
            m = full_matrix[rows]
            U_, S_, Vt = np.linalg.svd(m, full_matrices=False)
            kk = min(k, len(S_))
            dense = m.size * U.FLOAT
            fac = (m.shape[0] * kk + kk + kk * m.shape[1]) * U.FLOAT
            if fac < dense:
                payload[name] = ("svd_rows", U_[:, :kk].copy(), S_[:kk].copy(),
                                 Vt[:kk, :].copy(), w.shape, rows.copy(), transposed)
                nbytes += int(fac)
            else:
                payload[name] = ("svd_rows", np.eye(m.shape[0], dtype=np.float32),
                                 np.ones(m.shape[0], dtype=np.float32), m.copy(),
                                 w.shape, rows.copy(), transposed)
                nbytes += int(dense)
        else:
            p, nb = svd_compress({name: w}, k)
            payload.update(p); nbytes += nb
    return payload, nbytes

# ---------------- PowerSGD (low-rank P,Q; same payload family) ----------------
def powersgd_compress(state, k, q_memory=None, n_power_iter=1):
    """One-step PowerSGD-style factorisation with optional warm-started Q."""
    payload, nbytes = {}, 0
    q_next = {}
    for name, w in state.items():
        w = np.asarray(w, dtype=np.float32)
        if w.ndim >= 2:
            m = w.reshape(w.shape[0], -1)
            kk = min(k, min(m.shape))
            Q = None if q_memory is None else q_memory.get(name)
            if Q is None or Q.shape != (m.shape[1], kk):
                Q = np.random.standard_normal((m.shape[1], kk)).astype(np.float32)
            Q, _ = np.linalg.qr(Q, mode="reduced")
            for _ in range(max(1, n_power_iter)):
                P, _ = np.linalg.qr(m @ Q, mode="reduced")
                Q = m.T @ P
            P, _ = np.linalg.qr(P, mode="reduced")
            Q = m.T @ P
            payload[name] = ("pq", P.copy(), Q.copy(), w.shape)
            q_next[name] = Q.copy()
            nbytes += (P.size + Q.size) * U.FLOAT; continue
        payload[name] = ("dense", w.copy()); nbytes += w.size * U.FLOAT
    return payload, nbytes, q_next

def powersgd_decompress(payload):
    out = {}
    for name, item in payload.items():
        if item[0] == "pq":
            _, P, Q, sh = item; out[name] = (P @ Q.T).reshape(sh).astype(np.float32)
        else:
            out[name] = item[1].astype(np.float32)
    return out

def powersgd_p_bytes(payload):
    """Bytes in the intermediate orthogonalised-P broadcast."""
    return sum(item[1].size * U.FLOAT for item in payload.values() if item[0] == "pq")

# ---------------- QSGD (stochastic, s levels ~ 8-bit) ----------------
def qsgd_compress(state, s=255):
    payload, nbytes = {}, 0
    for name, w in state.items():
        v = np.asarray(w, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(v))
        if norm > 0:
            scaled = np.abs(v) / norm * s
            xi = np.floor(scaled).astype(np.int64)
            prob = scaled - xi
            q = xi + (np.random.rand(*xi.shape) < prob).astype(np.int64)
            q = np.clip(q, 0, s)
            signs = np.sign(v).astype(np.int8)
            payload[name] = ("qsgd", q.astype(np.uint8), signs, norm, w.shape)
            nbytes += U.FLOAT + v.size * U.UINT8 + U.bytes_bits(v.size)
        else:
            payload[name] = ("dense", w.copy()); nbytes += w.size * U.FLOAT
    return payload, nbytes

def qsgd_decompress(payload):
    out = {}
    for name, item in payload.items():
        if item[0] == "qsgd":
            _, q, signs, norm, sh = item
            v = signs * (q.astype(np.float32) / 255.0) * norm
            out[name] = v.reshape(sh).astype(np.float32)
        else:
            out[name] = item[1].astype(np.float32)
    return out

# ---------------- Top-k with error feedback ----------------
def topk_compress(residual, ratio, adaptive=False, lo=0.01, hi=0.20, energy=0.99):
    """residual: flat dict of numpy arrays + EF memory added by caller.
    Returns (payload, bytes, kept_mask_per_key)."""
    payload, nbytes, masks = {}, 0, {}
    for name, v in residual.items():
        original_shape = v.shape
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        mag = np.abs(v); order = np.argsort(-mag)
        if adaptive:
            tot = (mag ** 2).sum()
            cum = np.cumsum(mag[order] ** 2)
            kk = int(np.searchsorted(cum, energy * tot)) + 1
            kk = max(1, int(np.clip(kk, lo * v.size, hi * v.size)))
        else:
            kk = max(1, int(round(ratio * v.size)))
        idx = np.sort(order[:kk])
        vals = v[idx]
        masks[name] = (idx, original_shape)
        payload[name] = ("topk", vals.astype(np.float32), idx.astype(np.uint32), original_shape)
        nbytes += vals.size * U.FLOAT + idx.size * U.UINT32
    return payload, nbytes, masks

def topk_decompress(payload):
    out = {}
    for name, item in payload.items():
        if item[0] == "topk":
            _, vals, idx, sh = item
            full = np.zeros(int(np.prod(sh)), dtype=np.float32)
            full[idx] = vals
            out[name] = full.reshape(sh)
        else:
            out[name] = item[1].astype(np.float32)
    return out

# ---------------- SignSGD (1 bit / coord) ----------------
def sign_compress(state):
    payload, nbytes = {}, 0
    for name, w in state.items():
        s = np.sign(np.asarray(w, dtype=np.float32)).astype(np.int8)
        payload[name] = ("sign", s, w.shape); nbytes += U.bytes_bits(s.size)
    return payload, nbytes

def sign_decompress(payload):
    return {n: it[1].astype(np.float32) for n, it in payload.items()}

def sign_majority(sign_list, weights=None):
    """hierarchical majority vote over a list of sign dicts."""
    acc = None
    for s in sign_list:
        d = {k: np.asarray(v[1], dtype=np.float32) for k, v in s.items()}
        acc = d if acc is None else {k: acc[k] + d[k] for k in acc}
    return {k: np.sign(v).astype(np.float32) for k, v in acc.items()}
