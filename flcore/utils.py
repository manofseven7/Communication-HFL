import os, json, random, logging
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_logger(name: str):
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(); h.setFormatter(
            logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                              datefmt="%H:%M:%S"))
        log.addHandler(h); log.setLevel(logging.INFO)
    return log

# ---------- byte / float accounting (matches the paper's 3-link model) ----------
FLOAT = 4      # float32
UINT32 = 4
UINT8 = 1
BIT = 1 / 8

def numel(x) -> int:
    return int(np.asarray(x).size)

def bytes_floats(n: int) -> int:   return n * FLOAT
def bytes_uint32(n: int) -> int:   return n * UINT32
def bytes_bits(n_bits: float) -> int: return int(np.ceil(n_bits * BIT))

def to_np(state):
    return {k: v.detach().cpu().numpy() for k, v in state.items()}

def from_np(arrdict, template_state):
    out = {}
    for k, v in template_state.items():
        out[k] = torch.as_tensor(arrdict[k], dtype=v.dtype, device=v.device)
    return out

def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    path = str(path)
    with open(path + ".tmp", "w") as f: json.dump(obj, f, indent=2)
    os.replace(path + ".tmp", path)
