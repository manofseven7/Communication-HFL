from dataclasses import dataclass, asdict

@dataclass
class Cfg:
    method: str = "proposed"
    dataset: str = "har"
    seed: int = 42
    # federation (Table 2)
    clients: int = 50
    edges: int = 5
    rounds: int = 30
    local_epochs: int = 5
    lr: float = 0.05
    batch: int = 64
    rank: int = 10
    alpha: float = 0.1
    # feature selection (HAR)
    feature_selection: bool = False
    Tw: int = 5
    rho: float = 0.70
    fs_weights: tuple = (0.60, 0.25, 0.15)
    # method-specific
    mu: float = 0.01            # FedProx
    gamma: float = 1.5          # FedCOM
    fedpaq_k: int = 25          # FedPAQ sampled clients
    topk_ratio: float = 0.10
    mlp_hidden: int = 128
    cuda: bool = False
    # SCAFFOLD transmission model (single source of truth):
    #   "full"    = paper option-II: client sends Δx AND Δc_i, server broadcasts
    #               model AND c  -> total = 2x FedAvg  (matches the manuscript)
    #   "minimal" = client sends Δx only, server broadcasts model AND c
    #               -> total = 1.333x FedAvg  (optimised variant; document if used)
    scaffold_variant: str = "full"

def preset(dataset: str, fs: bool | None = None) -> Cfg:
    """fs=None  -> feature selection follows the dataset (on for HAR, off otherwise),
       unless overridden by the --tag/--fs CLI flags in run.py."""
    c = Cfg(dataset=dataset)
    c.feature_selection = (fs if fs is not None else (dataset == "har"))
    if dataset == "cifar":                 # Table 7 matched protocol
        c.clients, c.edges, c.rounds = 100, 5, 400
        c.feature_selection = False; c.cuda = True
    return c

METHODS = ["fedavg", "fedprox", "scaffold", "fednova", "fedkd",
           "qsgd", "fedpaq", "fedcom", "topk", "adaptive_topk",
           "sign", "powersgd", "proposed"]
