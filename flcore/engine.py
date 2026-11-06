import copy, os, random, hashlib, json, numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import f1_score
from . import utils as U, feature_selection as FS
from .methods import FLMethod, FedKDMethod

class CommTracker:
    def __init__(self): self.c2e = 0; self.e2c = 0; self.down = 0; self.setup = 0
    def total(self): return self.c2e + self.e2c + self.down + self.setup
    def mb(self): return self.total() / (1024 ** 2)

def _evaluate(model, loader, device):
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise FloatingPointError("Model parameters must be finite before evaluation")
    model.eval(); cor = tot = 0; ys = []; ps = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Model logits must be finite before evaluation")
            pred = logits.argmax(1)
            cor += (pred == y).sum().item(); tot += y.size(0)
            ys.append(y.cpu().numpy()); ps.append(pred.cpu().numpy())
    y_true, y_pred = np.concatenate(ys), np.concatenate(ps)
    return 100.0 * cor / tot, 100.0 * f1_score(y_true, y_pred, average="macro", zero_division=0)

def train(cfg, client_loader, test_loader, d, C_out, is_img, client_idxs, log):
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")
    from .models import build_model
    tmpl = build_model(cfg, d, C_out).to(device)
    view = U.to_np(tmpl.state_dict())
    method = (FedKDMethod(cfg, d, C_out) if cfg.method == "fedkd"
              else FLMethod(cfg, tmpl))

    M, N = cfg.edges, cfg.clients
    groups = [list(map(int, g)) for g in np.array_split(np.arange(N), M)]
    comm = CommTracker(); traj = []; mask = None; pending_mask = None

    # SCAFFOLD transmission model (single source of truth; see configs/presets.py)
    full_cv = (cfg.method == "scaffold" and
               getattr(cfg, "scaffold_variant", "full") == "full")

    start_round = 1
    checkpoint_path = getattr(cfg, "checkpoint_path", None)
    # Checkpoints are owned outputs from this run, not third-party model files.
    signature = hashlib.sha256(json.dumps({k:v for k,v in cfg.__dict__.items()
        if k != "checkpoint_path"}, sort_keys=True).encode()).hexdigest()
    if checkpoint_path and os.path.exists(checkpoint_path):
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved["signature"] != signature:
            raise ValueError("Checkpoint configuration differs; use a separate run path")
        view, method, comm = saved["view"], saved["method"], saved["comm"]
        traj, mask, pending_mask = saved["traj"], saved["mask"], saved["pending_mask"]
        start_round = saved["round"] + 1
        random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
        torch.random.set_rng_state(saved["torch_rng"].cpu())
        if torch.cuda.is_available() and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in saved["cuda_rng"]])
        log.info("Resuming after round %s", saved["round"])

    for r in range(start_round, cfg.rounds + 1):
        # A mask computed after the final warm-up round is activated only here.
        if pending_mask is not None:
            mask, pending_mask = pending_mask, None
            first_key = next(iter(view))
            view[first_key] = view[first_key] * mask[None, :]
            method.active_mask = mask
            if method.server_ef is not None:
                method.server_ef[first_key] *= mask[None, :]
        round_start = copy.deepcopy(view)
        sel = (sorted(np.random.choice(N, cfg.fedpaq_k, replace=False).tolist())
               if cfg.method == "fedpaq" else list(range(N)))

        # ---- client train + uplink (model part) ----
        uplinks, sizes, nsteps, cv_deltas = {}, {}, {}, {}
        round_c2e_model = 0; round_power_p_broadcast = 0
        for i in sel:
            loader, ni = client_loader(i)
            if cfg.method == "fedkd":
                method.ensure_teacher(i, loader, device); upd = method.client_train(view, loader, i)
            else:
                upd, ns, _ = method.client_train(view, loader, i, mask=mask); nsteps[i] = ns
                if cfg.method == "scaffold":
                    cv_deltas[i] = method.scaffold_update_cv(upd, ns, i)
            pay, nb = method.uplink(upd, i, mask=mask)
            if cfg.method == "powersgd":
                from .compression import powersgd_p_bytes
                round_power_p_broadcast += powersgd_p_bytes(pay)
            uplinks[i] = (pay, upd); sizes[i] = ni; round_c2e_model += nb
        if full_cv:                       # client also transmits Δc_i (== model shape)
            round_c2e_model *= 2
        comm.c2e += round_c2e_model
        comm.down += round_power_p_broadcast

        fednova_tau = None
        if cfg.method == "fednova":       # normalise by tau, restore weighted effective tau globally
            total_n = sum(sizes.values())
            fednova_tau = sum((sizes[i] / total_n) * max(nsteps[i], 1) for i in sel)
            for i in sel:
                pay, upd = uplinks[i]
                tau = max(nsteps[i], 1)
                uplinks[i] = (_scale_payload(pay, cfg.method, 1.0 / tau), upd)

        # ---- edge aggregate + recompress (model part) ----
        edge_pay, edge_sizes, round_e2c_model = [], [], 0
        for edge_id, g in enumerate(groups):
            gsel = [i for i in g if i in sel]
            if not gsel: continue
            ep, eb = method.edge_aggregate([uplinks[i][0] for i in gsel],
                                           [sizes[i] for i in gsel], edge_id=edge_id)
            edge_pay.append(ep); edge_sizes.append(sum(sizes[i] for i in gsel))
            round_e2c_model += eb
        if full_cv:                       # edge also forwards aggregated Δc (== agg shape)
            round_e2c_model *= 2
        comm.e2c += round_e2c_model

        # ---- central aggregate + downlink (scaffold: nb already = model + server cv) ----
        dpay, db = method.central_aggregate(edge_pay, edge_sizes)
        if fednova_tau is not None:
            dpay = _scale_payload(dpay, "fednova", fednova_tau)
        # The paper counts the complete central-to-client delivery, not a
        # single logical broadcast object. FedPAQ sends only to sampled clients.
        comm.down += db * len(sel)

        view = method.apply_downlink(view, dpay)

        # ---- SCAFFOLD control-variate bookkeeping (NO comm accounting here) ----
        if cfg.method == "scaffold":
            method.scaffold_server_cv([cv_deltas[i] for i in sel],
                                      [sizes[i] for i in sel])

        # ---- coordinated feature selection at T_w (HAR only) ----
        if cfg.feature_selection and mask is None and r == cfg.Tw:
            numpy_rng_state = np.random.get_state()
            torch_rng_state = torch.random.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            stats, szs = [], []; first_key = next(iter(view))
            for i in range(N):
                loader, ni = client_loader(i); Xb = _grab_inputs(loader, 512)
                if i not in uplinks:
                    raise RuntimeError("Feature-mask construction requires full participation at T_w")
                warmup_update = uplinks[i][1]
                G = FS.model_importance(round_start[first_key] + warmup_update[first_key],
                                        round_start[first_key])
                stats.append((G, FS.marginal_entropy(Xb),
                              FS.correlation_redundancy(Xb, seed=cfg.seed * 1000 + i)))
                szs.append(ni)
            pending_mask = FS.build_shared_mask(stats, szs, cfg.rho, w=cfg.fs_weights)
            comm.setup += FS.setup_bytes(N, len(pending_mask))
            # Reading score inputs must not perturb the shared training RNG.
            np.random.set_state(numpy_rng_state)
            torch.random.set_rng_state(torch_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

        # ---- evaluate ----
        tmpl.load_state_dict(U.from_np(view, tmpl.state_dict()))
        acc, f1 = _evaluate(tmpl, test_loader, device)
        traj.append(dict(round=r, acc=acc, f1=f1, comm_mb=comm.mb(), mask_active=mask is not None))
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            saved = dict(signature=signature, round=r, view=view, method=method,
                comm=comm, traj=traj, mask=mask, pending_mask=pending_mask,
                python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                torch_rng=torch.random.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
            torch.save(saved, checkpoint_path + ".tmp")
            os.replace(checkpoint_path + ".tmp", checkpoint_path)
        if r % 5 == 0 or r == cfg.rounds:
            log.info(f"[{cfg.method:14s}|{cfg.dataset:7s}|s{cfg.seed}] r={r:3d} "
                     f"acc={acc:5.2f}  f1={f1:5.2f}  comm={comm.mb():8.2f} MB")
    return traj, comm

def _scale_payload(pay, method, s):
    out = {}
    for n, it in pay.items():
        if it[0] == "dense": out[n] = ("dense", it[1] * s)
        elif it[0] == "svd": out[n] = ("svd", it[1], it[2]*s, it[3], it[4])
        else: out[n] = it
    return out

def _grab_inputs(loader, max_samples):
    xs = []
    for b, (x, y) in enumerate(loader):
        xs.append(x.numpy())
        if sum(len(z) for z in xs) >= max_samples: break
    return np.concatenate(xs, 0)[:max_samples]
