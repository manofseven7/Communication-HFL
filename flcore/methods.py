import copy, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from . import compression as C, utils as U

DENSE = "dense"

class FLMethod:
    """One engine, behaviour switched by cfg.method. Covers:
       fedavg, fedprox, scaffold, fednova, qsgd, fedpaq, fedcom,
       topk, adaptive_topk, sign, powersgd, proposed."""
    def __init__(self, cfg, model_template):
        self.cfg = cfg; self.m = cfg.method
        self.template = model_template
        # persistent states
        self.client_c = {}           # SCAFFOLD client control variates
        self.server_c = None         # SCAFFOLD server control variate
        self.client_ef = {}          # Top-k error feedback (uplink residual)
        self.server_ef = None        # downlink error feedback (central)
        self.powersgd_q = {}         # warm-start factors per client
        self.powersgd_edge_q = {}
        self.active_mask = None

    # ---------- local training ----------
    def client_train(self, view_state, loader, client_id, mask=None,
                     teacher=None, lam=0.0, T=1.0):
        cfg = self.cfg
        model = copy.deepcopy(self.template)
        model.load_state_dict(U.from_np(view_state, model.state_dict()))
        opt = torch.optim.SGD(model.parameters(), lr=cfg.lr)
        model.train()
        c_i = self.client_c.get(client_id)
        c   = self.server_c
        n_steps = 0
        for _ in range(cfg.local_epochs):
            for x, y in loader:
                x = x.to(next(model.parameters()).device)
                y = y.to(next(model.parameters()).device)
                if mask is not None:
                    x = x * torch.as_tensor(mask, dtype=x.dtype, device=x.device)
                opt.zero_grad(); out = model(x); loss = F.cross_entropy(out, y)
                # FedProx proximal term
                if self.m == "fedprox":
                    for (n, p), (_, g0) in zip(model.named_parameters(),
                            U.from_np(view_state, model.state_dict()).items()):
                        loss = loss + (cfg.mu / 2.0) * ((p - g0) ** 2).sum()
                # SCAFFOLD control-variate correction
                if self.m == "scaffold" and c_i is not None and c is not None:
                    corr = U.from_np({k: c[k] - c_i[k] for k in c}, model.state_dict())
                    for p, g in zip(model.parameters(), corr.values()):
                        loss = loss + (p * g).sum()
                loss.backward(); opt.step(); n_steps += 1
        # local update = trained - view
        upd = {k: (p.detach().cpu().numpy() - view_state[k]).astype(np.float32)
               for k, p in model.state_dict().items()}
        # PyTorch Linear stores features in columns; keep inactive columns zero.
        if mask is not None:
            first = next(iter(upd))
            upd[first] = upd[first] * mask[None, :]
        return upd, n_steps, model

    # ---------- uplink compression (client -> edge) ----------
    def uplink(self, upd, client_id, mask=None):
        k = self.cfg.rank
        if self.m == "proposed":
            if mask is not None:
                return C.svd_compress_compact_rows(upd, k, next(iter(upd)), mask)
            return C.svd_compress(upd, k)
        if self.m == "powersgd":
            ef = self.client_ef.get(client_id, {n: np.zeros_like(v) for n, v in upd.items()})
            res = {n: upd[n] + ef[n] for n in upd}
            pay, nb, q = C.powersgd_compress(res, k, self.powersgd_q.get(client_id))
            kept = C.powersgd_decompress(pay)
            self.client_ef[client_id] = {n: res[n] - kept[n] for n in res}
            self.powersgd_q[client_id] = q
            return pay, nb
        if self.m in ("qsgd", "fedpaq", "fedcom"):
            return C.qsgd_compress(upd, s=255)
        if self.m in ("topk", "adaptive_topk"):
            ef = self.client_ef.get(client_id, {n: np.zeros_like(v) for n, v in upd.items()})
            res = {n: upd[n] + ef[n] for n in upd}
            ratio = 0.10 if self.m == "topk" else None
            pay, nb, masks = C.topk_compress(res, ratio, adaptive=(self.m == "adaptive_topk"))
            kept = C.topk_decompress(pay)
            self.client_ef[client_id] = {n: res[n] - kept[n] for n in res}
            return pay, nb
        if self.m == "sign":
            return C.sign_compress(upd)
        # dense (fedavg/fedprox/scaffold/fednova): model part only here
        return {n: ("dense", v.copy()) for n, v in upd.items()}, \
               sum(v.size for v in upd.values()) * U.FLOAT

    # ---------- edge aggregation + recompression ----------
    def edge_aggregate(self, payloads, sizes, edge_id=None):
        w = np.asarray(sizes, dtype=np.float64); w = w / w.sum()
        if self.m == "sign":
            agg = C.sign_majority(payloads)
            return {n: ("sign", np.sign(v).astype(np.int8), v.shape) for n, v in agg.items()}, \
                   sum(U.bytes_bits(v.size) for v in agg.values())
        decs = [self._decomp(p) for p in payloads]
        agg = None
        for wi, d in zip(w, decs):
            agg = {n: wi * d[n] for n in d} if agg is None else \
                  {n: agg[n] + wi * d[n] for n in agg}
        if self.m == "proposed":
            if self.active_mask is not None:
                return C.svd_compress_compact_rows(
                    agg, self.cfg.rank, next(iter(agg)), self.active_mask)
            return C.svd_compress(agg, self.cfg.rank)
        if self.m == "powersgd":
            pay, nb, q = C.powersgd_compress(
                agg, self.cfg.rank, self.powersgd_edge_q.get(edge_id))
            self.powersgd_edge_q[edge_id] = q
            return pay, nb
        if self.m == "fedcom":
            agg = {n: self.cfg.gamma * v for n, v in agg.items()}
        # Quantised and sparse baselines use dense edge-to-central traffic in
        # the manuscript's declared accounting protocol.
        # dense (model part only; control-variate doubling handled in engine)
        return {n: ("dense", v.copy()) for n, v in agg.items()}, \
               sum(v.size for v in agg.values()) * U.FLOAT

    # ---------- central aggregation + downlink compression ----------
    def central_aggregate(self, edge_payloads, edge_sizes):
        w = np.asarray(edge_sizes, dtype=np.float64); w = w / w.sum()
        if self.m == "sign":
            agg = C.sign_majority(edge_payloads)
            return {n: ("sign", np.sign(v).astype(np.int8), v.shape) for n, v in agg.items()}, \
                   sum(U.bytes_bits(v.size) for v in agg.values())
        decs = [self._decomp(p) for p in edge_payloads]
        agg = None
        for wi, d in zip(w, decs):
            agg = {n: wi * d[n] for n in d} if agg is None else \
                  {n: agg[n] + wi * d[n] for n in agg}

        # ---- SCAFFOLD: downlink carries global model AND global server cv ----
        if self.m == "scaffold":
            model_bytes = sum(v.size for v in agg.values()) * U.FLOAT
            cv = self.server_c or {k: np.zeros_like(v) for k, v in agg.items()}
            cv_bytes = sum(v.size for v in cv.values()) * U.FLOAT
            pay = {n: ("dense", v.copy()) for n, v in agg.items()}
            return pay, model_bytes + cv_bytes     # downlink = 2x model-equivalent

        # ---- downlink error feedback for compressed methods ----
        if self.m == "proposed":
            ef = self.server_ef or {n: np.zeros_like(v) for n, v in agg.items()}
            payload_in = {n: agg[n] + ef[n] for n in agg}
            if self.active_mask is not None:
                pay, nb = C.svd_compress_compact_rows(
                    payload_in, self.cfg.rank, next(iter(payload_in)), self.active_mask)
            else:
                pay, nb = C.svd_compress(payload_in, self.cfg.rank)
            kept = self._decomp(pay)
            self.server_ef = {n: payload_in[n] - kept[n] for n in payload_in}
            return pay, nb

        # ---- dense (fedavg/fedprox/fednova) ----
        pay = {n: ("dense", v.copy()) for n, v in agg.items()}
        return pay, sum(v.size for v in agg.values()) * U.FLOAT

    # ---------- apply downlink to view ----------
    def apply_downlink(self, view_state, down_payload):
        dec = self._decomp(down_payload)
        return {k: (view_state[k] + dec[k]).astype(np.float32) for k in view_state}

    # ---------- helpers ----------
    def _decomp(self, payload):
        if self.m == "proposed":     return C.svd_decompress(payload)
        if self.m == "powersgd":     return C.powersgd_decompress(payload)
        if self.m in ("qsgd", "fedpaq", "fedcom"): return C.qsgd_decompress(payload)
        if self.m in ("topk", "adaptive_topk"):    return C.topk_decompress(payload)
        if self.m == "sign":         return C.sign_decompress(payload)
        return {n: it[1].astype(np.float32) for n, it in payload.items()}

    def scaffold_update_cv(self, local_update, n_steps, client_id):
        """Option-II client control update; returns delta-c for aggregation."""
        if self.m != "scaffold": return 0
        c = self.server_c or {k: np.zeros_like(v) for k, v in local_update.items()}
        ci = self.client_c.get(client_id, {k: np.zeros_like(v) for k, v in local_update.items()})
        denom = max(n_steps, 1) * max(self.cfg.lr, 1e-8)
        new_ci = {k: ci[k] - c[k] - local_update[k] / denom for k in local_update}
        delta_ci = {k: new_ci[k] - ci[k] for k in ci}
        self.client_c[client_id] = new_ci
        return delta_ci

    def scaffold_server_cv(self, client_deltas, sizes):
        if self.m != "scaffold": return 0
        w = np.full(len(client_deltas), 1.0 / len(client_deltas), dtype=np.float64)
        agg = None
        for wi, d in zip(w, client_deltas):
            agg = {k: wi * d[k] for k in d} if agg is None else \
                  {k: agg[k] + wi * d[k] for k in agg}
        if self.server_c is None:
            self.server_c = {k: np.zeros_like(v) for k, v in agg.items()}
        self.server_c = {k: self.server_c[k] + agg[k] for k in agg}
        return 0


class FedKDMethod:
    """Disclosed adaptation: private 256-teacher + shared 128-student; bidirectional KD."""
    def __init__(self, cfg, d, C_out):
        from .models import FedKD_Student, FedKD_Teacher
        self.cfg = cfg; self.d = d; self.C = C_out
        self.student_tmpl = FedKD_Student(d, C_out)
        self.teacher = {}
        self.lam = 0.3; self.T = 1.0

    def ensure_teacher(self, cid, loader, device):
        from .models import FedKD_Teacher
        if cid in self.teacher: return
        self.teacher[cid] = FedKD_Teacher(self.d, self.C).to(device)

    def client_train(self, view_state, loader, cid):
        stu = copy.deepcopy(self.student_tmpl)
        device = next(self.teacher[cid].parameters()).device
        stu = stu.to(device)
        stu.load_state_dict(U.from_np(view_state, stu.state_dict())); stu.train()
        tch = self.teacher[cid].train()
        opt_s = torch.optim.SGD(stu.parameters(), lr=self.cfg.lr)
        opt_t = torch.optim.SGD(tch.parameters(), lr=self.cfg.lr)
        for _ in range(self.cfg.local_epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt_s.zero_grad(); opt_t.zero_grad()
                ls, lt = stu(x), tch(x)
                ps = F.softmax(ls.detach() / self.T, dim=1)
                pt = F.softmax(lt.detach() / self.T, dim=1)
                loss_s = F.cross_entropy(ls, y) + self.lam * (self.T ** 2) * \
                    F.kl_div(F.log_softmax(ls / self.T, dim=1), pt, reduction="batchmean")
                loss_t = F.cross_entropy(lt, y) + self.lam * (self.T ** 2) * \
                    F.kl_div(F.log_softmax(lt / self.T, dim=1), ps, reduction="batchmean")
                (loss_s + loss_t).backward(); opt_s.step(); opt_t.step()
        upd = {k: (p.detach().cpu().numpy() - view_state[k]).astype(np.float32)
               for k, p in stu.state_dict().items()}
        return upd

    def uplink(self, upd, cid, mask=None):
        return C.svd_compress(upd, self.cfg.rank)

    def edge_aggregate(self, payloads, sizes, edge_id=None):
        w = np.asarray(sizes, dtype=np.float64); w = w / w.sum()
        decs = [C.svd_decompress(p) for p in payloads]; agg = None
        for wi, d in zip(w, decs):
            agg = {n: wi * d[n] for n in d} if agg is None else {n: agg[n] + wi * d[n] for n in agg}
        return {n: ("dense", v.copy()) for n, v in agg.items()}, \
               sum(v.size for v in agg.values()) * U.FLOAT

    def central_aggregate(self, ep, es):
        w = np.asarray(es, dtype=np.float64); w = w / w.sum()
        decs = [{n: it[1].astype(np.float32) for n, it in p.items()} for p in ep]; agg = None
        for wi, d in zip(w, decs):
            agg = {n: wi * d[n] for n in d} if agg is None else {n: agg[n] + wi * d[n] for n in agg}
        return {n: ("dense", v.copy()) for n, v in agg.items()}, \
               sum(v.size for v in agg.values()) * U.FLOAT

    def apply_downlink(self, view_state, pay):
        dec = {n: it[1].astype(np.float32) for n, it in pay.items()}
        return {k: (view_state[k] + dec[k]).astype(np.float32) for k in view_state}
