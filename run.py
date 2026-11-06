import os, argparse, time, resource
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from configs.presets import preset, METHODS
from flcore import utils as U
from flcore.data import make_loaders
from flcore.engine import train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--dataset", required=True, choices=["fashion", "har", "cifar"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--no-resume", action="store_true", help="Disable checkpoint creation/continuation")
    ap.add_argument("--tag", default="",
                    help="suffix for the output file; use 'fs'/'nofs' to disambiguate "
                         "the two HAR Proposed variants (and the unmasked Fashion path).")
    ap.add_argument("--fs", default=None, choices=["on", "off"],
                    help="override feature selection (on/off). If omitted, follows tag/dataset.")
    ap.add_argument("--out", default="results")
    ap.add_argument("--clients", type=int)
    ap.add_argument("--edges", type=int)
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--local-epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--rank", type=int)
    ap.add_argument("--rho", type=float)
    a = ap.parse_args()

    if a.fs is not None:
        fs = {"on": True, "off": False}[a.fs]
    elif a.tag in ("fs", "nofs"):
        fs = (a.tag == "fs")
    else:
        fs = (a.dataset == "har" and a.method == "proposed")
    cfg = preset(a.dataset, fs=fs); cfg.method = a.method; cfg.seed = a.seed
    for cli_name, cfg_name in (("clients","clients"),("edges","edges"),("rounds","rounds"),
                               ("local_epochs","local_epochs"),("batch","batch"),
                               ("lr","lr"),("alpha","alpha"),("rank","rank"),("rho","rho")):
        value = getattr(a, cli_name)
        if value is not None: setattr(cfg, cfg_name, value)
    if cfg.feature_selection and (cfg.dataset != "har" or cfg.method != "proposed"):
        ap.error("feature selection is implemented only for Proposed on UCI HAR")
    import torch
    torch.set_num_threads(a.threads)
    if a.device != "auto": cfg.cuda = a.device == "cuda"
    if cfg.cuda and not torch.cuda.is_available():
        ap.error("This preset requires CUDA; use --device cpu for an explicitly different host")
    U.set_seed(cfg.seed)
    effective_tag = a.tag
    if not effective_tag and a.method == "proposed" and a.dataset in ("fashion", "har"):
        effective_tag = "fs" if cfg.feature_selection else "nofs"
    log = U.get_logger("run")
    log.info(f"START {cfg.method}/{cfg.dataset}/s{cfg.seed} tag='{effective_tag}' fs={cfg.feature_selection}")

    import fcntl, json
    from pathlib import Path
    _suffix = f'_{effective_tag}' if effective_tag else ''
    _final = Path(a.out) / f'{cfg.dataset}_{cfg.method}{_suffix}_s{cfg.seed}.json'
    _lockdir = Path(a.out) / '.run_locks'; _lockdir.mkdir(parents=True, exist_ok=True)
    _lock = (_lockdir / (_final.name + '.lock')).open('a')
    fcntl.flock(_lock, fcntl.LOCK_EX)
    log.info('RUN_LOCK_ACQUIRED %s', _final.name)
    if _final.exists():
        _old = json.loads(_final.read_text())
        if len(_old.get('traj', [])) == cfg.rounds and _old.get('cfg') == json.loads(json.dumps(cfg.__dict__)):
            log.info('ALREADY_COMPLETE %s', _final.name); return
        raise ValueError('Existing final output has a different configuration: ' + str(_final))

    fn_base = _final.name
    t0 = time.perf_counter()
    cl, tl, d, C, is_img, cidx = make_loaders(cfg.dataset, cfg.batch, cfg.seed,
                                              cfg.alpha, cfg.clients)
    import hashlib
    signature = hashlib.sha256(json.dumps(cfg.__dict__, sort_keys=True).encode()).hexdigest()[:16]
    if not a.no_resume:
        cfg.checkpoint_path = str(Path(a.out) / 'checkpoints' / (Path(fn_base).stem + '_' + signature + '.pt'))
    traj, comm = train(cfg, cl, tl, d, C, is_img, cidx, log)
    if hasattr(cfg, 'checkpoint_path'): del cfg.checkpoint_path
    wall_time_s = time.perf_counter() - t0
    peak_rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_mb = peak_rss_raw / 1024.0  # Linux reports KiB

    os.makedirs(a.out, exist_ok=True)
    suf = f"_{effective_tag}" if effective_tag else ""
    fn = f"{a.out}/{cfg.dataset}_{cfg.method}{suf}_s{cfg.seed}.json"
    U.save_json(fn, dict(cfg=cfg.__dict__, tag=effective_tag, traj=traj,
                         final_acc=traj[-1]["acc"], final_f1=traj[-1]["f1"],
                         comm_mb=comm.mb(),
                         wall_time_s=wall_time_s, peak_rss_mb=peak_rss_mb,
                         c2e=comm.c2e/(1024**2), e2c=comm.e2c/(1024**2),
                         down=comm.down/(1024**2), setup=comm.setup/(1024**2)))
    log.info(f"DONE -> {fn}  acc={traj[-1]['acc']:.2f}  comm={comm.mb():.2f} MB  "
             f"time={wall_time_s:.1f}s  peakRSS={peak_rss_mb:.1f}MB")

if __name__ == "__main__":
    main()
