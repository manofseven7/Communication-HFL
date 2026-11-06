import logging
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from configs.presets import Cfg, METHODS
from flcore import compression as C
from flcore.engine import train
from flcore.feature_selection import build_shared_mask, model_importance
from flcore.models import MLP


def test_compact_feature_svd_uses_pytorch_input_columns():
    rng = np.random.default_rng(2)
    state = {
        "net.0.weight": rng.normal(size=(8, 12)).astype(np.float32),
        "net.0.bias": rng.normal(size=8).astype(np.float32),
    }
    mask = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1], dtype=bool)
    payload, nbytes = C.svd_compress_compact_rows(state, 3, "net.0.weight", mask)
    restored = C.svd_decompress(payload)
    assert restored["net.0.weight"].shape == state["net.0.weight"].shape
    assert np.all(restored["net.0.weight"][:, ~mask] == 0)
    assert nbytes > 0


def test_importance_and_mask_have_exact_input_dimension_and_retention():
    a = np.zeros((5, 11), np.float32); b = a.copy(); b[:, 3] = 2
    g = model_importance(b, a)
    assert g.shape == (11,) and np.argmax(g) == 3
    stats = [(g, np.arange(11), np.arange(11)[::-1]) for _ in range(3)]
    mask = build_shared_mask(stats, [2, 3, 5], rho=.7)
    assert mask.shape == (11,) and mask.sum() == round(.7 * 11)


@pytest.mark.parametrize("d,classes,expected", [(784,10,1222.90),(561,6,873.70)])
def test_dense_three_link_accounting_matches_article(d, classes, expected):
    model=MLP(d,128,classes)
    nbytes=sum(p.numel() for p in model.parameters()) * 4 * (50+5+50) * 30
    assert nbytes/(1024**2) == pytest.approx(expected, abs=.01)


def _synthetic_loaders(n_clients=4, d=6, classes=3):
    gen = torch.Generator().manual_seed(7)
    client_sets=[]
    for cid in range(n_clients):
        x=torch.randn(12,d,generator=gen); y=(torch.arange(12)+cid) % classes
        client_sets.append(TensorDataset(x,y))
    xt=torch.randn(24,d,generator=gen); yt=torch.arange(24) % classes
    def client_loader(i):
        ds=client_sets[i]
        return DataLoader(ds,batch_size=4,shuffle=True,generator=torch.Generator().manual_seed(100+i)),len(ds)
    return client_loader, DataLoader(TensorDataset(xt,yt),batch_size=8), [np.arange(12) for _ in range(n_clients)]


@pytest.mark.parametrize("method", METHODS)
def test_one_round_all_methods(method):
    cl, test, idx = _synthetic_loaders()
    cfg=Cfg(method=method,dataset="har",clients=4,edges=2,rounds=1,
            local_epochs=1,batch=4,rank=2,mlp_hidden=8,fedpaq_k=2,
            feature_selection=(method == "proposed"),Tw=1,rho=.7)
    traj,comm=train(cfg,cl,test,6,3,False,idx,logging.getLogger("test"))
    assert len(traj)==1
    assert set(("round","acc","f1","comm_mb")) <= set(traj[0])
    assert np.isfinite(traj[0]["acc"]) and np.isfinite(traj[0]["f1"])
    assert comm.total() > 0


def test_mask_is_pending_through_round_five(monkeypatch):
    """Warm-up models must match exactly, including the round-five evaluation."""
    from flcore import engine, utils
    observed = []
    original_eval = engine._evaluate
    def capture(model, loader, device):
        observed.append({k: p.detach().cpu().numpy().copy() for k, p in model.state_dict().items()})
        return original_eval(model, loader, device)
    monkeypatch.setattr(engine, '_evaluate', capture)
    trajectories = []
    for fs in (False, True):
        utils.set_seed(42)
        cl, test, idx = _synthetic_loaders()
        cfg = Cfg(method='proposed', dataset='har', clients=4, edges=2, rounds=6,
                  local_epochs=1, batch=4, rank=2, mlp_hidden=8,
                  feature_selection=fs, Tw=5, rho=.7)
        traj, _ = engine.train(cfg, cl, test, 6, 3, False, idx, logging.getLogger('timing'))
        trajectories.append(traj)
    for r in range(5):
        assert all(np.array_equal(observed[r][k], observed[r+6][k]) for k in observed[r])
    assert [x['mask_active'] for x in trajectories[1]] == [False]*5 + [True]


def test_local_epoch_budget_counts_all_minibatches():
    from flcore.methods import FLMethod
    from flcore import utils
    cl, _, _ = _synthetic_loaders()
    loader, _ = cl(0)
    cfg = Cfg(method='fedavg', dataset='har', local_epochs=5, mlp_hidden=8)
    model = MLP(6, 8, 3)
    _, steps, _ = FLMethod(cfg, model).client_train(utils.to_np(model.state_dict()), loader, 0)
    assert steps == 5 * len(loader)


def test_checkpoint_resume_matches_uninterrupted(monkeypatch, tmp_path):
    from flcore import engine, utils
    path = str(tmp_path / 'run.pt')
    cfg = Cfg(method='proposed', dataset='har', clients=4, edges=2, rounds=6,
              local_epochs=1, batch=4, rank=2, mlp_hidden=8,
              feature_selection=True, Tw=5, rho=.7)
    cfg.checkpoint_path = path
    original_save = engine.torch.save
    def stop_after_five(obj, target):
        original_save(obj, target)
        if obj['round'] == 5:
            import os
            os.replace(target, path)
            raise InterruptedError('simulated interruption')
    utils.set_seed(42); cl, test, idx = _synthetic_loaders()
    monkeypatch.setattr(engine.torch, 'save', stop_after_five)
    with pytest.raises(InterruptedError):
        engine.train(cfg, cl, test, 6, 3, False, idx, logging.getLogger('resume'))
    monkeypatch.setattr(engine.torch, 'save', original_save)
    # The helper uses explicit per-loader generators, recreated deterministically.
    cl, test, idx = _synthetic_loaders()
    resumed, comm = engine.train(cfg, cl, test, 6, 3, False, idx, logging.getLogger('resume'))
    del cfg.checkpoint_path
    utils.set_seed(42); cl, test, idx = _synthetic_loaders()
    fresh, fresh_comm = engine.train(cfg, cl, test, 6, 3, False, idx, logging.getLogger('resume'))
    assert resumed == fresh
    assert comm.total() == fresh_comm.total()


def test_checkpoint_resume_global_training_rng(monkeypatch, tmp_path):
    from flcore import engine, utils
    def factory():
        cl, test, idx = _synthetic_loaders()
        def global_loader(i):
            dl, n = cl(i)
            return DataLoader(dl.dataset,batch_size=4,shuffle=True), n
        return global_loader, test, idx
    cfg=Cfg(method='proposed',dataset='har',clients=4,edges=2,rounds=6,
        local_epochs=1,batch=4,rank=2,mlp_hidden=8,feature_selection=True,Tw=5,rho=.7)
    path=str(tmp_path/'global.pt');cfg.checkpoint_path=path
    save=engine.torch.save
    def interrupt(obj,target):
        save(obj,target)
        if obj['round']==5:
            import os
            os.replace(target,path);raise InterruptedError('simulated interruption')
    utils.set_seed(42);cl,te,idx=factory();monkeypatch.setattr(engine.torch,'save',interrupt)
    with pytest.raises(InterruptedError):engine.train(cfg,cl,te,6,3,False,idx,logging.getLogger('test'))
    monkeypatch.setattr(engine.torch,'save',save);cl,te,idx=factory()
    resumed,rc=engine.train(cfg,cl,te,6,3,False,idx,logging.getLogger('test'))
    del cfg.checkpoint_path;utils.set_seed(42);cl,te,idx=factory()
    fresh,fc=engine.train(cfg,cl,te,6,3,False,idx,logging.getLogger('test'))
    assert resumed==fresh and rc.total()==fc.total()


def test_nonfinite_model_is_not_an_accuracy_endpoint():
    from flcore.engine import _evaluate
    _,test,_=_synthetic_loaders()
    model=MLP(6,8,3)
    with torch.no_grad():next(model.parameters()).fill_(float('nan'))
    with pytest.raises(FloatingPointError):_evaluate(model,test,torch.device('cpu'))
