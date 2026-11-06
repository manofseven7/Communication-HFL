"""Training-only HAR retention sweep used to select rho (seeds 42--44).

The official test set is never loaded by this script. Fifteen percent of the
official training partition is held out with stratification and used only for
configuration selection.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from configs.presets import preset
from flcore.data import load_har, _dirichlet_split
from flcore.engine import train
from flcore.utils import get_logger, save_json, set_seed

SEEDS=(42,43,44); RHOS=(0.90,0.80,0.70); OUT="results/fs_validation"
os.makedirs(OUT,exist_ok=True); log=get_logger("fs-validation")
torch.set_num_threads(1)
X=torch.tensor(np.loadtxt('data/HAR/train/X_train.txt'),dtype=torch.float32)
y=torch.tensor(np.loadtxt('data/HAR/train/y_train.txt').astype(np.int64)-1)
d=X.shape[1];classes=int(y.max())+1

for seed in SEEDS:
    tr_idx,val_idx=train_test_split(np.arange(len(y)),test_size=.15,random_state=42,
                                    stratify=y.numpy())
    Xtr,ytr=X[tr_idx],y[tr_idx]; Xval,yval=X[val_idx],y[val_idx]
    parts=_dirichlet_split(ytr.numpy(),50,.1,np.random.default_rng(seed))
    def client_loader(i):
        ds=TensorDataset(Xtr[parts[i]],ytr[parts[i]])
        return DataLoader(ds,batch_size=64,shuffle=True),len(ds)
    val_loader=DataLoader(TensorDataset(Xval,yval),batch_size=256,shuffle=False)
    for rho in RHOS:
        cfg=preset("har",fs=True); cfg.method="proposed"; cfg.seed=seed; cfg.rho=rho
        set_seed(seed)
        traj,comm=train(cfg,client_loader,val_loader,d,classes,False,parts,log)
        save_json(f"{OUT}/rho{rho:.2f}_s{seed}.json",
                  dict(cfg=cfg.__dict__,split="fixed-stratified-training-85/15",
                       traj=traj,final_acc=traj[-1]["acc"],final_f1=traj[-1]["f1"],
                       comm_mb=comm.mb()))

print("rho mean_validation_accuracy mean_validation_macro_f1")
for rho in RHOS:
    recs=[json.load(open(f"{OUT}/rho{rho:.2f}_s{s}.json")) for s in SEEDS]
    print(f"{rho:.2f}",f"{np.mean([r['final_acc'] for r in recs]):.3f}",
          f"{np.mean([r['final_f1'] for r in recs]):.3f}")
