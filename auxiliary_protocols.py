import os,sys,pathlib,json,subprocess
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
r=pathlib.Path(__file__).parent;os.chdir(r)
import numpy as np,torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
from configs.presets import preset
from flcore.data import load_har,_dirichlet_split
from flcore.engine import train
from flcore.utils import set_seed,get_logger,save_json
torch.set_num_threads(1)
(X,y),_,d,classes=load_har();log=get_logger('validation')
tr,va=train_test_split(np.arange(len(y)),test_size=.15,random_state=42,stratify=y.numpy())
for seed in [42,43,44]:
    parts=_dirichlet_split(y[tr].numpy(),50,.1,np.random.default_rng(seed))
    def loader(i):
        ds=TensorDataset(X[tr][parts[i]],y[tr][parts[i]])
        return DataLoader(ds,batch_size=64,shuffle=True),len(ds)
    vl=DataLoader(TensorDataset(X[va],y[va]),batch_size=256,shuffle=False)
    for rho in [.9,.8,.7]:
        p=pathlib.Path(f'revision_results/fs_validation/rho{rho:.2f}_s{seed}.json')
        if p.exists():continue
        cfg=preset('har',fs=True);cfg.method='proposed';cfg.seed=seed;cfg.rho=rho
        set_seed(seed);os.environ['HFL_CHECKPOINT_SCOPE']=f'validation85_rho{rho:.2f}'
        traj,comm=train(cfg,loader,vl,d,classes,False,parts,log)
        save_json(str(p),dict(cfg=cfg.__dict__,split='fixed-stratified-training-85/15',traj=traj,final_acc=traj[-1]['acc'],final_f1=traj[-1]['f1'],comm_mb=comm.mb()))
        print('VALIDATION_DONE',rho,seed,flush=True)
for ds in ['har','fashion']:
    for method in ['fedavg','fednova','powersgd','proposed']:
        for seed in [42,43,44]:
            p=pathlib.Path(f'revision_results/profile/{ds}_{method}_profile_s{seed}.json')
            if p.exists():continue
            env=os.environ.copy();env['HFL_CHECKPOINT_SCOPE']='profile'
            subprocess.run([sys.executable,'run.py','--dataset',ds,'--method',method,'--seed',str(seed),'--rounds','1','--local-epochs','5','--tag','profile','--fs','off','--out','revision_results/profile'],env=env,check=True)
for n in [50,100,250,500]:
    for seed in [42,43,44]:
        p=pathlib.Path(f'revision_results/scalability/har_proposed_scale_n{n}_s{seed}.json')
        if p.exists():continue
        env=os.environ.copy();env['HFL_CHECKPOINT_SCOPE']=f'scale{n}'
        subprocess.run([sys.executable,'run.py','--dataset','har','--method','proposed','--seed',str(seed),'--clients',str(n),'--edges',str(n//10),'--rounds','1','--local-epochs','1','--alpha','1.0','--tag',f'scale_n{n}','--fs','off','--out','revision_results/scalability'],env=env,check=True)
print('ALL_AUXILIARY_COMPLETE',flush=True)
