"""Explicit, sequential rerun of the declared protocol; completed outputs are skipped."""
import argparse, subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--group',choices=['tabular','cifar','all'],required=True);p.add_argument('--out',default='new_results');a=p.parse_args()
sys.path.insert(0,str(root))
from configs.presets import METHODS
jobs=[]
if a.group in ['tabular','all']:
 for ds in ['fashion','har']:
  for seed in range(42,47):
   for method in METHODS:
    jobs.append((ds,method,seed,'fs' if ds=='har' and method=='proposed' else 'nofs' if method=='proposed' else ''))
   if ds=='har':jobs.append((ds,'proposed',seed,'nofs'))
if a.group in ['cifar','all']:
 for seed in [42,43,44]:
  for method in ['fedavg','powersgd','qsgd','proposed']:jobs.append(('cifar',method,seed,''))
for ds,method,seed,tag in jobs:
 cmd=[sys.executable,str(root/'run.py'),'--dataset',ds,'--method',method,'--seed',str(seed),'--out',a.out]
 if tag:cmd+=['--tag',tag,'--fs','on' if tag=='fs' else 'off']
 subprocess.run(cmd,cwd=root,check=True)
