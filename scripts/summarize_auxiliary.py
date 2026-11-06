"""Summarise complete auxiliary groups; never infer SD from missing repetitions."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('mode',choices=['profile','scalability']);p.add_argument('--root');a=p.parse_args()
root=Path(a.root or f'results/{a.mode}');groups={};expected_seeds=[42,43,44]
expected=({(d,m) for d in ['har','fashion'] for m in ['fedavg','fednova','powersgd','proposed']} if a.mode=='profile' else {(n,n//10) for n in [50,100,250,500]})
for path in sorted(root.glob('*.json')):
    r=json.loads(path.read_text());c=r['cfg'];key=(c['dataset'],c['method']) if a.mode=='profile' else (c['clients'],c['edges'])
    if len(r.get('traj',[]))!=1 or c['rounds']!=1:raise ValueError('Expected one complete round: '+str(path))
    checks=dict(local_epochs=5 if a.mode=='profile' else 1,rank=10,batch=64,lr=.05,alpha=.1 if a.mode=='profile' else 1.,cuda=False,feature_selection=False)
    if a.mode=='profile':checks.update(clients=50,edges=5)
    else:checks.update(dataset='har',method='proposed')
    for k,v in checks.items():
        if c[k]!=v:raise ValueError(f'{path}: {k} differs from declared auxiliary protocol')
    groups.setdefault(key,[]).append(r)
if set(groups)!=expected:raise ValueError(f'Incomplete or unexpected groups: expected {sorted(expected)}, found {sorted(groups)}')
print('dataset method n time_s_mean time_s_sd peak_rss_mib comm_mib' if a.mode=='profile' else 'clients edges n time_s_mean time_s_sd peak_rss_mib comm_mib')
for key,recs in sorted(groups.items()):
    if sorted(r['cfg']['seed'] for r in recs)!=expected_seeds:raise ValueError(f'Expected exactly seeds 42--44 for {key}')
    values=np.array([[r['wall_time_s'],r['peak_rss_mb'],r['comm_mb']] for r in recs],float)
    if not np.isfinite(values).all() or (values<=0).any():raise ValueError(f'Invalid measurements for {key}')
    print(*key,3,f'{values[:,0].mean():.2f}',f'{values[:,0].std(ddof=1):.2f}',f'{values[:,1].mean():.2f}',f'{values[:,2].mean():.2f}')
