"""Reproduce descriptive CIFAR audit; no training and no post-hoc hypothesis tests.
Run with Python + numpy + matplotlib. Outputs remain beside this script.
"""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'analysis_outputs'/'cifar'; OUT.mkdir(parents=True,exist_ok=True)
SEEDS=[42,43,44]; METHODS=['fedavg','powersgd','qsgd','proposed']
NAMES={'fedavg':'FedAvg','powersgd':'PowerSGD-EF','qsgd':'QSGD-8','proposed':'Proposed'}
def load(p): return json.loads(p.read_text())
def ms(x): return float(np.mean(x)),float(np.std(x,ddof=1))
def writecsv(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
data={}; rows=[]; per=[]; hashes={}
for m in METHODS:
 data[m]=[]
 for seed in SEEDS:
  p=ROOT/'results'/f'cifar_{m}_s{seed}.json';d=load(p)
  hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
  assert [t['round'] for t in d['traj']]==list(range(1,401))
  assert d['cfg']['seed']==seed and d['cfg']['rounds']==400
  assert not d['cfg']['feature_selection']
  assert d['final_acc']==d['traj'][-1]['acc'] and d['comm_mb']==d['traj'][-1]['comm_mb']
  failed=m=='fedavg' and seed==43
  data[m].append(d)
  per.append(dict(method=m,seed=seed,recorded_acc=d['final_acc'],recorded_f1=d['final_f1'],payload_mib=d['comm_mb'],status='confirmed_nonfinite_model' if failed else 'completed_no_recorded_numerical_failure'))
 valid=[d for d in data[m] if not(m=='fedavg' and d['cfg']['seed']==43)]
 am,asd=ms([d['final_acc'] for d in valid]);fm,fsd=ms([d['final_f1'] for d in valid])
 rows.append(dict(method=m,n_attempted=3,n_valid_endpoints=len(valid),failures=3-len(valid),accuracy_mean=am,accuracy_sd=asd,f1_mean=fm,f1_sd=fsd,payload_mib=float(np.mean([d['comm_mb'] for d in data[m]]))))
base=rows[0]['payload_mib']; prop=rows[-1]
for r in rows:r['payload_reduction_vs_fedavg_pct']=100*(1-r['payload_mib']/base)
writecsv('cifar400_summary.csv',rows);writecsv('cifar400_per_seed.csv',per)
abl={};arows=[];apairs=[];audits=[]
for variant,prefix in [('rank10','baseline'),('rank20','rank20'),('dense_down10','dense_down10')]:
 ds=[]
 for seed in SEEDS:
  p=ROOT/'diagnostics50'/f'{prefix}_s{seed}.json';d=load(p);hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
  assert [t['round'] for t in d['traj']]==list(range(1,51))
  if variant=='rank10':
   original=ROOT/'results'/f'cifar_proposed_s{seed}.json'
   assert hashlib.sha256(original.read_bytes()).hexdigest()==d['original_sha256']
   assert d['traj']==load(original)['traj'][:50]
  else:
   assert d['cfg']['rank']==(20 if variant=='rank20' else 10)
   other=load(ROOT/'diagnostics50'/f"{'dense_down10' if variant=='rank20' else 'rank20'}_s{seed}.json")
   assert d['partition_sha256']==other['partition_sha256']
   for a in d['reconstruction_audit']:
    audits.append(dict(variant=variant,seed=seed,round=a['round'],stage=a['stage'],relative_squared_error=a['relative_squared_error'],payload_bytes=a['bytes']))
  ds.append(d)
 abl[variant]=ds
 am,asd=ms([d['traj'][-1]['acc'] for d in ds]);fm,fsd=ms([d['traj'][-1]['f1'] for d in ds]);cm,csd=ms([d['traj'][-1]['comm_mb'] for d in ds])
 arows.append(dict(variant=variant,n=3,rounds=50,accuracy_mean=am,accuracy_sd=asd,f1_mean=fm,f1_sd=fsd,payload_mib=cm))
for v in ['rank20','dense_down10']:
 delta=[d['traj'][-1]['acc']-b['traj'][-1]['acc'] for d,b in zip(abl[v],abl['rank10'])]
 apairs.append(dict(variant=v,paired_accuracy_differences_pp=delta,mean_pp=ms(delta)[0],sample_sd_pp=ms(delta)[1],payload_multiplier=next(r for r in arows if r['variant']==v)['payload_mib']/arows[0]['payload_mib']))
writecsv('diagnostic50_summary.csv',arows);writecsv('sampled_reconstruction_errors.csv',audits)
report=dict(cifar400=rows,diagnostic50=arows,diagnostic_differences=apairs,proposed_accuracy_gap_vs_powersgd_pp=prop['accuracy_mean']-rows[1]['accuracy_mean'],proposed_accuracy_gap_vs_qsgd_pp=prop['accuracy_mean']-rows[2]['accuracy_mean'],proposed_payload_saving_vs_powersgd_pct=100*(1-prop['payload_mib']/rows[1]['payload_mib']),proposed_payload_saving_vs_qsgd_pct=100*(1-prop['payload_mib']/rows[2]['payload_mib']),fedavg_failure='seed43; nonfinite checkpoint verified separately; raw 10% is invalid model accuracy; valid-only n=2 summary is unbalanced and descriptive',inference='Descriptive, no post-hoc significance/equivalence claims. Reused rank10 prefixes, CUDA nondeterminism, three seeds, 50 rounds. Not a 400-round rank20 result.',sha256=hashes)
(OUT/'verified_cifar_interpretation.json').write_text(json.dumps(report,indent=2)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
colors=['#6b7280','#d97706','#2563eb','#059669']
fig,axs=plt.subplots(1,3,figsize=(11.2,3.6),gridspec_kw={'width_ratios':[1.35,1,1]})
for m,c in zip(METHODS,colors):
 valid=[d for d in data[m] if not(m=='fedavg' and d['cfg']['seed']==43)]
 y=np.array([[t['acc'] for t in d['traj']] for d in valid]);x=np.arange(1,401)
 label=NAMES[m]+(' (valid n=2)' if m=='fedavg' else ' (n=3)')
 axs[0].plot(x,y.mean(0),label=label,color=c,lw=1.5);axs[0].fill_between(x,y.mean(0)-y.std(0,ddof=1),y.mean(0)+y.std(0,ddof=1),color=c,alpha=.13)
axs[0].set(xlabel='Communication round',ylabel='Test accuracy (%)',ylim=(0,75),title='400-round trajectories')
axs[0].legend(fontsize=7,loc='lower right')
labels=['FedAvg\nvalid n=2','PowerSGD-EF\nn=3','QSGD-8\nn=3','Proposed\nn=3']
x=np.arange(4)
axs[1].errorbar(x,[r['accuracy_mean'] for r in rows],yerr=[r['accuracy_sd'] for r in rows],fmt='none',ecolor='#111827',capsize=4)
axs[1].scatter(x,[r['accuracy_mean'] for r in rows],c=colors,s=35,zorder=3)
axs[1].set_xticks(x,labels,rotation=35,ha='right',fontsize=7);axs[1].set(ylabel='Test accuracy (%)',ylim=(58,70),title='Round 400: mean ± sample SD')
for i,r in enumerate(rows):axs[1].annotate(f"{r['accuracy_mean']:.2f}",(i,r['accuracy_mean']+.6),ha='center',fontsize=8)
axs[2].bar(x,[r['payload_mib']/1024 for r in rows],color=colors,width=.65)
axs[2].set_xticks(x,labels,rotation=35,ha='right',fontsize=7);axs[2].set(ylabel='Cumulative modeled payload (GiB)',title='All three communication links',ylim=(0,365))
for i,r in enumerate(rows):axs[2].text(i,r['payload_mib']/1024+5,f"{r['payload_mib']/1024:.2f}",ha='center',fontsize=8)
fig.tight_layout();fig.savefig(OUT/'cifar400_verified.pdf',bbox_inches='tight');fig.savefig(OUT/'cifar400_verified.png',dpi=160,bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(8.5,3.3))
for v,c,l in [('rank10','#059669','Rank 10 (reused prefix)'),('rank20','#2563eb','Rank 20'),('dense_down10','#d97706','Rank 10, dense downlink')]:
 y=np.array([[t['acc'] for t in d['traj']] for d in abl[v]]);x=np.arange(1,51)
 axs[0].plot(x,y.mean(0),label=l,color=c);axs[0].fill_between(x,y.mean(0)-y.std(0,ddof=1),y.mean(0)+y.std(0,ddof=1),color=c,alpha=.12)
axs[0].set(xlabel='Communication round',ylabel='Test accuracy (%)',title='Exploratory 50-round diagnostics');axs[0].legend(fontsize=7)
axs[1].bar(np.arange(3),[r['payload_mib']/1024 for r in arows],color=['#059669','#2563eb','#d97706'])
axs[1].set_xticks(np.arange(3),['Rank 10','Rank 20','Dense downlink'],rotation=15,fontsize=8);axs[1].set(ylabel='Cumulative modeled payload (GiB)',title='Same horizon: 50 rounds')
fig.tight_layout();fig.savefig(OUT/'diagnostic50_verified.pdf',bbox_inches='tight');fig.savefig(OUT/'diagnostic50_verified.png',dpi=160,bbox_inches='tight');plt.close(fig)
print(json.dumps({k:v for k,v in report.items() if k!='sha256'},indent=2))
