"""Create publication tables/figures only from complete matched seed-level runs.

No missing seeds are dropped, no endpoints are supplied by hand, and no learning
curve is smoothed or interpolated. Run after revision_analysis.py for Figure 3.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np

ORDER=['fedavg','fedprox','scaffold','fednova','fedkd','powersgd','qsgd','fedpaq','fedcom','topk','adaptive_topk','sign','proposed']
LABEL=dict(zip(ORDER,['FedAvg','FedProx','SCAFFOLD','FedNova','FedKD-style','PowerSGD-EF','QSGD-8','FedPAQ-8','FedCOM-8','Top-k-EF','Adaptive-TopK-EF','SignSGD-MV','Proposed']))
COLORS=['#4477AA','#77AADD','#CC3311','#228833','#AA3377','#CC9900','#0077BB','#EE7733','#888888','#009988','#44AA99','#BB99BB','#111111']
STYLE={m:dict(color=c,linestyle=['-','--','-.',':'][i%4],lw=2.4 if m=='proposed' else 1.3) for i,(m,c) in enumerate(zip(ORDER,COLORS))}
STYLE['proposed']['linestyle']='-'

def load(root,ds,method,tag=None):
    if tag is None:tag=('fs' if ds=='har' else 'nofs') if method=='proposed' and ds!='cifar' else ''
    suffix='_'+tag if tag else ''; records=[]
    for seed in range(42,47):
        f=root/f'{ds}_{method}{suffix}_s{seed}.json'
        r=json.loads(f.read_text()); c=r['cfg'];tr=r['traj'];rounds=400 if ds=='cifar' else 30
        expected=dict(dataset=ds,method=method,seed=seed,rounds=rounds,clients=100 if ds=='cifar' else 50,edges=5,local_epochs=5,batch=64,rank=10,alpha=.1,lr=.05,cuda=(ds=='cifar'))
        for key,val in expected.items():
            if c[key]!=val:raise ValueError(f'{f}: {key}={c[key]}, expected {val}')
        if len(tr)!=rounds or [v['round'] for v in tr]!=list(range(1,rounds+1)):raise ValueError('Incomplete trajectory '+str(f))
        if r['final_acc']!=tr[-1]['acc'] or r['final_f1']!=tr[-1]['f1'] or r['comm_mb']!=tr[-1]['comm_mb']:raise ValueError('Endpoint mismatch '+str(f))
        if not np.isclose(sum(r[k] for k in ['c2e','e2c','down','setup']),r['comm_mb'],rtol=0,atol=1e-8):raise ValueError('Communication component sum mismatch '+str(f))
        values=np.array([[t[k] for k in ['acc','f1','comm_mb']] for t in tr])
        if not np.isfinite(values).all() or (values[:,2]<0).any() or (np.diff(values[:,2])<0).any():raise ValueError('Invalid trajectory '+str(f))
        if ds=='har' and method=='proposed':
            active=tag=='fs'
            if c['feature_selection']!=active:raise ValueError('Feature-selection label mismatch '+str(f))
            if [v.get('mask_active') for v in tr]!=([False]*5+[True]*25 if active else [False]*30):raise ValueError('Mask timing mismatch '+str(f))
        records.append(r)
    return records

def summary(ds,method,recs,tag=''):
    row=dict(dataset=ds,method=method,label=LABEL[method],variant=tag,n=5)
    for key in ['final_acc','final_f1','comm_mb']:
        vals=np.array([r[key] for r in recs]);row[key+'_mean']=float(vals.mean());row[key+'_sd']=float(vals.std(ddof=1))
    return row

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',default='revision_results');ap.add_argument('--out',default='revision_outputs');ap.add_argument('--stage',choices=['tabular','cifar','all'],default='tabular');args=ap.parse_args()
    root,out=Path(args.results),Path(args.out)
    # Load and validate an entire requested group before writing any outputs.
    groups={}
    if args.stage in ['tabular','all']:
        for ds in ['fashion','har']:
            for m in ORDER:groups[ds,m]=load(root,ds,m)
        nofs=load(root,'har','proposed','nofs')
        for a,b in zip(groups['har','proposed'],nofs):
            if any(a['traj'][i][k]!=b['traj'][i][k] for i in range(5) for k in ['acc','f1']):raise ValueError('Warm-up trajectories differ')
    if args.stage in ['cifar','all']:
        raise SystemExit('For the declared three-seed CIFAR group use: python scripts/analyze_cifar.py')
    if False:
        for m in ['fedavg','powersgd','qsgd','proposed']:groups['cifar',m]=load(root,'cifar',m)
    out.mkdir(parents=True,exist_ok=True)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8,'axes.labelsize':9,'axes.titlesize':10,'pdf.fonttype':42,'ps.fonttype':42})
    rows=[summary(ds,m,r,'fs' if ds=='har' and m=='proposed' else 'nofs' if ds=='fashion' and m=='proposed' else '') for (ds,m),r in groups.items()]
    if args.stage in ['tabular','all']:
        rows.append(summary('har','proposed',nofs,'nofs'))
        tabular=[r for r in rows if r['dataset']!='cifar']
        write_csv(out/'tabular_five_seed_summary.csv',tabular)
        (out/'tabular_five_seed_summary.json').write_text(json.dumps(tabular,indent=2))
        tex=[]
        for ds in ['fashion','har']:
            rr=[r for r in tabular if r['dataset']==ds];bestacc=max(r['final_acc_mean'] for r in rr);bestf1=max(r['final_f1_mean'] for r in rr);bestcomm=min(r['comm_mb_mean'] for r in rr)
            for i,r in enumerate(rr):
                label=r['label']
                if r['method']=='proposed':label+= ' (shared mask)' if r['variant']=='fs' else ' (unmasked)'
                vals=[]
                for metric,best in [('final_acc',bestacc),('final_f1',bestf1)]:
                    val=f"{r[metric+'_mean']:.2f}\\pm{r[metric+'_sd']:.2f}"
                    vals.append('$'+('\\mathbf{'+val+'}' if r[metric+'_mean']==best else val)+'$')
                comm=f"{r['comm_mb_mean']:.2f}";comm='\\textbf{'+comm+'}' if r['comm_mb_mean']==bestcomm else comm
                tex.append(('Fashion-MNIST' if ds=='fashion' else 'UCI HAR') if i==0 else '')
                tex[-1]+=' & '+label+' & '+' & '.join(vals)+' & '+comm+r'\\'
            if ds=='fashion':tex.append(r'\midrule')
        (out/'Table1_body.tex').write_text('\n'.join(tex)+'\n')
        fig,axes=plt.subplots(2,2,figsize=(7.2,6.0))
        for col,ds in enumerate(['fashion','har']):
            for row,metric in enumerate(['acc','f1']):
                ax=axes[row,col]
                for m in ORDER:
                    rr=groups[ds,m];x=np.mean([[t['comm_mb'] for t in r['traj']] for r in rr],axis=0);v=np.array([[t[metric] for t in r['traj']] for r in rr]);mean=v.mean(0);sd=v.std(0,ddof=1)
                    ax.plot(x,mean,label=LABEL[m],**STYLE[m]);ax.fill_between(x,mean-sd,mean+sd,color=STYLE[m]['color'],alpha=.10 if m=='proposed' else .045)
                ax.set_xscale('log');ax.grid(alpha=.18);ax.set_xlabel('Cumulative three-link payload (MiB)');ax.set_ylabel('Test accuracy (%)' if metric=='acc' else 'Macro-F1 (%)')
                if row==0:ax.set_title('Fashion-MNIST' if ds=='fashion' else 'UCI HAR')
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='lower center',fontsize=7.5,frameon=False)
        fig.tight_layout(rect=(0,.17,1,1));fig.savefig(out/'all_13_methods_performance_composite.pdf');fig.savefig(out/'all_13_methods_performance_composite.png',dpi=180);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(7.2,4.9))
        for ax,ds in zip(axes,['fashion','har']):
            vals=[next(r['comm_mb_mean'] for r in rows if r['dataset']==ds and r['method']==m and not (ds=='har' and r['variant']=='nofs')) for m in ORDER]
            ax.barh(np.arange(13),vals,color=COLORS);ax.set_xscale('log');ax.set_yticks(np.arange(13),[LABEL[m] for m in ORDER],fontsize=8);ax.invert_yaxis();ax.set_xlabel('Cumulative three-link payload (MiB)');ax.set_title('Fashion-MNIST' if ds=='fashion' else 'UCI HAR');ax.grid(axis='x',alpha=.18)
        fig.tight_layout();fig.savefig(out/'all_13_methods_final_communication.pdf');fig.savefig(out/'all_13_methods_final_communication.png',dpi=180);plt.close(fig)
    if args.stage in ['cifar','all']:
        raise SystemExit('For the declared three-seed CIFAR group use: python scripts/analyze_cifar.py')
    if False:
        rr=[r for r in rows if r['dataset']=='cifar'];write_csv(out/'cifar_five_seed_summary.csv',rr);(out/'cifar_five_seed_summary.json').write_text(json.dumps(rr,indent=2))
        methods=['fedavg','powersgd','qsgd','proposed'];fig,axes=plt.subplots(1,2,figsize=(7.2,3.2));x=np.arange(4)
        axes[0].bar(x,[r['final_acc_mean'] for r in rr],yerr=[r['final_acc_sd'] for r in rr],color=[STYLE[m]['color'] for m in methods],capsize=4);axes[0].set_ylabel('Test accuracy after 400 rounds (%)')
        axes[1].bar(x,[r['comm_mb_mean'] for r in rr],color=[STYLE[m]['color'] for m in methods]);axes[1].set_yscale('log');axes[1].set_ylabel('Cumulative three-link payload (MiB)')
        for ax in axes:ax.set_xticks(x,[LABEL[m] for m in methods],rotation=25,ha='right');ax.grid(axis='y',alpha=.18)
        fig.tight_layout();fig.savefig(out/'cifar10_four_methods.pdf');fig.savefig(out/'cifar10_four_methods.png',dpi=180);plt.close(fig)
    print(json.dumps({'stage':args.stage,'complete_seed_runs':sum(len(r) for r in groups.values())+(5 if args.stage in ['tabular','all'] else 0),'summary_rows':len(rows),'output':str(out)},indent=2))
if __name__=='__main__':main()
