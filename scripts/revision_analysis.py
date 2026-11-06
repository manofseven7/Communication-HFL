"""Strict five-seed analysis; no imputation, trajectory interpolation or hand-set endpoints."""
import argparse,json,csv
from pathlib import Path
import numpy as np
from scipy import stats

def holm(ps):
    order=np.argsort(ps); out=np.empty(len(ps)); running=0.
    for j,i in enumerate(order):
        running=max(running,(len(ps)-j)*ps[i]);out[i]=min(1.,running)
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',default='revision_results');ap.add_argument('--out',default='revision_analysis');a=ap.parse_args()
    root,out=Path(a.results),Path(a.out);out.mkdir(parents=True,exist_ok=True)
    expected=[42,43,44,45,46]
    def load(ds,m,tag=''):
        suffix='_'+tag if tag else ''; records=[]
        for seed in expected:
            p=root/f'{ds}_{m}{suffix}_s{seed}.json'
            if not p.exists():raise FileNotFoundError(f'Missing seed {seed}: {p}')
            rec=json.loads(p.read_text()); cfg=rec['cfg']
            if cfg['seed']!=seed or cfg['dataset']!=ds or cfg['method']!=m:raise ValueError('Metadata mismatch '+str(p))
            if len(rec['traj'])!=(400 if ds=='cifar' else 30):raise ValueError('Incomplete trajectory '+str(p))
            records.append(rec)
        return records
    # Reviewer-requested six-contrast verification family. This is not labelled
    # as a prospectively registered family or as evidence of equivalence.
    rows=[]
    for ds,m,tag,baseline in [('fashion','proposed','nofs','fedavg'),('fashion','proposed','nofs','powersgd'),('har','proposed','nofs','fedavg'),('har','proposed','nofs','powersgd'),('har','proposed','fs','fedavg'),('har','proposed','fs','powersgd')]:
        x,y=load(ds,m,tag),load(ds,baseline)
        d=np.array([u['final_acc']-v['final_acc'] for u,v in zip(x,y)])
        half=stats.t.ppf(.975,4)*d.std(ddof=1)/np.sqrt(5)
        rows.append(dict(dataset=ds,variant=tag,baseline=baseline,n=5,difference_pp=float(d.mean()),ci_low=float(d.mean()-half),ci_high=float(d.mean()+half),p=float(stats.ttest_1samp(d,0).pvalue)))
    for r,p in zip(rows,holm([r['p'] for r in rows])):r['holm_p']=float(p)
    (out/'paired_six_contrasts.json').write_text(json.dumps(rows,indent=2))
    x,y=load('har','proposed','fs'),load('har','proposed','nofs')
    # Warm-up validation occurs before making the reviewer-requested figure.
    for seed,u,v in zip(expected,x,y):
        for j in range(5):
            for metric in ['acc','f1']:
                if u['traj'][j][metric]!=v['traj'][j][metric]:raise ValueError(f'Warm-up mismatch seed={seed},round={j+1},metric={metric}')
        if any(t.get('mask_active') is not False for t in u['traj'][:5]):raise ValueError('Missing/incorrect mask timing provenance')
        if u['traj'][5].get('mask_active') is not True:raise ValueError('Mask not active in round 6')
    d=np.array([u['final_acc']-v['final_acc'] for u,v in zip(x,y)]); half=stats.t.ppf(.975,4)*d.std(ddof=1)/np.sqrt(5)
    component=dict(comparison='HAR masked minus unmasked',n=5,difference_pp=float(d.mean()),ci_low=float(d.mean()-half),ci_high=float(d.mean()+half),p=float(stats.ttest_1samp(d,0).pvalue),family='Separate component question; no equivalence claim')
    (out/'har_component.json').write_text(json.dumps(component,indent=2))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8,'axes.labelsize':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.0),layout='constrained')
    for records,label,style,color in [(y,'Without feature mask','--','#D55E00'),(x,'Shared feature mask','-','#111111')]:
        acc=np.array([[t['acc'] for t in r['traj']] for r in records]); comm=np.array([[t['comm_mb'] for t in r['traj']] for r in records]).mean(0)
        mean,sd=acc.mean(0),acc.std(0,ddof=1)
        for ax,xx in zip(axes,[np.arange(1,31),comm]):
            ax.plot(xx,mean,style,color=color,label=label,lw=2);ax.fill_between(xx,mean-sd,mean+sd,color=color,alpha=.12)
            ax.set_ylabel('Test accuracy (%)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].axvspan(.5,5.5,color='#888888',alpha=.08);axes[0].axvline(5.5,color='#777777',ls=':',lw=1);axes[0].text(.03,.97,'Mask active from round 6',transform=axes[0].transAxes,va='top',fontsize=8);axes[0].set_xlabel('Communication round');axes[0].set_xlim(1,30)
    axes[1].set_xlabel('Cumulative three-link payload (MiB)')
    fig.savefig(out/'Figure3_HAR_verified.pdf');fig.savefig(out/'Figure3_HAR_verified.png',dpi=180);plt.close(fig)
    print(json.dumps(rows,indent=2));print(json.dumps(component,indent=2))
if __name__=='__main__':main()
