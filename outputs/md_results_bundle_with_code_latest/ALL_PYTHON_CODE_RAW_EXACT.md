# ALL PYTHON CODE (RAW EXACT)

Total files: 4

## 1. `scripts/rf_mcl_gmm_v4_router.py`

```python
    give me all the results#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RF-MCL GMM v4: train real experts, then train a deployable router q(k|x_t).

Outputs in --outdir:
  calibration.json, metrics_all.csv, router_final_metrics.csv, SUMMARY.txt,
  checkpoint_<variant>.pt.

Key metrics:
  oracle_best_mse   : min_k cost, uses y at test, not deployable.
  soft_oracle_mse   : MCL responsibilities q*(k|x_t,y), uses y, not deployable.
  router_mix_mse    : deployable mixture sum_k q_theta(k|x_t) f_k(x_t).
  bayes_router_mse  : GMM-only exact p(class|x_t), mapped to learned experts.
"""
from __future__ import annotations
import argparse, csv, itertools, json, math, random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pandas as pd
except Exception:
    pd = None
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def device_of(s:str):
    return torch.device('cuda' if s == 'auto' and torch.cuda.is_available() else ('cpu' if s == 'auto' else s))

def ensure_dir(p):
    p=Path(p); p.mkdir(parents=True, exist_ok=True); return p

def norm_entropy(probs:torch.Tensor, eps=1e-12)->float:
    K=probs.numel(); p=probs.clamp_min(eps); return float((-(p*p.log()).sum()/math.log(K)).cpu())

def mi_norm(assign:torch.Tensor, labels:torch.Tensor, K:int, C:int, eps=1e-12)->float:
    a=assign.detach().cpu().long(); y=labels.detach().cpu().long(); n=max(1, y.numel())
    joint=torch.zeros(C,K,dtype=torch.float64)
    for c in range(C):
        for k in range(K): joint[c,k]=((y==c)&(a==k)).sum().item()
    joint/=n; pc=joint.sum(1,keepdim=True); pk=joint.sum(0,keepdim=True); den=pc@pk
    m=joint>0
    mi=(joint[m]*(joint[m]/den[m]).log()).sum()
    hc=-(pc[pc>0]*pc[pc>0].log()).sum(); hk=-(pk[pk>0]*pk[pk>0].log()).sum()
    return float((mi/torch.minimum(hc,hk).clamp_min(eps)).item())

def purity(assign:torch.Tensor, labels:torch.Tensor, K:int, C:int)->float:
    a=assign.detach().cpu().long(); y=labels.detach().cpu().long(); n=max(1,y.numel()); total=0
    for k in range(K):
        m=a==k
        if m.sum(): total += int(torch.bincount(y[m], minlength=C).max())
    return total/n - 1.0/C

def best_perm(score:np.ndarray)->List[int]:
    C,K=score.shape; best=None; val=-1e100
    for p in itertools.permutations(range(K), min(C,K)):
        s=sum(score[c,p[c]] for c in range(len(p)))
        if s>val: val=s; best=list(p)
    return best if best is not None else list(range(min(C,K)))

# ---------------- data ----------------

def simplex_means(C:int,d:int,mu:float,device)->torch.Tensor:
    if C==2:
        M=torch.zeros(C,d,device=device); M[0,0]=-mu*math.sqrt(d); M[1,0]=mu*math.sqrt(d); return M
    G=torch.randn(C,d,device=device); G=G-G.mean(0,keepdim=True)
    Q,_=torch.linalg.qr(G.T, mode='reduced')
    M=Q.T[:C]; M=M-M.mean(0,keepdim=True)
    M=M/M.norm(dim=1,keepdim=True).clamp_min(1e-12)*(mu*math.sqrt(d))
    return M

def sample_gmm(n:int, means:torch.Tensor, sigma0:float, t:float, device):
    C,d=means.shape; lab=torch.randint(0,C,(n,),device=device)
    x0=means[lab]+sigma0*torch.randn(n,d,device=device)
    g=math.exp(-t); xt=g*x0+math.sqrt(max(1-g*g,1e-12))*torch.randn_like(x0)
    return xt,x0,lab

def bayes_post(xt:torch.Tensor, means:torch.Tensor, sigma0:float, t:float):
    g=math.exp(-t); st2=g*g*sigma0*sigma0+max(1-g*g,1e-12)
    dist=((xt[:,None,:]-g*means[None,:,:])**2).sum(-1)
    return torch.softmax(-0.5*dist/st2, dim=1)

# ---------------- model ----------------

class RandomFeatures(nn.Module):
    def __init__(self,d:int,p:int,activation='erf',rf_scale=1.0,seed=0):
        super().__init__(); gen=torch.Generator(device='cpu'); gen.manual_seed(seed+12345)
        self.register_buffer('W', torch.randn(p,d,generator=gen)*rf_scale/math.sqrt(d))
        self.register_buffer('b', 2*math.pi*torch.rand(p,generator=gen))
        self.p=p; self.activation=activation
    def forward(self,x):
        z=x@self.W.T.to(x.device)
        if self.activation=='erf': h=torch.erf(z)
        elif self.activation=='tanh': h=torch.tanh(z)
        elif self.activation=='relu': h=F.relu(z)
        elif self.activation=='cos': h=torch.cos(z+self.b.to(x.device))
        elif self.activation=='sin': h=torch.sin(z+self.b.to(x.device))
        else: raise ValueError(self.activation)
        return h/math.sqrt(self.p)

class RFExperts(nn.Module):
    def __init__(self,p:int,dout:int,K:int,init_std:float):
        super().__init__(); self.A=nn.Parameter(init_std*torch.randn(K,p,dout))
    def forward(self,phi): return torch.einsum('bp,kpd->bkd', phi, self.A)

class Router(nn.Module):
    def __init__(self,p:int,K:int,hidden:int=0,dropout:float=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(p,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,K)) if hidden>0 else nn.Linear(p,K)
    def forward(self,phi): return self.net(phi)

def costs(pred,y): return ((pred-y[:,None,:])**2).mean(-1)

def mcl_loss(e,beta:float,balance=0.0,entropy=0.0):
    K=e.shape[1]
    if beta<=1e-12:
        q=torch.full_like(e,1.0/K); loss=e.mean()
    else:
        logits=-beta*e; q=torch.softmax(logits,1)
        loss=-(torch.logsumexp(logits,1)-math.log(K)).mean()/beta
    if balance>0:
        u=q.mean(0); loss=loss+balance*((u-1.0/K)**2).sum()
    if entropy>0:
        ent=-(q.clamp_min(1e-12)*q.clamp_min(1e-12).log()).sum(1).mean()/math.log(K)
        loss=loss-entropy*ent
    return loss,q

def sched(step:int,variant:str,beta_final:float,warm:int,ramp:int):
    if variant=='uniform': return 0.0
    if variant.startswith('fixed') or variant.startswith('grid') or variant=='hard_cold': return beta_final
    if variant=='theory_anneal':
        if step<warm: return 0.0
        u=1.0 if ramp<=0 else min(1.0,max(0.0,(step-warm)/ramp)); u=u*u*(3-2*u)
        return beta_final*u
    raise ValueError(variant)

# ---------------- calibration ----------------

@torch.no_grad()
def ridge_fit(phi,y,ridge):
    n,p=phi.shape; C=(phi.T@phi)/n+ridge*torch.eye(p,device=phi.device); B=(phi.T@y)/n
    return torch.linalg.solve(C,B)

@torch.no_grad()
def whiten(phi,ridge):
    n,p=phi.shape; C=(phi.T@phi)/n+ridge*torch.eye(p,device=phi.device)
    ev,V=torch.linalg.eigh(C); return phi@(V@torch.diag(ev.clamp_min(ridge).rsqrt())@V.T)

@torch.no_grad()
def power_free(S,R,iters):
    if iters<=0: return 0.0
    n,p=S.shape; d=R.shape[1]
    B=torch.randn(p,d,device=S.device); B=B/B.norm().clamp_min(1e-12); lam=torch.tensor(0.,device=S.device)
    for _ in range(iters):
        delta=S@B; a=(R*delta).sum(1,keepdim=True)/math.sqrt(d); M=a*R/math.sqrt(d)
        GB=(S.T@M)/n; lam=(B*GB).sum(); B=GB/GB.norm().clamp_min(1e-12)
    return max(float(lam.cpu()),0.0)

@torch.no_grad()
def lambda_dir(S,R,v):
    n=S.shape[0]; a=(R@v).pow(2); H=(S.T*a[None,:])@S/n
    return max(float(torch.linalg.eigvalsh(H)[-1].cpu()),0.0)

@torch.no_grad()
def class_basis(means):
    M=means-means.mean(0,keepdim=True); U,S,Vh=torch.linalg.svd(M, full_matrices=False)
    rank=int((S>1e-8*S.max().clamp_min(1e-8)).sum())
    return Vh[:rank].T.contiguous()

@torch.no_grad()
def calibrate(rf,x,y,labels,means,ridge,power_iters,beta_mult,glass_safety,trans_samples=12):
    phi=rf(x); A0=ridge_fit(phi,y,ridge); R=y-phi@A0; S=whiten(phi,ridge)
    lf=power_free(S,R,power_iters)
    B=class_basis(means).to(x.device); lc=max([lambda_dir(S,R,B[:,j]) for j in range(B.shape[1])] or [0.0])
    d=y.shape[1]; P=B@B.T if B.numel() else torch.zeros(d,d,device=x.device); vals=[]
    for _ in range(trans_samples):
        v=torch.randn(d,device=x.device); v=v-P@v
        if v.norm()>1e-8: vals.append(lambda_dir(S,R,v/v.norm()))
    lt=max(vals or [0.0])
    beta=lambda l: 0.5/max(l,1e-12)
    E=R.pow(2).mean(1); v_emp=float(E.var(unbiased=False).cpu()); alpha=math.log(max(2,x.shape[0]))/max(1,d)
    bg=math.sqrt(2*alpha/max(v_emp,1e-12)); bc=beta(lc); bt=beta(lt)
    target=min(beta_mult*bc, glass_safety*bg)
    return dict(lambda_free=lf,lambda_class=lc,lambda_trans=lt,beta_free=beta(lf),beta_class=bc,beta_trans=bt,
                alpha_log_n_over_d=alpha,v_emp=v_emp,beta_glass_emp=bg,beta_target=target,
                residual_mse=float(E.mean().cpu()),class_basis_rank=int(B.shape[1]),
                window_class_before_trans=float(bc<bt),window_class_before_glass=float(bc<bg))

# ---------------- eval/router ----------------

@torch.no_grad()
def eval_all(experts,rf,x,y,labels,beta,C,router=None,means=None,sigma0=None,t=None,c2k=None,batch=1024,prefix=''):
    K=experts.A.shape[0]; sums={k:0.0 for k in ['best','soft','mean','rmix','rsoft','bmix','bsoft']}; n=0
    ao=[]; ar=[]; ab=[]; qt=[]; qr=[]; qb=[]
    for i in range(0,x.shape[0],batch):
        xb=x[i:i+batch]; yb=y[i:i+batch]; phi=rf(xb); pred=experts(phi); e=costs(pred,yb); _,q=mcl_loss(e,beta)
        Bsz=xb.shape[0]; n+=Bsz
        sums['best']+=float(e.min(1).values.sum()); sums['soft']+=float((q*e).sum(1).sum()); sums['mean']+=float(e.mean(1).sum())
        ao.append(e.argmin(1).cpu()); qt.append(q.cpu())
        if router is not None:
            rq=torch.softmax(router(phi),1); mix=torch.einsum('bk,bkd->bd',rq,pred)
            sums['rmix']+=float(((mix-yb)**2).mean(1).sum()); sums['rsoft']+=float((rq*e).sum(1).sum())
            ar.append(rq.argmax(1).cpu()); qr.append(rq.cpu())
        if means is not None and c2k is not None:
            pc=bayes_post(xb,means,sigma0,t); bq=torch.zeros(Bsz,K,device=xb.device)
            for c,k in enumerate(c2k):
                if k<K: bq[:,k]+=pc[:,c]
            mix=torch.einsum('bk,bkd->bd',bq,pred)
            sums['bmix']+=float(((mix-yb)**2).mean(1).sum()); sums['bsoft']+=float((bq*e).sum(1).sum())
            ab.append(bq.argmax(1).cpu()); qb.append(bq.cpu())
    out={prefix+'oracle_best_mse':sums['best']/n,prefix+'soft_oracle_mse':sums['soft']/n,prefix+'mean_expert_mse':sums['mean']/n}
    ycpu=labels.cpu()
    def add(name,alist,qlist):
        if not alist: return
        a=torch.cat(alist); q=torch.cat(qlist); u=torch.bincount(a,minlength=K).float(); u=u/u.sum().clamp_min(1)
        out[prefix+name+'_usage_entropy']=norm_entropy(u); out[prefix+name+'_eff_frac_min']=float((u.min()*K).item())
        out[prefix+name+'_class_mi_norm']=mi_norm(a,ycpu,K,C); out[prefix+name+'_class_purity']=purity(a,ycpu,K,C)
        out[prefix+name+'_soft_usage_entropy']=norm_entropy(q.mean(0))
    add('oracle',ao,qt)
    if router is not None:
        out[prefix+'router_mix_mse']=sums['rmix']/n; out[prefix+'router_soft_mse']=sums['rsoft']/n
        Q=torch.cat(qt).clamp_min(1e-12); Rq=torch.cat(qr).clamp_min(1e-12)
        out[prefix+'router_vs_teacher_ce']=float((-(Q*Rq.log()).sum(1).mean()).item())
        out[prefix+'router_vs_teacher_kl']=float((Q*(Q.log()-Rq.log())).sum(1).mean().item())
        add('router',ar,qr)
    if ab:
        out[prefix+'bayes_router_mix_mse']=sums['bmix']/n; out[prefix+'bayes_router_soft_mse']=sums['bsoft']/n; add('bayes',ab,qb)
    return out

@torch.no_grad()
def teacher_q(experts,rf,x,y,beta,batch=1024):
    qs=[]
    for i in range(0,x.shape[0],batch):
        phi=rf(x[i:i+batch]); e=costs(experts(phi),y[i:i+batch]); _,q=mcl_loss(e,beta); qs.append(q.cpu())
    return torch.cat(qs)

def class_to_expert(q,labels,C,K):
    qn=q.numpy(); yn=labels.detach().cpu().numpy(); score=np.zeros((C,K))
    for c in range(C):
        m=yn==c
        if m.sum(): score[c]=qn[m].mean(0)
    return best_perm(score)

def train_router(experts,rf,x,y,beta,params,device):
    router=Router(params.p,params.K,params.router_hidden,params.router_dropout).to(device)
    opt=torch.optim.AdamW(router.parameters(),lr=params.router_lr)
    experts.eval()
    for _ in range(params.router_steps):
        idx=torch.randint(0,x.shape[0],(params.router_batch_size,),device=device); xb=x[idx]; yb=y[idx]
        with torch.no_grad():
            phi=rf(xb); e=costs(experts(phi),yb); _,q=mcl_loss(e,beta)
        loss=-(q*torch.log_softmax(router(phi),1)).sum(1).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return router

# ---------------- main training ----------------

@dataclass
class Params:
    d:int=64; p:int=256; C:int=4; K:int=4; mu:float=1.0; sigma0:float=0.5; t:float=2.05
    n_train:int=6000; n_test:int=3000; n_calib:int=2500; batch_size:int=256; steps:int=4000
    lr:float=3e-3; weight_decay:float=0.0; init_std:float=1e-3; activation:str='erf'; rf_scale:float=1.0
    seed:int=0; warmup_steps:int=800; ramp_steps:int=1000; eval_every:int=200; power_iters:int=30; ridge:float=1e-5
    beta_target_mult:float=3.0; beta_glass_safety:float=0.45; cold_mult_glass:float=1.3; fixed_class_mult:float=1.2
    balance_weight_theory:float=0.0; entropy_weight_theory:float=0.0
    router_steps:int=1200; router_lr:float=2e-3; router_hidden:int=0; router_dropout:float=0.0; router_batch_size:int=256
    beta_grid:bool=False; device:str='auto'; outdir:str='./gmm_v4_router'

def train_variant(name,beta_final,params,rf,data,device):
    ex=RFExperts(params.p,params.d,params.K,params.init_std).to(device)
    opt=torch.optim.AdamW(ex.parameters(),lr=params.lr,weight_decay=params.weight_decay); rows=[]
    xt,yt,lt=data['x_train'],data['y_train'],data['labels_train']; xv,yv,lv=data['x_test'],data['y_test'],data['labels_test']
    for step in range(params.steps+1):
        if step%params.eval_every==0 or step==params.steps:
            b=sched(step,name,beta_final,params.warmup_steps,params.ramp_steps)
            row={'variant':name,'step':step,'beta':b}; row.update(eval_all(ex,rf,xv,yv,lv,b,params.C,batch=2048,prefix='test_'))
            m=min(2048,xt.shape[0]); row.update(eval_all(ex,rf,xt[:m],yt[:m],lt[:m],b,params.C,batch=2048,prefix='train_')); rows.append(row)
        if step==params.steps: break
        idx=torch.randint(0,xt.shape[0],(params.batch_size,),device=device); phi=rf(xt[idx]); e=costs(ex(phi),yt[idx])
        b=sched(step,name,beta_final,params.warmup_steps,params.ramp_steps)
        bw=params.balance_weight_theory if name=='theory_anneal' else 0.0; ew=params.entropy_weight_theory if name=='theory_anneal' else 0.0
        loss,_=mcl_loss(e,b,bw,ew); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return ex,rows

def write_csv(path,rows):
    if not rows: return
    keys=sorted(set().union(*[r.keys() for r in rows]));
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]

def plot(outdir,rows,router_rows):
    if plt is None or pd is None: return
    if rows:
        df=pd.DataFrame(rows)
        for y in ['test_oracle_best_mse','test_oracle_class_mi_norm']:
            if y not in df: continue
            plt.figure()
            for v,s in df.groupby('variant'): plt.plot(s['step'],s[y],label=v)
            plt.xlabel('step'); plt.ylabel(y); plt.legend(); plt.tight_layout(); plt.savefig(outdir/f'{y}.png',dpi=180); plt.close()
    if router_rows:
        df=pd.DataFrame(router_rows); plt.figure(); x=np.arange(len(df)); plt.bar(x,df['router_mix_mse'])
        plt.xticks(x,df['variant'],rotation=45,ha='right'); plt.ylabel('router_mix_mse'); plt.tight_layout(); plt.savefig(outdir/'final_router_mix_mse.png',dpi=180); plt.close()

def run(params:Params):
    set_seed(params.seed); device=device_of(params.device); out=ensure_dir(params.outdir)
    means=simplex_means(params.C,params.d,params.mu,device)
    xtr,ytr,ltr=sample_gmm(params.n_train,means,params.sigma0,params.t,device); xte,yte,lte=sample_gmm(params.n_test,means,params.sigma0,params.t,device)
    xca,yca,lca=sample_gmm(params.n_calib,means,params.sigma0,params.t,device)
    rf=RandomFeatures(params.d,params.p,params.activation,params.rf_scale,params.seed).to(device).eval()
    cal=calibrate(rf,xca,yca,lca,means,params.ridge,params.power_iters,params.beta_target_mult,params.beta_glass_safety)
    (out/'params.json').write_text(json.dumps(asdict(params),indent=2)); (out/'calibration.json').write_text(json.dumps(cal,indent=2))
    bc,bg,bt,target=cal['beta_class'],cal['beta_glass_emp'],cal['beta_trans'],cal['beta_target']
    variants=[('uniform',0.0),('fixed_class',params.fixed_class_mult*bc),('fixed_good',target),('hard_cold',params.cold_mult_glass*bg),('theory_anneal',target)]
    if params.beta_grid:
        for b in sorted(set([.25*bc,.75*bc,1.25*bc,target,.8*bt,1.2*bt,.5*bg])):
            if math.isfinite(b) and b>0: variants.append((f'grid_{b:.6g}',b))
    data=dict(x_train=xtr,y_train=ytr,labels_train=ltr,x_test=xte,y_test=yte,labels_test=lte)
    all_rows=[]; router_rows=[]
    for name,beta_final in variants:
        print(f'=== {name} beta={beta_final:.6g} ===',flush=True)
        ex,rows=train_variant(name,beta_final,params,rf,data,device); all_rows.extend(rows)
        beta_eval=beta_final if name!='uniform' else max(bc,1e-6)
        router=train_router(ex,rf,xtr,ytr,beta_eval,params,device)
        q=teacher_q(ex,rf,xtr[:min(4096,xtr.shape[0])],ytr[:min(4096,ytr.shape[0])],beta_eval)
        c2k=class_to_expert(q,ltr[:q.shape[0]],params.C,params.K)
        final=eval_all(ex,rf,xte,yte,lte,beta_eval,params.C,router=router,means=means,sigma0=params.sigma0,t=params.t,c2k=c2k,batch=2048)
        final.update(variant=name,beta_eval=beta_eval,class_to_expert=str(c2k)); router_rows.append(final)
        torch.save(dict(experts=ex.state_dict(),router=router.state_dict(),rf=rf.state_dict(),params=asdict(params),calibration=cal,variant=name,beta_eval=beta_eval,class_to_expert=c2k),out/f'checkpoint_{name}.pt')
        write_csv(out/'metrics_all.csv',all_rows); write_csv(out/'router_final_metrics.csv',router_rows)
    plot(out,all_rows,router_rows)
    lines=['RF-MCL GMM v4 router',json.dumps(asdict(params),indent=2),'\nCalibration:']+[f'  {k}: {v}' for k,v in cal.items()]+['\nFinal router metrics:']
    for r in router_rows:
        lines.append(f"\n[{r['variant']}] beta_eval={r['beta_eval']}")
        for k in ['oracle_best_mse','soft_oracle_mse','router_mix_mse','bayes_router_mix_mse','oracle_class_mi_norm','router_class_mi_norm','bayes_class_mi_norm','oracle_usage_entropy','router_usage_entropy','bayes_usage_entropy','router_vs_teacher_ce','router_vs_teacher_kl','class_to_expert']:
            if k in r: lines.append(f'  {k}: {r[k]}')
    (out/'SUMMARY.txt').write_text('\n'.join(lines)); print('\n'.join(lines))

def parse():
    P=argparse.ArgumentParser()
    for k,v in asdict(Params()).items():
        arg='--'+k.replace('_','-')
        if isinstance(v,bool): P.add_argument(arg,action='store_true' if not v else 'store_false')
        elif isinstance(v,int): P.add_argument(arg,type=int,default=v)
        elif isinstance(v,float): P.add_argument(arg,type=float,default=v)
        else: P.add_argument(arg,type=str,default=v)
    return Params(**vars(P.parse_args()))

if __name__=='__main__': run(parse())

```

## 2. `scripts/rf_mcl_cifar_v3_router.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RF-MCL CIFAR v3: train real experts, then train a deployable router q(k|x_t).

Outputs in --outdir:
  data_info.json, calibration.json, metrics_all.csv, router_final_metrics.csv,
  SUMMARY.txt, checkpoint_<variant>.pt.

Key metrics:
  oracle_best_mse : min_k cost, uses y at test, not deployable.
  soft_oracle_mse : MCL responsibilities q*(k|x_t,y), uses y, not deployable.
  router_mix_mse  : deployable mixture sum_k q_theta(k|x_t) f_k(x_t).

Recommended:
  python rf_mcl_cifar_v3_router.py --device cuda --d-mode pca --pca-dim 512 --p 512 --outdir ./cifar_v3_pca
"""
from __future__ import annotations
import argparse, csv, json, math, random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision
    import torchvision.transforms as T
except Exception:
    torchvision=None; T=None
try:
    import pandas as pd
except Exception:
    pd=None
try:
    import matplotlib.pyplot as plt
except Exception:
    plt=None

CIFAR10_CLASSES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def device_of(s:str):
    return torch.device('cuda' if s=='auto' and torch.cuda.is_available() else ('cpu' if s=='auto' else s))

def ensure_dir(p):
    p=Path(p); p.mkdir(parents=True,exist_ok=True); return p

def norm_entropy(probs:torch.Tensor, eps=1e-12)->float:
    K=probs.numel(); p=probs.clamp_min(eps); return float((-(p*p.log()).sum()/math.log(K)).cpu())

def mi_norm(assign:torch.Tensor, labels:torch.Tensor, K:int, C:int, eps=1e-12)->float:
    a=assign.detach().cpu().long(); y=labels.detach().cpu().long(); n=max(1,y.numel())
    joint=torch.zeros(C,K,dtype=torch.float64)
    for c in range(C):
        for k in range(K): joint[c,k]=((y==c)&(a==k)).sum().item()
    joint/=n; pc=joint.sum(1,keepdim=True); pk=joint.sum(0,keepdim=True); den=pc@pk; m=joint>0
    mi=(joint[m]*(joint[m]/den[m]).log()).sum(); hc=-(pc[pc>0]*pc[pc>0].log()).sum(); hk=-(pk[pk>0]*pk[pk>0].log()).sum()
    return float((mi/torch.minimum(hc,hk).clamp_min(eps)).item())

def purity(assign:torch.Tensor, labels:torch.Tensor, K:int, C:int)->float:
    a=assign.detach().cpu().long(); y=labels.detach().cpu().long(); n=max(1,y.numel()); total=0
    for k in range(K):
        m=a==k
        if m.sum(): total+=int(torch.bincount(y[m],minlength=C).max())
    return total/n-1.0/C

# ---------------- data ----------------

def parse_classes(classes:str)->List[int]:
    out=[]
    for c in [z.strip() for z in classes.split(',') if z.strip()]:
        out.append(int(c) if c.isdigit() else CIFAR10_CLASSES.index(c))
    return out

def load_cifar(data_root, class_ids, n_train, n_test, no_download, seed):
    if torchvision is None: raise ImportError('torchvision is required for CIFAR.')
    tr=torchvision.datasets.CIFAR10(data_root,train=True,download=not no_download,transform=T.ToTensor())
    te=torchvision.datasets.CIFAR10(data_root,train=False,download=not no_download,transform=T.ToTensor())
    def collect(ds,n,offset):
        local={c:i for i,c in enumerate(class_ids)}; per=max(1,n//len(class_ids)); cnt={c:0 for c in class_ids}
        idx=list(range(len(ds))); rng=random.Random(seed+offset); rng.shuffle(idx); xs=[]; ys=[]
        for j in idx:
            x,y=ds[j]
            if y not in local: continue
            if len(xs)<n or cnt[y]<per:
                xs.append(x.flatten()); ys.append(local[y]); cnt[y]+=1
            if len(xs)>=n and all(cnt[c]>=per for c in class_ids): break
        X=torch.stack(xs)[:n]; Y=torch.tensor(ys[:n],dtype=torch.long); return X,Y
    Xtr,ytr=collect(tr,n_train,1); Xte,yte=collect(te,n_test,2)
    mean=Xtr.mean(0,keepdim=True); std=Xtr.std().clamp_min(1e-6)
    Xtr=(Xtr-mean)/std; Xte=(Xte-mean)/std
    info=dict(classes=[CIFAR10_CLASSES[i] for i in class_ids],raw_dim=int(Xtr.shape[1]),global_std=float(std),train_n=int(Xtr.shape[0]),test_n=int(Xte.shape[0]))
    return Xtr,ytr,Xte,yte,info

@dataclass
class FeatureMap:
    mode:str; mean:Optional[torch.Tensor]=None; pca:Optional[torch.Tensor]=None; rp:Optional[torch.Tensor]=None
    def transform(self,X):
        if self.mode=='full': return X
        if self.mode=='pca': return (X-self.mean.to(X.device))@self.pca.to(X.device).T
        if self.mode=='rp': return X@self.rp.to(X.device).T
        raise ValueError(self.mode)

def build_feature_map(X,mode,pca_dim,rp_dim,seed):
    if mode=='full': return FeatureMap('full')
    if mode=='pca':
        q=min(pca_dim,X.shape[1],X.shape[0]-1); mean=X.mean(0,keepdim=True); Xc=(X-mean).cpu()
        U,S,V=torch.pca_lowrank(Xc,q=q,center=False,niter=4); return FeatureMap('pca',mean.cpu(),V[:,:q].T.contiguous().cpu(),None)
    if mode=='rp':
        gen=torch.Generator(device='cpu'); gen.manual_seed(seed+999); R=torch.randn(rp_dim,X.shape[1],generator=gen)/math.sqrt(rp_dim)
        return FeatureMap('rp',None,None,R)
    raise ValueError(mode)

def diffuse(x0,t):
    g=math.exp(t*-1); return g*x0+math.sqrt(max(1-g*g,1e-12))*torch.randn_like(x0)

# ---------------- model ----------------

class RandomFeatures(nn.Module):
    def __init__(self,d,p,activation='erf',rf_scale=1.0,seed=0):
        super().__init__(); gen=torch.Generator(device='cpu'); gen.manual_seed(seed+12345)
        self.register_buffer('W',torch.randn(p,d,generator=gen)*rf_scale/math.sqrt(d)); self.register_buffer('b',2*math.pi*torch.rand(p,generator=gen))
        self.p=p; self.activation=activation
    def forward(self,x):
        z=x@self.W.T.to(x.device)
        if self.activation=='erf': h=torch.erf(z)
        elif self.activation=='tanh': h=torch.tanh(z)
        elif self.activation=='relu': h=F.relu(z)
        elif self.activation=='cos': h=torch.cos(z+self.b.to(x.device))
        elif self.activation=='sin': h=torch.sin(z+self.b.to(x.device))
        else: raise ValueError(self.activation)
        return h/math.sqrt(self.p)

class RFExperts(nn.Module):
    def __init__(self,p,dout,K,init_std):
        super().__init__(); self.A=nn.Parameter(init_std*torch.randn(K,p,dout))
    def forward(self,phi): return torch.einsum('bp,kpd->bkd',phi,self.A)

class Router(nn.Module):
    def __init__(self,p,K,hidden=0,dropout=0.0):
        super().__init__(); self.net=nn.Sequential(nn.Linear(p,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,K)) if hidden>0 else nn.Linear(p,K)
    def forward(self,phi): return self.net(phi)

def costs(pred,y): return ((pred-y[:,None,:])**2).mean(-1)

def mcl_loss(e,beta,balance=0.0,entropy=0.0):
    K=e.shape[1]
    if beta<=1e-12:
        q=torch.full_like(e,1.0/K); loss=e.mean()
    else:
        logits=-beta*e; q=torch.softmax(logits,1); loss=-(torch.logsumexp(logits,1)-math.log(K)).mean()/beta
    if balance>0:
        u=q.mean(0); loss=loss+balance*((u-1.0/K)**2).sum()
    if entropy>0:
        ent=-(q.clamp_min(1e-12)*q.clamp_min(1e-12).log()).sum(1).mean()/math.log(K); loss=loss-entropy*ent
    return loss,q

def sched(step,variant,beta_final,warm,ramp):
    if variant=='uniform': return 0.0
    if variant.startswith('fixed') or variant.startswith('grid') or variant=='hard_cold': return beta_final
    if step<warm: return 0.0
    u=1.0 if ramp<=0 else min(1.0,max(0.0,(step-warm)/ramp)); u=u*u*(3-2*u); return beta_final*u

# ---------------- calibration ----------------

@torch.no_grad()
def ridge_fit(phi,y,ridge):
    n,p=phi.shape; C=(phi.T@phi)/n+ridge*torch.eye(p,device=phi.device); B=(phi.T@y)/n; return torch.linalg.solve(C,B)

@torch.no_grad()
def whiten(phi,ridge):
    n,p=phi.shape; C=(phi.T@phi)/n+ridge*torch.eye(p,device=phi.device); ev,V=torch.linalg.eigh(C)
    return phi@(V@torch.diag(ev.clamp_min(ridge).rsqrt())@V.T)

@torch.no_grad()
def power_free(S,R,iters):
    if iters<=0: return 0.0
    n,p=S.shape; d=R.shape[1]; B=torch.randn(p,d,device=S.device); B=B/B.norm().clamp_min(1e-12); lam=torch.tensor(0.,device=S.device)
    for _ in range(iters):
        delta=S@B; a=(R*delta).sum(1,keepdim=True)/math.sqrt(d); M=a*R/math.sqrt(d); GB=(S.T@M)/n
        lam=(B*GB).sum(); B=GB/GB.norm().clamp_min(1e-12)
    return max(float(lam.cpu()),0.0)

@torch.no_grad()
def lambda_dir(S,R,v):
    n=S.shape[0]; a=(R@v).pow(2); H=(S.T*a[None,:])@S/n; return max(float(torch.linalg.eigvalsh(H)[-1].cpu()),0.0)

@torch.no_grad()
def class_basis(y,labels,C):
    M=[]
    for c in range(C):
        m=labels==c
        if m.sum(): M.append(y[m].mean(0))
    M=torch.stack(M); M=M-M.mean(0,keepdim=True); U,S,Vh=torch.linalg.svd(M,full_matrices=False)
    rank=int((S>1e-8*S.max().clamp_min(1e-8)).sum()); return Vh[:rank].T.contiguous()

@torch.no_grad()
def calibrate(rf,x,y,labels,C,ridge,power_iters,beta_mult,glass_safety,trans_samples=12):
    phi=rf(x); A0=ridge_fit(phi,y,ridge); R=y-phi@A0; S=whiten(phi,ridge)
    lf=power_free(S,R,power_iters); B=class_basis(y,labels,C).to(x.device)
    lc=max([lambda_dir(S,R,B[:,j]) for j in range(B.shape[1])] or [0.0])
    d=y.shape[1]; P=B@B.T if B.numel() else torch.zeros(d,d,device=x.device); vals=[]
    for _ in range(trans_samples):
        v=torch.randn(d,device=x.device); v=v-P@v
        if v.norm()>1e-8: vals.append(lambda_dir(S,R,v/v.norm()))
    lt=max(vals or [0.0]); beta=lambda l:0.5/max(l,1e-12)
    E=R.pow(2).mean(1); v_emp=float(E.var(unbiased=False).cpu()); alpha=math.log(max(2,x.shape[0]))/max(1,d)
    bg=math.sqrt(2*alpha/max(v_emp,1e-12)); bc=beta(lc); bt=beta(lt); target=min(beta_mult*bc,glass_safety*bg)
    return dict(lambda_free=lf,lambda_class=lc,lambda_trans=lt,beta_free=beta(lf),beta_class=bc,beta_trans=bt,
                alpha_log_n_over_d=alpha,v_emp=v_emp,beta_glass_emp=bg,beta_target=target,residual_mse=float(E.mean().cpu()),
                class_basis_rank=int(B.shape[1]),window_class_before_trans=float(bc<bt),window_class_before_glass=float(bc<bg))

# ---------------- eval/router ----------------

@torch.no_grad()
def evaluate(experts,rf,x,y,labels,beta,C,router=None,batch=1024,prefix=''):
    K=experts.A.shape[0]; sums={k:0.0 for k in ['best','soft','mean','rmix','rsoft']}; n=0; ao=[]; ar=[]; qt=[]; qr=[]
    for i in range(0,x.shape[0],batch):
        xb=x[i:i+batch]; yb=y[i:i+batch]; phi=rf(xb); pred=experts(phi); e=costs(pred,yb); _,q=mcl_loss(e,beta); B=xb.shape[0]; n+=B
        sums['best']+=float(e.min(1).values.sum()); sums['soft']+=float((q*e).sum(1).sum()); sums['mean']+=float(e.mean(1).sum())
        ao.append(e.argmin(1).cpu()); qt.append(q.cpu())
        if router is not None:
            rq=torch.softmax(router(phi),1); mix=torch.einsum('bk,bkd->bd',rq,pred)
            sums['rmix']+=float(((mix-yb)**2).mean(1).sum()); sums['rsoft']+=float((rq*e).sum(1).sum())
            ar.append(rq.argmax(1).cpu()); qr.append(rq.cpu())
    out={prefix+'oracle_best_mse':sums['best']/n,prefix+'soft_oracle_mse':sums['soft']/n,prefix+'mean_expert_mse':sums['mean']/n}
    ycpu=labels.cpu()
    def add(name,alist,qlist):
        if not alist: return
        a=torch.cat(alist); q=torch.cat(qlist); u=torch.bincount(a,minlength=K).float(); u=u/u.sum().clamp_min(1)
        out[prefix+name+'_usage_entropy']=norm_entropy(u); out[prefix+name+'_eff_frac_min']=float((u.min()*K).item())
        out[prefix+name+'_class_mi_norm']=mi_norm(a,ycpu,K,C); out[prefix+name+'_class_purity']=purity(a,ycpu,K,C)
        out[prefix+name+'_soft_usage_entropy']=norm_entropy(q.mean(0))
    add('oracle',ao,qt)
    if router is not None:
        out[prefix+'router_mix_mse']=sums['rmix']/n; out[prefix+'router_soft_mse']=sums['rsoft']/n
        Q=torch.cat(qt).clamp_min(1e-12); Rq=torch.cat(qr).clamp_min(1e-12)
        out[prefix+'router_vs_teacher_ce']=float((-(Q*Rq.log()).sum(1).mean()).item())
        out[prefix+'router_vs_teacher_kl']=float((Q*(Q.log()-Rq.log())).sum(1).mean().item())
        add('router',ar,qr)
    return out

def train_router(experts,rf,x,y,beta,params,device):
    router=Router(params.p,params.K,params.router_hidden,params.router_dropout).to(device); opt=torch.optim.AdamW(router.parameters(),lr=params.router_lr); experts.eval()
    for _ in range(params.router_steps):
        idx=torch.randint(0,x.shape[0],(params.router_batch_size,),device=device); xb=x[idx]; yb=y[idx]
        with torch.no_grad():
            phi=rf(xb); e=costs(experts(phi),yb); _,q=mcl_loss(e,beta)
        loss=-(q*torch.log_softmax(router(phi),1)).sum(1).mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return router

# ---------------- main ----------------

@dataclass
class Params:
    data_root:str='./data'; classes:str='automobile,horse'; d_mode:str='pca'; pca_dim:int=512; rp_dim:int=512; p:int=512; K:int=4; t:float=1.5
    n_train:int=6000; n_test:int=2000; n_calib:int=2048; batch_size:int=192; steps:int=3000; lr:float=2e-3; weight_decay:float=0.0
    init_std:float=1e-3; activation:str='erf'; rf_scale:float=1.0; seed:int=0; warmup_steps:int=600; ramp_steps:int=900; eval_every:int=200
    power_iters:int=8; ridge:float=1e-5; beta_target_mult:float=3.0; beta_glass_safety:float=0.45; cold_mult_glass:float=1.2; fixed_class_mult:float=1.2
    balance_weight_theory:float=0.0; entropy_weight_theory:float=0.0; router_steps:int=1000; router_lr:float=2e-3; router_hidden:int=0; router_dropout:float=0.0; router_batch_size:int=192
    beta_grid:bool=False; no_download:bool=False; device:str='auto'; outdir:str='./cifar_v3_router'

def train_variant(name,beta_final,params,rf,data,device,d,C):
    ex=RFExperts(params.p,d,params.K,params.init_std).to(device); opt=torch.optim.AdamW(ex.parameters(),lr=params.lr,weight_decay=params.weight_decay); rows=[]
    xt,yt,lt=data['x_train'],data['y_train'],data['labels_train']; xv,yv,lv=data['x_test'],data['y_test'],data['labels_test']
    for step in range(params.steps+1):
        if step%params.eval_every==0 or step==params.steps:
            b=sched(step,name,beta_final,params.warmup_steps,params.ramp_steps); row={'variant':name,'step':step,'beta':b}
            row.update(evaluate(ex,rf,xv,yv,lv,b,C,batch=1024,prefix='test_'))
            m=min(2048,xt.shape[0]); row.update(evaluate(ex,rf,xt[:m],yt[:m],lt[:m],b,C,batch=1024,prefix='train_')); rows.append(row)
        if step==params.steps: break
        idx=torch.randint(0,xt.shape[0],(params.batch_size,),device=device); phi=rf(xt[idx]); e=costs(ex(phi),yt[idx]); b=sched(step,name,beta_final,params.warmup_steps,params.ramp_steps)
        bw=params.balance_weight_theory if name=='theory_anneal' else 0.0; ew=params.entropy_weight_theory if name=='theory_anneal' else 0.0
        loss,_=mcl_loss(e,b,bw,ew); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return ex,rows

def write_csv(path,rows):
    if not rows: return
    keys=sorted(set().union(*[r.keys() for r in rows]));
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]

def plot(outdir,rows,router_rows):
    if plt is None or pd is None: return
    if rows:
        df=pd.DataFrame(rows)
        for y in ['test_oracle_best_mse','test_oracle_class_mi_norm']:
            if y not in df: continue
            plt.figure()
            for v,s in df.groupby('variant'): plt.plot(s['step'],s[y],label=v)
            plt.xlabel('step'); plt.ylabel(y); plt.legend(); plt.tight_layout(); plt.savefig(outdir/f'{y}.png',dpi=180); plt.close()
    if router_rows:
        df=pd.DataFrame(router_rows); plt.figure(); x=np.arange(len(df)); plt.bar(x,df['router_mix_mse'])
        plt.xticks(x,df['variant'],rotation=45,ha='right'); plt.ylabel('router_mix_mse'); plt.tight_layout(); plt.savefig(outdir/'final_router_mix_mse.png',dpi=180); plt.close()

def run(params:Params):
    set_seed(params.seed); device=device_of(params.device); out=ensure_dir(params.outdir); ids=parse_classes(params.classes); C=len(ids)
    Xtr_raw,ytr,Xte_raw,yte,info=load_cifar(params.data_root,ids,params.n_train,params.n_test,params.no_download,params.seed)
    fmap=build_feature_map(Xtr_raw,params.d_mode,params.pca_dim,params.rp_dim,params.seed)
    X0=fmap.transform(Xtr_raw).float(); Xt0=fmap.transform(Xte_raw).float(); lat_mean=X0.mean(0,keepdim=True); lat_std=X0.std().clamp_min(1e-6)
    X0=(X0-lat_mean)/lat_std; Xt0=(Xt0-lat_mean)/lat_std; X=diffuse(X0,params.t); Xv=diffuse(Xt0,params.t)
    xtr=X.to(device); ytr0=X0.to(device); ltr=ytr.to(device); xte=Xv.to(device); yte0=Xt0.to(device); lte=yte.to(device); d=ytr0.shape[1]
    info.update(d_mode=params.d_mode,d_latent=int(d),p=params.p,K=params.K,t=params.t)
    rf=RandomFeatures(d,params.p,params.activation,params.rf_scale,params.seed).to(device).eval()
    idx=torch.randperm(xtr.shape[0],device=device)[:min(params.n_calib,xtr.shape[0])]
    cal=calibrate(rf,xtr[idx],ytr0[idx],ltr[idx],C,params.ridge,params.power_iters,params.beta_target_mult,params.beta_glass_safety)
    (out/'params.json').write_text(json.dumps(asdict(params),indent=2)); (out/'data_info.json').write_text(json.dumps(info,indent=2)); (out/'calibration.json').write_text(json.dumps(cal,indent=2))
    bc,bg,bt,target=cal['beta_class'],cal['beta_glass_emp'],cal['beta_trans'],cal['beta_target']
    if not (bc<min(bg,bt)): print(f'WARNING: narrow/absent window beta_class={bc:.4g}, beta_trans={bt:.4g}, beta_glass={bg:.4g}',flush=True)
    variants=[('uniform',0.0),('fixed_class',params.fixed_class_mult*bc),('fixed_good',target),('hard_cold',params.cold_mult_glass*bg),('theory_anneal',target)]
    if params.beta_grid:
        for b in sorted(set([.25*bc,.75*bc,1.25*bc,target,.8*bt,1.2*bt,.5*bg])):
            if math.isfinite(b) and b>0: variants.append((f'grid_{b:.6g}',b))
    data=dict(x_train=xtr,y_train=ytr0,labels_train=ltr,x_test=xte,y_test=yte0,labels_test=lte)
    all_rows=[]; router_rows=[]
    for name,beta_final in variants:
        print(f'=== {name} beta={beta_final:.6g} ===',flush=True)
        ex,rows=train_variant(name,beta_final,params,rf,data,device,d,C); all_rows.extend(rows); beta_eval=beta_final if name!='uniform' else max(bc,1e-6)
        router=train_router(ex,rf,xtr,ytr0,beta_eval,params,device)
        final=evaluate(ex,rf,xte,yte0,lte,beta_eval,C,router=router,batch=1024); final.update(variant=name,beta_eval=beta_eval); router_rows.append(final)
        torch.save(dict(experts=ex.state_dict(),router=router.state_dict(),rf=rf.state_dict(),params=asdict(params),data_info=info,calibration=cal,variant=name,beta_eval=beta_eval,
                        latent_mean=lat_mean.cpu(),latent_std=lat_std.cpu(),feature_map_mode=fmap.mode,pca_components=None if fmap.pca is None else fmap.pca.cpu(),rp_matrix=None if fmap.rp is None else fmap.rp.cpu(),raw_mean=None if fmap.mean is None else fmap.mean.cpu()),out/f'checkpoint_{name}.pt')
        write_csv(out/'metrics_all.csv',all_rows); write_csv(out/'router_final_metrics.csv',router_rows)
    plot(out,all_rows,router_rows)
    lines=['RF-MCL CIFAR v3 router',json.dumps(asdict(params),indent=2),'\nData info:',json.dumps(info,indent=2),'\nCalibration:']+[f'  {k}: {v}' for k,v in cal.items()]
    if not (bc<min(bg,bt)): lines.append('\nWARNING: class/glass/transverse window narrow or absent.')
    lines.append('\nFinal router metrics:')
    for r in router_rows:
        lines.append(f"\n[{r['variant']}] beta_eval={r['beta_eval']}")
        for k in ['oracle_best_mse','soft_oracle_mse','router_mix_mse','router_soft_mse','oracle_class_mi_norm','router_class_mi_norm','oracle_usage_entropy','router_usage_entropy','router_vs_teacher_ce','router_vs_teacher_kl']:
            if k in r: lines.append(f'  {k}: {r[k]}')
    (out/'SUMMARY.txt').write_text('\n'.join(lines)); print('\n'.join(lines))

def parse():
    P=argparse.ArgumentParser()
    for k,v in asdict(Params()).items():
        arg='--'+k.replace('_','-')
        if isinstance(v,bool): P.add_argument(arg,action='store_true' if not v else 'store_false')
        elif isinstance(v,int): P.add_argument(arg,type=int,default=v)
        elif isinstance(v,float): P.add_argument(arg,type=float,default=v)
        else: P.add_argument(arg,type=str,default=v)
    return Params(**vars(P.parse_args()))

if __name__=='__main__': run(parse())

```

## 3. `scripts/rf_mcl_gmm_v5_router.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pandas as pd
except Exception:
    pd = None
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_of(s: str):
    return torch.device("cuda" if s == "auto" and torch.cuda.is_available() else ("cpu" if s == "auto" else s))


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def norm_entropy(probs: torch.Tensor, eps: float = 1e-12) -> float:
    k = probs.numel()
    p = probs.clamp_min(eps)
    return float((-(p * p.log()).sum() / math.log(k)).cpu())


def mi_norm(assign: torch.Tensor, labels: torch.Tensor, k: int, c: int, eps: float = 1e-12) -> float:
    a = assign.detach().cpu().long()
    y = labels.detach().cpu().long()
    n = max(1, y.numel())
    joint = torch.zeros(c, k, dtype=torch.float64)
    for ci in range(c):
        for ki in range(k):
            joint[ci, ki] = ((y == ci) & (a == ki)).sum().item()
    joint /= n
    pc = joint.sum(1, keepdim=True)
    pk = joint.sum(0, keepdim=True)
    den = pc @ pk
    m = joint > 0
    mi = (joint[m] * (joint[m] / den[m]).log()).sum()
    hc = -(pc[pc > 0] * pc[pc > 0].log()).sum()
    hk = -(pk[pk > 0] * pk[pk > 0].log()).sum()
    return float((mi / torch.minimum(hc, hk).clamp_min(eps)).item())


def best_perm(score: np.ndarray):
    c, k = score.shape
    best = None
    val = -1e100
    for p in itertools.permutations(range(k), min(c, k)):
        s = sum(score[ci, p[ci]] for ci in range(len(p)))
        if s > val:
            val = s
            best = list(p)
    return best if best is not None else list(range(min(c, k)))


def simplex_means(c: int, d: int, mu: float, device):
    if c == 2:
        m = torch.zeros(c, d, device=device)
        m[0, 0] = -mu * math.sqrt(d)
        m[1, 0] = mu * math.sqrt(d)
        return m
    g = torch.randn(c, d, device=device)
    g = g - g.mean(0, keepdim=True)
    q, _ = torch.linalg.qr(g.T, mode="reduced")
    m = q.T[:c]
    m = m - m.mean(0, keepdim=True)
    m = m / m.norm(dim=1, keepdim=True).clamp_min(1e-12) * (mu * math.sqrt(d))
    return m


def sample_gmm(n: int, means: torch.Tensor, sigma0: float, t: float, device):
    c, _ = means.shape
    labels = torch.randint(0, c, (n,), device=device)
    x0 = means[labels] + sigma0 * torch.randn(n, means.shape[1], device=device)
    gamma = math.exp(-t)
    eps = torch.randn_like(x0)
    xt = gamma * x0 + math.sqrt(max(1.0 - gamma * gamma, 1e-12)) * eps
    return xt, eps, labels


def bayes_post(xt: torch.Tensor, means: torch.Tensor, sigma0: float, t: float):
    g = math.exp(-t)
    st2 = g * g * sigma0 * sigma0 + max(1 - g * g, 1e-12)
    dist = ((xt[:, None, :] - g * means[None, :, :]) ** 2).sum(-1)
    return torch.softmax(-0.5 * dist / st2, dim=1)


class RandomFeatures(nn.Module):
    def __init__(self, d: int, p: int, activation: str = "erf", seed: int = 0):
        super().__init__()
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + 12345)
        self.register_buffer("W", torch.randn(p, d, generator=gen) / math.sqrt(d))
        self.register_buffer("b", 2 * math.pi * torch.rand(p, generator=gen))
        self.p = p
        self.activation = activation

    def forward(self, x):
        z = x @ self.W.T.to(x.device)
        if self.activation == "erf":
            h = torch.erf(z)
        elif self.activation == "tanh":
            h = torch.tanh(z)
        elif self.activation == "relu":
            h = F.relu(z)
        elif self.activation == "cos":
            h = torch.cos(z + self.b.to(x.device))
        else:
            raise ValueError(self.activation)
        return h / math.sqrt(self.p)


class RFExperts(nn.Module):
    def __init__(self, p: int, d: int, k: int, init_std: float):
        super().__init__()
        self.A = nn.Parameter(init_std * torch.randn(k, p, d))

    def forward(self, phi):
        return torch.einsum("bp,kpd->bkd", phi, self.A)


class Router(nn.Module):
    def __init__(self, p: int, k: int):
        super().__init__()
        self.fc = nn.Linear(p, k)

    def forward(self, phi):
        return self.fc(phi)


def mcl_loss(e, beta: float):
    k = e.shape[1]
    if beta <= 1e-12:
        q = torch.full_like(e, 1.0 / k)
        loss = e.mean()
    else:
        logits = -beta * e
        q = torch.softmax(logits, 1)
        loss = -(torch.logsumexp(logits, 1) - math.log(k)).mean() / beta
    return loss, q


@torch.no_grad()
def ridge_fit(phi, y, ridge: float):
    n, p = phi.shape
    c = (phi.T @ phi) / n + ridge * torch.eye(p, device=phi.device)
    b = (phi.T @ y) / n
    return torch.linalg.solve(c, b)


@torch.no_grad()
def whiten(phi, ridge: float):
    n, p = phi.shape
    c = (phi.T @ phi) / n + ridge * torch.eye(p, device=phi.device)
    ev, v = torch.linalg.eigh(c)
    return phi @ (v @ torch.diag(ev.clamp_min(ridge).rsqrt()) @ v.T)


@torch.no_grad()
def power_free(s, r, iters: int):
    if iters <= 0:
        return 0.0
    n, p = s.shape
    d = r.shape[1]
    b = torch.randn(p, d, device=s.device)
    b = b / b.norm().clamp_min(1e-12)
    lam = torch.tensor(0.0, device=s.device)
    for _ in range(iters):
        delta = s @ b
        a = (r * delta).sum(1, keepdim=True) / math.sqrt(d)
        m = a * r / math.sqrt(d)
        gb = (s.T @ m) / n
        lam = (b * gb).sum()
        b = gb / gb.norm().clamp_min(1e-12)
    return max(float(lam.cpu()), 0.0)


@torch.no_grad()
def lambda_dir(s, r, v):
    n = s.shape[0]
    a = (r @ v).pow(2)
    h = (s.T * a[None, :]) @ s / n
    return max(float(torch.linalg.eigvalsh(h)[-1].cpu()), 0.0)


@torch.no_grad()
def class_basis(means: torch.Tensor):
    m = means - means.mean(0, keepdim=True)
    _, s, vh = torch.linalg.svd(m, full_matrices=False)
    rank = int((s > 1e-8 * s.max().clamp_min(1e-8)).sum())
    return vh[:rank].T.contiguous()


@torch.no_grad()
def calibrate(rf, x, y, labels, means, ridge: float, power_iters: int):
    phi = rf(x)
    a0 = ridge_fit(phi, y, ridge)
    r = y - phi @ a0
    s = whiten(phi, ridge)
    lf = power_free(s, r, power_iters)
    b = class_basis(means).to(x.device)
    lc = max([lambda_dir(s, r, b[:, j]) for j in range(b.shape[1])] or [0.0])
    d = y.shape[1]
    pmat = b @ b.T if b.numel() else torch.zeros(d, d, device=x.device)
    vals = []
    for _ in range(12):
        v = torch.randn(d, device=x.device)
        v = v - pmat @ v
        if v.norm() > 1e-8:
            vals.append(lambda_dir(s, r, v / v.norm()))
    lt = max(vals or [0.0])
    beta = lambda l: 0.5 / max(l, 1e-12)
    e = r.pow(2).mean(1)
    v_emp = float(e.var(unbiased=False).cpu())
    alpha = math.log(max(2, x.shape[0])) / max(1, d)
    bg = math.sqrt(2 * alpha / max(v_emp, 1e-12))
    return dict(
        lambda_free=lf,
        lambda_class=lc,
        lambda_trans=lt,
        beta_free=beta(lf),
        beta_class=beta(lc),
        beta_trans=beta(lt),
        beta_glass_emp=bg,
        alpha_log_n_over_d=alpha,
        residual_mse=float(e.mean().cpu()),
        v_emp=v_emp,
    )


def sched(step: int, variant: str, beta_final: float, warm: int, ramp: int):
    if variant == "uniform":
        return 0.0
    if variant.startswith("fixed") or variant.startswith("grid") or variant == "hard_cold":
        return beta_final
    if step < warm:
        return 0.0
    u = min(1.0, max(0.0, (step - warm) / max(1, ramp)))
    u = u * u * (3 - 2 * u)
    return beta_final * u


@dataclass
class Params:
    d: int = 64
    p: int = 256
    C: int = 4
    K: int = 4
    mu: float = 1.0
    sigma0: float = 0.5
    t: float = 2.05
    n_train: int = 6000
    n_test: int = 3000
    n_calib: int = 2500
    batch_size: int = 256
    steps: int = 4000
    lr: float = 3e-3
    init_std: float = 1e-3
    activation: str = "erf"
    seed: int = 0
    warmup_steps: int = 800
    ramp_steps: int = 1000
    eval_every: int = 200
    power_iters: int = 30
    ridge: float = 1e-5
    router_steps: int = 1200
    router_lr: float = 2e-3
    router_batch_size: int = 256
    beta_grid: bool = False
    quick: bool = False
    device: str = "auto"
    outdir: str = "./gmm_v5_router"


def train_variant(name: str, beta_final: float, params: Params, rf, data, device):
    ex = RFExperts(params.p, params.d, params.K, params.init_std).to(device)
    opt = torch.optim.AdamW(ex.parameters(), lr=params.lr)
    rows = []
    xt, eps_t, ltr = data["x_train"], data["y_train"], data["labels_train"]
    xv, eps_v, lte = data["x_test"], data["y_test"], data["labels_test"]
    phi_train = data["phi_train"]
    phi_test = data["phi_test"]
    f0_train = data["f0_train"]
    f0_test = data["f0_test"]
    for step in range(params.steps + 1):
        if step % params.eval_every == 0 or step == params.steps:
            b = sched(step, name, beta_final, params.warmup_steps, params.ramp_steps)
            with torch.no_grad():
                pred_test = f0_test[:, None, :] + ex(phi_test)
                e_test = ((pred_test - eps_v[:, None, :]) ** 2).mean(-1)
                _, q_test = mcl_loss(e_test, b)
                row = {
                    "variant": name,
                    "step": step,
                    "beta": b,
                    "test_oracle_best_mse": float(e_test.min(1).values.mean().item()),
                    "test_soft_oracle_mse": float((q_test * e_test).sum(1).mean().item()),
                    "test_mean_expert_mse": float(e_test.mean(1).mean().item()),
                    "test_teacher_entropy_norm": float((-(q_test.clamp_min(1e-12) * q_test.clamp_min(1e-12).log()).sum(1).mean() / math.log(params.K)).item()),
                    "test_oracle_class_mi_norm": mi_norm(e_test.argmin(1), lte, params.K, params.C),
                }
                rows.append(row)
        if step == params.steps:
            break
        idx = torch.randint(0, xt.shape[0], (params.batch_size,), device=device)
        phi = phi_train[idx]
        pred = f0_train[idx][:, None, :] + ex(phi)
        e = ((pred - eps_t[idx][:, None, :]) ** 2).mean(-1)
        b = sched(step, name, beta_final, params.warmup_steps, params.ramp_steps)
        loss, _ = mcl_loss(e, b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return ex, rows


@torch.no_grad()
def teacher_q(experts, phi, y, f0, beta: float):
    pred = f0[:, None, :] + experts(phi)
    e = ((pred - y[:, None, :]) ** 2).mean(-1)
    _, q = mcl_loss(e, beta)
    return q


def train_router(experts, phi, y, f0, beta: float, params: Params, device):
    router = Router(params.p, params.K).to(device)
    opt = torch.optim.AdamW(router.parameters(), lr=params.router_lr)
    experts.eval()
    for _ in range(params.router_steps):
        idx = torch.randint(0, phi.shape[0], (params.router_batch_size,), device=device)
        with torch.no_grad():
            pred = f0[idx][:, None, :] + experts(phi[idx])
            e = ((pred - y[idx][:, None, :]) ** 2).mean(-1)
            _, q = mcl_loss(e, beta)
        loss = -(q * torch.log_softmax(router(phi[idx]), 1)).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return router


@torch.no_grad()
def eval_router(experts, router, phi, y, labels, f0, beta: float, c: int):
    pred = f0[:, None, :] + experts(phi)
    e = ((pred - y[:, None, :]) ** 2).mean(-1)
    _, q = mcl_loss(e, beta)
    rq = torch.softmax(router(phi), 1)
    mix = torch.einsum("bk,bkd->bd", rq, pred)
    u = torch.bincount(rq.argmax(1), minlength=pred.shape[1]).float()
    u = u / u.sum().clamp_min(1)
    return dict(
        oracle_best_mse=float(e.min(1).values.mean().item()),
        soft_oracle_mse=float((q * e).sum(1).mean().item()),
        mean_expert_mse=float(e.mean(1).mean().item()),
        router_mix_mse=float(((mix - y) ** 2).mean(1).mean().item()),
        router_soft_mse=float((rq * e).sum(1).mean().item()),
        oracle_class_mi_norm=mi_norm(e.argmin(1), labels, pred.shape[1], c),
        router_class_mi_norm=mi_norm(rq.argmax(1), labels, pred.shape[1], c),
        oracle_usage_entropy=norm_entropy(torch.bincount(e.argmin(1), minlength=pred.shape[1]).float().div(max(1, y.shape[0]))),
        router_usage_entropy=norm_entropy(u),
        teacher_entropy_norm=float((-(q.clamp_min(1e-12) * q.clamp_min(1e-12).log()).sum(1).mean() / math.log(pred.shape[1])).item()),
        router_vs_teacher_ce=float((-(q.clamp_min(1e-12) * rq.clamp_min(1e-12).log()).sum(1).mean()).item()),
        router_vs_teacher_kl=float((q.clamp_min(1e-12) * (q.clamp_min(1e-12).log() - rq.clamp_min(1e-12).log())).sum(1).mean().item()),
    )


def write_csv(path: Path, rows):
    if not rows:
        return
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run(params: Params):
    if params.quick:
        params.n_train = min(params.n_train, 2500)
        params.n_test = min(params.n_test, 1200)
        params.n_calib = min(params.n_calib, 1200)
        params.steps = min(params.steps, 1500)
        params.eval_every = min(params.eval_every, 150)
        params.router_steps = min(params.router_steps, 600)
    set_seed(params.seed)
    device = device_of(params.device)
    out = ensure_dir(params.outdir)
    means = simplex_means(params.C, params.d, params.mu, device)
    xtr, eps_tr, ltr = sample_gmm(params.n_train, means, params.sigma0, params.t, device)
    xte, eps_te, lte = sample_gmm(params.n_test, means, params.sigma0, params.t, device)
    xca, eps_ca, lca = sample_gmm(params.n_calib, means, params.sigma0, params.t, device)
    rf = RandomFeatures(params.d, params.p, params.activation, params.seed).to(device).eval()
    phi_tr = rf(xtr)
    phi_te = rf(xte)
    a0 = ridge_fit(phi_tr, eps_tr, params.ridge)
    f0_tr = phi_tr @ a0
    f0_te = phi_te @ a0
    cal = calibrate(rf, xca, eps_ca, lca, means, params.ridge, params.power_iters)
    base_grid = [0.2, 0.4, 0.7, 1.1]
    bc = cal["beta_class"]
    bg = cal["beta_glass_emp"]
    variants = [
        ("uniform", 0.0),
        ("fixed_class", max(0.2, 1.2 * bc)),
        ("fixed_good", min(max(0.2, 3.0 * bc), 0.45 * bg)),
        ("hard_cold", 1.2 * bg),
        ("theory_anneal", min(max(0.2, 3.0 * bc), 0.45 * bg)),
    ]
    if params.beta_grid:
        for b in base_grid:
            variants.append((f"grid_{b:.3f}", b))
    data = dict(x_train=xtr, y_train=eps_tr, labels_train=ltr, x_test=xte, y_test=eps_te, labels_test=lte, phi_train=phi_tr, phi_test=phi_te, f0_train=f0_tr, f0_test=f0_te)
    all_rows = []
    final_rows = []
    for name, bfin in variants:
        print(f"=== {name} beta={bfin:.6g} ===", flush=True)
        ex, rows = train_variant(name, bfin, params, rf, data, device)
        all_rows.extend(rows)
        beval = bfin if name != "uniform" else max(0.2, bc)
        router = train_router(ex, phi_tr, eps_tr, f0_tr, beval, params, device)
        final = eval_router(ex, router, phi_te, eps_te, lte, f0_te, beval, params.C)
        final.update(variant=name, beta_eval=beval)
        final_rows.append(final)
        torch.save(dict(experts=ex.state_dict(), router=router.state_dict(), rf=rf.state_dict(), ridge_head=a0, params=asdict(params), calibration=cal, variant=name, beta_eval=beval), out / f"checkpoint_{name}.pt")
        write_csv(out / "metrics_all.csv", all_rows)
        write_csv(out / "router_final_metrics.csv", final_rows)
    if plt is not None and pd is not None:
        df = pd.DataFrame(all_rows)
        if not df.empty:
            for y in ["test_oracle_best_mse", "test_oracle_class_mi_norm", "test_teacher_entropy_norm"]:
                if y in df:
                    plt.figure()
                    for v, s in df.groupby("variant"):
                        plt.plot(s["step"], s[y], label=v)
                    plt.xlabel("step")
                    plt.ylabel(y)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(out / f"{y}.png", dpi=180)
                    plt.close()
    lines = ["RF-MCL GMM v5 router (target=eps, residualized)", json.dumps(asdict(params), indent=2), "\nCalibration:"]
    lines.extend([f"  {k}: {v}" for k, v in cal.items()])
    lines.append("\nFinal router metrics:")
    for r in final_rows:
        lines.append(f"\n[{r['variant']}] beta_eval={r['beta_eval']}")
        for k in ["oracle_best_mse", "soft_oracle_mse", "mean_expert_mse", "router_mix_mse", "router_soft_mse", "oracle_class_mi_norm", "router_class_mi_norm", "teacher_entropy_norm", "router_vs_teacher_ce", "router_vs_teacher_kl"]:
            lines.append(f"  {k}: {r[k]}")
        if r["teacher_entropy_norm"] > 0.95:
            lines.append("  teacher_status: near_uniform (router likely uninformative)")
        elif r["teacher_entropy_norm"] < 0.05:
            lines.append("  teacher_status: near_collapse")
    (out / "SUMMARY.txt").write_text("\n".join(lines))
    print("\n".join(lines))


def parse():
    p = argparse.ArgumentParser()
    for k, v in asdict(Params()).items():
        arg = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            p.add_argument(arg, action="store_true" if not v else "store_false")
        elif isinstance(v, int):
            p.add_argument(arg, type=int, default=v)
        elif isinstance(v, float):
            p.add_argument(arg, type=float, default=v)
        else:
            p.add_argument(arg, type=str, default=v)
    return Params(**vars(p.parse_args()))


if __name__ == "__main__":
    run(parse())


```

## 4. `scripts/rf_mcl_cifar_v5_router.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision
    import torchvision.transforms as T
except Exception:
    torchvision = None
    T = None
try:
    import pandas as pd
except Exception:
    pd = None
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_of(s: str):
    return torch.device("cuda" if s == "auto" and torch.cuda.is_available() else ("cpu" if s == "auto" else s))


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def norm_entropy(probs: torch.Tensor, eps: float = 1e-12) -> float:
    k = probs.numel()
    p = probs.clamp_min(eps)
    return float((-(p * p.log()).sum() / math.log(k)).cpu())


def mi_norm(assign: torch.Tensor, labels: torch.Tensor, k: int, c: int, eps: float = 1e-12) -> float:
    a = assign.detach().cpu().long()
    y = labels.detach().cpu().long()
    n = max(1, y.numel())
    joint = torch.zeros(c, k, dtype=torch.float64)
    for ci in range(c):
        for ki in range(k):
            joint[ci, ki] = ((y == ci) & (a == ki)).sum().item()
    joint /= n
    pc = joint.sum(1, keepdim=True)
    pk = joint.sum(0, keepdim=True)
    den = pc @ pk
    m = joint > 0
    mi = (joint[m] * (joint[m] / den[m]).log()).sum()
    hc = -(pc[pc > 0] * pc[pc > 0].log()).sum()
    hk = -(pk[pk > 0] * pk[pk > 0].log()).sum()
    return float((mi / torch.minimum(hc, hk).clamp_min(eps)).item())


def parse_classes(classes: str) -> List[int]:
    out = []
    for c in [z.strip() for z in classes.split(",") if z.strip()]:
        out.append(int(c) if c.isdigit() else CIFAR10_CLASSES.index(c))
    return out


def load_cifar(data_root: str, class_ids: List[int], n_train: int, n_test: int, no_download: bool, seed: int):
    if torchvision is None:
        raise ImportError("torchvision is required for CIFAR.")
    tr = torchvision.datasets.CIFAR10(data_root, train=True, download=not no_download, transform=T.ToTensor())
    te = torchvision.datasets.CIFAR10(data_root, train=False, download=not no_download, transform=T.ToTensor())

    def collect(ds, n, offset):
        local = {c: i for i, c in enumerate(class_ids)}
        per = max(1, n // len(class_ids))
        cnt = {c: 0 for c in class_ids}
        idx = list(range(len(ds)))
        rng = random.Random(seed + offset)
        rng.shuffle(idx)
        xs = []
        ys = []
        for j in idx:
            x, y = ds[j]
            if y not in local:
                continue
            if len(xs) < n or cnt[y] < per:
                xs.append(x.flatten())
                ys.append(local[y])
                cnt[y] += 1
            if len(xs) >= n and all(cnt[c] >= per for c in class_ids):
                break
        x = torch.stack(xs)[:n]
        y = torch.tensor(ys[:n], dtype=torch.long)
        return x, y

    xtr, ytr = collect(tr, n_train, 1)
    xte, yte = collect(te, n_test, 2)
    mean = xtr.mean(0, keepdim=True)
    std = xtr.std().clamp_min(1e-6)
    xtr = (xtr - mean) / std
    xte = (xte - mean) / std
    info = dict(classes=[CIFAR10_CLASSES[i] for i in class_ids], raw_dim=int(xtr.shape[1]), global_std=float(std), train_n=int(xtr.shape[0]), test_n=int(xte.shape[0]))
    return xtr, ytr, xte, yte, info


@dataclass
class FeatureMap:
    mode: str
    mean: Optional[torch.Tensor] = None
    pca: Optional[torch.Tensor] = None
    rp: Optional[torch.Tensor] = None

    def transform(self, x):
        if self.mode == "full":
            return x
        if self.mode == "pca":
            return (x - self.mean.to(x.device)) @ self.pca.to(x.device).T
        if self.mode == "rp":
            return x @ self.rp.to(x.device).T
        raise ValueError(self.mode)


def build_feature_map(x: torch.Tensor, mode: str, pca_dim: int, rp_dim: int, seed: int):
    if mode == "full":
        return FeatureMap("full")
    if mode == "pca":
        q = min(pca_dim, x.shape[1], x.shape[0] - 1)
        mean = x.mean(0, keepdim=True)
        xc = (x - mean).cpu()
        _, _, v = torch.pca_lowrank(xc, q=q, center=False, niter=4)
        return FeatureMap("pca", mean.cpu(), v[:, :q].T.contiguous().cpu(), None)
    if mode == "rp":
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + 999)
        r = torch.randn(rp_dim, x.shape[1], generator=gen) / math.sqrt(rp_dim)
        return FeatureMap("rp", None, None, r)
    raise ValueError(mode)


def diffuse_with_target(x0: torch.Tensor, t: float):
    gamma = math.exp(-t)
    eps = torch.randn_like(x0)
    xt = gamma * x0 + math.sqrt(max(1.0 - gamma * gamma, 1e-12)) * eps
    return xt, eps


class RandomFeatures(nn.Module):
    def __init__(self, d: int, p: int, activation: str = "erf", seed: int = 0):
        super().__init__()
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + 12345)
        self.register_buffer("W", torch.randn(p, d, generator=gen) / math.sqrt(d))
        self.register_buffer("b", 2 * math.pi * torch.rand(p, generator=gen))
        self.p = p
        self.activation = activation

    def forward(self, x):
        z = x @ self.W.T.to(x.device)
        if self.activation == "erf":
            h = torch.erf(z)
        elif self.activation == "tanh":
            h = torch.tanh(z)
        elif self.activation == "relu":
            h = F.relu(z)
        elif self.activation == "cos":
            h = torch.cos(z + self.b.to(x.device))
        else:
            raise ValueError(self.activation)
        return h / math.sqrt(self.p)


class RFExperts(nn.Module):
    def __init__(self, p: int, d: int, k: int, init_std: float):
        super().__init__()
        self.A = nn.Parameter(init_std * torch.randn(k, p, d))

    def forward(self, phi):
        return torch.einsum("bp,kpd->bkd", phi, self.A)


class Router(nn.Module):
    def __init__(self, p: int, k: int):
        super().__init__()
        self.fc = nn.Linear(p, k)

    def forward(self, phi):
        return self.fc(phi)


def mcl_loss(e, beta: float):
    k = e.shape[1]
    if beta <= 1e-12:
        q = torch.full_like(e, 1.0 / k)
        loss = e.mean()
    else:
        logits = -beta * e
        q = torch.softmax(logits, 1)
        loss = -(torch.logsumexp(logits, 1) - math.log(k)).mean() / beta
    return loss, q


@torch.no_grad()
def ridge_fit(phi, y, ridge: float):
    n, p = phi.shape
    c = (phi.T @ phi) / n + ridge * torch.eye(p, device=phi.device)
    b = (phi.T @ y) / n
    return torch.linalg.solve(c, b)


@torch.no_grad()
def whiten(phi, ridge: float):
    n, p = phi.shape
    c = (phi.T @ phi) / n + ridge * torch.eye(p, device=phi.device)
    ev, v = torch.linalg.eigh(c)
    return phi @ (v @ torch.diag(ev.clamp_min(ridge).rsqrt()) @ v.T)


@torch.no_grad()
def power_free(s, r, iters: int):
    if iters <= 0:
        return 0.0
    n, p = s.shape
    d = r.shape[1]
    b = torch.randn(p, d, device=s.device)
    b = b / b.norm().clamp_min(1e-12)
    lam = torch.tensor(0.0, device=s.device)
    for _ in range(iters):
        delta = s @ b
        a = (r * delta).sum(1, keepdim=True) / math.sqrt(d)
        m = a * r / math.sqrt(d)
        gb = (s.T @ m) / n
        lam = (b * gb).sum()
        b = gb / gb.norm().clamp_min(1e-12)
    return max(float(lam.cpu()), 0.0)


@torch.no_grad()
def lambda_dir(s, r, v):
    n = s.shape[0]
    a = (r @ v).pow(2)
    h = (s.T * a[None, :]) @ s / n
    return max(float(torch.linalg.eigvalsh(h)[-1].cpu()), 0.0)


@torch.no_grad()
def class_basis(y, labels, c: int):
    mats = []
    for ci in range(c):
        m = labels == ci
        if m.sum():
            mats.append(y[m].mean(0))
    m = torch.stack(mats)
    m = m - m.mean(0, keepdim=True)
    _, s, vh = torch.linalg.svd(m, full_matrices=False)
    rank = int((s > 1e-8 * s.max().clamp_min(1e-8)).sum())
    return vh[:rank].T.contiguous()


@torch.no_grad()
def calibrate(rf, x, y, labels, c: int, ridge: float, power_iters: int):
    phi = rf(x)
    a0 = ridge_fit(phi, y, ridge)
    r = y - phi @ a0
    s = whiten(phi, ridge)
    lf = power_free(s, r, power_iters)
    b = class_basis(y, labels, c).to(x.device)
    lc = max([lambda_dir(s, r, b[:, j]) for j in range(b.shape[1])] or [0.0])
    d = y.shape[1]
    pmat = b @ b.T if b.numel() else torch.zeros(d, d, device=x.device)
    vals = []
    for _ in range(12):
        v = torch.randn(d, device=x.device)
        v = v - pmat @ v
        if v.norm() > 1e-8:
            vals.append(lambda_dir(s, r, v / v.norm()))
    lt = max(vals or [0.0])
    beta = lambda l: 0.5 / max(l, 1e-12)
    e = r.pow(2).mean(1)
    v_emp = float(e.var(unbiased=False).cpu())
    alpha = math.log(max(2, x.shape[0])) / max(1, d)
    bg = math.sqrt(2 * alpha / max(v_emp, 1e-12))
    return dict(
        lambda_free=lf,
        lambda_class=lc,
        lambda_trans=lt,
        beta_free=beta(lf),
        beta_class=beta(lc),
        beta_trans=beta(lt),
        beta_glass_emp=bg,
        alpha_log_n_over_d=alpha,
        residual_mse=float(e.mean().cpu()),
        v_emp=v_emp,
    )


def sched(step: int, variant: str, beta_final: float, warm: int, ramp: int):
    if variant == "uniform":
        return 0.0
    if variant.startswith("fixed") or variant.startswith("grid") or variant == "hard_cold":
        return beta_final
    if step < warm:
        return 0.0
    u = min(1.0, max(0.0, (step - warm) / max(1, ramp)))
    u = u * u * (3 - 2 * u)
    return beta_final * u


@dataclass
class Params:
    data_root: str = "./data"
    classes: str = "automobile,horse"
    d_mode: str = "pca"
    pca_dim: int = 512
    rp_dim: int = 512
    p: int = 512
    K: int = 4
    t: float = 1.5
    n_train: int = 8000
    n_test: int = 2500
    n_calib: int = 2048
    batch_size: int = 192
    steps: int = 3000
    lr: float = 2e-3
    init_std: float = 1e-3
    activation: str = "erf"
    seed: int = 0
    warmup_steps: int = 600
    ramp_steps: int = 900
    eval_every: int = 200
    power_iters: int = 30
    ridge: float = 1e-5
    router_steps: int = 1000
    router_lr: float = 2e-3
    router_batch_size: int = 192
    beta_grid: bool = False
    quick: bool = False
    no_download: bool = False
    device: str = "auto"
    outdir: str = "./cifar_v5_router"


def train_variant(name: str, beta_final: float, params: Params, rf, data, device, c: int):
    ex = RFExperts(params.p, data["y_train"].shape[1], params.K, params.init_std).to(device)
    opt = torch.optim.AdamW(ex.parameters(), lr=params.lr)
    rows = []
    phi_train, phi_test = data["phi_train"], data["phi_test"]
    f0_train, f0_test = data["f0_train"], data["f0_test"]
    ytr, yte = data["y_train"], data["y_test"]
    ltr, lte = data["labels_train"], data["labels_test"]
    for step in range(params.steps + 1):
        if step % params.eval_every == 0 or step == params.steps:
            b = sched(step, name, beta_final, params.warmup_steps, params.ramp_steps)
            with torch.no_grad():
                pred_test = f0_test[:, None, :] + ex(phi_test)
                e_test = ((pred_test - yte[:, None, :]) ** 2).mean(-1)
                _, q_test = mcl_loss(e_test, b)
                row = {
                    "variant": name,
                    "step": step,
                    "beta": b,
                    "test_oracle_best_mse": float(e_test.min(1).values.mean().item()),
                    "test_soft_oracle_mse": float((q_test * e_test).sum(1).mean().item()),
                    "test_mean_expert_mse": float(e_test.mean(1).mean().item()),
                    "test_teacher_entropy_norm": float((-(q_test.clamp_min(1e-12) * q_test.clamp_min(1e-12).log()).sum(1).mean() / math.log(params.K)).item()),
                    "test_oracle_class_mi_norm": mi_norm(e_test.argmin(1), lte, params.K, c),
                }
                rows.append(row)
        if step == params.steps:
            break
        idx = torch.randint(0, phi_train.shape[0], (params.batch_size,), device=device)
        pred = f0_train[idx][:, None, :] + ex(phi_train[idx])
        e = ((pred - ytr[idx][:, None, :]) ** 2).mean(-1)
        b = sched(step, name, beta_final, params.warmup_steps, params.ramp_steps)
        loss, _ = mcl_loss(e, b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return ex, rows


def train_router(experts, phi, y, f0, beta: float, params: Params, device):
    router = Router(params.p, params.K).to(device)
    opt = torch.optim.AdamW(router.parameters(), lr=params.router_lr)
    experts.eval()
    for _ in range(params.router_steps):
        idx = torch.randint(0, phi.shape[0], (params.router_batch_size,), device=device)
        with torch.no_grad():
            pred = f0[idx][:, None, :] + experts(phi[idx])
            e = ((pred - y[idx][:, None, :]) ** 2).mean(-1)
            _, q = mcl_loss(e, beta)
        loss = -(q * torch.log_softmax(router(phi[idx]), 1)).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return router


@torch.no_grad()
def eval_router(experts, router, phi, y, labels, f0, beta: float, c: int, k: int):
    pred = f0[:, None, :] + experts(phi)
    e = ((pred - y[:, None, :]) ** 2).mean(-1)
    _, q = mcl_loss(e, beta)
    rq = torch.softmax(router(phi), 1)
    mix = torch.einsum("bk,bkd->bd", rq, pred)
    return dict(
        oracle_best_mse=float(e.min(1).values.mean().item()),
        soft_oracle_mse=float((q * e).sum(1).mean().item()),
        mean_expert_mse=float(e.mean(1).mean().item()),
        router_mix_mse=float(((mix - y) ** 2).mean(1).mean().item()),
        router_soft_mse=float((rq * e).sum(1).mean().item()),
        oracle_class_mi_norm=mi_norm(e.argmin(1), labels, k, c),
        router_class_mi_norm=mi_norm(rq.argmax(1), labels, k, c),
        oracle_usage_entropy=norm_entropy(torch.bincount(e.argmin(1), minlength=k).float().div(max(1, y.shape[0]))),
        router_usage_entropy=norm_entropy(torch.bincount(rq.argmax(1), minlength=k).float().div(max(1, y.shape[0]))),
        teacher_entropy_norm=float((-(q.clamp_min(1e-12) * q.clamp_min(1e-12).log()).sum(1).mean() / math.log(k)).item()),
        router_vs_teacher_ce=float((-(q.clamp_min(1e-12) * rq.clamp_min(1e-12).log()).sum(1).mean()).item()),
        router_vs_teacher_kl=float((q.clamp_min(1e-12) * (q.clamp_min(1e-12).log() - rq.clamp_min(1e-12).log())).sum(1).mean().item()),
    )


def write_csv(path: Path, rows):
    if not rows:
        return
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run(params: Params):
    if params.quick:
        params.n_train = min(params.n_train, 3000)
        params.n_test = min(params.n_test, 1000)
        params.n_calib = min(params.n_calib, 1000)
        params.steps = min(params.steps, 1200)
        params.eval_every = min(params.eval_every, 150)
        params.router_steps = min(params.router_steps, 500)
        params.p = min(params.p, 256)
        params.pca_dim = min(params.pca_dim, 256)
    set_seed(params.seed)
    device = device_of(params.device)
    out = ensure_dir(params.outdir)
    ids = parse_classes(params.classes)
    c = len(ids)
    xtr_raw, ytr, xte_raw, yte, info = load_cifar(params.data_root, ids, params.n_train, params.n_test, params.no_download, params.seed)
    fmap = build_feature_map(xtr_raw, params.d_mode, params.pca_dim, params.rp_dim, params.seed)
    x0 = fmap.transform(xtr_raw).float()
    xt0 = fmap.transform(xte_raw).float()
    lat_mean = x0.mean(0, keepdim=True)
    lat_std = x0.std().clamp_min(1e-6)
    x0 = (x0 - lat_mean) / lat_std
    xt0 = (xt0 - lat_mean) / lat_std
    xtr_t, eps_tr = diffuse_with_target(x0, params.t)
    xte_t, eps_te = diffuse_with_target(xt0, params.t)
    xtr = xtr_t.to(device)
    ytr_eps = eps_tr.to(device)
    ltr = ytr.to(device)
    xte = xte_t.to(device)
    yte_eps = eps_te.to(device)
    lte = yte.to(device)
    d = ytr_eps.shape[1]
    rf = RandomFeatures(d, params.p, params.activation, params.seed).to(device).eval()
    phi_tr = rf(xtr)
    phi_te = rf(xte)
    a0 = ridge_fit(phi_tr, ytr_eps, params.ridge)
    f0_tr = phi_tr @ a0
    f0_te = phi_te @ a0
    idx = torch.randperm(xtr.shape[0], device=device)[: min(params.n_calib, xtr.shape[0])]
    cal = calibrate(rf, xtr[idx], ytr_eps[idx], ltr[idx], c, params.ridge, params.power_iters)
    (out / "params.json").write_text(json.dumps(asdict(params), indent=2))
    info.update(d_mode=params.d_mode, d_latent=int(d), p=params.p, K=params.K, t=params.t, target="eps")
    (out / "data_info.json").write_text(json.dumps(info, indent=2))
    (out / "calibration.json").write_text(json.dumps(cal, indent=2))
    bc = cal["beta_class"]
    bg = cal["beta_glass_emp"]
    variants = [
        ("uniform", 0.0),
        ("fixed_class", max(0.2, 1.2 * bc)),
        ("fixed_good", min(max(0.2, 3.0 * bc), 0.45 * bg)),
        ("hard_cold", 1.2 * bg),
        ("theory_anneal", min(max(0.2, 3.0 * bc), 0.45 * bg)),
    ]
    if params.beta_grid:
        for b in [0.2, 0.4, 0.7, 1.1]:
            variants.append((f"grid_{b:.3f}", b))
    all_rows = []
    final_rows = []
    for name, bfin in variants:
        print(f"=== {name} beta={bfin:.6g} ===", flush=True)
        ex, rows = train_variant(name, bfin, params, rf, dict(phi_train=phi_tr, phi_test=phi_te, f0_train=f0_tr, f0_test=f0_te, y_train=ytr_eps, y_test=yte_eps, labels_train=ltr, labels_test=lte), device, c)
        all_rows.extend(rows)
        beval = bfin if name != "uniform" else max(0.2, bc)
        router = train_router(ex, phi_tr, ytr_eps, f0_tr, beval, params, device)
        final = eval_router(ex, router, phi_te, yte_eps, lte, f0_te, beval, c, params.K)
        final.update(variant=name, beta_eval=beval)
        final_rows.append(final)
        torch.save(dict(experts=ex.state_dict(), router=router.state_dict(), rf=rf.state_dict(), ridge_head=a0, params=asdict(params), data_info=info, calibration=cal, variant=name, beta_eval=beval), out / f"checkpoint_{name}.pt")
        write_csv(out / "metrics_all.csv", all_rows)
        write_csv(out / "router_final_metrics.csv", final_rows)
    if plt is not None and pd is not None:
        df = pd.DataFrame(all_rows)
        if not df.empty:
            for y in ["test_oracle_best_mse", "test_oracle_class_mi_norm", "test_teacher_entropy_norm"]:
                if y in df:
                    plt.figure()
                    for v, s in df.groupby("variant"):
                        plt.plot(s["step"], s[y], label=v)
                    plt.xlabel("step")
                    plt.ylabel(y)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(out / f"{y}.png", dpi=180)
                    plt.close()
    lines = ["RF-MCL CIFAR v5 router (target=eps, residualized)", json.dumps(asdict(params), indent=2), "\nData info:", json.dumps(info, indent=2), "\nCalibration:"]
    lines.extend([f"  {k}: {v}" for k, v in cal.items()])
    lines.append("\nFinal router metrics:")
    for r in final_rows:
        lines.append(f"\n[{r['variant']}] beta_eval={r['beta_eval']}")
        for k in ["oracle_best_mse", "soft_oracle_mse", "mean_expert_mse", "router_mix_mse", "router_soft_mse", "oracle_class_mi_norm", "router_class_mi_norm", "teacher_entropy_norm", "router_vs_teacher_ce", "router_vs_teacher_kl"]:
            lines.append(f"  {k}: {r[k]}")
        if r["teacher_entropy_norm"] > 0.95:
            lines.append("  teacher_status: near_uniform (router likely uninformative)")
        elif r["teacher_entropy_norm"] < 0.05:
            lines.append("  teacher_status: near_collapse")
    (out / "SUMMARY.txt").write_text("\n".join(lines))
    print("\n".join(lines))


def parse():
    p = argparse.ArgumentParser()
    for k, v in asdict(Params()).items():
        arg = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            p.add_argument(arg, action="store_true" if not v else "store_false")
        elif isinstance(v, int):
            p.add_argument(arg, type=int, default=v)
        elif isinstance(v, float):
            p.add_argument(arg, type=float, default=v)
        else:
            p.add_argument(arg, type=str, default=v)
    return Params(**vars(p.parse_args()))


if __name__ == "__main__":
    run(parse())


```
