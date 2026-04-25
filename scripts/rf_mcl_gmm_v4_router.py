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
