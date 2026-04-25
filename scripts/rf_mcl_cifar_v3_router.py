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
