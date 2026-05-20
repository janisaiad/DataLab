#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayes beta(t) and theoretical router tests for MNIST/CIFAR.

Tests the closed-form schedules
  beta_route_dot(t)=a_t/v_t,
  beta_x_mse(t)=d/(2 v_t),
  beta_eps_mse(t)=d/(2 tau_t^2),
under the isotropic latent GMM approximation
  z0 | c ~ N(m_c, sigma0^2 I),  x_t=a_t z0+b_t eps.

Also tests the deployable risk router
  k*(x_t,t)=argmin_k sum_c p(c|x_t) A[c,k](t)
with class-Bayes eps experts by default.
"""
from __future__ import annotations

import argparse, csv, json, math, random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

MNIST_CLASSES = [str(i) for i in range(10)]
CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def device_of(s:str):
    return torch.device("cuda" if s == "auto" and torch.cuda.is_available() else ("cpu" if s == "auto" else s))

def ensure_dir(p):
    p=Path(p); p.mkdir(parents=True, exist_ok=True); return p

def parse_floats(s:str)->List[float]:
    return [float(x.strip()) for x in s.split(',') if x.strip()]

def parse_classes(dataset:str, classes:str)->List[int]:
    if classes.strip().lower() in ("all", "*"):
        return list(range(10))
    names = MNIST_CLASSES if dataset.lower()=="mnist" else CIFAR10_CLASSES
    out=[]
    for z in [x.strip() for x in classes.split(',') if x.strip()]:
        out.append(int(z) if z.isdigit() else names.index(z))
    return out

def remap_labels(y:torch.Tensor, class_ids:List[int])->torch.Tensor:
    mp={c:i for i,c in enumerate(class_ids)}
    return torch.tensor([mp[int(v)] for v in y.tolist()], dtype=torch.long)

def write_csv(path:Path, rows:List[Dict]):
    if not rows: return
    keys=sorted(set().union(*[r.keys() for r in rows]))
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow(r)


def load_dataset(dataset:str, data_root:str, class_ids:List[int], n_train:int, n_test:int, no_download:bool, seed:int):
    if torchvision is None:
        raise ImportError("torchvision is required")
    dataset=dataset.lower(); class_set=set(class_ids)
    if dataset == "mnist":
        names=MNIST_CLASSES
        transform=T.Compose([T.ToTensor(), T.Normalize([0.5],[0.5])])
        tr=torchvision.datasets.MNIST(data_root, train=True, download=not no_download, transform=transform)
        te=torchvision.datasets.MNIST(data_root, train=False, download=not no_download, transform=transform)
    elif dataset == "cifar10":
        names=CIFAR10_CLASSES
        transform=T.Compose([T.ToTensor(), T.Normalize([0.5]*3,[0.5]*3)])
        tr=torchvision.datasets.CIFAR10(data_root, train=True, download=not no_download, transform=transform)
        te=torchvision.datasets.CIFAR10(data_root, train=False, download=not no_download, transform=transform)
    else:
        raise ValueError("dataset must be mnist or cifar10")

    def collect(ds, n:int, offset:int):
        order=list(range(len(ds))); rng=random.Random(seed+offset); rng.shuffle(order)
        xs=[]; ys=[]
        for i in order:
            x,y=ds[i]
            if int(y) not in class_set: continue
            xs.append(x.flatten()); ys.append(int(y))
            if n>0 and len(xs)>=n: break
        X=torch.stack(xs).float(); y=remap_labels(torch.tensor(ys), class_ids)
        return X,y

    Xtr,ytr=collect(tr,n_train,1); Xte,yte=collect(te,n_test,2)
    info=dict(dataset=dataset, classes=[names[i] for i in class_ids], raw_dim=int(Xtr.shape[1]), train_n=int(Xtr.shape[0]), test_n=int(Xte.shape[0]), normalization="[-1,1]")
    return Xtr,ytr,Xte,yte,info


@dataclass
class FeatureMap:
    mode:str
    mean:Optional[torch.Tensor]=None
    pca:Optional[torch.Tensor]=None
    rp:Optional[torch.Tensor]=None
    def transform(self, X:torch.Tensor)->torch.Tensor:
        if self.mode == "raw": return X
        if self.mode == "pca": return (X-self.mean.to(X.device)) @ self.pca.to(X.device).T
        if self.mode == "rp": return X @ self.rp.to(X.device).T
        raise ValueError(self.mode)

@torch.no_grad()
def build_feature_map(X:torch.Tensor, mode:str, pca_dim:int, rp_dim:int, seed:int)->FeatureMap:
    mode=mode.lower()
    if mode == "raw": return FeatureMap("raw")
    if mode == "pca":
        q=min(pca_dim, X.shape[1], X.shape[0]-1)
        mean=X.mean(0, keepdim=True); Xc=(X-mean).cpu()
        _,_,V=torch.pca_lowrank(Xc, q=q, center=False, niter=4)
        return FeatureMap("pca", mean.cpu(), V[:,:q].T.contiguous().cpu(), None)
    if mode == "rp":
        gen=torch.Generator(device="cpu"); gen.manual_seed(seed+999)
        R=torch.randn(rp_dim, X.shape[1], generator=gen)/math.sqrt(rp_dim)
        return FeatureMap("rp", None, None, R.cpu())
    raise ValueError("feature_mode must be raw,pca,rp")

@torch.no_grad()
def standardize(Ztr:torch.Tensor, Zte:torch.Tensor):
    mu=Ztr.mean(0, keepdim=True); std=Ztr.std(0, keepdim=True).clamp_min(1e-6)
    return (Ztr-mu)/std, (Zte-mu)/std, mu.cpu(), std.cpu()

@dataclass
class IsoGMM:
    means:torch.Tensor
    priors:torch.Tensor
    sigma2:float
    @property
    def C(self): return int(self.means.shape[0])
    @property
    def d(self): return int(self.means.shape[1])

@torch.no_grad()
def fit_iso_gmm(Z:torch.Tensor, y:torch.Tensor, C:int)->IsoGMM:
    means=[]; priors=[]; ss=0.0; count=0
    for c in range(C):
        m=(y==c); Zc=Z[m]
        if Zc.numel()==0: raise ValueError(f"empty class {c}")
        mu=Zc.mean(0); means.append(mu); priors.append(float(m.float().mean().item()))
        ss += float(((Zc-mu)**2).sum().item()); count += int(Zc.numel())
    pri=torch.tensor(priors).float(); pri=pri/pri.sum()
    return IsoGMM(torch.stack(means), pri, ss/max(1,count))


def diffusion_coeffs(t:float):
    a=math.exp(-t); b=math.sqrt(max(1-a*a, 1e-12)); return a,b

def beta_info(t:float, d:int, sigma2:float)->Dict[str,float]:
    a,b=diffusion_coeffs(t); v=b*b+a*a*sigma2; tau2=a*a*sigma2/max(v,1e-30)
    return dict(a=a,b=b,v=v,tau2=tau2,
                beta_route_dot=a/max(v,1e-30),
                beta_x_mse=d/(2*max(v,1e-30)),
                beta_eps_mse=d/(2*max(tau2,1e-30)),
                snr_class=a*a/max(v,1e-30))

@torch.no_grad()
def diffuse(Z0:torch.Tensor, t:float, seed:int, device:torch.device):
    gen=torch.Generator(device=device); gen.manual_seed(seed)
    Z0=Z0.to(device); eps=torch.randn(Z0.shape, generator=gen, device=device)
    a,b=diffusion_coeffs(t); return a*Z0+b*eps, eps

@torch.no_grad()
def log_px_iso(Xt:torch.Tensor, gmm:IsoGMM, t:float)->torch.Tensor:
    means=gmm.means.to(Xt.device); pri=gmm.priors.to(Xt.device).clamp_min(1e-30)
    a,b=diffusion_coeffs(t); v=b*b+a*a*gmm.sigma2
    dist2=((Xt[:,None,:]-a*means[None,:,:])**2).sum(-1)
    logits=pri.log()[None,:]-0.5*dist2/max(v,1e-30)
    return logits - torch.logsumexp(logits,1,keepdim=True)

@torch.no_grad()
def log_px_beta_energy(Xt:torch.Tensor, gmm:IsoGMM, t:float, beta:float)->torch.Tensor:
    means=gmm.means.to(Xt.device); pri=gmm.priors.to(Xt.device).clamp_min(1e-30)
    a,_=diffusion_coeffs(t)
    E=((Xt[:,None,:]-a*means[None,:,:])**2).mean(-1)
    logits=pri.log()[None,:]-float(beta)*E
    return logits - torch.logsumexp(logits,1,keepdim=True)

@torch.no_grad()
def eps_means(Xt:torch.Tensor, gmm:IsoGMM, t:float)->torch.Tensor:
    means=gmm.means.to(Xt.device); a,b=diffusion_coeffs(t); v=b*b+a*a*gmm.sigma2
    return (b/max(v,1e-30))*(Xt[:,None,:]-a*means[None,:,:])

@torch.no_grad()
def log_qeps(Xt:torch.Tensor, eps:torch.Tensor, gmm:IsoGMM, t:float, include_px_prior:bool=True, beta_override:Optional[float]=None)->torch.Tensor:
    E=((eps[:,None,:]-eps_means(Xt,gmm,t))**2).mean(-1)
    beta=beta_info(t,gmm.d,gmm.sigma2)["beta_eps_mse"] if beta_override is None else float(beta_override)
    logits=-beta*E
    if include_px_prior:
        logits=logits+log_px_iso(Xt,gmm,t)
    else:
        logits=logits+gmm.priors.to(Xt.device).clamp_min(1e-30).log()[None,:]
    return logits - torch.logsumexp(logits,1,keepdim=True)

@torch.no_grad()
def acc(logp:torch.Tensor, y:torch.Tensor)->float:
    return float((logp.argmax(1).cpu()==y.cpu()).float().mean().item())

@torch.no_grad()
def nll(logp:torch.Tensor, y:torch.Tensor)->float:
    yd=y.to(logp.device); return float((-logp[torch.arange(yd.numel(),device=logp.device),yd]).mean().item())

@torch.no_grad()
def entropy_norm(logp:torch.Tensor)->float:
    p=logp.exp().clamp_min(1e-30); return float((-(p*p.log()).sum(1).mean()/math.log(p.shape[1])).item())

@torch.no_grad()
def ece(logp:torch.Tensor, y:torch.Tensor, bins:int=15)->float:
    p=logp.exp(); conf,pred=p.max(1); corr=(pred.cpu()==y.cpu()).to(conf.device).float()
    out=torch.tensor(0.0,device=p.device); edges=torch.linspace(0,1,bins+1,device=p.device)
    for i in range(bins):
        m=(conf>=edges[i]) & ((conf<edges[i+1]) if i<bins-1 else (conf<=edges[i+1]))
        if m.any(): out += m.float().mean()*(conf[m].mean()-corr[m].mean()).abs()
    return float(out.item())

@torch.no_grad()
def kl(logq:torch.Tensor, logp:torch.Tensor)->float:
    q=logq.exp().clamp_min(1e-30); return float((q*(logq-logp)).sum(1).mean().item())

@torch.no_grad()
def losses_class_bayes(Xt:torch.Tensor, eps:torch.Tensor, gmm:IsoGMM, t:float)->torch.Tensor:
    return ((eps[:,None,:]-eps_means(Xt,gmm,t))**2).mean(-1)

@torch.no_grad()
def empirical_A(losses:torch.Tensor, y:torch.Tensor, C:int)->torch.Tensor:
    y=y.to(losses.device); A=torch.zeros(C,losses.shape[1],device=losses.device)
    for c in range(C):
        m=(y==c); A[c]=losses[m].mean(0) if m.any() else losses.mean(0)
    return A

@torch.no_grad()
def analytic_risk(Xt:torch.Tensor, gmm:IsoGMM, t:float, logp:torch.Tensor)->torch.Tensor:
    mu=eps_means(Xt,gmm,t); p=logp.exp(); tau2=beta_info(t,gmm.d,gmm.sigma2)["tau2"]
    D=((mu[:,:,None,:]-mu[:,None,:,:])**2).mean(-1)
    return tau2 + torch.einsum('nc,nck->nk',p,D)

@torch.no_grad()
def route_metrics(name:str, route:torch.Tensor, losses:torch.Tensor, oracle:torch.Tensor, y:torch.Tensor)->Dict[str,float]:
    chosen=losses[torch.arange(losses.shape[0],device=losses.device), route.to(losses.device)]
    best=losses.min(1).values
    return {f"{name}_mse":float(chosen.mean().item()),
            f"{name}_excess_vs_oracle":float((chosen-best).mean().item()),
            f"{name}_agree_oracle":float((route.cpu()==oracle.cpu()).float().mean().item()),
            f"{name}_class_acc":float((route.cpu()==y.cpu()).float().mean().item())}


def stratified_subset(Z:torch.Tensor,y:torch.Tensor,n:int,seed:int):
    if n<=0 or n>=Z.shape[0]: return Z,y
    gen=torch.Generator(device='cpu'); gen.manual_seed(seed); idx=[]; C=int(y.max())+1; per=max(1,n//C)
    for c in range(C):
        ids=torch.where(y==c)[0]; ids=ids[torch.randperm(ids.numel(),generator=gen)[:per]]; idx.append(ids)
    idx=torch.cat(idx); idx=idx[torch.randperm(idx.numel(),generator=gen)[:n]]
    return Z[idx], y[idx]

@dataclass
class Params:
    dataset:str="mnist"
    data_root:str="./data"
    classes:str="all"
    feature_mode:str="pca"
    pca_dim:int=64
    rp_dim:int=128
    n_train:int=20000
    n_test:int=5000
    train_for_A:int=5000
    times:str="0.2,0.5,0.85,1.1,1.5,2.05,2.5,3.0"
    beta_grid_mult_min:float=0.05
    beta_grid_mult_max:float=20.0
    beta_grid_points:int=61
    seed:int=0
    device:str="auto"
    no_download:bool=False
    outdir:str="./outputs/theory_beta_router_tests"


def run(P:Params):
    set_seed(P.seed); device=device_of(P.device); out=ensure_dir(P.outdir)
    class_ids=parse_classes(P.dataset,P.classes); C=len(class_ids)
    Xtr,ytr,Xte,yte,info=load_dataset(P.dataset,P.data_root,class_ids,P.n_train,P.n_test,P.no_download,P.seed)
    fmap=build_feature_map(Xtr,P.feature_mode,P.pca_dim,P.rp_dim,P.seed)
    Ztr=fmap.transform(Xtr).float(); Zte=fmap.transform(Xte).float(); Ztr,Zte,lat_mu,lat_std=standardize(Ztr,Zte)
    gmm=fit_iso_gmm(Ztr,ytr,C); d=gmm.d; times=parse_floats(P.times)
    Z_A,y_A=stratified_subset(Ztr,ytr,P.train_for_A,P.seed+77)

    info.update(dict(feature_mode=P.feature_mode, latent_dim=d, C=C, sigma2_iso=gmm.sigma2, priors=gmm.priors.tolist(), times=times))
    (out/'params.json').write_text(json.dumps(asdict(P),indent=2),encoding='utf-8')
    (out/'data_info.json').write_text(json.dumps(info,indent=2),encoding='utf-8')

    summary=[]; sweep_x=[]; sweep_eps=[]; routing=[]
    mults=np.geomspace(P.beta_grid_mult_min,P.beta_grid_mult_max,P.beta_grid_points)
    for j,t in enumerate(times):
        binfo=beta_info(t,d,gmm.sigma2)
        Xt,eps=diffuse(Zte,t,P.seed+1000+j,device); ytd=yte.to(device)
        XtA,epsA=diffuse(Z_A,t,P.seed+2000+j,device); yAd=y_A.to(device)
        px=log_px_iso(Xt,gmm,t); qeps=log_qeps(Xt,eps,gmm,t,True); qeps_nopx=log_qeps(Xt,eps,gmm,t,False)
        row=dict(dataset=P.dataset,feature_mode=P.feature_mode,t=t,d=d,C=C,sigma2_iso=gmm.sigma2,**binfo)
        row.update(px_acc=acc(px,yte),px_nll=nll(px,yte),px_ece=ece(px,yte),px_entropy=entropy_norm(px))
        row.update(qeps_acc=acc(qeps,yte),qeps_nll=nll(qeps,yte),qeps_ece=ece(qeps,yte),qeps_entropy=entropy_norm(qeps))
        row.update(qeps_nopx_acc=acc(qeps_nopx,yte),qeps_nopx_nll=nll(qeps_nopx,yte),qeps_nopx_entropy=entropy_norm(qeps_nopx))
        row['kl_qeps_to_px']=kl(qeps,px); row['kl_qeps_nopx_to_px']=kl(qeps_nopx,px)

        best=(1e99,None,None,None)
        for m in mults:
            beta=float(binfo['beta_x_mse']*m); lp=log_px_beta_energy(Xt,gmm,t,beta)
            rr=dict(dataset=P.dataset,feature_mode=P.feature_mode,t=t,beta=beta,beta_theory=binfo['beta_x_mse'],mult=float(m),nll=nll(lp,yte),acc=acc(lp,yte),ece=ece(lp,yte))
            sweep_x.append(rr)
            if rr['nll']<best[0]: best=(rr['nll'],beta,rr['acc'],rr['ece'])
        row['beta_x_emp_nll_min']=best[1]; row['beta_x_emp_over_theory']=best[1]/binfo['beta_x_mse']; row['beta_x_emp_nll']=best[0]; row['beta_x_emp_acc']=best[2]; row['beta_x_emp_ece']=best[3]

        best=(1e99,None,None,None)
        for m in mults:
            beta=float(binfo['beta_eps_mse']*m); lq=log_qeps(Xt,eps,gmm,t,True,beta)
            rr=dict(dataset=P.dataset,feature_mode=P.feature_mode,t=t,beta=beta,beta_theory=binfo['beta_eps_mse'],mult=float(m),nll=nll(lq,yte),acc=acc(lq,yte),ece=ece(lq,yte))
            sweep_eps.append(rr)
            if rr['nll']<best[0]: best=(rr['nll'],beta,rr['acc'],rr['ece'])
        row['beta_eps_emp_nll_min']=best[1]; row['beta_eps_emp_over_theory']=best[1]/binfo['beta_eps_mse']; row['beta_eps_emp_nll']=best[0]; row['beta_eps_emp_acc']=best[2]; row['beta_eps_emp_ece']=best[3]

        LA=losses_class_bayes(XtA,epsA,gmm,t); A=empirical_A(LA,yAd,C)
        L=losses_class_bayes(Xt,eps,gmm,t); oracle=L.argmin(1)
        r_px=px.argmax(1); r_emp=(px.exp()@A).argmin(1); r_an=analytic_risk(Xt,gmm,t,px).argmin(1); r_q=qeps.argmax(1)
        rt=dict(dataset=P.dataset,feature_mode=P.feature_mode,t=t,d=d,C=C,A_diag=float(A.diag().mean().item()),A_offdiag=float(((A.sum()-A.diag().sum())/max(1,C*C-C)).item()),oracle_mse=float(L.min(1).values.mean().item()),mean_expert_mse=float(L.mean(1).mean().item()))
        rt.update(route_metrics('posterior_argmax',r_px,L,oracle,yte))
        rt.update(route_metrics('risk_empA',r_emp,L,oracle,yte))
        rt.update(route_metrics('risk_analytic',r_an,L,oracle,yte))
        rt.update(route_metrics('qeps_argmax',r_q,L,oracle,yte))
        routing.append(rt)
        for k,v in rt.items():
            if isinstance(v,(int,float)): row['route_'+k]=v
        summary.append(row)
        print(f"[{P.dataset} t={t:.3g}] px_acc={row['px_acc']:.4f} qeps_acc={row['qeps_acc']:.4f} KL={row['kl_qeps_to_px']:.3g} beta_x_emp/theory={row['beta_x_emp_over_theory']:.3g} risk_excess={rt['risk_empA_excess_vs_oracle']:.3e}", flush=True)

    write_csv(out/'summary_by_t.csv',summary); write_csv(out/'beta_sweep_x.csv',sweep_x); write_csv(out/'beta_sweep_eps.csv',sweep_eps); write_csv(out/'routing_by_t.csv',routing)

    if pd is not None and plt is not None:
        df=pd.DataFrame(summary)
        plots=[('px_acc','deployable p(c|x_t) acc','px_acc_vs_t.png'),('qeps_acc','oracle p(c|x_t,eps) acc','qeps_acc_vs_t.png'),('kl_qeps_to_px','KL oracle||router','kl_oracle_to_router_vs_t.png'),('beta_x_emp_over_theory','empirical beta_x / theory','beta_x_ratio_vs_t.png'),('route_risk_empA_excess_vs_oracle','risk-router excess MSE','risk_excess_vs_t.png')]
        for col,ylabel,name in plots:
            if col in df:
                plt.figure(); plt.plot(df['t'],df[col],marker='o'); plt.xlabel('t'); plt.ylabel(ylabel); plt.tight_layout(); plt.savefig(out/name,dpi=180); plt.close()
        sx=pd.DataFrame(sweep_x)
        for t in times:
            sub=sx[np.isclose(sx['t'],t)]
            if not sub.empty:
                plt.figure(); plt.semilogx(sub['mult'],sub['nll'],marker='o'); plt.axvline(1.0,linestyle='--'); plt.xlabel('beta/beta_x(t)'); plt.ylabel('NLL'); plt.title(f'{P.dataset} beta_x sweep t={t:g}'); plt.tight_layout(); plt.savefig(out/f'beta_x_sweep_t_{t:g}.png',dpi=160); plt.close()

    lines=['# Bayes beta(t) + router tests','', '## Data','```json',json.dumps(info,indent=2),'```','', '## Readout','', '| t | px acc | qeps acc | KL(qeps||px) | beta_x emp/theory | risk excess |','|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['t']:.3g} | {r['px_acc']:.4f} | {r['qeps_acc']:.4f} | {r['kl_qeps_to_px']:.4g} | {r['beta_x_emp_over_theory']:.4g} | {r['route_risk_empA_excess_vs_oracle']:.4g} |")
    lines += ['', '## Files', '- `summary_by_t.csv`', '- `beta_sweep_x.csv`', '- `beta_sweep_eps.csv`', '- `routing_by_t.csv`', '- plots `*.png` if matplotlib/pandas are installed']
    (out/'README_SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f"\nSaved outputs to {out.resolve()}")


def parse()->Params:
    p=argparse.ArgumentParser()
    for k,v in asdict(Params()).items():
        arg='--'+k.replace('_','-')
        if isinstance(v,bool): p.add_argument(arg, action='store_false' if v else 'store_true', default=v)
        elif isinstance(v,int): p.add_argument(arg,type=int,default=v)
        elif isinstance(v,float): p.add_argument(arg,type=float,default=v)
        else: p.add_argument(arg,type=str,default=v)
    return Params(**vars(p.parse_args()))

if __name__=='__main__':
    run(parse())
