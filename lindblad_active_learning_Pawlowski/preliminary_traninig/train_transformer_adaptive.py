
"""
train_transformer_adaptive_phys.py

Train the 80-setting Transformer on the DEPLOYMENT distribution:

  25% uniform
  25% static global Bayesian A-optimal
  50% closed-loop adaptive:
      Bayesian warmup -> theta_hat -> accumulated local Fisher -> new shots -> ...

Loss:lind
    L = L_supervised + lambda_phys * L_binomial_physics

The physics loss uses differentiable Lindblad propagation at theta_hat and the
ACTUAL accumulated (n_m, N_m), so unequal allocations are handled correctly.

No oracle allocations are used for training.

Expected files:
  lindblad_probability_pool.pt
  oracle_fisher_settings_pool.pt

Typical run:
  python train_transformer_adaptive_phys.py --epochs 100 --n-test 512

Faster sanity:
  python train_transformer_adaptive_phys.py --epochs 20 --steps-per-epoch 40 --n-test 128
"""

import argparse, copy, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import MID, HALF, LOW, HIGH

HBAR_MEV_PS = 0.6582119569
PSI_NP = [
    -0.0245922 - 0.22576801j,
     0.46178448 + 0.25162446j,
    -0.49144100 + 0.13995686j,
    -0.49401122 - 0.40830746j,
]


def theta_to_z(theta):
    return (theta - MID.to(theta.device)) / HALF.to(theta.device)

def z_to_theta(z):
    return MID.to(z.device) + HALF.to(z.device) * z


def load_pool(fname):
    o = torch.load(fname, map_location="cpu", weights_only=False)
    if isinstance(o, dict):
        theta = next((o[k] for k in ("theta","theta_pool","thetas") if k in o), None)
        p = next((o[k] for k in ("p","p_pool","probs","probabilities") if k in o), None)
    else:
        theta,p = o[:2]
    if theta is None or p is None:
        raise KeyError("Cannot identify theta/probability tensors.")
    return torch.as_tensor(theta).float(), torch.as_tensor(p).float()


class MeasurementTransformer(nn.Module):
    def __init__(self,S,d=128):
        super().__init__()
        self.value_embed=nn.Sequential(nn.Linear(2,d),nn.SiLU(),nn.Linear(d,d))
        self.setting_embed=nn.Embedding(S,d)
        layer=nn.TransformerEncoderLayer(
            d_model=d,nhead=8,dim_feedforward=256,dropout=.05,
            batch_first=True,norm_first=True,activation="gelu")
        self.encoder=nn.TransformerEncoder(layer,3,enable_nested_tensor=False)
        self.head=nn.Sequential(
            nn.LayerNorm(d),nn.Linear(d,128),nn.SiLU(),
            nn.Linear(128,5),nn.Tanh())

    def forward(self,x):
        ids=torch.arange(x.shape[1],device=x.device)
        h=self.value_embed(x)+self.setting_embed(ids)[None]
        return self.head(self.encoder(h).mean(1))


def sample_counts(p,N):
    return torch.distributions.Binomial(
        total_count=N.float(),probs=p).sample()


def features_from_counts(n,N,K,p_mean,p_std):
    Nf=N.float()
    phat=torch.where(N>0,n/Nf.clamp_min(1),torch.zeros_like(n))
    x=(phat-p_mean.to(n.device))/p_std.to(n.device)
    x=torch.where(N>0,x,torch.zeros_like(x))
    return torch.stack([x,Nf/float(K)],-1)


def equal_counts(B,S,K,device):
    N=torch.full((B,S),K//S,dtype=torch.long,device=device)
    if K%S:
        N[:,:K%S]+=1
    return N


@torch.no_grad()
def weights_to_counts(w,K):
    if w.ndim==1:
        w=w[None]
    raw=w*K
    N=torch.floor(raw).long()
    rem=K-N.sum(1)
    frac=raw-N.float()
    order=torch.argsort(frac,dim=1,descending=True)
    for b in range(len(N)):
        r=int(rem[b].item())
        if r:
            N[b,order[b,:r]]+=1
    return N


@torch.no_grad()
def bayesian_aopt_weights(Ftrain,K,prior,iters,explore,chunk):
    M,S=Ftrain.shape[:2]
    dev=Ftrain.device
    w=torch.full((S,),1/S,device=dev)
    I=torch.eye(5,device=dev)
    for _ in range(iters):
        gain=torch.zeros(S,device=dev)
        for lo in range(0,M,chunk):
            Fc=Ftrain[lo:lo+chunk]
            FF=prior*I[None]+K*(w[None,:,None,None]*Fc).sum(1)
            Fi=torch.linalg.pinv(FF,rcond=1e-6)
            A=Fi[:,None]@Fc@Fi[:,None]
            gain += torch.diagonal(A,dim1=-2,dim2=-1).sum(-1).sum(0)
        gain=(gain/M).clamp_min(1e-20)
        w *= torch.sqrt(gain/gain.mean())
        w /= w.sum()
    w=(1-explore)*w+explore/S
    return w/w.sum()


@torch.no_grad()
def sequential_local_aopt_weights(Fset,Nold,deltaK,prior,iters):
    B,S=Fset.shape[:2]
    dev=Fset.device
    I=torch.eye(5,device=dev,dtype=Fset.dtype)[None]
    Fold=prior*I+(Nold.float()[:,:,None,None]*Fset).sum(1)
    w=torch.full((B,S),1/S,device=dev,dtype=Fset.dtype)
    for _ in range(iters):
        FF=Fold+deltaK*(w[:,:,None,None]*Fset).sum(1)
        Fi=torch.linalg.pinv(FF,rcond=1e-6)
        A=Fi[:,None]@Fset@Fi[:,None]
        gain=torch.diagonal(A,dim1=-2,dim2=-1).sum(-1).clamp_min(1e-20)
        w *= torch.sqrt(gain/gain.mean(1,keepdim=True))
        w /= w.sum(1,keepdim=True)
    return w


@torch.no_grad()
def nearest_fisher(theta_hat,theta_train,Ftrain):
    zh=theta_to_z(theta_hat).cpu()
    zt=theta_to_z(theta_train).cpu()
    out=[]
    for b in range(len(zh)):
        j=torch.argmin(((zt-zh[b])**2).sum(1))
        out.append(Ftrain[j])
    return torch.stack(out)


def build_system(device,J_meV=0.0):
    cd=torch.complex64
    I=torch.eye(2,dtype=cd,device=device)
    Z=torch.tensor([[1,0],[0,-1]],dtype=cd,device=device)
    SM=torch.tensor([[0,1],[0,0]],dtype=cd,device=device)
    L=torch.stack([
        torch.kron(SM,I),torch.kron(I,SM),
        torch.kron(Z,I),torch.kron(I,Z),
        torch.kron(Z,I)+torch.kron(I,Z)])
    Hs=torch.tensor([1.,0.],dtype=cd,device=device)
    Vs=torch.tensor([0.,1.],dtype=cd,device=device)
    Ds=torch.tensor([1.,1.],dtype=cd,device=device)/math.sqrt(2)
    Rs=torch.tensor([1.,-1j],dtype=cd,device=device)/math.sqrt(2)
    P=[]
    for a in (Hs,Vs,Ds,Rs):
        for b in (Hs,Vs,Ds,Rs):
            q=torch.kron(a,b)
            P.append(torch.outer(q,q.conj()))
    P=torch.stack(P)
    psi=torch.tensor(PSI_NP,dtype=cd,device=device)
    psi=psi/torch.linalg.vector_norm(psi)
    rho0=torch.outer(psi,psi.conj())
    M=torch.tensor([[1,0,0,0],[0,-1,2,0],[0,2,-1,0],[0,0,0,1]],
                   dtype=cd,device=device)
    H=(J_meV/(4*HBAR_MEV_PS))*M
    return H,L,P,rho0


def theta_to_gamma(theta):
    # Same convention as the probability/Fisher cache.
    g=torch.empty_like(theta)
    g[:,0]=1/theta[:,0]
    g[:,1]=1/theta[:,1]
    g[:,2]=1/(2*theta[:,2])
    g[:,3]=1/(2*theta[:,3])
    g[:,4]=1/theta[:,4]
    return g


def rhs(rho,gamma,H,Lops):
    out=-1j*(H@rho-rho@H)
    for k in range(5):
        L=Lops[k]
        Ld=L.conj().T
        LdL=Ld@L
        D=L@rho@Ld-.5*(LdL@rho+rho@LdL)
        out=out+gamma[:,k,None,None]*D
    return out


def evolve_differentiable(theta,t,H,Lops,rho0,max_step):
    B=len(theta)
    rho=rho0[None].expand(B,-1,-1).clone()
    gamma=theta_to_gamma(theta)
    nsteps=max(1,int(math.ceil(float(t)/max_step)))
    h=float(t)/nsteps
    for _ in range(nsteps):
        k1=rhs(rho,gamma,H,Lops)
        k2=rhs(rho+.5*h*k1,gamma,H,Lops)
        k3=rhs(rho+.5*h*k2,gamma,H,Lops)
        k4=rhs(rho+h*k3,gamma,H,Lops)
        rho=rho+h*(k1+2*k2+2*k3+k4)/6
    return .5*(rho+rho.conj().transpose(-1,-2))


def probs_differentiable(theta,times,H,Lops,rho0,P,max_step):
    chunks=[]
    for t in times:
        rho=evolve_differentiable(theta,t,H,Lops,rho0,max_step)
        pp=torch.einsum("kij,bji->bk",P,rho).real
        chunks.append(pp.clamp(1e-6,1-1e-6))
    return torch.cat(chunks,1)


def binomial_physics_nll(theta_pred,n,N,system,times,max_step):
    H,Lops,P,rho0=system
    pp=probs_differentiable(theta_pred,times,H,Lops,rho0,P,max_step)
    nf=N.float()
    # Per-shot mean NLL: stable across different total K.
    ll=n*torch.log(pp)+(nf-n)*torch.log1p(-pp)
    return -ll.sum()/nf.sum().clamp_min(1.0)


@torch.no_grad()
def make_static_batch(pp,K,kind,bayes_counts,device):
    B,S=pp.shape
    if kind=="uniform":
        N=equal_counts(B,S,K,device)
    else:
        N=bayes_counts[K].to(device)[None].repeat(B,1)
    n=sample_counts(pp,N)
    return n,N


@torch.no_grad()
def make_adaptive_batch(model,pp,targetK,theta_train,Ftrain,bayes_counts,
                        bayes_weights,p_mean,p_std,args,device):
    """
    Generate the SAME closed-loop family used at deployment.
    The current model generates theta_hat, but no gradient is propagated through
    allocation decisions.
    """
    B,S=pp.shape
    warm=min(args.warmup,targetK)
    Ntot=bayes_counts[warm].to(device)[None].repeat(B,1).clone()
    ntot=sample_counts(pp,Ntot)

    if targetK<=warm:
        return ntot,Ntot

    pred=z_to_theta(model(features_from_counts(
        ntot,Ntot,warm,p_mean,p_std))).detach()

    prev=warm
    for K in args.adaptive_Ks:
        if K<=warm or K>targetK:
            continue
        delta=K-prev
        Flocal=nearest_fisher(pred.cpu(),theta_train,Ftrain).to(device)
        wl=sequential_local_aopt_weights(
            Flocal,Ntot,delta,args.prior_precision,args.design_iters)
        wb=bayes_weights[K].to(device)[None].repeat(B,1)
        wnew=(1-args.epsilon)*wl+args.epsilon*wb
        wnew/=wnew.sum(1,keepdim=True)
        Nnew=weights_to_counts(wnew,delta).to(device)
        ntot += sample_counts(pp,Nnew)
        Ntot += Nnew
        pred=z_to_theta(model(features_from_counts(
            ntot,Ntot,K,p_mean,p_std))).detach()
        prev=K

    # If targetK is not itself an adaptive checkpoint, finish one last batch.
    if prev<targetK:
        delta=targetK-prev
        Flocal=nearest_fisher(pred.cpu(),theta_train,Ftrain).to(device)
        wl=sequential_local_aopt_weights(
            Flocal,Ntot,delta,args.prior_precision,args.design_iters)
        wb=bayes_weights[targetK].to(device)[None].repeat(B,1)
        wnew=(1-args.epsilon)*wl+args.epsilon*wb
        wnew/=wnew.sum(1,keepdim=True)
        Nnew=weights_to_counts(wnew,delta).to(device)
        ntot += sample_counts(pp,Nnew)
        Ntot += Nnew

    return ntot,Ntot


@torch.no_grad()
def make_oracle_batch(model,pp,Ftrue,targetK,bayes_counts,bayes_weights,
                      p_mean,p_std,args,device):
    """
    Diagnostic only: same sequential accumulated-Fisher experiment as adaptive,
    but the local Fisher tensor is evaluated at theta_true (Ftrue).

    Oracle is NEVER used for training.
    """
    B,S=pp.shape
    warm=min(args.warmup,targetK)
    Ntot=bayes_counts[warm].to(device)[None].repeat(B,1).clone()
    ntot=sample_counts(pp,Ntot)

    if targetK<=warm:
        return ntot,Ntot

    prev=warm
    for K in args.adaptive_Ks:
        if K<=warm or K>targetK:
            continue
        delta=K-prev
        wl=sequential_local_aopt_weights(
            Ftrue.to(device),Ntot,delta,args.prior_precision,args.design_iters)
        wb=bayes_weights[K].to(device)[None].repeat(B,1)
        wnew=(1-args.epsilon)*wl+args.epsilon*wb
        wnew/=wnew.sum(1,keepdim=True)
        Nnew=weights_to_counts(wnew,delta).to(device)
        ntot += sample_counts(pp,Nnew)
        Ntot += Nnew
        prev=K

    if prev<targetK:
        delta=targetK-prev
        wl=sequential_local_aopt_weights(
            Ftrue.to(device),Ntot,delta,args.prior_precision,args.design_iters)
        wb=bayes_weights[targetK].to(device)[None].repeat(B,1)
        wnew=(1-args.epsilon)*wl+args.epsilon*wb
        wnew/=wnew.sum(1,keepdim=True)
        Nnew=weights_to_counts(wnew,delta).to(device)
        ntot += sample_counts(pp,Nnew)
        Ntot += Nnew

    return ntot,Ntot


@torch.no_grad()
def true_fisher_ascore(Ftrue,N,prior):
    """Mean/median Tr[(prior I + sum_m N_m F_m(theta_true))^-1]."""
    Ftrue=Ftrue.float().cpu()
    N=N.float().cpu()
    I=torch.eye(5,dtype=Ftrue.dtype)[None]
    Ft=prior*I+(N[:,:,None,None]*Ftrue).sum(1)
    Fi=torch.linalg.pinv(Ft,rcond=1e-6)
    A=torch.diagonal(Fi,dim1=-2,dim2=-1).sum(-1)
    return A.mean().item(),A.median().item()


def score(true,pred):
    rm=torch.sqrt(((pred-true)**2).mean(0))
    prior=(HIGH-LOW)/math.sqrt(12.)
    s=torch.sqrt(((rm/prior)**2).mean()).item()
    return s,rm


@torch.no_grad()
def evaluate(model,theta,p,Ftrue,K,mode,bayes_counts,bayes_weights,
             theta_train,Ftrain,p_mean,p_std,args,device):
    model.eval()
    preds=[]
    Ns=[]
    for lo in range(0,len(theta),args.eval_batch):
        th=theta[lo:lo+args.eval_batch]
        pp=p[lo:lo+args.eval_batch].to(device)
        Ft=Ftrue[lo:lo+args.eval_batch]
        if mode in ("uniform","bayes"):
            n,N=make_static_batch(pp,K,mode,bayes_counts,device)
        elif mode=="adaptive":
            n,N=make_adaptive_batch(
                model,pp,K,theta_train,Ftrain,bayes_counts,bayes_weights,
                p_mean,p_std,args,device)
        elif mode=="oracle":
            n,N=make_oracle_batch(
                model,pp,Ft,K,bayes_counts,bayes_weights,
                p_mean,p_std,args,device)
        else:
            raise ValueError(mode)
        x=features_from_counts(n,N,K,p_mean,p_std)
        preds.append(z_to_theta(model(x)).cpu())
        Ns.append(N.cpu())
    return score(theta,torch.cat(preds)),torch.cat(Ns)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache",default="lindblad_probability_pool.pt")
    ap.add_argument("--fisher-cache",default="oracle_fisher_settings_pool.pt")
    ap.add_argument("--checkpoint",default="adaptive_phys_transformer_v3.pt")
    ap.add_argument("--out",default="adaptive_phys_transformer_v3_results.json")

    ap.add_argument("--epochs",type=int,default=100)
    ap.add_argument("--steps-per-epoch",type=int,default=80)
    ap.add_argument("--batch-size",type=int,default=256)
    ap.add_argument("--eval-batch",type=int,default=128)
    ap.add_argument("--lr",type=float,default=1e-3)

    ap.add_argument("--train-Ks",type=int,nargs="+",
                    default=[4096,8192,16384,32768,65536])
    ap.add_argument("--Ks",type=int,nargs="+",
                    default=[4096,8192,16384,32768,65536])
    ap.add_argument("--adaptive-Ks",type=int,nargs="+",
                    default=[8192,16384,32768,65536])
    ap.add_argument("--warmup",type=int,default=4096)
    ap.add_argument("--p-uniform",type=float,default=.25)
    ap.add_argument("--p-bayes",type=float,default=.25)
    # remainder = adaptive, default .50

    ap.add_argument("--epsilon",type=float,default=.20)
    ap.add_argument("--bayes-samples",type=int,default=2048)
    ap.add_argument("--design-iters",type=int,default=32)
    ap.add_argument("--prior-precision",type=float,default=3.0)
    ap.add_argument("--explore",type=float,default=.10)
    ap.add_argument("--bayes-chunk",type=int,default=512)

    # Physics loss: exact differentiable Lindblad, but only on a small subset
    # every few optimizer steps to keep training practical.
    ap.add_argument("--lambda-phys",type=float,default=.20)
    ap.add_argument("--phys-every",type=int,default=4)
    ap.add_argument("--phys-batch",type=int,default=32)
    ap.add_argument("--times",type=float,nargs="+",default=[0.5,1,2,5,10])
    ap.add_argument("--max-step",type=float,default=.10)
    ap.add_argument("--J-meV",type=float,default=0.0)

    ap.add_argument("--n-test",type=int,default=512)
    ap.add_argument("--seed",type=int,default=1234)
    args=ap.parse_args()

    if args.p_uniform+args.p_bayes>1:
        raise ValueError("p-uniform + p-bayes must be <= 1")
    args.adaptive_Ks=sorted(set(args.adaptive_Ks))
    all_Ks=sorted(set(args.train_Ks+args.Ks+args.adaptive_Ks+[args.warmup]))

    torch.manual_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:",device)
    print("allocation mixture U/B/A:",
          args.p_uniform,args.p_bayes,1-args.p_uniform-args.p_bayes)
    print("physics: lambda=",args.lambda_phys,
          "every",args.phys_every,"steps, batch",args.phys_batch)

    theta,p=load_pool(args.cache)
    fo=torch.load(args.fisher_cache,map_location="cpu",weights_only=False)
    Fall=torch.as_tensor(fo["Fset"] if isinstance(fo,dict) else fo).float()
    if len(Fall)!=len(theta):
        raise RuntimeError("Fisher/probability cache length mismatch")
    if p.shape[1] != 16*len(args.times):
        raise RuntimeError(
            f"Expected {16*len(args.times)} settings from times={args.times}, "
            f"but cache has {p.shape[1]}. Use the same times as cache generation.")

    g=torch.Generator().manual_seed(args.seed)
    perm=torch.randperm(len(theta),generator=g)
    nv=max(1,int(.2*len(theta)))
    iv,it=perm[:nv],perm[nv:]
    th_tr,p_tr,F_tr=theta[it],p[it],Fall[it]
    th_va,p_va,F_va=theta[iv],p[iv],Fall[iv]

    p_mean=p_tr.mean(0)
    p_std=p_tr.std(0).clamp_min(1e-8)
    S=p.shape[1]

    # Global Bayesian allocations, computed once from TRAIN only.
    gg=torch.Generator().manual_seed(args.seed+991)
    M=min(args.bayes_samples,len(F_tr))
    bi=torch.randperm(len(F_tr),generator=gg)[:M]
    Fb=F_tr[bi].to(device)
    bayes_counts={}
    bayes_weights={}
    for K in all_Ks:
        print("Bayesian allocation K=",K)
        w=bayesian_aopt_weights(
            Fb,K,args.prior_precision,args.design_iters,args.explore,args.bayes_chunk
        ).cpu()
        bayes_weights[K]=w
        bayes_counts[K]=weights_to_counts(w,K)[0].cpu()

    ds=TensorDataset(th_tr,p_tr)
    dl=DataLoader(ds,batch_size=args.batch_size,shuffle=True,drop_last=True)

    model=MeasurementTransformer(S).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-6)
    system=build_system(device,args.J_meV)

    best=float("inf")
    best_state=None

    for ep in range(1,args.epochs+1):
        model.train()
        itr=iter(dl)
        tsum=dsum=psum=0.0
        nphys=0
        modes={"U":0,"B":0,"A":0}

        for step in range(args.steps_per_epoch):
            try:
                th,pp=next(itr)
            except StopIteration:
                itr=iter(dl); th,pp=next(itr)
            th=th.to(device); pp=pp.to(device)
            B=len(th)
            K=args.train_Ks[torch.randint(len(args.train_Ks),(1,)).item()]

            r=torch.rand(()).item()
            if r<args.p_uniform:
                mode="U"
                n,N=make_static_batch(pp,K,"uniform",bayes_counts,device)
            elif r<args.p_uniform+args.p_bayes:
                mode="B"
                n,N=make_static_batch(pp,K,"bayes",bayes_counts,device)
            else:
                mode="A"
                # Allocation decisions are discrete/no-grad; decoder training below is grad-enabled.
                model.eval()
                n,N=make_adaptive_batch(
                    model,pp,K,th_tr,F_tr,bayes_counts,bayes_weights,
                    p_mean,p_std,args,device)
                model.train()
            modes[mode]+=1

            x=features_from_counts(n,N,K,p_mean,p_std)
            zpred=model(x)
            loss_data=F.mse_loss(zpred,theta_to_z(th))
            loss=loss_data
            loss_phys=torch.zeros((),device=device)

            if args.lambda_phys>0 and (step%args.phys_every==0):
                q=min(args.phys_batch,B)
                # Random subset prevents physics loss from always seeing first samples.
                jj=torch.randperm(B,device=device)[:q]
                theta_pred=z_to_theta(zpred[jj])
                loss_phys=binomial_physics_nll(
                    theta_pred,n[jj],N[jj],system,args.times,args.max_step)
                loss=loss+args.lambda_phys*loss_phys
                psum+=loss_phys.item()
                nphys+=1

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.0)
            opt.step()

            tsum+=loss.item()
            dsum+=loss_data.item()

        if ep==1 or ep%5==0 or ep==args.epochs:
            nq=min(256,len(th_va))
            # Validate all three deployment-relevant distributions at K=8192.
            valK=8192 if 8192 in all_Ks else args.warmup
            (su,_),_=evaluate(model,th_va[:nq],p_va[:nq],F_va[:nq],valK,"uniform",
                          bayes_counts,bayes_weights,th_tr,F_tr,p_mean,p_std,args,device)
            (sb,_),_=evaluate(model,th_va[:nq],p_va[:nq],F_va[:nq],valK,"bayes",
                          bayes_counts,bayes_weights,th_tr,F_tr,p_mean,p_std,args,device)
            (sa,_),_=evaluate(model,th_va[:nq],p_va[:nq],F_va[:nq],valK,"adaptive",
                          bayes_counts,bayes_weights,th_tr,F_tr,p_mean,p_std,args,device)
            metric=(su+sb+sa)/3
            print(
                f"ep {ep:4d} total={tsum/args.steps_per_epoch:.5e} "
                f"data={dsum/args.steps_per_epoch:.5e} "
                f"phys={(psum/max(nphys,1)):.5e} "
                f"modes={modes} | val@{valK}: "
                f"U={su:.4f} B={sb:.4f} A={sa:.4f} mean={metric:.4f}")
            if metric<best:
                best=metric
                best_state=copy.deepcopy(model.state_dict())
                print("  new best")

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({
        "model_state":model.state_dict(),
        "p_mean":p_mean.cpu(),
        "p_std":p_std.cpu(),
        "S":S,
        "bayes_counts":bayes_counts,
        "args":vars(args),
        "best_val_metric":best,
    },args.checkpoint)
    print("\nSaved checkpoint:",args.checkpoint)

    nt=min(args.n_test,len(th_va))
    tht,pt,Ft=th_va[:nt],p_va[:nt],F_va[:nt]
    rows=[]
    fisher_rows=[]
    print("\n"+"="*112)
    print("FINAL: UNIFORM vs STATIC BAYES vs CLOSED-LOOP ADAPTIVE vs ORACLE")
    print("="*112)
    print(f"{'K':>8} {'uniform':>12} {'bayes':>12} {'adaptive':>12} {'oracle':>12} {'B/A':>10}")

    for K in args.Ks:
        (su,ru),Nu=evaluate(model,tht,pt,Ft,K,"uniform",bayes_counts,bayes_weights,
                            th_tr,F_tr,p_mean,p_std,args,device)
        (sb,rb),Nb=evaluate(model,tht,pt,Ft,K,"bayes",bayes_counts,bayes_weights,
                            th_tr,F_tr,p_mean,p_std,args,device)
        (sa,ra),Na=evaluate(model,tht,pt,Ft,K,"adaptive",bayes_counts,bayes_weights,
                            th_tr,F_tr,p_mean,p_std,args,device)
        (so,ro),No=evaluate(model,tht,pt,Ft,K,"oracle",bayes_counts,bayes_weights,
                            th_tr,F_tr,p_mean,p_std,args,device)

        print(f"{K:8d} {su:12.6f} {sb:12.6f} {sa:12.6f} {so:12.6f} {sb/sa:10.3f}")
        print("  U:", " ".join(f"{x:.4f}" for x in ru))
        print("  B:", " ".join(f"{x:.4f}" for x in rb))
        print("  A:", " ".join(f"{x:.4f}" for x in ra))
        print("  O:", " ".join(f"{x:.4f}" for x in ro))

        Au=true_fisher_ascore(Ft,Nu,args.prior_precision)
        Ab=true_fisher_ascore(Ft,Nb,args.prior_precision)
        Aa=true_fisher_ascore(Ft,Na,args.prior_precision)
        Ao=true_fisher_ascore(Ft,No,args.prior_precision)

        rows.append({
            "K":K,"uniform":su,"bayes":sb,"adaptive":sa,"oracle":so,
            "bayes_over_adaptive":sb/sa,
            "uniform_rmse":[float(x) for x in ru],
            "bayes_rmse":[float(x) for x in rb],
            "adaptive_rmse":[float(x) for x in ra],
            "oracle_rmse":[float(x) for x in ro],
        })
        fisher_rows.append({
            "K":K,
            "uniform_mean":Au[0],"uniform_median":Au[1],
            "bayes_mean":Ab[0],"bayes_median":Ab[1],
            "adaptive_mean":Aa[0],"adaptive_median":Aa[1],
            "oracle_mean":Ao[0],"oracle_median":Ao[1],
        })

    print("\n"+"="*112)
    print("TRUE-FISHER A-SCORE OF ACTUAL TEST ALLOCATIONS (LOWER = BETTER)")
    print("="*112)
    print(f"{'K':>8} {'uniform':>14} {'bayes':>14} {'adaptive':>14} {'oracle':>14} {'B/A':>10} {'A/O':>10}")
    for r in fisher_rows:
        u,b,a,o=(r["uniform_mean"],r["bayes_mean"],
                 r["adaptive_mean"],r["oracle_mean"])
        print(f"{r['K']:8d} {u:14.6e} {b:14.6e} {a:14.6e} {o:14.6e} "
              f"{b/a:10.3f} {a/o:10.3f}")
    with open(args.out,"w") as f:
        json.dump({
            "rows":rows,"true_fisher_rows":fisher_rows,"best_val_metric":best,
            "allocation_mix":{
                "uniform":args.p_uniform,
                "bayes":args.p_bayes,
                "adaptive":1-args.p_uniform-args.p_bayes},
            "lambda_phys":args.lambda_phys,
            "epsilon":args.epsilon,
            "warmup":args.warmup,
        },f,indent=2)
    print("Saved results:",args.out)


if __name__=="__main__":
    main()
