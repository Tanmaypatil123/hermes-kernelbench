"""Chunk-parallel KDA with a nilpotent triangular solve and FP32 state."""
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _prepare(Q,K,V,G,Beta,W,U,QG,KE,AQ,D,T:tl.constexpr,H:tl.constexpr,DK:tl.constexpr,DV:tl.constexpr,C:tl.constexpr,SCALE:tl.constexpr):
    chunk=tl.program_id(0)
    bh=tl.program_id(1)
    b=bh//H; h=bh%H
    nt=T//C
    r=tl.arange(0,C)
    d=tl.arange(0,DK)
    x=((b*T+chunk*C+r[:,None])*H+h)*DK+d[None,:]
    kk=tl.load(K+x).to(tl.float32)
    gg=tl.cumsum(tl.load(G+x),axis=0)
    eg=tl.exp(gg)
    kp=kk*eg
    km=kk*tl.exp(-gg)
    be=tl.load(Beta+(b*T+chunk*C+r)*H+h).to(tl.float32)
    gram=tl.dot(kp,tl.trans(km),input_precision='tf32')
    lower=tl.where(r[:,None]>r[None,:],-gram*be[:,None],0.)
    identity=tl.where(r[:,None]==r[None,:],1.,0.)
    inv=identity+lower
    power=lower
    for step in range(5):
        power=tl.dot(power,power,input_precision='tf32')
        inv=inv+tl.dot(inv,power,input_precision='tf32')
    inv=inv*be[None,:]
    ww=tl.dot(inv,kp,input_precision='tf32')
    vd=tl.arange(0,DV)
    vv=tl.load(V+((b*T+chunk*C+r[:,None])*H+h)*DV+vd[None,:]).to(tl.float32)
    uu=tl.dot(inv,vv,input_precision='tf32')
    qq=tl.load(Q+x).to(tl.float32)*SCALE*eg
    aq=tl.dot(qq,tl.trans(km),input_precision='tf32')
    aq=tl.where(r[:,None]>=r[None,:],aq,0.)
    last=tl.sum(tl.where(r[:,None]==C-1,gg,0.),axis=0)
    ke=kk*tl.exp(last[None,:]-gg)
    base=(bh*nt+chunk)*C
    tl.store(W+base*DK+r[:,None]*DK+d[None,:],ww)
    tl.store(QG+base*DK+r[:,None]*DK+d[None,:],qq)
    tl.store(KE+base*DK+r[:,None]*DK+d[None,:],ke)
    tl.store(U+base*DV+r[:,None]*DV+vd[None,:],uu)
    tl.store(AQ+base*C+r[:,None]*C+r[None,:],aq)
    tl.store(D+(bh*nt+chunk)*DK+d,tl.exp(last))

@triton.jit
def _recur(W,U,QG,KE,AQ,D,O,T:tl.constexpr,H:tl.constexpr,DK:tl.constexpr,DV:tl.constexpr,C:tl.constexpr,BV:tl.constexpr):
    bh=tl.program_id(0)
    vi=tl.program_id(1)
    b=bh//H; h=bh%H
    r=tl.arange(0,C)
    d=tl.arange(0,DK)
    v=vi*BV+tl.arange(0,BV)
    state=tl.full((DK,BV),0,tl.float32)
    for chunk in range(T//C):
        base=(bh*(T//C)+chunk)*C
        w=tl.load(W+base*DK+r[:,None]*DK+d[None,:])
        u=tl.load(U+base*DV+r[:,None]*DV+v[None,:])
        residual=u-tl.dot(w,state,input_precision='tf32')
        q=tl.load(QG+base*DK+r[:,None]*DK+d[None,:])
        aq=tl.load(AQ+base*C+r[:,None]*C+r[None,:])
        out=tl.dot(q,state,input_precision='tf32')+tl.dot(aq,residual,input_precision='tf32')
        tl.store(O+((b*T+chunk*C+r[:,None])*H+h)*DV+v[None,:],out)
        ke=tl.load(KE+base*DK+r[:,None]*DK+d[None,:])
        decay=tl.load(D+(bh*(T//C)+chunk)*DK+d)
        state=state*decay[:,None]+tl.dot(tl.trans(ke),residual,input_precision='tf32')

class Model(nn.Module):
    def __init__(self,B,T,H,K,V,chunk_size=64):
        super().__init__()
        self.B,self.T,self.H,self.K,self.V=B,T,H,K,V
        self.chunk_size=chunk_size
        self.scale=float(K)**-0.5
        self.register_buffer('_dummy',torch.zeros(1),persistent=False)
    def forward(self,q,k,v,g,beta):
        B,T,H,K=q.shape; V=v.shape[-1]; C=min(self.chunk_size,32); N=T//C
        w=torch.empty((B*H*N,C,K),device=q.device,dtype=torch.float32)
        qg=torch.empty_like(w); ke=torch.empty_like(w)
        u=torch.empty((B*H*N,C,V),device=q.device,dtype=torch.float32)
        aq=torch.empty((B*H*N,C,C),device=q.device,dtype=torch.float32)
        decay=torch.empty((B*H*N,K),device=q.device,dtype=torch.float32)
        o=torch.empty_like(v)
        _prepare[(N,B*H)](q,k,v,g,beta,w,u,qg,ke,aq,decay,T,H,K,V,C,self.scale,num_warps=8,num_stages=1)
        _recur[(B*H,V//32)](w,u,qg,ke,aq,decay,o,T,H,K,V,C,32,num_warps=4,num_stages=1)
        return o

B=2
T=1024
H=8
K=128
V=128
CHUNK_SIZE=64

def get_inputs():
    torch.manual_seed(0)
    q=torch.randn(B,T,H,K,dtype=torch.bfloat16)*0.1
    k=torch.randn(B,T,H,K,dtype=torch.bfloat16)*0.1
    v=torch.randn(B,T,H,V,dtype=torch.bfloat16)*0.1
    g=torch.randn(B,T,H,K,dtype=torch.float32)*0.1-0.05
    beta=torch.sigmoid(torch.randn(B,T,H,dtype=torch.bfloat16))
    return [q,k,v,g,beta]

def get_init_inputs():
    return [B,T,H,K,V,CHUNK_SIZE]
