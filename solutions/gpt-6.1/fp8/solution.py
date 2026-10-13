import torch
import triton
import triton.language as tl
from reference import Model as ReferenceModel, get_inputs, get_init_inputs

@triton.jit
def gemm(X,W,S,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,SPLIT:tl.constexpr):
    p=tl.program_id(0)
    z=tl.program_id(1)
    nm=tl.cdiv(M,BM)
    nn=tl.cdiv(N,BN)
    group=p//(8*nn)
    first=group*8
    size=tl.minimum(nm-first,8)
    pm=first+p%size
    pn=(p%(8*nn))//size
    m=pm*BM+tl.arange(0,BM)
    n=pn*BN+tl.arange(0,BN)
    k=z*BK+tl.arange(0,BK)
    acc=tl.full((BM,BN),0,tl.float32)
    for i in range(tl.cdiv(K,BK*SPLIT)):
        a=tl.load(X+m[:,None]*K+k[None,:],(m[:,None]<M)&(k[None,:]<K),0.0)
        b=tl.load(W+n[None,:]*K+k[:,None],(n[None,:]<N)&(k[:,None]<K),0.0)
        acc=tl.dot(a,b,acc,max_num_imprecise_acc=0)
        k+=BK*SPLIT
    if SPLIT==1:
        scale=tl.load(S+n,n<N,0)
        out=acc.to(tl.bfloat16).to(tl.float32)*scale[None,:]
        tl.store(Y+m[:,None]*N+n[None,:],out,(m[:,None]<M)&(n[None,:]<N))
    else:
        tl.store(Y+z*M*N+m[:,None]*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))

@triton.jit
def reduce(P,S,Y,M:tl.constexpr,N:tl.constexpr,SPLIT:tl.constexpr,B:tl.constexpr):
    t=tl.program_id(0)*B+tl.arange(0,B)
    a=tl.full((B,),0,tl.float32)
    for z in range(SPLIT):
        a+=tl.load(P+z*M*N+t,t<M*N,0)
    s=tl.load(S+t%N,t<M*N,0)
    y=a.to(tl.bfloat16).to(tl.float32)*s
    tl.store(Y+t,y,t<M*N)

@triton.jit
def pad(A,O,R:tl.constexpr,K:tl.constexpr,P:tl.constexpr,B:tl.constexpr):
    t=tl.program_id(0)*B+tl.arange(0,B)
    r=t//P
    c=t%P
    v=tl.load(A+r*K+c,(r<R)&(c<K),0.0)
    tl.store(O+t,v,r<R)

class Model(ReferenceModel):
    def forward(self,x):
        m,k=x.shape
        n=self.weight.shape[0]
        y=torch.empty((m,n),device=x.device,dtype=torch.bfloat16)
        split=4 if m<=32 else 1
        bm,bn,bk=(32,128,256) if m<=32 else (128,256,128)
        tmp=torch.empty((split,m,n),device=x.device,dtype=torch.float32) if split>1 else y
        w=self.weight
        if k%128:
            pk=triton.cdiv(k,128)*128
            xp=torch.empty((m,pk),device=x.device,dtype=x.dtype)
            wp=torch.empty((n,pk),device=x.device,dtype=w.dtype)
            pad[(triton.cdiv(m*pk,4096),)](x,xp,m,k,pk,4096)
            pad[(triton.cdiv(n*pk,4096),)](w,wp,n,k,pk,4096)
            x,w,k=xp,wp,pk
        gemm[(triton.cdiv(m,bm)*triton.cdiv(n,bn),split)](x,w,self.weight_scale,tmp,m,n,k,bm,bn,bk,split,num_warps=4 if m<=32 else 8,num_stages=3)
        if split>1:
            reduce[(triton.cdiv(m*n,1024),)](tmp,self.weight_scale,y,m,n,split,1024)
        return y
