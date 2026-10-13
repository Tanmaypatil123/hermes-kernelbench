import torch
import triton
import triton.language as tl
from reference import Model as ReferenceModel, get_inputs, get_init_inputs

@triton.jit
def decode(X,W,S,Z,P,N:tl.constexpr,K:tl.constexpr,G:tl.constexpr,BN:tl.constexpr=128,BK:tl.constexpr=256, SPLIT:tl.constexpr=16):
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    part=tl.program_id(1)
    r=part*(K//SPLIT)+2*tl.arange(0,BK//2)
    q=tl.load(W+(r[:,None]//2)*N+n[None,:],n[None,:]<N,0).to(tl.int32)
    s=tl.load(S+(r[:,None]//G)*N+n[None,:],n[None,:]<N,0).to(tl.float32)
    z=tl.load(Z+(r[:,None]//G)*N+n[None,:],n[None,:]<N,0).to(tl.float32)
    b0=(((q&15).to(tl.float32)-z)*s).to(tl.bfloat16).to(tl.float32)
    b1=(((q>>4).to(tl.float32)-z)*s).to(tl.bfloat16).to(tl.float32)
    a0=tl.load(X+r).to(tl.float32)
    a1=tl.load(X+r+1).to(tl.float32)
    v=tl.sum(a0[:,None]*b0+a1[:,None]*b1,axis=0)
    tl.store(P+part*N+n,v,n<N)

@triton.jit
def gemm(X,W,S,Z,P,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,G:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,SP:tl.constexpr):
    m=tl.program_id(0)*BM+tl.arange(0,BM)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    p=tl.program_id(2)
    rk=tl.arange(0,BK)
    acc=tl.full((BM,BN),0,tl.float32)
    for start in range(p*(K//SP),(p+1)*(K//SP),BK):
        k=start+rk
        a=tl.load(X+m[:,None]*K+k[None,:],m[:,None]<M,0)
        q=tl.load(W+(k[:,None]//2)*N+n[None,:],n[None,:]<N,0).to(tl.int32)
        q=(q>>((k[:,None]%2)*4))&15
        s=tl.load(S+(start//G)*N+n,n<N,0).to(tl.float32)[None,:]
        z=tl.load(Z+(start//G)*N+n,n<N,0).to(tl.float32)[None,:]
        b=((q.to(tl.float32)-z)*s).to(tl.bfloat16)
        acc=tl.dot(a,b,acc)
    tl.store(P+p*M*N+m[:,None]*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))

@triton.jit
def reduce(P,Y,T:tl.constexpr,SP:tl.constexpr,B:tl.constexpr=512):
    t=tl.program_id(0)*B+tl.arange(0,B)
    s=tl.arange(0,SP)
    v=tl.load(P+s[:,None]*T+t[None,:],t[None,:]<T,0)
    tl.store(Y+t,tl.sum(v,axis=0),t<T)

class Model(ReferenceModel):
    def forward(self,x):
        m,k=x.shape
        n=self.N
        sp=16 if m==1 else (4 if m<=32 else 1)
        p=torch.empty((sp,m,n),device=x.device,dtype=torch.float32)
        y=torch.empty((m,n),device=x.device,dtype=torch.bfloat16)
        if m==1:
            decode[(triton.cdiv(n,64),sp)](x,self.w_q,self.scales,self.zeros,p,n,k,self.group_size,BN=64,BK=256,SPLIT=sp,num_warps=4)
        else:
            bm=16 if m<=16 else (32 if m<=32 else 64)
            gemm[(triton.cdiv(m,bm),triton.cdiv(n,64),sp)](x,self.w_q,self.scales,self.zeros,p,m,n,k,self.group_size,bm,64,128,sp,num_warps=4,num_stages=1 if m<=32 else 2)
        reduce[(triton.cdiv(m*n,512),)](p,y,m*n,sp)
        return y
