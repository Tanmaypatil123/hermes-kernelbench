import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def partial(X,T,N:tl.constexpr,K:tl.constexpr,C:tl.constexpr,S:tl.constexpr):
    r=tl.program_id(0); c=tl.program_id(1)
    a=tl.arange(0,C); i=c*C+a
    v=tl.load(X+r*N+i,i<N,other=-float('inf'))
    b=v.to(tl.uint32,bitcast=True)
    ordered=tl.where((b & 0x80000000)!=0,~b,b ^ 0x80000000).to(tl.uint32)
    key=(ordered.to(tl.uint64)<<32) | i.to(tl.uint64)
    key=tl.where(i<N,key,0)
    key=tl.topk(key,K)
    out=tl.arange(0,K)
    tl.store(T+r*S*K+c*K+out,key)

@triton.jit
def finish(X,T,V,I,N:tl.constexpr,K:tl.constexpr,M:tl.constexpr,B:tl.constexpr):
    r=tl.program_id(0); a=tl.arange(0,B)
    key=tl.load(T+r*M+a,a<M,other=0)
    key=tl.topk(key,K)
    out=tl.arange(0,K)
    idx=(key & 0xffffffff).to(tl.int64)
    val=tl.load(X+r*N+idx)
    tl.store(V+r*K+out,val)
    tl.store(I+r*K+out,idx)

@triton.jit
def argmax_kernel(X,V,I,N:tl.constexpr,B:tl.constexpr):
    r=tl.program_id(0); a=tl.arange(0,B)
    v=tl.load(X+r*N+a,a<N,other=-float('inf'))
    idx=tl.argmax(v,0)
    val=tl.max(v,0)
    tl.store(V+r,val); tl.store(I+r,idx.to(tl.int64))

@triton.jit
def direct(X,V,I,N:tl.constexpr,K:tl.constexpr,B:tl.constexpr):
    r=tl.program_id(0); a=tl.arange(0,B)
    v=tl.load(X+r*N+a,a<N,other=-float('inf'))
    b=v.to(tl.uint32,bitcast=True)
    ordered=tl.where((b & 0x80000000)!=0,~b,b ^ 0x80000000).to(tl.uint32)
    key=(ordered.to(tl.uint64)<<32)|a.to(tl.uint64)
    key=tl.where(a<N,key,0)
    key=tl.topk(key,K)
    o=tl.arange(0,K); idx=(key & 0xffffffff).to(tl.int64)
    val=tl.load(X+r*N+idx)
    tl.store(V+r*K+o,val); tl.store(I+r*K+o,idx)

class Model(nn.Module):
    def __init__(self,batch,n,k):
        super().__init__(); self.batch=batch; self.n=n; self.k=k
        self.register_buffer('_dummy',torch.zeros(1))
    def forward(self,x):
        b,n=x.shape; k=self.k
        v=torch.empty((b,k),device=x.device,dtype=torch.float32)
        i=torch.empty((b,k),device=x.device,dtype=torch.int64)
        if k==1:
            argmax_kernel[(b,)](x,v,i,n,triton.next_power_of_2(n))
        else:
            c=4096 if b==1 else 1024; s=triton.cdiv(n,c)
            t=torch.empty((b,s*k),device=x.device,dtype=torch.uint64)
            partial[(b,s)](x,t,n,k,c,s,num_warps=4)
            finish[(b,)](x,t,v,i,n,k,s*k,triton.next_power_of_2(s*k),num_warps=4)
        return v,i

batch=64
n=8192
k=8
def get_inputs(): return [torch.randn(batch,n,dtype=torch.float32)]
def get_init_inputs(): return [batch,n,k]
