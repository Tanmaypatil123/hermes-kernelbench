import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _metadata(O, P, E:tl.constexpr, B:tl.constexpr, S:tl.constexpr):
    e=tl.arange(0,S)
    a=tl.load(O+e, e<E, 0)
    b=tl.load(O+e+1,e<E,0)
    n=tl.where(e<E,tl.cdiv(b-a,B),0)
    p=tl.cumsum(n)
    tl.store(P+e+1,p,e<E)
    tl.store(P,0)

@triton.jit
def _gemm(X,G,U,O,P,Y,H:tl.constexpr,N:tl.constexpr,E:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pid=tl.program_id(0)
    nt=tl.cdiv(N,BN)
    mt=tl.num_programs(0)//nt
    group=pid//(8*nt)
    first=group*8
    size=tl.minimum(mt-first,8)
    tile=first+(pid%(8*nt))%size
    ni=(pid%(8*nt))//size
    total=tl.load(P+E)
    if tile<total:
        lo=0
        hi=E
        while lo<hi:
            mid=(lo+hi)//2
            v=tl.load(P+mid+1)
            if v<=tile:
                lo=mid+1
            else:
                hi=mid
        expert=lo
        local=tile-tl.load(P+expert)
        start=tl.load(O+expert)
        end=tl.load(O+expert+1)
        m=start+local*BM+tl.arange(0,BM)
        n=ni*BN+tl.arange(0,BN)
        k=tl.arange(0,BK)
        xp=X+m[:,None]*H+k[None,:]
        gp=G+expert*H*N+k[:,None]*N+n[None,:]
        up=U+expert*H*N+k[:,None]*N+n[None,:]
        ag=tl.zeros((BM,BN),tl.float32)
        au=tl.zeros((BM,BN),tl.float32)
        for block in range(tl.cdiv(H,BK)):
            x=tl.load(xp, (m[:,None]<end)&(k[None,:]+block*BK<H),0)
            g=tl.load(gp,(n[None,:]<N)&(k[:,None]+block*BK<H),0)
            u=tl.load(up,(n[None,:]<N)&(k[:,None]+block*BK<H),0)
            ag=tl.dot(x,g,ag)
            au=tl.dot(x,u,au)
            xp+=BK
            gp+=BK*N
            up+=BK*N
        g=ag.to(tl.bfloat16).to(tl.float32)
        u=au.to(tl.bfloat16).to(tl.float32)
        s=(g/(1.+tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
        y=s*u
        tl.store(Y+m[:,None]*N+n[None,:],y,(m[:,None]<end)&(n[None,:]<N))

class Model(nn.Module):
    def __init__(self,T_total,H,I,E,K):
        super().__init__()
        self.T_total,self.H,self.I,self.E,self.K=T_total,H,I,E,K
        self.W_gate=nn.Parameter(torch.empty(E,H,I,dtype=torch.bfloat16))
        self.W_up=nn.Parameter(torch.empty(E,H,I,dtype=torch.bfloat16))
        nn.init.normal_(self.W_gate,std=0.02)
        nn.init.normal_(self.W_up,std=0.02)
    def forward(self,hidden_states,expert_offsets):
        t,h=hidden_states.shape
        bm,bn,bk=64,128,32
        p=torch.empty(self.E+1,dtype=torch.int32,device=hidden_states.device)
        out=torch.empty((t,self.I),dtype=torch.bfloat16,device=hidden_states.device)
        _metadata[(1,)](expert_offsets,p,self.E,bm,triton.next_power_of_2(self.E))
        _gemm[((triton.cdiv(t,bm)+self.E)*triton.cdiv(self.I,bn),)](hidden_states,self.W_gate,self.W_up,expert_offsets,p,out,h,self.I,self.E,bm,bn,bk,num_warps=4,num_stages=3)
        return out

T_total=32768
H=4096
I=1536
E=128
K=8

def get_inputs():
    t=T_total*K
    x=torch.randn(t,H,dtype=torch.bfloat16)*0.1
    counts=torch.full((E,),t//E,dtype=torch.int32)
    counts[:t%E]+=1
    offsets=torch.zeros(E+1,dtype=torch.int32)
    offsets[1:]=torch.cumsum(counts,0)
    return [x,offsets]

def get_init_inputs():
    return [T_total,H,I,E,K]
