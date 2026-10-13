import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

BATCH=8
NUM_HEADS=32
NUM_KV_HEADS=8
HEAD_DIM=128
SEQ_LEN=1024
PAGE_SIZE=16

@triton.jit
def partial(Q,KV,BT,Lens,Tmp,Stats,H:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,P:tl.constexpr,NB:tl.constexpr,S:tl.constexpr,C:tl.constexpr):
    bk=tl.program_id(0)
    s=tl.program_id(1)
    b=bk//HK
    kh=bk%HK
    g=tl.arange(0,16)
    h=kh*(H//HK)+g
    bh=b*H+h
    t=s*C+tl.arange(0,C)
    d=tl.arange(0,D)
    length=tl.load(Lens+b)
    valid=t<length
    page=tl.load(BT+b*NB+t//P,valid,0)
    ptr=((page*P+t%P)*HK+kh)*2*D
    q=tl.load(Q+bh[:,None]*D+d[None,:],g[:,None]<(H//HK),0)
    k=tl.load(KV+ptr[None,:]+d[:,None],valid[None,:],0)
    scores=tl.dot(q,k)*(D**-0.5)
    scores=tl.where(valid[None,:],scores,float('-inf'))
    m=tl.max(scores,1)
    m=tl.where(m==float('-inf'),0.,m)
    w=tl.exp(scores-m[:,None])
    z=tl.sum(w,1)
    v=tl.load(KV+ptr[:,None]+D+d[None,:],valid[:,None],0)
    o=tl.dot(w.to(v.dtype),v)
    tl.store(Tmp+(bh[:,None]*S+s)*D+d[None,:],o,g[:,None]<(H//HK))
    tl.store(Stats+(bh*S+s)*2,m,g<(H//HK))
    tl.store(Stats+(bh*S+s)*2+1,z,g<(H//HK))

@triton.jit
def merge(Tmp,Stats,Out,D:tl.constexpr,S:tl.constexpr,R:tl.constexpr):
    bh=tl.program_id(0)
    s=tl.arange(0,R)
    d=tl.arange(0,D)
    m=tl.load(Stats+(bh*S+s)*2,s<S,float('-inf'))
    z=tl.load(Stats+(bh*S+s)*2+1,s<S,0.)
    mx=tl.max(tl.where(z>0,m,float('-inf')),0)
    mx=tl.where(mx==float('-inf'),0.,mx)
    w=tl.exp(m-mx)
    vals=tl.load(Tmp+(bh*S+s[:,None])*D+d[None,:],s[:,None]<S,0.)
    den=tl.sum(z*w,0)
    out=tl.sum(vals*w[:,None],0)/tl.maximum(den,1.e-20)
    tl.store(Out+bh*D+d,out)

class Model(nn.Module):
    def __init__(self,batch,num_heads,num_kv_heads,head_dim,seq_len,page_size):
        super().__init__()
        self.num_kv_heads=num_kv_heads
        self.page_size=page_size
        self.register_buffer('_dummy',torch.zeros(1,dtype=torch.bfloat16),persistent=False)
    def forward(self,query,kv_cache,block_table,seq_lens):
        b,h,d=query.shape
        c=128
        s=triton.cdiv(block_table.shape[1]*self.page_size,c)
        tmp=torch.empty((b*h,s,d),device=query.device,dtype=torch.float32)
        stats=torch.empty((b*h,s,2),device=query.device,dtype=torch.float32)
        out=torch.empty_like(query)
        partial[(b*self.num_kv_heads,s)](query,kv_cache,block_table,seq_lens,tmp,stats,h,self.num_kv_heads,d,self.page_size,block_table.shape[1],s,c,num_warps=4)
        merge[(b*h,)](tmp,stats,out,d,s,triton.next_power_of_2(s),num_warps=4)
        return out

def get_inputs():
    B,H,HK,D,L,P=BATCH,NUM_HEADS,NUM_KV_HEADS,HEAD_DIM,SEQ_LEN,PAGE_SIZE
    pages=(L+P-1)//P
    total=max(B*pages+8,64)
    q=torch.randn(B,H,D,dtype=torch.bfloat16)*0.1
    kv=torch.randn(total,P,HK,2*D,dtype=torch.bfloat16)*0.1
    bt=torch.randperm(total)[:B*pages].reshape(B,pages).int().contiguous()
    lens=torch.full((B,),L,dtype=torch.int32)
    return [q,kv,bt,lens]

def get_init_inputs():
    return [BATCH,NUM_HEADS,NUM_KV_HEADS,HEAD_DIM,SEQ_LEN,PAGE_SIZE]
