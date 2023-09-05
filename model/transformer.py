import torch
import torch.nn as nn


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def Rotary_posembedding(xq,xk,freqs_cos,freqs_sin):
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self,*args,**kwargs):

        self.model_parallel = 1
        self.hidden_dim = args.hidden_dim
        self.n_head=args.n_head
        self.n_kv_head=args.n_kv_head if args.n_kv_head is None else args.n_head

        self.n_local_head=self.n_head/self.model_parallel
        self.n_local_kv_head = self.n_kv_head / self.model_parallel
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

        self.weight_q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.weight_k = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.weight_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.weight_o = nn.Linear(self.hidden_dim, self.hidden_dim)


    def forward(self,x,freqs_cos,freqs_sin):
        B,L,D=x.size()

        q = self.weight_q(x).view(B, L, self.n_head, -1)
        k = self.weight_q(x).view(B, L, self.n_kv_head, -1)
        v = self.weight_q(x).view(B, L, self.n_kv_head, -1)

        q, k = Rotary_posembedding(q, k,freqs_cos,freqs_sin)



        score=torch.matmul()

class Tranformer(nn.Module):
    def __init__(self,n_head,n_layers):
        self.n_head=n_head
        self.n_layers=n_layers
        self.attention=
