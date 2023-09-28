import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


class RMSNorm(nn.Module):
    def __init__(self,dim,eps):
        super().__init__()
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))

    def _norm(self,x):
        return x*torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self,x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def Rotary_posembedding(xq, xk, freqs_cos, freqs_sin):
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

def percomputr_freqs_cis(dim,sqe_length,theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t=torch.arange(sqe_length,device=freqs.device)
    freqs=torch.outer(t,freqs).float()
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)
    return freqs_cos,freqs_sin


def repeat_kv(x,n_rep):
    b,l,h,d=x.size()
    if n_rep==1:
        return x
    else:
        return x[:,:,:,None,:].expand(b,l,h,n_rep,d).reshape(b,l,h*n_rep,d)
class Attention(nn.Module):
    def __init__(self, hidden_dim,n_head,n_kv_head,max_seq_length,dropout):
        self.model_parallel = 1
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is None else n_head
        self.max_seq_length=max_seq_length

        self.n_local_head = self.n_head / self.model_parallel
        self.n_local_kv_head = self.n_kv_head / self.model_parallel
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

        self.weight_q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.weight_k = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.weight_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.weight_o = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Pytorch>=2.0
        self.flash=hasattr(torch.nn.functional,"scaled_dot_product_attention")
        if not self.flash:
            mask=torch.full((1,1,self.max_seq_length,self.max_seq_len),float("-inf"))
            mask = torch.triu(mask,diagonal=1)
            self.register_buffer("mask",mask)


    def forward(self, x, freqs_cos, freqs_sin):
        B, L, D = x.size()

        q = self.weight_q(x).view(B, L, self.n_head, -1)
        k = self.weight_q(x).view(B, L, self.n_kv_head, -1)
        v = self.weight_q(x).view(B, L, self.n_kv_head, -1)

        q, k = Rotary_posembedding(q, k, freqs_cos, freqs_sin)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                                      attn_mask=None,
                                                                      dropout_p=self.dropout if self.training else 0.0,
                                                                      is_causal=True)
        else:
            scores = torch.matmul(q,k.transpose(2,3))/math.sqrt(self.hidden_dim//self.n_head)
            assert hasattr(self,'mask')
            scores=scores+self.mask[:,:,:L,:L]
            scores=F.softmax(scores,dim=-1).type_as(q)
            scores=self.dropout1(scores)
            output=torch.matmul(scores,v)

        output=output.transpose(1,2).contiguous().view(B,L,-1)

        output=self.dropout2(self.weight_o(output))
        return output

class FeedForward(nn.Module):
    def __init__(self,input_dim,output_dim,multiple,dropout):
        super().__init__()
        output_dim=int(2*output_dim/3)
        output_dim=multiple*((output_dim+multiple-1)//multiple)
        self.linear1 = nn.Linear(input_dim,output_dim)
        self.linear2 = nn.Linear(output_dim,input_dim)
        self.linear3 = nn.Linear(input_dim,output_dim)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        return self.dropout(self.linear2(F.silu(self.linear1(x))*self.linear3(x)))

class TransformerBlock(nn.Module):
    def __init__(self,n_head,hidden_dim,eps,layer_id):
        super().__init__()
        self.n_head=n_head
        self.hidden_dim=hidden_dim
        self.attention=Attention()
        self.feed_forward=FeedForward(hidden_dim,hidden_dim*4)

        self.layer_id = layer_id
        self.layer_norm1 = RMSNorm(hidden_dim,eps)
        self.layer_norm2 = RMSNorm(hidden_dim, eps)
    def forward(self,x,freq_cos,freq_sin):
        attention_out=x+self.layer_norm1(self.attention(x,freq_cos,freq_sin))
        ffn_out =attention_out + self.layer_norm2(self.feed_forward(attention_out))
        return ffn_out

class Tranformer(nn.Module):
    def __init__(self, n_head,
                 n_layers,
                 vocab_size,
                 hidden_dim,
                 max_seq_length=1024,
                 dropout=0.2,
                 norm_eps=1e-5,
                ):
        self.n_head = n_head
        self.n_layers = n_layers
        self.norm_eps=norm_eps
        self.hidden_dim=hidden_dim
        self.max_seq_length=max_seq_length
        self.vocab_size=vocab_size
        self.attention = Attention()
        self.dropout=nn.Dropout(dropout)
        self.layers=nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(
                TransformerBlock(
                hidden_dim=self.hidden_dim,
                n_head=self.n_head,
                eps=self.norm_eps,
                layer_id=i
            )
        )
        self.layer_norm=RMSNorm(hidden_dim,eps=self.eps)
        self.token_embedding=nn.Embedding(vocab_size,hidden_dim)
        self.output_gate=nn.Linear(self.hidden_dim,self.vocab_size)

        freqs_cos,freqs_sin=percomputr_freqs_cis(self.hidden_dim//self.n_head,
                                                self.max_seq_length
                                                   )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.apply(self._init_weights)
        for n,p in self.named_parameters():
            if n.endswith('w3.weight') or n.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

        self.loss=None
    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self,tokens,targets):
        batch_size,length,dim = tokens.size()
        hidden_tensor=self.dropout(self.token_embedding(tokens))
        freqs_cos = self.freqs_cos[:length]
        freqs_sin = self.freqs_sin[:length]

        for layer in self.layers:
            hidden_tensor=layer(hidden_tensor,freqs_cos,freqs_sin)
        hidden_tensor=self.layer_norm(hidden_tensor)

        if targets is not None:
            logits=self.output(hidden_tensor)
            self.loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1),ignore_index=-1)
        else:
            logits = self.output(hidden_tensor[:,[-1],:])
            self.loss = None

        return logits





if __name__ == "__main__":
    percomputr_freqs_cis(512,1024)

