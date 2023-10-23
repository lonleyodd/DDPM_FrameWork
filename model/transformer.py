import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import inspect

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
    def __init__(self, hidden_dim,n_head,max_seq_length,dropout,n_kv_head=None):
        super().__init__()
        self.model_parallel = 1
        self.hidden_dim = hidden_dim

        self.n_head = n_head
        self.n_kv_head = self.n_head if n_kv_head is None else self.n_head
        self.max_seq_length=max_seq_length

        self.n_local_head = self.n_head / self.model_parallel
        self.n_local_kv_head = self.n_kv_head / self.model_parallel
        self.n_rep = self.n_local_head // self.n_local_kv_head

        self.weight_q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.weight_k = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.weight_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.weight_o = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Pytorch>=2.0
        self.flash=hasattr(torch.nn.functional,"scaled_dot_product_attention")
        if not self.flash:
            mask=torch.full((1,1,self.max_seq_length,self.max_seq_length),float("-inf"))
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
    def __init__(self,n_head,hidden_dim,eps,layer_id,max_seq_length,drop_out,multiple):
        super().__init__()
        self.n_head=n_head
        self.hidden_dim=hidden_dim
        self.max_seq_length=max_seq_length
        self.drop_out=drop_out
        self.attention=Attention(
            hidden_dim=self.hidden_dim,
            n_head=self.n_head,
            max_seq_length=self.max_seq_length,
            dropout=self.drop_out
        )
        self.feed_forward=FeedForward(hidden_dim,hidden_dim*4,multiple,drop_out)

        self.layer_id = layer_id
        self.layer_norm1 = RMSNorm(hidden_dim,eps)
        self.layer_norm2 = RMSNorm(hidden_dim, eps)
    def forward(self,x,freq_cos,freq_sin):
        attention_out=x+self.layer_norm1(self.attention(x,freq_cos,freq_sin))
        ffn_out =attention_out + self.layer_norm2(self.feed_forward(attention_out))
        return ffn_out

class Tranformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_layer = cfg.n_layer
        self.norm_eps=cfg.norm_eps
        self.hidden_dim=cfg.hidden_dim
        self.max_seq_length=cfg.max_seq_length
        self.vocab_size=cfg.vocab_size

        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.dropout=nn.Dropout(cfg.drop_out)
        self.layers=nn.ModuleList()
        for i in range(self.n_layer):
            self.layers.append(
                TransformerBlock(
                    hidden_dim=self.hidden_dim,
                    n_head=self.n_head,
                    eps=self.norm_eps,
                    drop_out=cfg.drop_out,
                    max_seq_length=self.max_seq_length,
                    layer_id=i,
                    multiple=cfg.multiple,

            )
        )
        self.layer_norm=RMSNorm(self.hidden_dim,eps=self.norm_eps)
        self.output_gate=nn.Linear(self.hidden_dim,self.vocab_size)

        freqs_cos,freqs_sin=percomputr_freqs_cis(self.hidden_dim//self.n_head,
                                                self.max_seq_length
                                                )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.apply(self._init_weights)
        for n,p in self.named_parameters():
            if n.endswith('w3.weight') or n.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
     

    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,tokens,targets):
        

        batch_size,length = tokens.size()
        hidden_tensor=self.dropout(self.token_embedding(tokens))
        freqs_cos = self.freqs_cos[:length]
        freqs_sin = self.freqs_sin[:length]

        for layer in self.layers:
            hidden_tensor=layer(hidden_tensor,freqs_cos,freqs_sin)
        hidden_tensor=self.layer_norm(hidden_tensor)

        if targets is not None:
            logits = self.output_gate(hidden_tensor)
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1),ignore_index=-1)
        else:
            logits = self.output_gate(hidden_tensor[:,[-1],:])
            loss = None

        output={
            "loss":loss,
            "logits":logits
        }
        return output


    def param_configure(self,weight_decay,learning_rate,betas,device_type):
        param_dict={}
        for n, p in self.named_parameters():
            if p.requires_grad:
                param_dict[n]=p
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optimizer_groups=[
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
      

        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
       
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer = torch.optim.AdamW(optimizer_groups, lr=learning_rate, betas=betas, **extra_args)

        print(f"using fused AdamW: {use_fused}")
        return optimizer


class CustomLRScheduler():
    def __init__(self, cfg, optimizer):
        self.lr_max = cfg.lr_max
        self.warmup_iters =cfg.warmup_iters
        self.lr_decay_iters = cfg.lr_decay_iters
        self.lr_min = cfg.lr_min
        self.optimizer = optimizer
        self.cur_T=0
    
    def get_lr(self):
        # 1) linear warmup for warmup_iters steps
        if self.cur_T < self.warmup_iters:
            return self.lr_max * self.cur_T / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if self.cur_T  > self.lr_decay_iters:
            return self.lr_min
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (self.cur_T  - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.lr_min + coeff * (self.lr_max - self.lr_min)

    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.cur_T+=1
    

if __name__ == "__main__":
    from easydict import EasyDict
    model_cfg=EasyDict(dict(
        n_head=8,
        n_layer=1,
        hidden_dim=512,
        vocab_size=64794,
        max_seq_length=1024,
        drop_out=0.1,
        norm_eps=0.0001,
        multiple=32
    ))
    model=Tranformer(model_cfg)
    x=torch.randint(0,64793,(10,256))
    y=torch.randint(0,64793,(10,256))
    print(model(x,y))

