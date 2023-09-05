import torch
import torch.nn as nn
from torch import Tensor
from torch.functional import einsum
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import ColumnParallelLinear


class Conv2dAttention(nn.Module):
    def __init__(self, input_dim, n_head=8, dim_head=32):
        super().__init__()
        self.n_head = n_head
        self.dim_head = dim_head
        self.dim_hidden = self.n_head * self.dim_head
        self.weight_query = nn.Conv2d(input_dim, self.dim_hidden)
        self.weight_key = nn.Conv2d(input_dim, self.dim_hidden)
        self.weight_value = nn.Conv2d(input_dim, self.dim_hidden)
        self.weight_hidden = nn.Conv2d(self.dim_hidden, input_dim)
        self.scaler = self.dim_head ** -0.5

    def forward(self, x: Tensor):
        b, c, h, w = x.size()
        query = self.weight_query(x).reshape(b, self.n_head, c // self.n_head, h * w)
        key = self.weight_key(x).reshape(b, self.n_head, c // self.n_head, h * w)
        value = self.weight_value(x).reshape(b, self.n_head, c // self.n_head, h * w)

        # x: b,head_num,head_dim,h*w
        query = query * self.scaler
        sim = einsum('b n d l,b n d m -> b n l m', query, key)
        att = sim.softmax(dim=-1, keepdim=True)

        out = einsum('b n l d,b n m d', sim, value)
        out = self.weight_hidden(out.reshape(b, c, h, -1))
        return out


class WeightStandardizedConv2d(nn.Conv2d):
    '''
    https://github.com/joe-siyuan-qiao/WeightStandardization
    '''

    def forward(self, input: Tensor) -> Tensor:
        eps = 1e-5 if input.dtype == torch.float32 else 1e-3
        weight = self.weight

        mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + eps
        weight = weight / std.expand_as(weight)

        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.norm(self.proj(x))

        if scale_shift:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, time_emb_dim, group=8):
        super().__init__()
        self.block1 = Block(dim_in, dim_out)
        self.block2 = Block(dim_in, dim_out)
        self.res_cov2d = nn.Conv2d(dim_in, dim_out)
        self.mlp = (nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out)
        ) if time_emb_dim is not None else None)

    def forward(self, x, time_emb):
        scale_shift = None
        if self.mlp and time_emb:
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.rearrange(time_emb, 'b c ->b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        y = self.block2(self.block1(x, scale_shift=scale_shift))
        resdiual_x = self.res_conv(x)

        return y + resdiual_x


class Unet(nn.Module):
    def __init__(self, channels, ):
        super(Unet).__init__()
        self.channel = channels

    def forward(self):
        pass


class DDMP_Model(nn.Module):
    def __init__(self):
        super(DDMP_Model).__init__()
        self.encoder = None
        self.decoder = None

    def genrate_block(self, tag=None):
        block = nn.ModuleList()
        if tag == 'encoder':
            block.append(
                [

                ]
            )

    def forward(self):
        pass


if __name__ == "__main__":
    x = torch.rand(16, 256, 256, 3)
    linear = nn.Linear(3, 18)
    print(linear(x).size())
