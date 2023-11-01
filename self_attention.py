import torch
import math
from torch.nn import Module, Linear, Softmax
from icecream import ic


# M ... num_memories x size_memory


class MapToQKV(Module):
    def __init__(self, size_memory):
        super().__init__()
        self.q = Linear(size_memory, size_memory)
        self.k = Linear(size_memory, size_memory)
        self.v = Linear(size_memory, size_memory)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return q, k, v


class ScaleDotProductAttention(Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = Softmax(dim=-1)

    def forward(self, q, k, v):
        # input is of size (batch_size, nr_heads, num_memories, size_memory/nr_heads)
        d_tensor = k.size()[-1]
        # do dot-product between Query and Key^T
        score = (q @ k.transpose(-1, -2)) / math.sqrt(d_tensor)
        score = self.softmax(score)
        output = score @ v
        return output


class MultiHeadAttention(Module):
    def __init__(self, size_memory, n_head, device):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.qkv = MapToQKV(size_memory)
        self.w_concat = Linear(size_memory, size_memory, device=device)

    def forward(self, x):
        q, k, v = self.qkv(x)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out = self.attention(q, k, v)
        out = self.concat(out)
        out = self.w_concat(out)
        return out

    def split(self, x):
        *batch_size, num_memories, size_memory = x.size()
        if size_memory % self.n_head != 0:
            raise AssertionError('Model dimension size_memory must be dividable by number of attention heads n_head.')
        d_tensor = size_memory // self.n_head
        # this splits a big vector of size (x,y,z) into (x,y,n_head,z/n_head)
        x = x.view(*batch_size, num_memories, self.n_head, d_tensor)
        # transpose to (batch_size, nr_heads, num_memories, size_memory/nr_heads)
        x = x.transpose(-2, -3)
        return x

    def concat(self, x):
        # d_tensor is size_memory / nr_heads
        *batch_size, n_head, num_memories, d_tensor = x.size()
        size_memory = n_head * d_tensor
        x = x.transpose(-2, -3).contiguous().view(*batch_size, num_memories, size_memory)
        return x


if __name__ =='__main__':

    n_heads = 5
    size_memory = 50
    num_memories = 20

    W = torch.zeros([100, 100, num_memories, size_memory])

    MultiHeadAttention(size_memory, n_heads, 'cpu')(W)

