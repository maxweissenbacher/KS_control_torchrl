import torch
import torch.nn as nn
import math
from torch.nn import Module, Linear, Softmax
from tensordict.nn import InteractionType, TensorDictModule, TensorDictSequential, TensorDictModuleBase
from tensordict.tensordict import TensorDict, TensorDictBase
from tensordict.tensordict import NO_DEFAULT
from torchrl.modules import MLP
from models.memoryless.base import tqc_critic_net
from utils.device_finder import network_device


class MapToQKV(Module):
    def __init__(self, size_memory, device):
        super().__init__()
        self.q = Linear(size_memory, size_memory, bias=False, device=device)
        self.k = Linear(size_memory, size_memory, bias=False, device=device)
        self.v = Linear(size_memory, size_memory, bias=False, device=device)

    def forward(self, M, x):
        """
        M is assumed to be of shape batch_size x num_memories x size_memory
        x is assumed to be of shape batch_size x size_memory
        """
        q = self.q(M)
        k = self.k(torch.cat([M, torch.unsqueeze(x, dim=-2)], dim=-2))
        v = self.v(torch.cat([M, torch.unsqueeze(x, dim=-2)], dim=-2))
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
        self.qkv = MapToQKV(size_memory, device=device)
        self.w_concat = Linear(size_memory, size_memory, device=device)

    def forward(self, M, x):
        q, k, v = self.qkv(M, x)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out = self.attention(q, k, v)
        out = self.concatenate(out)
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

    @staticmethod
    def concatenate(x):
        # d_tensor is size_memory / nr_heads
        *batch_size, n_head, num_memories, d_tensor = x.size()
        size_memory = n_head * d_tensor
        x = x.transpose(-2, -3).contiguous().view(*batch_size, num_memories, size_memory)
        return x


class SelfAttentionLayer(Module):
    def __init__(self, size_memory, n_head, attention_mlp_depth, device):
        super(SelfAttentionLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(size_memory, n_head, device)
        self.norm = nn.LayerNorm(normalized_shape=size_memory)
        self.attention_mlp = MLP(
            num_cells=attention_mlp_depth*[size_memory],
            out_features=size_memory,
            activation_class=nn.ReLU,
            device=device
        )

    def forward(self, memory, input):
        x = self.multi_head_attention(memory, input)  # Multi head attention
        x += memory  # Residual connection
        x_ = self.norm(x)  # Layer norm
        x = self.attention_mlp(x_)  # Row/memory-wise MLP
        x += x_  # Residual connection
        return x


class SelfAttentionLayerIdentityReordered(Module):
    def __init__(self, size_memory, n_head, attention_mlp_depth, device):
        super(SelfAttentionLayerIdentityReordered, self).__init__()
        self.multi_head_attention = MultiHeadAttention(size_memory, n_head, device)
        self.norm = nn.LayerNorm(normalized_shape=size_memory)
        self.attention_mlp = MLP(
            num_cells=attention_mlp_depth*[size_memory],
            out_features=size_memory,
            activation_class=nn.ReLU,
            device=device
        )
        self.relu = nn.ReLU()

    def forward(self, memory, input):
        x = self.norm(memory)
        x = self.multi_head_attention(x, input)  # Multi head attention
        x = self.relu(x) + memory  # Residual connection
        x_ = self.norm(x)  # Layer norm
        x = self.attention_mlp(x_)  # Row/memory-wise MLP
        x = self.relu(x) + x_  # Residual connection
        return x


class SelfAttentionLayerSimplified(Module):
    def __init__(self, size_memory, n_head, device):
        super(SelfAttentionLayerSimplified, self).__init__()
        self.multi_head_attention = MultiHeadAttention(size_memory, n_head, device)
        self.norm = nn.LayerNorm(normalized_shape=size_memory)

    def forward(self, memory, input):
        x = self.multi_head_attention(memory, input)  # Multi head attention
        x = self.norm(x)
        return x


class Gate(Module):
    def __init__(self, input_size, size_memory):
        super().__init__()
        self.linear_input = Linear(in_features=input_size, out_features=size_memory, bias=False)
        self.linear_memory = Linear(in_features=size_memory, out_features=size_memory, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, memory, input):
        input_transformed = torch.unsqueeze(self.linear_input(input), dim=-2)
        memory_transformed = self.linear_memory(self.tanh(memory))
        return self.sigmoid(input_transformed + memory_transformed)


class SigmoidLinear(Module):
    def __init__(self, size_memory, init_weight):
        super().__init__()
        self.weights = nn.Parameter(torch.logit(init_weight * torch.eye(size_memory)))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x @ self.sigmoid(self.weights)

