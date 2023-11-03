import torch
import torch.nn as nn
import math
from torch.nn import Module, Linear, Softmax
from tensordict.nn import InteractionType, TensorDictModule, TensorDictSequential, TensorDictModuleBase
from tensordict.tensordict import TensorDict, TensorDictBase
from tensordict.tensordict import NO_DEFAULT
from torchrl.modules import MLP
from torchrl.envs.transforms.transforms import TensorDictPrimer
from torchrl.data import UnboundedContinuousTensorSpec
from icecream import ic


class MapToQKV(Module):
    def __init__(self, size_memory):
        super().__init__()
        self.q = Linear(size_memory, size_memory, bias=False)
        self.k = Linear(size_memory, size_memory, bias=False)
        self.v = Linear(size_memory, size_memory, bias=False)

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
        self.qkv = MapToQKV(size_memory)
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
    def __init__(self, size_memory, n_head, device):
        super(SelfAttentionLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(size_memory, n_head, device)

    def forward(self, M, x):
        return self.multi_head_attention(M, x)


class SelfAttentionMemoryActor(TensorDictModuleBase):
    def __init__(
            self,
            num_memories,
            size_memory,
            n_heads,
            action_spec,
            out_key,
            device,
    ):
        super().__init__()
        self.memory_key = "memory"
        self.in_keys = ["observation", self.memory_key]
        self.out_keys = [out_key, ("next", self.memory_key)]

        self.num_memories = num_memories
        self.size_memory = size_memory
        self.action_spec = action_spec
        self.n_heads = n_heads
        self.memory_reset_std = 0.1
        self.device = device

        self.action_mlp = MLP(
            num_cells=[256],
            out_features=2 * self.action_spec.shape[-1],
            activation_class=nn.ReLU,
            device=self.device
        )
        self.feature = MLP(
            num_cells=[128, 128],
            out_features=self.size_memory,
            activation_class=nn.ReLU,
            device=self.device
        )
        self.attention = SelfAttentionLayer(
            size_memory=self.size_memory,
            n_head=self.n_heads,
            device=self.device,
        )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        defaults = [NO_DEFAULT, None]  # We want to get an error if the value input is missing, but not the memory
        is_init = tensordict.get("is_init").squeeze(-1)
        observation, memory = (
            tensordict.get(key, default)
            for key, default in zip(self.in_keys, defaults)
        )
        batch_size = is_init.shape

        #memory = torch.ones([*batch_size, self.num_memories, self.size_memory])
        # When env gets reset, need to reset memory
        if is_init.any() and memory is not None:
            memory[is_init] = 0. # reset memory to zero - should be random instead
        # Reset memory if not existent in tensordict
        if memory is None:
            memory = torch.zeros([*batch_size, self.num_memories, self.size_memory])
            tensordict.set(self.memory_key, memory)  # probably not necessary to do this

        # Compute the "action" (whatever is processed into the action) for this step
        # This uses the current observation and memory state
        action_out = self.action_mlp(memory)

        # Preprocess the observation into a vector of the right size for the memory
        observation_feature = self.feature(observation)

        # Compute proposed memory update
        memory_update = self.attention(memory, observation_feature)

        # Now conduct the ACTUAL memory update (i.e. autoregression, LSTM where input is proposed update and
        # output is real update, the special LSTM architecture from the Deepmind paper, ...)
        # For now, simple autoregressive update for memory:
        next_memory = 0.9 * memory + 0.1 * memory_update

        # Write output to tensordict
        tensordict.set(self.out_keys[0], action_out)
        tensordict.set(self.out_keys[1], next_memory)

        return tensordict

