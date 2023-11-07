import torch
import torch.nn as nn
import math
from torch.nn import Module, Linear, Softmax
from tensordict.nn import InteractionType, TensorDictModule, TensorDictSequential, TensorDictModuleBase
from tensordict.tensordict import TensorDict, TensorDictBase
from tensordict.tensordict import NO_DEFAULT
from torchrl.modules import MLP
#from models.memoryless.base import tqc_critic_net


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
    def __init__(self, size_memory, n_head, device):
        super(SelfAttentionLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(size_memory, n_head, device)

    def forward(self, M, x):
        return self.multi_head_attention(M, x)


class SelfAttentionMemoryActor(TensorDictModuleBase):
    def __init__(self, cfg, action_spec, in_keys=None, out_keys=None):
        super().__init__()
        if out_keys is None:
            out_keys = ["_actor_net_out", ("next", "memory")]
        if in_keys is None:
            in_keys = ["observation", "memory"]
        assert(out_keys[1][1] == in_keys[1])

        self.memory_key = in_keys[1]
        self.in_keys = in_keys
        self.out_keys = out_keys

        self.num_memories = cfg.network.attention.num_memories
        self.size_memory = cfg.network.attention.size_memory
        self.action_spec = action_spec
        self.batch_size = action_spec
        self.n_heads = cfg.network.attention.n_heads
        self.device = cfg.network.device

        print('stop here')

        self.action_mlp = MLP(
            num_cells=[256],
            out_features=2 * self.action_spec.shape[-1],
            activation_class=nn.ReLU,
            device=self.device
        )

        """
        class MemoryReader(Module):
            def __init__(self, batch_size, action_spec, device):
                super(MemoryReader, self).__init__()
                self.mlp = MLP(
                    num_cells=[256],
                    out_features=2 * action_spec.shape[-1],
                    activation_class=nn.ReLU,
                    device=device
                )
                self.batch_size = batch_size

            def forward(self, memory):
                batch_size = memory.size()[:-2]
                reshaped_memory = torch.reshape(memory, [*batch_size, -1]).contiguous()
                return self.mlp(reshaped_memory)

        self.memory_reader = MemoryReader([256], self.action_spec, self.device)
        """

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
        defaults = [NO_DEFAULT, NO_DEFAULT]  # We want to get an error if either memory or value are missing.
        is_init = tensordict.get("is_init").squeeze(-1)
        observation, memory = (
            tensordict.get(key, default)
            for key, default in zip(self.in_keys, defaults)
        )
        batch_size = is_init.size()

        # Compute the "action" (whatever is processed into the action) for this step
        # This uses the current observation and memory state (concatenate and pass into MLP?)
        # Currently only uses memory for testing purposes...
        action_out = self.action_mlp(torch.reshape(memory, [*batch_size, -1]))
        # action_out = self.action_mlp(torch.reshape(observation, [*batch_size, -1]))

        # Preprocess the observation into a vector of the right size for the memory
        observation_feature = self.feature(observation)

        # Compute proposed memory update
        memory_update = self.attention(memory, observation_feature)
        # memory_update = memory

        # Now conduct the ACTUAL memory update (i.e. autoregression, LSTM where input is proposed update and
        # output is real update, the special LSTM architecture from the Deepmind paper, ...)
        # For now, simple autoregressive update for memory:
        alpha = 1.0
        next_memory = (1-alpha) * memory + alpha * memory_update

        # Write output to tensordict
        tensordict.set(self.out_keys[0], action_out)
        tensordict.set(self.out_keys[1], next_memory)

        return tensordict


"""
class SelfAttentionMemoryCritic(TensorDictModuleBase):
    def __init__(self, cfg, in_keys=None, out_keys=None):
        super().__init__()
        if out_keys is None:
            out_keys = ["state_action_value"]
        if in_keys is None:
            in_keys = ["observation", "action"]
        self.in_keys = in_keys
        self.out_keys = out_keys

        self.net = tqc_critic_net(cfg)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return None
"""

