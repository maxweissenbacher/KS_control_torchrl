import torch
import math
from torch.nn import Module, Linear, Softmax
from tensordict.nn import InteractionType, TensorDictModule, TensorDictSequential, TensorDictModuleBase
from tensordict.tensordict import TensorDict, TensorDictBase
from tensordict.tensordict import NO_DEFAULT
from icecream import ic
from utils import make_ks_env
import hydra


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

    def forward(self, x):
        return self.multi_head_attention(x)


class SelfAttentionMemoryActor(TensorDictModuleBase):
    def __init__(self):
        super().__init__()
        self.in_keys = ['observation', 'memory']
        self.num_memories = 2
        self.size_memory = 5

    def forward(self, tensordict: TensorDictBase):
        # we want to get an error if the value input is missing, but not the hidden states
        defaults = [NO_DEFAULT, None]

        is_init = tensordict.get("is_init").squeeze(-1)

        observation, memory = (
            tensordict.get(key, default)
            for key, default in zip(self.in_keys, defaults)
        )

        ic(is_init)
        ic(is_init.shape)
        ic(memory)
        ic(observation.shape)

        batch_size = is_init.shape
        ic(batch_size)

        #memory = torch.ones([*batch_size, self.num_memories, self.size_memory])

        if is_init.any() and memory is not None:
            memory[is_init] = 0. # reset memory to zero - should be random instead

        # reset memory to zero - should be random instead
        if memory is None:
            memory = torch.zeros([*batch_size, self.num_memories, self.size_memory])

        # Do self attention here to retrieve a SUGGESTED memory update

        # Update the memory here
        # For instance, pass the suggested update as the "C" layer to an LSTM

        ic(memory)


@hydra.main(version_base="1.1", config_path="", config_name="config")
def main(cfg):
    n_heads = 5
    size_memory = 50
    num_memories = 20

    # M is the memory
    # M ... batch_size x num_memories x size_memory
    M = torch.zeros([100, 100, num_memories, size_memory])

    MultiHeadAttention(size_memory, n_heads, 'cpu')(M)

    train_env, eval_env = make_ks_env(cfg)

    # Doing a simple rollout with zero actions
    td = eval_env.reset()
    rollout = eval_env.rollout(max_steps=3)

    ic(rollout)

    actor = SelfAttentionMemoryActor()

    actor(rollout)


if __name__ == '__main__':
    main()
