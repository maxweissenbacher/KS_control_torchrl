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


class SelfAttentionMemoryActor2(TensorDictModuleBase):
    def __init__(self, cfg, action_spec, in_keys=None, out_keys=None):
        super().__init__()
        if out_keys is None:
            out_keys = ["_actor_net_out", ("next", str(cfg.network.attention.critic_memory_key))]
        if in_keys is None:
            in_keys = ["observation", str(cfg.network.attention.critic_memory_key)]
        assert(out_keys[1][1] == in_keys[1])

        self.memory_key = in_keys[1]
        self.in_keys = in_keys
        self.out_keys = out_keys

        self.action_spec = action_spec
        self.observation_size = cfg.env.num_sensors
        self.device = network_device(cfg)

        self.action_mlp = MLP(
            num_cells=cfg.network.attention.actor_mlp_hidden_sizes,
            out_features=2 * self.action_spec.shape[-1],
            activation_class=nn.ReLU,
            device=self.device
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
        # This uses the observation and memory state
        memory_observation = torch.cat((memory.view([*batch_size, -1]), observation.view([*batch_size, -1])), dim=-1)
        action_out = self.action_mlp(memory_observation)

        # Write output to tensordict
        tensordict.set(self.out_keys[0], action_out)

        return tensordict


class SelfAttentionMemoryCritic2(TensorDictModuleBase):
    def __init__(self, cfg, action_spec, in_keys=None, out_keys=None):
        super().__init__()
        if out_keys is None:
            out_keys = ["state_action_value", ("next", str(cfg.network.attention.critic_memory_key))]
        if in_keys is None:
            in_keys = ["observation", "action", str(cfg.network.attention.critic_memory_key)]
        assert (out_keys[1][1] == in_keys[2])

        self.memory_key = in_keys[2]
        self.in_keys = in_keys
        self.out_keys = out_keys

        self.num_memories = cfg.network.attention.num_memories
        self.size_memory = cfg.network.attention.size_memory
        self.action_spec = action_spec
        self.n_heads = cfg.network.attention.n_heads
        self.attention_mlp_depth = cfg.network.attention.attention_mlp_depth
        self.observation_size = cfg.env.num_sensors
        self.num_actions = cfg.env.num_actuators
        self.device = network_device(cfg)

        self.critic_net = tqc_critic_net(cfg, model='attention')
        self.feature = MLP(
            num_cells=[128],
            out_features=self.size_memory,
            activation_class=nn.ReLU,
            device=self.device
        )
        # self.feature = Linear(in_features=self.observation_size+self.num_actions, out_features=self.size_memory)
        self.attention = SelfAttentionLayer(
            size_memory=self.size_memory,
            n_head=self.n_heads,
            attention_mlp_depth=self.attention_mlp_depth,
            device=self.device,
        )
        self.forget_gate = Gate(input_size=self.size_memory, size_memory=self.size_memory)
        self.input_gate = Gate(input_size=self.size_memory, size_memory=self.size_memory)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        defaults = [NO_DEFAULT, NO_DEFAULT, NO_DEFAULT]
        is_init = tensordict.get("is_init").squeeze(-1)
        observation, action, memory = (
            tensordict.get(key, default)
            for key, default in zip(self.in_keys, defaults)
        )
        batch_size = is_init.size()
        observation_action = torch.cat((observation.view([*batch_size, -1]), action.view([*batch_size, -1])), dim=-1)

        # Preprocess the observation into a vector of the right size for the memory
        observation_action_feature = self.feature(observation_action)

        # Input and forget gates
        forget_gate = self.forget_gate(memory, observation_action_feature)
        input_gate = self.input_gate(memory, observation_action_feature)

        # Compute memory update
        next_memory = self.attention(memory, observation_action_feature)
        next_memory = forget_gate * memory + input_gate * nn.Tanh()(next_memory)

        # Compute the critic output from memory, observation and action
        memory_observation_action = torch.cat((next_memory.view([*batch_size, -1]), observation_action), dim=-1)
        state_action_value = self.critic_net(memory_observation_action)

        # Write output to tensordict
        tensordict.set(self.out_keys[0], state_action_value)
        tensordict.set(self.out_keys[1], next_memory)

        return tensordict

