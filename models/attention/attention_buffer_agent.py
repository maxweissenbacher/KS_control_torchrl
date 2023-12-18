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
from models.attention.transformers import SelfAttentionLayer, SelfAttentionLayerIdentityReordered, Gate, SigmoidLinear


class SelfAttentionBufferMemoryActor(TensorDictModuleBase):
    def __init__(self, cfg, action_spec, in_keys=None, out_keys=None):
        super().__init__()
        if out_keys is None:
            out_keys = ["_actor_net_out", ("next", str(cfg.network.attention.actor_memory_key))]
        if in_keys is None:
            in_keys = [
                "observation",
                str(cfg.network.attention.actor_memory_key),
                str(cfg.network.buffer.buffer_observation_key),
            ]
        assert(out_keys[1][1] == in_keys[1])

        self.memory_key = in_keys[1]
        self.buffer_key = in_keys[2]
        self.in_keys = in_keys
        self.out_keys = out_keys

        self.num_memories = cfg.network.attention.num_memories
        self.size_memory = cfg.network.attention.size_memory
        self.action_spec = action_spec
        self.n_heads = cfg.network.attention.n_heads
        self.attention_mlp_depth = cfg.network.attention.attention_mlp_depth
        self.observation_size = cfg.env.num_sensors
        self.buffer_size = cfg.network.buffer.size
        self.device = network_device(cfg)
        self.reset_memory = cfg.network.attention.reset_memory

        self.action_mlp = MLP(
            num_cells=cfg.network.attention.actor_mlp_hidden_sizes,
            out_features=2 * self.action_spec.shape[-1],
            activation_class=nn.ReLU,
            device=self.device
        )
        self.feature = MLP(
            num_cells=[128],
            out_features=self.size_memory,
            activation_class=nn.ReLU,
            device=self.device
        )
        # self.feature = Linear(in_features=self.observation_size, out_features=self.size_memory)
        if cfg.network.attention.identity_reordering:
            self.attention = SelfAttentionLayerIdentityReordered(
                size_memory=self.size_memory,
                n_head=self.n_heads,
                attention_mlp_depth=self.attention_mlp_depth,
                device=self.device,
            )
        else:
            self.attention = SelfAttentionLayer(
                size_memory=self.size_memory,
                n_head=self.n_heads,
                attention_mlp_depth=self.attention_mlp_depth,
                device=self.device,
            )
        self.forget_gate = Gate(input_size=self.size_memory, size_memory=self.size_memory)
        self.input_gate = Gate(input_size=self.size_memory, size_memory=self.size_memory)

        init_weight = 0.5
        self.linear_update_previous = SigmoidLinear(size_memory=self.size_memory, init_weight=init_weight)
        self.linear_update_new = SigmoidLinear(size_memory=self.size_memory, init_weight=1-init_weight)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        defaults = [NO_DEFAULT, NO_DEFAULT, NO_DEFAULT]  # We want to get an error if either memory or value are missing.
        is_init = tensordict.get("is_init").squeeze(-1)
        observation, memory, buffer = (
            tensordict.get(key, default)
            for key, default in zip(self.in_keys, defaults)
        )
        batch_size = is_init.size()

        if self.reset_memory:
            # TO-DO: Implement a non-zero, non-random init routine
            memory = 0.1*torch.randn((*batch_size, self.num_memories, self.size_memory), device=memory.device)

        buffer = buffer.view(*batch_size, self.buffer_size, self.observation_size)
        for i in range(self.buffer_size):
            # Preprocess the observation into a vector of the right size for the memory
            observation_feature = self.feature(buffer[..., i, :])
            # Input and forget gates
            forget_gate = self.forget_gate(memory, observation_feature)
            input_gate = self.input_gate(memory, observation_feature)
            # Compute memory update
            next_memory = self.attention(memory, observation_feature)
            next_memory = nn.Tanh()(next_memory)
            # LSTM memory update:
            # next_memory = forget_gate * memory + input_gate * next_memory
            # Instead, try something simpler:
            memory = self.linear_update_previous(memory) + self.linear_update_new(next_memory)

        # Compute the "action" (whatever is processed into the action) for this step
        # This uses the observation and memory state
        memory_observation = torch.cat((memory.view([*batch_size, -1]), observation.view([*batch_size, -1])), dim=-1)
        action_out = self.action_mlp(memory_observation)

        # Write output to tensordict
        tensordict.set(self.out_keys[0], action_out)
        tensordict.set(self.out_keys[1], memory)

        return tensordict


class SelfAttentionBufferMemoryCritic(TensorDictModuleBase):
    def __init__(self, cfg, action_spec, in_keys=None, out_keys=None):
        super().__init__()
        if out_keys is None:
            out_keys = ["state_action_value", ("next", str(cfg.network.attention.critic_memory_key))]
        if in_keys is None:
            in_keys = [
                "observation",
                "action",
                str(cfg.network.attention.critic_memory_key),
                str(cfg.network.buffer.buffer_observation_key),
            ]
        assert (out_keys[1][1] == in_keys[2])

        self.memory_key = in_keys[2]
        self.buffer_key = in_keys[3]
        self.in_keys = in_keys
        self.out_keys = out_keys

        self.num_memories = cfg.network.attention.num_memories
        self.size_memory = cfg.network.attention.size_memory
        self.action_spec = action_spec
        self.n_heads = cfg.network.attention.n_heads
        self.attention_mlp_depth = cfg.network.attention.attention_mlp_depth
        self.observation_size = cfg.env.num_sensors
        self.num_actions = cfg.env.num_actuators
        self.buffer_size = cfg.network.buffer.size
        self.device = network_device(cfg)
        self.reset_memory = cfg.network.attention.reset_memory

        self.critic_net = tqc_critic_net(cfg, model='attention')
        self.feature = MLP(
            num_cells=[128],
            out_features=self.size_memory,
            activation_class=nn.ReLU,
            device=self.device
        )
        # self.feature = Linear(in_features=self.observation_size+self.num_actions, out_features=self.size_memory)
        if cfg.network.attention.identity_reordering:
            self.attention = SelfAttentionLayerIdentityReordered(
                size_memory=self.size_memory,
                n_head=self.n_heads,
                attention_mlp_depth=self.attention_mlp_depth,
                device=self.device,
            )
        else:
            self.attention = SelfAttentionLayer(
                size_memory=self.size_memory,
                n_head=self.n_heads,
                attention_mlp_depth=self.attention_mlp_depth,
                device=self.device,
            )

        self.forget_gate = Gate(input_size=self.size_memory, size_memory=self.size_memory)
        self.input_gate = Gate(input_size=self.size_memory, size_memory=self.size_memory)

        init_weight = 0.5
        self.linear_update_previous = SigmoidLinear(size_memory=self.size_memory, init_weight=init_weight)
        self.linear_update_new = SigmoidLinear(size_memory=self.size_memory, init_weight=1 - init_weight)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        defaults = [NO_DEFAULT, NO_DEFAULT, NO_DEFAULT, NO_DEFAULT]
        is_init = tensordict.get("is_init").squeeze(-1)
        observation, action, memory, buffer = (
            tensordict.get(key, default)
            for key, default in zip(self.in_keys, defaults)
        )
        batch_size = is_init.size()
        observation_action = torch.cat((observation.view([*batch_size, -1]), action.view([*batch_size, -1])), dim=-1)

        if self.reset_memory:
            # TO-DO: Implement a non-zero, non-random init routine
            memory = 0.1 * torch.randn((*batch_size, self.num_memories, self.size_memory), device=memory.device)

        buffer = buffer.view(*batch_size, self.buffer_size, self.observation_size)
        for i in range(self.buffer_size):
            # Preprocess the observation into a vector of the right size for the memory
            observation_feature = self.feature(buffer[..., i, :])
            # Input and forget gates
            forget_gate = self.forget_gate(memory, observation_feature)
            input_gate = self.input_gate(memory, observation_feature)
            # Compute memory update
            next_memory = self.attention(memory, observation_feature)
            next_memory = nn.Tanh()(next_memory)
            # LSTM memory update:
            # next_memory = forget_gate * memory + input_gate * next_memory
            # Instead, try something simpler:
            memory = self.linear_update_previous(memory) + self.linear_update_new(next_memory)

        # Compute the critic output from memory, observation and action
        memory_observation_action = torch.cat((memory.view([*batch_size, -1]), observation_action), dim=-1)
        state_action_value = self.critic_net(memory_observation_action)

        # Write output to tensordict
        tensordict.set(self.out_keys[0], state_action_value)
        tensordict.set(self.out_keys[1], memory)

        return tensordict
