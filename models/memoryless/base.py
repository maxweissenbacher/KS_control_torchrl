import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.modules import MLP, ValueOperator
from typing import Tuple


def basic_tqc_actor(cfg, action_spec, in_keys=None, out_keys=None):
    if out_keys is None:
        out_keys = ["_actor_net_out"]
    if in_keys is None:
        in_keys = ["observation"]
    activation = nn.ReLU

    actor_module = TensorDictModule(
        MLP(num_cells=cfg.network.actor_hidden_sizes,
            out_features=2 * action_spec.shape[-1],
            activation_class=activation),
        in_keys=in_keys,
        out_keys=out_keys,
    )

    return actor_module


class tqc_critic_net(nn.Module):
    def __init__(self, cfg, model=None):
        super().__init__()
        activation = nn.ReLU
        self.nets = []
        if model is None:  # Memoryless base model
            critic_hidden_sizes = cfg.network.critic_hidden_sizes
            n_quantiles = cfg.network.n_quantiles
            n_nets = cfg.network.n_nets
        elif model == 'lstm':  # LSTM model
            critic_hidden_sizes = cfg.network.lstm.critic_hidden_sizes
            n_quantiles = cfg.network.lstm.n_quantiles
            n_nets = cfg.network.lstm.n_nets
        elif model == 'attention':  # Attention model
            critic_hidden_sizes = cfg.network.attention.critic_hidden_sizes
            n_quantiles = cfg.network.attention.n_quantiles
            n_nets = cfg.network.attention.n_nets
        qvalue_net_kwargs = {
            "num_cells": critic_hidden_sizes,
            "out_features": n_quantiles,
            "activation_class": activation,
        }
        for i in range(n_nets):
            net = MLP(**qvalue_net_kwargs)
            self.add_module(f'critic_net_{i}', net)
            self.nets.append(net)

    def forward(self, *inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        if len(inputs) > 1:
            inputs = (torch.cat([*inputs], -1),)
        quantiles = torch.stack(tuple(net(*inputs) for net in self.nets), dim=-2)  # batch x n_nets x n_quantiles
        return quantiles


def basic_tqc_critic(cfg, in_keys=None, out_keys=None):
    if out_keys is None:
        out_keys = ["state_action_value"]
    if in_keys is None:
        in_keys = ["action", "observation"]

    qvalue_net = tqc_critic_net(cfg)
    qvalue = ValueOperator(
        in_keys=in_keys,
        out_keys=out_keys,
        module=qvalue_net,
    )

    return qvalue