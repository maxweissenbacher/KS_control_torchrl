import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.modules import MLP, ValueOperator
from typing import Tuple


def buffer_tqc_actor(cfg, action_spec, in_keys=None, out_keys=None):
    if out_keys is None:
        out_keys = ["_actor_net_out"]
    if in_keys is None:
        in_keys = [str(cfg.network.buffer.buffer_observation_key)]
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
    def __init__(self, cfg):
        super().__init__()
        activation = nn.ReLU
        self.nets = []
        qvalue_net_kwargs = {
            "num_cells": cfg.network.critic_hidden_sizes,
            "out_features": cfg.network.n_quantiles,
            "activation_class": activation,
        }
        for i in range(cfg.network.n_nets):
            net = MLP(**qvalue_net_kwargs)
            self.add_module(f'critic_net_{i}', net)
            self.nets.append(net)

    def forward(self, *inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        if len(inputs) > 1:
            inputs = (torch.cat([*inputs], -1),)
        quantiles = torch.stack(tuple(net(*inputs) for net in self.nets), dim=-2)  # batch x n_nets x n_quantiles
        return quantiles


def buffer_tqc_critic(cfg, in_keys=None, out_keys=None):
    if out_keys is None:
        out_keys = ["state_action_value"]
    if in_keys is None:
        in_keys = ["action", str(cfg.network.buffer.buffer_observation_key)]

    qvalue_net = tqc_critic_net(cfg)
    qvalue = ValueOperator(
        in_keys=in_keys,
        out_keys=out_keys,
        module=qvalue_net,
    )

    return qvalue

