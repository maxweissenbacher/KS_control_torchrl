import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.modules import MLP, ValueOperator
from typing import Tuple


def basic_tqc_actor(cfg, in_keys, out_keys, action_spec):
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