# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy

import tempfile
from contextlib import nullcontext
import torch
import numpy as np
from tensordict.nn import InteractionType, TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn, optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs import Compose, DoubleToFloat, EnvCreator, ParallelEnv, TransformedEnv
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter
from torchrl.envs.transforms import FiniteTensorDictCheck, ObservationNorm, FrameSkipTransform
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator, LSTMModule
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import SoftUpdate
from torchrl.data import CompositeSpec, TensorSpec
from torchrl.objectives.common import LossModule
from tensordict.tensordict import TensorDict, TensorDictBase
from typing import Tuple
from solver.KS_environment import KSenv


# ====================================================================
# Environment utils
# -----------------


def make_ks_env(cfg):
    transforms = Compose(
        InitTracker(),
        StepCounter(cfg.env.max_episode_steps // cfg.env.frame_skip),
        # DoubleToFloat(),
        RewardSum(),
        FiniteTensorDictCheck(),
        #FrameSkipTransform(frame_skip=cfg.env.frame_skip),
        # ObservationNorm(in_keys=["observation"], loc=0., scale=10.),
    )
    device = cfg.collector.collector_device
    actuator_locs = torch.tensor(np.linspace(start=0.0, stop=2*torch.pi, num=cfg.env.num_actuators, endpoint=False), device=device)
    sensor_locs = torch.tensor(np.linspace(start=0.0, stop=2*torch.pi, num=cfg.env.num_sensors, endpoint=False), device=device)

    train_env = TransformedEnv(
        ParallelEnv(
            cfg.collector.env_per_collector,
            EnvCreator(
                lambda: KSenv(
                    nu=float(cfg.env.nu),
                    actuator_locs=actuator_locs,
                    sensor_locs=sensor_locs,
                    burn_in=int(cfg.env.burnin),
                    frame_skip=int(cfg.env.frame_skip),
                    soft_action=bool(cfg.env.soft_action),
                    autoreg_weight=float(cfg.env.autoreg_action),
                    actuator_loss_weight=float(cfg.optim.actuator_loss_weight),
                    device=cfg.collector.collector_device,
                )
            )
        ),
        transforms
    )
    train_env.set_seed(cfg.env.seed)

    eval_env = TransformedEnv(
        ParallelEnv(
            cfg.collector.env_per_collector,
            EnvCreator(
                lambda: KSenv(
                    nu=float(cfg.env.nu),
                    actuator_locs=actuator_locs,
                    sensor_locs=sensor_locs,
                    burn_in=int(cfg.env.burnin),
                    frame_skip=int(cfg.env.frame_skip),
                    soft_action=bool(cfg.env.soft_action),
                    autoreg_weight=float(cfg.env.autoreg_action),
                    actuator_loss_weight=float(cfg.optim.actuator_loss_weight),
                    device=cfg.collector.collector_device,
                )
            )
        ),
        train_env.transform.clone()
    )

    return train_env, eval_env


# ====================================================================
# Collector and replay buffer
# ---------------------------


def make_collector(cfg, train_env, actor_model_explore):
    """Make collector."""
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        init_random_frames=cfg.collector.init_random_frames // cfg.env.frame_skip,
        frames_per_batch=cfg.collector.frames_per_batch // cfg.env.frame_skip,
        total_frames=cfg.collector.total_frames // cfg.env.frame_skip,
        device=cfg.collector.collector_device,
    )
    collector.set_seed(cfg.env.seed)
    return collector


def make_replay_buffer(
    batch_size,
    prb=False,
    buffer_size=1000000,
    buffer_scratch_dir=None,
    device="cpu",
    prefetch=3,
):
    with (
        tempfile.TemporaryDirectory()
        if buffer_scratch_dir is None
        else nullcontext(buffer_scratch_dir)
    ) as scratch_dir:
        if prb:
            replay_buffer = TensorDictPrioritizedReplayBuffer(
                alpha=0.7,
                beta=0.5,
                pin_memory=False,
                prefetch=prefetch,
                storage=LazyMemmapStorage(
                    buffer_size,
                    scratch_dir=scratch_dir,
                    device=device,
                ),
                batch_size=batch_size,
            )
        else:
            replay_buffer = TensorDictReplayBuffer(
                pin_memory=False,
                prefetch=prefetch,
                storage=LazyMemmapStorage(
                    buffer_size,
                    scratch_dir=scratch_dir,
                    device=device,
                ),
                batch_size=batch_size,
            )
        return replay_buffer


# ====================================================================
# Model architecture for critic
# -----------------------------


class tqc_critic_net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.nets = []
        qvalue_net_kwargs = {
            "num_cells": cfg.network.critic_hidden_sizes,
            "out_features": cfg.network.n_quantiles,
            "activation_class": get_activation(cfg),
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


# ====================================================================
# Actor architectures
# -------------------


def basic_tqc_actor(cfg, in_keys, out_keys, action_spec):
    actor_module = TensorDictModule(
        MLP(num_cells=cfg.network.actor_hidden_sizes,
            out_features=2 * action_spec.shape[-1],
            activation_class=get_activation(cfg)),
        in_keys=in_keys,
        out_keys=out_keys,
    )
    return actor_module


def lstm_tqc_actor(cfg, in_keys, out_keys, action_spec):
    lstm_key = "embed"

    feature = TensorDictModule(
        MLP(num_cells=cfg.network.lstm.feature_hidden_sizes,
            out_features=cfg.network.lstm.feature_out_size,
            activation_class=get_activation(cfg)),
        in_keys=in_keys,
        out_keys=[lstm_key],
    )

    lstm = LSTMModule(
        input_size=feature.module[-1].out_features,
        hidden_size=cfg.network.lstm.hidden_size,
        device=cfg.network.device,
        in_key=lstm_key,
        out_key=lstm_key,
    )

    final_net = MLP(
        num_cells=[cfg.network.lstm.hidden_size // 2],
        out_features=2 * action_spec.shape[-1],
        activation_class=get_activation(cfg)
    )
    final_net[-1].bias.data.fill_(0.0)
    final_mlp = TensorDictModule(
        final_net,
        in_keys=lstm_key,
        out_keys=out_keys,
    )

    actor_module = TensorDictSequential(feature, lstm, final_mlp)

    # Look at the cuDNN optimisation options (for computing loss)

    return actor_module


# ====================================================================
# Model
# -----


def make_tqc_agent(cfg, train_env, eval_env, device):
    """Make TQC agent."""
    # Define Actor Network
    in_keys_actor = ["observation"]
    out_keys_actor = ["actor_net_out"]
    action_spec = train_env.action_spec
    if train_env.batch_size:
        action_spec = action_spec[(0,) * len(train_env.batch_size)]

    actor_net = None
    if cfg.network.architecture == 'base':
        actor_net = basic_tqc_actor(cfg, in_keys=in_keys_actor, out_keys=out_keys_actor, action_spec=action_spec)
    elif cfg.network.architecture == 'lstm':
        actor_net = lstm_tqc_actor(cfg, in_keys=in_keys_actor, out_keys=out_keys_actor, action_spec=action_spec)

    actor_extractor = TensorDictModule(
        NormalParamExtractor(
            scale_mapping=f"biased_softplus_{cfg.network.default_policy_scale}",
            scale_lb=cfg.network.scale_lb,
        ),
        in_keys=out_keys_actor,
        out_keys=["loc", "scale"],
    )

    actor_module = TensorDictSequential(actor_net, actor_extractor)

    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        module=actor_module,
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": action_spec.space.low,
            "max": action_spec.space.high,
            "tanh_loc": False,  # can be omitted since this is default value
        },
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=True,
    )

    # Define Critic Network
    qvalue_net = tqc_critic_net(cfg)
    qvalue = ValueOperator(
        in_keys=["action"] + in_keys_actor,
        module=qvalue_net,
    )

    model = nn.ModuleList([actor, qvalue]).to(device)

    # init nets
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = eval_env.reset()
        td = td.to(device)
        for net in model:
            net(td)
    del td
    eval_env.close()

    return model, model[0]


# ====================================================================
# Quantile Huber Loss
# -------------------


def quantile_huber_loss_f(quantiles, samples):
    """
    Quantile Huber loss from the original PyTorch TQC implementation.
    See: https://github.com/SamsungLabs/tqc_pytorch/blob/master/tqc/functions.py
    """
    pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1,
                             abs_pairwise_delta - 0.5,
                             pairwise_delta ** 2 * 0.5)

    n_quantiles = quantiles.shape[2]
    tau = torch.arange(n_quantiles, device=quantiles.device).float() / n_quantiles + 1 / 2 / n_quantiles
    loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
    return loss


# ====================================================================
# TQC Loss
# ---------

class TQCLoss(LossModule):
    def __init__(
            self,
            actor_network,
            qvalue_network,
            gamma,
            top_quantiles_to_drop,
            alpha_init,
            device
    ):
        super(type(self), self).__init__()
        super().__init__()

        self.convert_to_functional(
            actor_network,
            "actor",
            create_target_params=False,
            funs_to_decorate=["forward", "get_dist"],
        )

        self.convert_to_functional(
            qvalue_network,
            "critic",
            create_target_params=True
        )  # do we need to specify the compare_against argument here? Check!

        self.device = device
        self.log_alpha = torch.tensor([np.log(alpha_init)], requires_grad=True, device=self.device)
        self.gamma = gamma
        self.top_quantiles_to_drop = top_quantiles_to_drop

        # Compute target entropy
        action_spec = getattr(self.actor, "spec", None)
        if action_spec is None:
            print("Could not deduce action spec from actor network.")
        if not isinstance(action_spec, CompositeSpec):
            action_spec = CompositeSpec({"action": action_spec})
        action_container_len = len(action_spec.shape)
        self.target_entropy = -float(action_spec["action"].shape[action_container_len:].numel())

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        td_next = tensordict["next"]
        reward = td_next["reward"]
        not_done = tensordict["done"].logical_not()

        alpha = torch.exp(self.log_alpha)

        # Q-loss
        with torch.no_grad():
            # get policy action
            td_next = self.actor(td_next, params=self.actor_params)
            td_next = self.critic(td_next, params=self.target_critic_params)  # check if this works!!
            # At initialisation (no update steps), this outputs the same as using params = self.critic_params

            next_log_pi = td_next["sample_log_prob"]
            next_log_pi = torch.unsqueeze(next_log_pi, dim=-1)

            # compute and cut quantiles at the next state
            next_z = td_next["state_action_value"]
            sorted_z, _ = torch.sort(next_z.reshape(next_z.shape[0], -1))
            sorted_z_part = sorted_z[:, :-self.top_quantiles_to_drop]

            # compute target
            target = reward + not_done * self.gamma * (sorted_z_part - alpha * next_log_pi)

        td_cur = tensordict
        td_cur = self.critic(td_cur, params=self.critic_params)
        cur_z = td_cur["state_action_value"]
        critic_loss = quantile_huber_loss_f(cur_z, target)

        # --- Policy and alpha loss ---
        td_new = tensordict
        td_new = self.actor(td_new, params=self.actor_params)
        td_new = self.critic(td_new, params=self.critic_params)
        new_log_pi = td_new["sample_log_prob"]
        alpha_loss = -self.log_alpha * (new_log_pi + self.target_entropy).detach().mean()
        actor_loss = (alpha * new_log_pi - td_new["state_action_value"].mean(2).mean(1, keepdim=True)).mean()

        # --- Entropy ---
        with set_exploration_type(ExplorationType.RANDOM):
            dist = self.actor.get_dist(
                tensordict,
                params=self.actor_params,
            )
            a_reparm = dist.rsample()
        log_prob = dist.log_prob(a_reparm).detach()
        entropy = -log_prob.mean()

        return TensorDict(
            {
                "loss_critic": critic_loss,
                "loss_actor": actor_loss,
                "loss_alpha": alpha_loss,
                "alpha": alpha,
                "entropy": entropy,
            },
            batch_size=[]
        )


def make_loss_module(cfg, model):
    """Make loss module and target network updater."""
    # Create TQC loss
    loss_module = TQCLoss(
        actor_network=model[0],
        qvalue_network=model[1],
        device=cfg.network.device,
        gamma=cfg.optim.gamma,
        top_quantiles_to_drop=cfg.network.top_quantiles_to_drop_per_net * cfg.network.n_nets,
        alpha_init=cfg.optim.alpha_init
    )

    # Define Target Network Updater
    target_net_updater = SoftUpdate(loss_module, eps=cfg.optim.target_update_polyak)
    return loss_module, target_net_updater


def make_tqc_optimizer(cfg, loss_module):
    critic_params = list(loss_module.critic_params.flatten_keys().values())
    actor_params = list(loss_module.actor_params.flatten_keys().values())

    optimizer_actor = optim.Adam(
        actor_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps,
    )
    optimizer_critic = optim.Adam(
        critic_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps,
    )
    optimizer_alpha = optim.Adam(
        [loss_module.log_alpha],
        lr=3.0e-4,
    )
    return optimizer_actor, optimizer_critic, optimizer_alpha


# ====================================================================
# General utils
# ---------


def log_metrics(logger, metrics, step):
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)


# Remove this once the logging is fixed, just for testing now
def log_metrics_2(logs, metrics):
    for metric_name, metric_value in metrics.items():
        if metric_name in logs.keys():
            logs[metric_name].append(metric_value)
        else:
            logs[metric_name] = [metric_value]


def get_activation(cfg):
    if cfg.network.activation == "relu":
        return nn.ReLU
    elif cfg.network.activation == "tanh":
        return nn.Tanh
    elif cfg.network.activation == "leaky_relu":
        return nn.LeakyReLU
    else:
        raise NotImplementedError
