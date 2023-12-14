# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
from torchrl.envs import Compose, TransformedEnv, UnsqueezeTransform, CatFrames, FlattenObservation
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter
from torchrl.envs.transforms import FiniteTensorDictCheck, ObservationNorm
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator, LSTMModule
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import SoftUpdate
from torchrl.data import CompositeSpec
from torchrl.objectives.common import LossModule
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.envs.transforms.transforms import TensorDictPrimer
from torchrl.data import UnboundedContinuousTensorSpec
from typing import Tuple
from solver.KS_environment import KSenv
from models.attention.attention_agent import SelfAttentionMemoryActor, SelfAttentionMemoryCritic
from models.attention.attention_onememory_agent import SelfAttentionMemoryActor2, SelfAttentionMemoryCritic2
from models.attention.attention_buffer_agent import SelfAttentionBufferMemoryActor, SelfAttentionBufferMemoryCritic
from models.lstm.lstm import lstm_actor, lstm_critic
from models.gru.gru import gru_actor, gru_critic
from models.memoryless.base import basic_tqc_actor, basic_tqc_critic
from models.buffer.buffer import buffer_tqc_actor, buffer_tqc_critic
from models.cnn.cnn_agent import cnn_actor, cnn_critic
from utils.device_finder import network_device
from utils.rng import env_seed
import wandb


# ====================================================================
# Environment utils
# -----------------


def make_ks_env(cfg):
    # Make transforms
    transform_list = [
        InitTracker(),
        # StepCounter(cfg.env.max_episode_steps // cfg.env.frame_skip),
        RewardSum(),
        FiniteTensorDictCheck(),
        ObservationNorm(in_keys=["observation"], loc=0., scale=10.),
    ]
    # For the self attention memory, add a TensorDictPrimer
    if cfg.network.architecture == 'attention' or cfg.network.architecture == 'attentionBuffer':
        transform_list.append(
            TensorDictPrimer(
                {
                    str(cfg.network.attention.actor_memory_key): UnboundedContinuousTensorSpec(
                        shape=(cfg.network.attention.num_memories, cfg.network.attention.size_memory),
                        dtype=torch.float32,
                        device=cfg.collector.collector_device,
                    ),
                    str(cfg.network.attention.critic_memory_key): UnboundedContinuousTensorSpec(
                        shape=(cfg.network.attention.num_memories, cfg.network.attention.size_memory),
                        dtype=torch.float32,
                        device=cfg.collector.collector_device,
                    ),
                },
                default_value=0.0,
                random=bool(cfg.network.attention.initialise_random_memory),
            )
        )

    # THIS IS ONLY FOR SELF ATTENTION MEMORY WITH ONE SHARED MEMORY
    # For the self attention memory with one memory, add a TensorDictPrimer
    if cfg.network.architecture == 'attention2':
        transform_list.append(
            TensorDictPrimer(
                {
                    str(cfg.network.attention.critic_memory_key): UnboundedContinuousTensorSpec(
                        shape=(cfg.network.attention.num_memories, cfg.network.attention.size_memory),
                        dtype=torch.float32,
                        device=cfg.collector.collector_device,
                    ),
                },
                default_value=0.0,
                random=bool(cfg.network.attention.initialise_random_memory),
            )
        )

    # For the buffer memory, append the Buffer Transforms
    buffer_required = cfg.network.architecture == 'buffer' or \
                      cfg.network.architecture == 'cnn' or \
                      cfg.network.architecture == 'lstm' or \
                      cfg.network.architecture == 'attentionBuffer'
    if buffer_required:
        transform_list.append(
            CatFrames(dim=-1,
                      N=int(cfg.network.buffer.size),
                      in_keys=["observation"],
                      out_keys=[str(cfg.network.buffer.buffer_observation_key)])
        )
        """
        transform_list.append(
            CatFrames(dim=-1,
                      N=int(cfg.network.buffer.size),
                      in_keys=["action"],
                      out_keys=[str(cfg.network.buffer.buffer_observation_key)])
        )
        """
    env_transforms = Compose(*transform_list)

    # Set environment hyperparameters
    device = cfg.collector.collector_device
    actuator_locs = torch.tensor(
        np.linspace(
            start=0.0,
            stop=2 * torch.pi,
            num=cfg.env.num_actuators,
            endpoint=False
        ),
        device=device
    )
    sensor_locs = torch.tensor(
        np.linspace(start=0.0,
                    stop=2 * torch.pi,
                    num=cfg.env.num_sensors,
                    endpoint=False
                    ),
        device=device
    )
    env_params = {
        "nu": float(cfg.env.nu),
        "actuator_locs": actuator_locs,
        "sensor_locs": sensor_locs,
        "burn_in": int(cfg.env.burnin),
        "frame_skip": int(cfg.env.frame_skip),
        "soft_action": bool(cfg.env.soft_action),
        "autoreg_weight": float(cfg.env.autoreg_action),
        "actuator_loss_weight": float(cfg.optim.actuator_loss_weight),
        "device": cfg.collector.collector_device,
    }

    # Create environments
    train_env = TransformedEnv(KSenv(**env_params), env_transforms)
    train_env.set_seed(env_seed(cfg))
    eval_env = TransformedEnv(KSenv(**env_params), train_env.transform.clone())

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
        max_frames_per_traj=cfg.env.max_episode_steps_train,
    )
    collector.set_seed(env_seed(cfg))
    return collector


def make_replay_buffer(cfg, prefetch=3):
    batch_size = cfg.optim.batch_size
    buffer_size = cfg.replay_buffer.size // cfg.env.frame_skip
    buffer_scratch_dir = cfg.replay_buffer.scratch_dir
    device = network_device(cfg)

    # Transforms for replay buffer
    transform_list = []
    buffer_observation_key = str(cfg.network.buffer.buffer_observation_key)
    if cfg.network.architecture == 'buffer':
        transform_list.append(
            CatFrames(
                dim=-1,
                N=int(cfg.network.buffer.size),
                in_keys=["observation", ("next", "observation")],
                out_keys=[buffer_observation_key, ("next", buffer_observation_key)]
            )
        )
        transform_list.append(
            FlattenObservation(
                first_dim=-2,
                last_dim=-1,
                in_keys=[buffer_observation_key, ("next", buffer_observation_key)],
            )
        )
    rpb_transforms = Compose(*transform_list)

    with (
            tempfile.TemporaryDirectory()
            if buffer_scratch_dir is None
            else nullcontext(buffer_scratch_dir)
    ) as scratch_dir:
        if cfg.replay_buffer.prb:
            replay_buffer = TensorDictPrioritizedReplayBuffer(
                alpha=0.7,
                beta=0.5,
                pin_memory=False,
                prefetch=prefetch,
                transform=rpb_transforms,
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
                transform=rpb_transforms,
                storage=LazyMemmapStorage(
                    buffer_size,
                    scratch_dir=scratch_dir,
                    device=device,
                ),
                batch_size=batch_size,
            )
        return replay_buffer


# ====================================================================
# Model
# -----


def make_tqc_agent(cfg, train_env, eval_env):
    """Make TQC agent."""
    device = network_device(cfg)
    # Define Actor Network
    in_keys_actor = ["observation"]
    out_keys_actor = ["_actor_net_out"]
    action_spec = train_env.action_spec
    if train_env.batch_size:
        action_spec = action_spec[(0,) * len(train_env.batch_size)]

    actor_net = None
    if cfg.network.architecture == 'base':
        actor_net = basic_tqc_actor(cfg, action_spec)
    elif cfg.network.architecture == 'lstm':
        actor_net = lstm_actor(cfg, action_spec)
    elif cfg.network.architecture == 'gru':
        actor_net = gru_actor(cfg, action_spec)
    elif cfg.network.architecture == 'attention':
        actor_net = SelfAttentionMemoryActor(cfg, action_spec)
    elif cfg.network.architecture == 'attention2':
        actor_net = SelfAttentionMemoryActor2(cfg, action_spec)
    elif cfg.network.architecture == 'attentionBuffer':
        actor_net = SelfAttentionBufferMemoryActor(cfg, action_spec)
    elif cfg.network.architecture == 'buffer':
        actor_net = buffer_tqc_actor(cfg, action_spec)
    elif cfg.network.architecture == 'cnn':
        actor_net = cnn_actor(cfg, action_spec)

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
    critic = None
    if cfg.network.architecture == 'base':
        critic = basic_tqc_critic(cfg)
    if cfg.network.architecture == 'lstm':
        critic = lstm_critic(cfg)
    if cfg.network.architecture == 'gru':
        critic = gru_critic(cfg)
    if cfg.network.architecture == 'attention':
        critic = SelfAttentionMemoryCritic(cfg, action_spec)
    if cfg.network.architecture == 'attention2':
        critic = SelfAttentionMemoryCritic2(cfg, action_spec)
    if cfg.network.architecture == 'attentionBuffer':
        critic = SelfAttentionBufferMemoryCritic(cfg, action_spec)
    if cfg.network.architecture == 'buffer':
        critic = buffer_tqc_critic(cfg)
    if cfg.network.architecture == 'cnn':
        critic = cnn_critic(cfg)

    model = nn.ModuleList([actor, critic]).to(device)

    # Initialise models
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
        device=network_device(cfg),
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


def log_metrics_wandb(metrics, step):
    wandb.log(data=metrics, step=step)


def log_metrics_offline(logs, metrics):
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
