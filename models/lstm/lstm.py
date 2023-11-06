import torch
import numpy as np
from tensordict.nn import InteractionType, TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn, optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs import Compose, EnvCreator, ParallelEnv, TransformedEnv
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
from models.attention.self_attention import SelfAttentionMemoryActor


def lstm_actor(cfg, in_keys, out_keys, action_spec):
    """
    We assume that in_keys has two elements:
    in_keys[0] is the key for observations
    in_keys[1] is the key for the previous action
    """
    observation_key = in_keys[0]
    previous_action_key = in_keys[1]
    lstm_key = "_embed"
    final_layer_observation_key = "_observation_mlp"

    activation = nn.ReLU

    mlp_observation_residual = TensorDictModule(
        MLP(num_cells=cfg.network.lstm.feature_for_final_layer_sizes,
            out_features=cfg.network.lstm.feature_size,
            activation_class=activation),
        in_keys=[observation_key],
        out_keys=[final_layer_observation_key],
    )

    feature_for_lstm = TensorDictModule(
        MLP(num_cells=cfg.network.lstm.feature_for_final_layer_sizes,
            out_features=cfg.network.lstm.feature_size,
            activation_class=activation),
        #nn.Linear(cfg.env.num_sensors, cfg.network.lstm.feature_size, bias=False),
        in_keys=[observation_key, previous_action_key],
        out_keys=[lstm_key],
    )

    lstm = LSTMModule(
        input_size=cfg.network.lstm.feature_size,
        hidden_size=cfg.network.lstm.hidden_size,
        device=cfg.network.device,
        in_key=lstm_key,
        out_key=lstm_key,
    )

    final_net = MLP(
        num_cells=cfg.network.lstm.final_net_sizes,
        out_features=2 * action_spec.shape[-1],
        activation_class=activation
    )
    #final_net[-1].bias.data.fill_(0.0)
    final_mlp = TensorDictModule(
        final_net,
        in_keys=[lstm_key, final_layer_observation_key],  # final net sees the original observation and the LSTM state
        out_keys=out_keys,
    )

    actor_module = TensorDictSequential(mlp_observation_residual, feature_for_lstm, lstm, final_mlp)

    # TO-DO: Look at the cuDNN optimisation options (for computing loss)

    return actor_module


def lstm_critic():
    return None

