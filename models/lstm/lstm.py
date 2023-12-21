from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torchrl.modules import MLP, LSTMModule
from models.memoryless.base import tqc_critic_net
from utils.device_finder import network_device


def lstm_actor(cfg, action_spec, in_keys=None, out_keys=None):
    """
    We assume that in_keys has two elements:
    in_keys[0] is the key for observations
    in_keys[1] is the key for the previous action
    """
    if out_keys is None:
        out_keys = ["_actor_net_out"]
    if in_keys is None:
        in_keys = ["observation", "prev_action"]
    observation_key = in_keys[0]
    previous_action_key = in_keys[1]
    lstm_key = "_embed"
    final_layer_observation_key = "_observation_mlp"

    activation = nn.ReLU

    """
    mlp_observation_residual = TensorDictModule(
        MLP(num_cells=cfg.network.lstm.preprocessing_mlp_sizes,
            out_features=cfg.network.lstm.feature_size,
            activation_class=activation),
        in_keys=[observation_key],
        out_keys=[final_layer_observation_key],
    )
    """

    feature_for_lstm = TensorDictModule(
        MLP(num_cells=cfg.network.lstm.preprocessing_mlp_sizes,
            out_features=cfg.network.lstm.feature_size,
            activation_class=activation),
        in_keys=[observation_key],
        out_keys=[lstm_key],
    )

    lstm = LSTMModule(
        input_size=cfg.network.lstm.feature_size,
        hidden_size=cfg.network.lstm.hidden_size,
        device=network_device(cfg),
        in_key=lstm_key,
        out_key=lstm_key,
    )

    final_net = MLP(
        num_cells=cfg.network.lstm.final_net_sizes,
        out_features=2 * action_spec.shape[-1],
        activation_class=activation
    )
    final_net[-1].bias.data.fill_(0.0)
    final_mlp = TensorDictModule(
        final_net,
        in_keys=[lstm_key, observation_key],  # final net sees the original observation and the LSTM state
        out_keys=out_keys,
    )

    actor_module = TensorDictSequential(feature_for_lstm, lstm, final_mlp)

    return actor_module


def lstm_critic(cfg, in_keys=None, out_keys=None):

    # TO-DO : introduce proper hyperparameters and network layouts...
    # Graph should be connected correctly

    if out_keys is None:
        out_keys = ["state_action_value"]
    if in_keys is None:
        in_keys = ["observation", "action", "prev_action"]
    observation_key = in_keys[0]
    action_key = in_keys[1]
    previous_action_key = in_keys[2]
    lstm_key = "_embed_critic"
    state_action_embed_key = "_state_action_embed"

    activation = nn.ReLU

    """
    mlp_state_action = TensorDictModule(
        MLP(num_cells=cfg.network.lstm.preprocessing_mlp_sizes,
            out_features=cfg.network.lstm.feature_size,
            activation_class=activation),
        in_keys=[observation_key, action_key],
        out_keys=[state_action_embed_key],
    )
    """

    mlp_state_prev_action = TensorDictModule(
        MLP(num_cells=cfg.network.lstm.preprocessing_mlp_sizes,
            out_features=cfg.network.lstm.feature_size,
            activation_class=activation),
        in_keys=[observation_key, action_key],
        out_keys=[lstm_key],
    )

    lstm = LSTMModule(
        input_size=cfg.network.lstm.feature_size,
        hidden_size=cfg.network.lstm.hidden_size,
        device=network_device(cfg),
        in_key=lstm_key,
        out_key=lstm_key,
    )

    tqc_critic_mlp = TensorDictModule(
        tqc_critic_net(cfg, model='lstm'),
        in_keys=[lstm_key, observation_key, action_key],
        out_keys=out_keys,
    )

    critic_module = TensorDictSequential(
        # mlp_state_action,
        mlp_state_prev_action,
        lstm,
        tqc_critic_mlp
    )

    return critic_module
