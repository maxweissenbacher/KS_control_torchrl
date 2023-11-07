from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torchrl.modules import MLP, LSTMModule
from models.memoryless.base import tqc_critic_net


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


def lstm_critic(cfg, in_keys, out_keys, action_spec):

    # TO-DO : introduce proper hyperparameters and network layouts...
    # Graph should be connected correctly

    observation_key = in_keys[0]
    action_key = in_keys[1]
    previous_action_key = in_keys[2]
    lstm_key = "_embed_critic"
    state_action_embed_key = "_state_action_embed"
    critic_in_key = "_critic_input_key"

    activation = nn.ReLU

    mlp_state_action = TensorDictModule(
        MLP(num_cells=cfg.network.lstm.feature_for_final_layer_sizes,
            out_features=cfg.network.lstm.feature_size,
            activation_class=activation),
        in_keys=[observation_key, action_key],
        out_keys=[state_action_embed_key],
    )

    mlp_state_prev_action = TensorDictModule(
        MLP(num_cells=cfg.network.lstm.feature_for_final_layer_sizes,
            out_features=cfg.network.lstm.feature_size,
            activation_class=activation),
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

    final_mlp = TensorDictModule(
        MLP(num_cells=cfg.network.lstm.feature_for_final_layer_sizes,
            out_features=cfg.network.lstm.feature_size,
            activation_class=activation),
        in_keys=[lstm_key, state_action_embed_key],
        out_keys=[critic_in_key],
    )

    tqc_critic_mlp = TensorDictModule(
        tqc_critic_net(cfg),
        in_keys=[critic_in_key],
        out_keys=[critic_in_key],
    )

    critic_module = TensorDictSequential(
        mlp_state_action,
        mlp_state_prev_action,
        lstm,
        final_mlp,
        tqc_critic_mlp
    )

    return critic_module
