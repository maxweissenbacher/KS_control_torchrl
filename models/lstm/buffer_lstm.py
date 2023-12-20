from tensordict.nn import TensorDictModule, TensorDictSequential, TensorDictModuleBase
from torch import nn
from torch.nn import Module
from torchrl.modules import MLP, LSTMModule
from tensordict.tensordict import TensorDictBase, NO_DEFAULT
from models.memoryless.base import tqc_critic_net
from utils.device_finder import network_device


class LSTMBufferModule(TensorDictModuleBase):
    def __init__(self, cfg, in_keys=None, out_keys=None):
        super().__init__()

        if in_keys is None:
            in_keys = [str(cfg.network.buffer.buffer_observation_key)]
        if out_keys is None:
            raise ValueError("Out key must be specified. out_keys argument expects a list of strings.")
        self.in_keys = in_keys
        self.out_keys = out_keys
        self.buffer_key = in_keys[0]
        self.hidden_key = '_buffer_lstm_hidden_state'
        self.cell_key = '_buffer_lstm_cell_state'

        self.buffer_size = cfg.network.buffer.size
        self.observation_size = cfg.env.num_sensors
        self.hidden_size = cfg.network.buffer_lstm.hidden_size
        self.num_layers = cfg.network.buffer_lstm.num_layers

        # this variable determines whether the hidden and cell state are passed from one iteration to the next
        self.use_hidden = bool(cfg.network.buffer_lstm.use_hidden)

        self.lstm = nn.LSTM(
            input_size=self.observation_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            device=network_device(cfg),
            dropout=cfg.network.buffer_lstm.dropout,
            batch_first=True,
        )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        buffer = tensordict.get(self.buffer_key, NO_DEFAULT)
        prev_hidden = tensordict.get(self.hidden_key, None)
        prev_cell = tensordict.get(self.cell_key, None)

        buffer = buffer.view(-1, self.buffer_size, self.observation_size)
        if prev_hidden is None or prev_cell is None or not self.use_hidden:
            output, (hidden, cell) = self.lstm(buffer)
        else:
            prev_hidden = prev_hidden.view(self.num_layers, -1, self.hidden_size)
            prev_cell = prev_cell.view(self.num_layers, -1, self.hidden_size)
            output, (hidden, cell) = self.lstm(buffer, (prev_hidden, prev_cell))
        output = output[..., -1, :]
        if output.shape[0] == 1:
            output = output.view(-1)
        if hidden.shape[0] == 1:
            hidden = hidden.squeeze(dim=0)
        if cell.shape[0] == 1:
            cell = cell.squeeze(dim=0)

        tensordict.set(self.out_keys[0], output)
        tensordict.set(self.hidden_key, hidden)
        tensordict.set(self.cell_key, cell)

        return tensordict


def buffer_lstm_actor(cfg, action_spec, in_keys=None, out_keys=None):
    """
    We assume that in_keys has two elements:
    in_keys[0] is the key for observations
    in_keys[1] is the key for the previous action
    """
    if out_keys is None:
        out_keys = ["_actor_net_out"]
    if in_keys is None:
        in_keys = ["observation", "prev_action", str(cfg.network.buffer.buffer_observation_key)]
    observation_key = in_keys[0]
    previous_action_key = in_keys[1]
    buffer_key = in_keys[2]
    lstm_key = "_embed"
    final_layer_observation_key = "_observation_mlp"

    activation = nn.ReLU

    mlp_observation_residual = TensorDictModule(
        MLP(num_cells=cfg.network.buffer_lstm.preprocessing_mlp_sizes,
            out_features=cfg.network.buffer_lstm.feature_size,
            activation_class=activation),
        in_keys=[observation_key, previous_action_key],
        out_keys=[final_layer_observation_key],
    )

    """
    feature_for_lstm = TensorDictModule(
        MLP(num_cells=cfg.network.buffer_lstm.preprocessing_mlp_sizes,
            out_features=cfg.network.buffer_lstm.feature_size,
            activation_class=activation),
        in_keys=[observation_key, previous_action_key],
        out_keys=[lstm_key],
    )
    """

    lstm = LSTMBufferModule(cfg, in_keys=[buffer_key], out_keys=[lstm_key])

    """
    lstm = LSTMModule(
        input_size=cfg.network.buffer_lstm.feature_size,
        hidden_size=cfg.network.buffer_lstm.hidden_size,
        num_layers=cfg.network.buffer_lstm.num_layers,
        device=network_device(cfg),
        dropout=cfg.network.buffer_lstm.dropout,
        in_key=lstm_key,
        out_key=lstm_key,
        batch_first=True,
    )
    """

    final_net = MLP(
        num_cells=cfg.network.buffer_lstm.final_net_sizes,
        out_features=2 * action_spec.shape[-1],
        activation_class=activation
    )
    final_net[-1].bias.data.fill_(0.0)
    final_mlp = TensorDictModule(
        final_net,
        in_keys=[lstm_key, final_layer_observation_key],  # final net sees the original observation and the LSTM state
        out_keys=out_keys,
    )

    actor_module = TensorDictSequential(mlp_observation_residual, lstm, final_mlp)

    # TO-DO: Look at the cuDNN optimisation options (for computing loss)

    return actor_module


def buffer_lstm_critic(cfg, in_keys=None, out_keys=None):

    # TO-DO : introduce proper hyperparameters and network layouts...
    # Graph should be connected correctly

    if out_keys is None:
        out_keys = ["state_action_value"]
    if in_keys is None:
        in_keys = ["observation", "action", "prev_action", str(cfg.network.buffer.buffer_observation_key)]
    observation_key = in_keys[0]
    action_key = in_keys[1]
    previous_action_key = in_keys[2]
    buffer_key = in_keys[3]
    lstm_key = "_embed_critic"
    state_action_embed_key = "_state_action_embed"

    activation = nn.ReLU

    mlp_state_action = TensorDictModule(
        MLP(num_cells=cfg.network.buffer_lstm.preprocessing_mlp_sizes,
            out_features=cfg.network.buffer_lstm.feature_size,
            activation_class=activation),
        in_keys=[observation_key, action_key],
        out_keys=[state_action_embed_key],
    )

    """
    mlp_state_prev_action = TensorDictModule(
        MLP(num_cells=cfg.network.buffer_lstm.preprocessing_mlp_sizes,
            out_features=cfg.network.buffer_lstm.feature_size,
            activation_class=activation),
        in_keys=[observation_key, previous_action_key],
        out_keys=[lstm_key],
    )

    lstm = LSTMModule(
        input_size=cfg.network.buffer_lstm.feature_size,
        hidden_size=cfg.network.buffer_lstm.hidden_size,
        num_layers=cfg.network.buffer_lstm.num_layers,
        device=network_device(cfg),
        in_key=lstm_key,
        out_key=lstm_key,
    )
    """

    lstm = LSTMBufferModule(cfg, in_keys=[buffer_key], out_keys=[lstm_key])

    tqc_critic_mlp = TensorDictModule(
        tqc_critic_net(cfg, model='lstm'),
        in_keys=[lstm_key, state_action_embed_key],
        out_keys=out_keys,
    )

    critic_module = TensorDictSequential(
        mlp_state_action,
        # mlp_state_prev_action,
        lstm,
        tqc_critic_mlp
    )

    return critic_module
