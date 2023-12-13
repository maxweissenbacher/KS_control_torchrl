import torch
from torch.nn import Module, Linear, Softmax
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torchrl.modules import MLP, ValueOperator
from typing import Tuple
from models.memoryless.base import tqc_critic_net
import hydra


class CNNBuffer(Module):
    def __init__(self, cfg):
        super().__init__()
        self.buffer_size = cfg.network.buffer.size
        self.observation_size = cfg.env.num_sensors
        self.cnn1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=3,
            kernel_size=2,
        )
        self.cnn2 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=2,
            kernel_size=2,
        )
        self.batchnorm1 = torch.nn.BatchNorm2d(num_features=3)
        self.batchnorm2 = torch.nn.BatchNorm2d(num_features=2)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        y = x.view(-1, 1, self.buffer_size, self.observation_size)
        y = self.cnn1(y)
        y = self.batchnorm1(y)
        y = self.relu(y)
        y = self.maxpool1(y)
        #print(y.shape)
        y = self.cnn2(y)
        y = self.batchnorm2(y)
        y = self.relu(y)
        #y = self.maxpool2(y)
        #print(y.shape)

        if y.shape[0] == 1:
            y = y.view(-1)
        else:
            y = y.view(y.shape[0], -1)
        #print(y.shape)
        return y


def cnn_actor(cfg, action_spec, in_keys=None, out_keys=None):
    if out_keys is None:
        out_keys = ["_actor_net_out"]
    if in_keys is None:
        in_keys = [str(cfg.network.buffer.buffer_observation_key)]
    activation = nn.ReLU

    buffer_cnn = CNNBuffer(cfg)
    mlp = MLP(
        num_cells=cfg.network.actor_hidden_sizes,
        out_features=2 * action_spec.shape[-1],
        activation_class=activation
    )

    actor_module = TensorDictModule(
        nn.Sequential(buffer_cnn, mlp),
        in_keys=in_keys,
        out_keys=out_keys,
    )

    return actor_module


def cnn_critic(cfg, in_keys=None, out_keys=None):
    if out_keys is None:
        out_keys = ["state_action_value"]
    if in_keys is None:
        in_keys = ["action", str(cfg.network.buffer.buffer_observation_key)]

    buffer_cnn = CNNBuffer(cfg)
    cnn_module = TensorDictModule(
        buffer_cnn,
        in_keys=[str(cfg.network.buffer.buffer_observation_key)],
        out_keys=["_cnn_critic_buffer_output"]
    )
    qvalue_net = tqc_critic_net(cfg)
    qvalue = ValueOperator(
        in_keys=["action", "_cnn_critic_buffer_output"],
        out_keys=out_keys,
        module=qvalue_net
    )
    critic_module = TensorDictSequential(cnn_module, qvalue)

    return critic_module


@hydra.main(version_base="1.1", config_path="../../", config_name="config")
def main(cfg):
    test_input = torch.zeros([256, 100])
    buffer_cnn = CNNBuffer(cfg)
    output = buffer_cnn(test_input)
    print(output.shape)

if __name__=='__main__':
    main()
