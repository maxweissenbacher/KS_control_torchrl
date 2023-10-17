import time
import hydra
import numpy as np
import torch
import torch.cuda
import tqdm
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from torchrl.record.loggers import generate_exp_name, get_logger
from utils import (
    log_metrics,
    make_collector,
    make_environment,
    make_loss_module,
    make_replay_buffer,
    make_tqc_agent,
    make_tqc_optimizer,
)


@hydra.main(version_base="1.1", config_path=".", config_name="config")
def main(cfg):
    device = torch.device(cfg.network.device)

    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    # Create environments
    train_env, eval_env = make_environment(cfg)

    # Create agent
    model, exploration_policy = make_tqc_agent(cfg, train_env, eval_env, device)

    # Test evaluation rollout
    eval_steps = 500
    eval_rollout = eval_env.rollout(
        eval_steps,
        model[0],
        auto_cast_to_device=True,
        break_when_any_done=True,
    )

    print(f'Output of critic for rollout with {eval_steps} steps has shape:')
    print(model[1].forward(eval_rollout["observation"], eval_rollout["action"]).shape)

    print('Check if I understand model transition dynamics...')
    print((eval_rollout["loc"] - model[0](eval_rollout)["loc"]).mean())


if __name__ == '__main__':
    main()