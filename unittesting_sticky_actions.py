import copy
import time
import hydra
import numpy as np
import torch
import torch.cuda
import tqdm
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.envs import Compose, DoubleToFloat, EnvCreator, ParallelEnv, TransformedEnv
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter
from torchrl.envs.transforms import FiniteTensorDictCheck, ObservationNorm, FrameSkipTransform
from solver.KS_solver import KS
from torchrl.record.loggers import generate_exp_name, get_logger
from utils import (
    log_metrics,
    make_collector,
    make_loss_module,
    make_replay_buffer,
    make_tqc_agent,
    make_tqc_optimizer,
    make_ks_env,
)


@hydra.main(version_base="1.1", config_path="", config_name="config")
def main(cfg):
    device = torch.device(cfg.network.device)

    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    # Create environments
    train_env, eval_env = make_ks_env(cfg)

    # Create agent
    model, exploration_policy = make_tqc_agent(cfg, train_env, eval_env, device)

    # Doing a simple rollout with zero actions
    td = eval_env.reset()
    td_copy = copy.deepcopy(td)

    def simple_rollout(td, env, eval_steps):
        rewards = []
        uu = [td["u"]]
        for _ in range(eval_steps):
            td.set("action", torch.zeros(eval_env.action_spec.shape))
            td = env.step(td).get("next")
            rewards.append(td.get("reward").item())
            uu.append(td["u"])
        #print(f'Simple rollout reward sum {reward_sum:.2f}')
        #rewards = np.array(rewards)
        return uu, rewards

    def simplest_rollout(td, eval_steps):
        actuator_locs = torch.tensor(
            np.linspace(start=0.0, stop=2 * torch.pi, num=cfg.env.num_actuators, endpoint=False))
        solver_step = KS(nu=cfg.env.nu,
                         N=64,
                         dt=0.05,
                         actuator_locs=actuator_locs
                         ).advance
        u = td["u"]
        uu = [u]
        rewards = []
        for _ in range(eval_steps):
            u = solver_step(u, torch.zeros(actuator_locs.shape))
            uu.append(u)
            reward = - torch.linalg.norm(u)
            rewards.append(reward)
        return uu, rewards

    len_rollout = 100
    uu_simplest, rewards_simplest = simplest_rollout(td, len_rollout)
    uu_simple, rewards_simple = simple_rollout(td, eval_env, len_rollout)

    print('stop here')

    print('--- u ---')
    for i in range(20):
        diff = uu_simplest[cfg.env.frame_skip * i] - uu_simple[i]
        print(torch.mean(torch.abs(diff)))

    print('--- Rewards ---')
    num = 9
    assert (num * cfg.env.frame_skip < len_rollout)
    rew_simplest = torch.mean(torch.tensor(rewards_simplest[:cfg.env.frame_skip * num]))
    rew_simple = torch.mean(torch.tensor(rewards_simple[:num]))
    print(rew_simplest-rew_simple)

    eval_env.rollout(3)


if __name__ == '__main__':
    main()



from utils import (
    log_metrics,
    make_collector,
    make_loss_module,
    make_replay_buffer,
    make_tqc_agent,
    make_tqc_optimizer,
    make_ks_env,
)