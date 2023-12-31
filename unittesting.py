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

    print('stop here')

    # Test evaluation rollout
    eval_steps = 500
    eval_rollout = eval_env.rollout(
        eval_steps,
        model[0],
        auto_cast_to_device=True,
        break_when_any_done=True,
    )

    print('here')

    print(f'Output of critic for rollout with {eval_steps} steps has shape:')
    print(model[1].forward(eval_rollout["observation"], eval_rollout["action"]).shape)

    print('Check if I understand model transition dynamics...')
    print((eval_rollout["loc"] - model[0](eval_rollout)["loc"]).mean())

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
        reward_sum = np.sum(np.array(rewards))
        print(f'Simple rollout reward sum {reward_sum:.2f}')
        rewards = np.array(rewards)
        if False:
            l = []
            for i in range(10):
                l.append(np.sum(rewards[2*i:2*(i+1)]))
                #print(rewards[2*i:2*(i+1)])
                #print(np.sum(rewards[2*i:2*(i+1)]))
                #print("\n")
        else:
            l = rewards[:10]
        print(l)
        print(f'original rewards {rewards[:20]}')
        return uu

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
        for _ in range(eval_steps):
            u = solver_step(u, torch.zeros(actuator_locs.shape))
            uu.append(u)
        return uu

    uu_simplest = simplest_rollout(td, 100)

    uu_simple = simple_rollout(td, eval_env, 100)

    # Comparing outputs of frame skipped env with normal env
    eval_env_skip = TransformedEnv(eval_env, FrameSkipTransform(frame_skip=2))



if __name__ == '__main__':
    main()
