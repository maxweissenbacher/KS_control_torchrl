# Here we wrap the numerical KS solver into a TorchRL environment

from typing import Optional
import numpy as np
import torch
from tensordict.nn import TensorDictModule
from tensordict.tensordict import TensorDict, TensorDictBase
from torch import nn
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, BoundedTensorSpec
from torchrl.envs import EnvBase, Compose, DoubleToFloat, EnvCreator, ParallelEnv, TransformedEnv
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter, FiniteTensorDictCheck, ObservationNorm
from torchrl.envs.utils import check_env_specs
from solver.KS_solver import KS
from plot.plotting import contourplot_KS


class KSenv(EnvBase):
    metadata = {}
    batch_locked = False

    def __init__(self, nu, actuator_locs, sensor_locs, burn_in=1000, initial_amplitude=1e-2, seed=None, device="cpu"):
        # Specify simulation parameters
        self.nu = nu
        self.N = 64
        self.dt = 0.05
        self.action_size = actuator_locs.size()[-1]
        self.actuator_locs = actuator_locs
        self.burn_in = burn_in
        self.initial_amplitude = initial_amplitude
        self.observation_inds = [int(x) for x in (self.N / (2 * np.pi)) * sensor_locs]
        self.num_observations = len(self.observation_inds)
        assert len(self.observation_inds) == len(set(self.observation_inds))
        self.termination_threshold = 20.  # Terminate the simulation if max(u) exceeds this threshold
        self.action_low = -1.0  # Minimum allowed actuation (per actuator)
        self.action_high = 1.0  # Maximum allowed actuation (per actuator)
        self.device = device

        super().__init__(device=device, batch_size=[])
        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

        self.solver_step = KS(nu=self.nu,
                              N=self.N,
                              dt=self.dt,
                              actuator_locs=self.actuator_locs
                              ).advance

    def _step(self, tensordict):
        u = tensordict["u"]  # Solution at previous timestep
        action = tensordict["action"]  # Extract action from tensordict
        new_u = self.solver_step(u, action)  # Take a step using the PDE solver
        new_observation = new_u[self.observation_inds]  # Evaluate at desired indices
        # To allow for batched computations, use this instead:
        # ... however the KS solver needs to be compatible with torch.vmap!
        # new_u = torch.vmap(self.solver_step)(u, action)

        # reward = - torch.linalg.norm(new_u, dim=-1) / self.N  # normalised L2 norm of solution
        reward = - torch.linalg.norm(new_u, dim=-1)  # L2 norm of solution
        reward = reward.view(*tensordict.shape, 1)

        done = (torch.isfinite(new_u).all().logical_not()) or \
               (new_u.abs().max() > self.termination_threshold) or \
               (torch.isfinite(reward).all().logical_not())
        done = done.view(*tensordict.shape, 1)  # Env terminates if NaN value encountered or very large values
        out = TensorDict(
            {
                "u": new_u,
                "observation": new_observation,
                "reward": reward,
                "done": done,
            },
            tensordict.shape,
        )
        return out

    def _reset(self, tensordict):
        # Uniformly random initial data
        # u = torch.rand([*tensordict.shape, tensordict["params", "N"]], generator=self.rng, device=self.device)
        # u = 0.01 * u
        # u = u-u.mean(dim=-1).unsqueeze(-1)

        # Initial data drawn from IID normal distributions
        zrs = torch.zeros([*self.batch_size, self.N])
        ons = torch.ones([*self.batch_size, self.N])
        u = torch.normal(mean=zrs, std=ons, generator=self.rng)
        u = self.initial_amplitude * u
        u = u - u.mean(dim=-1).unsqueeze(-1)

        # Burn in
        for _ in range(self.burn_in):
            u = self.solver_step(u, torch.zeros(self.action_size))

        out = TensorDict(
            {
                "u": u,
                "observation": u[self.observation_inds],
            },
            self.batch_size,
        )
        return out

    def _make_spec(self):
        self.observation_spec = CompositeSpec(
            u=UnboundedContinuousTensorSpec(shape=(*self.batch_size, self.N), dtype=torch.float32),
            observation=UnboundedContinuousTensorSpec(shape=(*self.batch_size, self.num_observations),
                                                      dtype=torch.float32),
            shape=()
        )
        self.state_spec = CompositeSpec(
            u=UnboundedContinuousTensorSpec(shape=(*self.batch_size, self.N), dtype=torch.float32),
            shape=()
        )
        self.action_spec = BoundedTensorSpec(low=self.action_low,
                                             high=self.action_high,
                                             shape=[self.action_size],
                                             dtype=torch.float32)
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(*self.batch_size, 1), dtype=torch.float32)

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng


if __name__ == '__main__':

    # Defining env
    env = KSenv(nu=0.08,
                actuator_locs=torch.tensor([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]),
                sensor_locs=torch.tensor([0.0, 1.0, 2.0]),
                burn_in=0)
    env.reset()
    print('hi')

    transformed_env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            StepCounter(1000),
            DoubleToFloat(),
            RewardSum(),
            FiniteTensorDictCheck(),
            ObservationNorm(in_keys=["observation"], loc=0., scale=10.),
        ),
    )

    check_env_specs(env)

    # print("observation_spec:", env.observation_spec)
    # print("state_spec:", env.state_spec)
    # print("reward_spec:", env.reward_spec)

    td = env.reset()
    print("reset tensordict", td)

    td = env.rand_step(td)
    print("random step tensordict", td)

    # Print the solution with 0 actuation to check for consistency
    # This policy outputs constant zero for the actions
    zeros = nn.Linear(env.observation_spec['observation'].shape.numel(), env.action_size, bias=False)
    zeros.weight = torch.nn.Parameter(torch.zeros(zeros.weight.shape))
    policy = TensorDictModule(
        zeros,
        in_keys=["observation"],
        out_keys=["action"],
    )

    rollout = env.rollout(1000, policy)
    contourplot_KS(rollout["next", "u"].detach().numpy())
    u_env_rollout = rollout["next", "u"].detach().numpy()

    # check if outputs are the same as from the KS solver class directly... Yes they do!
    ks = KS(nu=env.nu, N=env.N, dt=env.dt, actuator_locs=env.actuator_locs)
    u0 = torch.tensor(u_env_rollout[0])
    u = u0
    uu = [u0]
    for _ in range(999):
        u = ks.advance(u, torch.zeros(env.action_size))
        uu.append(u.detach().numpy())
    uu = np.array(uu)

    # Test if the done state works and execution terminates early
    # This policy outputs a constant value for the actions and should drive the system into blow-up.
    const = nn.Linear(env.observation_spec['observation'].shape.numel(), env.action_size, bias=False)
    const.weight = torch.nn.Parameter(1.0 * torch.ones(zeros.weight.shape))
    policy = TensorDictModule(
        const,
        in_keys=["observation"],
        out_keys=["action"],
    )

    rollout = env.rollout(1000, policy)
    contourplot_KS(rollout["next", "u"].detach().numpy())

    print('here')

    # Check if batching computations works
    # Currently, batching is not supported - because the solver code is not compatible with torch.vmap
    # batch_size = 10  # number of environments to be executed in batch
    # env = KSenv(nu=0.001,
    #            actuator_locs=torch.tensor([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]),
    #            burn_in=1000,
    #            batch_size=batch_size)
    # td = env.reset()
    # print(f"reset (batch size of {batch_size})", td)
    # td = env.rand_step(td)
    # print(f"rand step (batch size of {batch_size})", td)

    print("observation_spec:", env.observation_spec)
    print("state_spec:", env.state_spec)
