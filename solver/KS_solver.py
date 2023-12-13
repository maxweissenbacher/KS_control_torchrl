# -------------------------------------------------------------------------------------
# Implementation based on Bucci-Semeraro-Allauzen-Wisniewski-Cordier-Mathelin (2019)
# https://doi.org/10.1098/rspa.2019.0351
# -------------------------------------------------------------------------------------
# The code has been refactored into PyTorch and is compatible with the torchrl package.
# The actuation is implemented in a different fashion, to allow more control over
# actuator locations and spread.


import torch
from functools import partial
import matplotlib.pyplot as plt


def normal_pdf(x, loc, scale):
    return torch.exp(torch.distributions.normal.Normal(loc, scale).log_prob(x))


class KS:

    def __init__(self, actuator_locs, nu=1.0, N=256, dt=0.5, device='cpu'):
        """
        :param nu: 'Viscosity' parameter of the KS equation.
        :param N: Number of collocation points
        :param dt: Time step
        :param actuator_locs: Torch tensor. Specifies the locations of the actuators in the interval [0, 2*pi].
                              Cannot be empty or unspecified. Must be of shape [n] for some n > 0.
        """
        # torch.set_default_dtype(torch.float64)
        self.device = torch.device(device)

        # Convert the 'viscosity' parameter to a length parameter - this is numerically more stable
        self.L = (2 * torch.pi / torch.sqrt(torch.tensor(nu))).to(self.device)
        self.n = torch.tensor(N, dtype=torch.int, device=self.device)  # Ensure that N is integer
        self.dt = torch.tensor(dt, device=self.device)
        self.x = torch.arange(self.n, device=self.device) * self.L / self.n
        self.k = (self.n * torch.fft.fftfreq(self.n)[0:self.n // 2 + 1] * 2 * torch.pi / self.L).to(self.device)
        self.ik = 1j * self.k  # spectral derivative operator
        self.lin = self.k ** 2 - self.k ** 4  # Fourier multipliers for linear term

        # Actuation set-up
        self.num_actuators = actuator_locs.size()[-1]
        self.scale = self.L/(2*torch.pi) * 0.1  # Rescale so that we represent the same actuator shape in [0, 2*pi]
        # This should really be doable with vmap...
        B_list = []
        for loc in actuator_locs:
            B_list.append(self.normal_pdf_periodic(self.L / (2 * torch.pi) * loc))
        self.B = torch.stack(B_list, dim=1).to(self.device)

    def nlterm(self, u, f):
        # compute tendency from nonlinear term. advection + forcing
        ur = torch.fft.irfft(u, axis=-1)
        return -0.5 * self.ik * torch.fft.rfft(ur ** 2, axis=-1) + f

    def advance(self, u0, action):
        """
        :param u0: np.array or torch.tensor. Solution at previous timestep.
        :param action: np.array or torch.tensor of shape [len(sensor_locations)].
        :return: Same type and shape as u0. The solution at the next timestep.
        """
        # print(self.B.shape)
        # print(action.shape)

        if action.dtype != torch.float32 or self.B.dtype != torch.float32:
            print(f'Action dtype {action.dtype} || B dtype {self.B.dtype}')
        f0 = self.B @ action

        # semi-implicit third-order runge kutta update.
        # ref: http://journals.ametsoc.org/doi/pdf/10.1175/MWR3214.1
        u = torch.fft.rfft(u0, axis=-1)
        f = torch.fft.rfft(f0, axis=-1)
        u_save = u.clone()
        for n in range(3):
            dt = self.dt / (3 - n)
            # explicit RK3 step for nonlinear term
            u = u_save + dt * self.nlterm(u, f)
            # implicit trapezoidal adjustment for linear term
            u = (u + 0.5 * self.lin * dt * u_save) / (1. - 0.5 * self.lin * dt)
        u = torch.fft.irfft(u, axis=-1)
        return u

    def normal_pdf_periodic(self, loc):
        """
        Return the pdf of the normal distribution centred at loc with variance self.scale,
        wrapped around the circle of length self.L
        :param loc: Float
        :return: torch.tensor of shape self.x.shape
        """
        y = torch.zeros(self.x.size(), device=self.device)
        for shift in range(-3, 3):
            y += torch.exp(torch.distributions.normal.Normal(loc, self.scale).log_prob(self.x + shift*self.L))
        y = y/torch.max(y)
        return y

if __name__ == '__main__':
    ks = KS(actuator_locs=torch.tensor([0., torch.pi]), device='cuda')
    u = torch.zeros(ks.n, device='cuda')
    ks.advance(u, torch.zeros(2, device='cuda'))

    print('here')
