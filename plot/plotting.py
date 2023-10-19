import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from solver.KS_solver import KS
import torch


def contourplot_KS(uu, N=256, dt=0.5, num_plot_points=1000):
    # Make contour plot of solution
    plt.figure()
    tt = np.arange(uu.shape[0]) * dt
    x = np.arange(0, 2 * np.pi, 2 * np.pi / N)
    plt.contourf(x, tt[:num_plot_points], uu[:num_plot_points], 31, extend='both', cmap=cm.plasma)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.colorbar()
    plt.title('Solution of the KS equation')
    plt.show()


if __name__ == '__main__':

    N = 512  # number of collocation points
    dt = 0.5 # timestep size
    actuator_locs = torch.tensor(np.linspace(0.0, 2*np.pi, num=12, endpoint=False))
    ks = KS(nu=0.01, N=N, dt=dt, actuator_locs=actuator_locs)

    # Random initial data
    u = 0.0001 * np.random.normal(size=N)  # noisy intial data
    u = u - u.mean()
    u = torch.tensor(u)

    action = torch.zeros(ks.num_actuators)
    action1 = torch.ones(ks.num_actuators)
    action2 = torch.tensor(np.random.uniform(size=ks.num_actuators), dtype=torch.float32)

    # Plot the profile of the actuators
    xx = np.linspace(0.0,2*np.pi,num=N)
    plt.plot(xx, ks.B @ action1)
    #plt.plot(xx, torch.sum(ks.B, dim=1))
    plt.title('Actuator profile in domain')
    plt.show()

    # Burn-in
    for _ in range(1000):
        u = ks.advance(u, action)
    # Advance solver
    uu = []
    for _ in range(5000):
        u = ks.advance(u, action)
        uu.append(u.detach().numpy())
    uu = np.array(uu)

    contourplot_KS(uu, N=N, dt=dt)