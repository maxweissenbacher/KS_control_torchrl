import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from solver.KS_solver import KS
import torch


def contourplot_KS(uu, N=256, dt=0.5, num_plot_points=1000, plot_frame=True):
    # Make contour plot of solution
    fig, ax = plt.subplots()
    tt = np.arange(uu.shape[0]) * dt
    x = np.arange(0, 2 * np.pi, 2 * np.pi / N)
    ax.contourf(x, tt[:num_plot_points], uu[:num_plot_points], 31, extend='both', cmap=cm.plasma)
    if plot_frame:
        ax.xlabel('x')
        ax.ylabel('t')
        ax.colorbar()
        ax.title('Solution of the KS equation')
    else:
        ax.axis('off')

    plt.show()


if __name__ == '__main__':

    N = 512  # number of collocation points
    dt = 0.1 # timestep size
    actuator_locs = torch.tensor(np.linspace(0.0, 2*np.pi, num=5, endpoint=False))
    ks = KS(nu=0.01, N=N, dt=dt, actuator_locs=actuator_locs)

    # Random initial data
    u = 0.0001 * np.random.normal(size=N)  # noisy intial data
    u = u - u.mean()
    u = torch.tensor(u)

    action = torch.zeros(ks.num_actuators)
    action1 = torch.ones(ks.num_actuators)
    action2 = torch.tensor(np.random.uniform(size=ks.num_actuators), dtype=torch.float32)

    # Plot the profile of the actuators
    show_frame = False
    xx = np.linspace(0.0, 2*np.pi, num=N)
    fig, ax = plt.subplots()
    ax.plot(xx, ks.B @ action2)
    #plt.plot(xx, torch.sum(ks.B, dim=1))
    if show_frame:
        ax.title('Actuator profile in domain')
    else:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    plt.show()

    # Burn-in
    burnin = 10
    for _ in range(burnin):
        u = ks.advance(u, action)
    # Advance solver
    uu = []
    for _ in range(5000):
        u = ks.advance(u, action)
        uu.append(u.detach().numpy())
    uu = np.array(uu)

    contourplot_KS(uu, N=N, dt=dt, plot_frame=False)