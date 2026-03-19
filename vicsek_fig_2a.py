import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from multiprocessing import Pool
from functools import partial

def promedio_vel(angulos):
    x = np.sum(np.cos(angulos))
    y = np.sum(np.sin(angulos))
    return np.sqrt(x*x + y*y) / len(angulos)

def run_simulation(run_idx, N, T, L, v, r, dt, eta):
    """Una corrida independiente del modelo de Vicsek."""
    X = np.zeros((N, T))
    Y = np.zeros((N, T))
    Th = np.zeros((N, T))

    X[:, 0] = np.random.uniform(0, L, N)
    Y[:, 0] = np.random.uniform(0, L, N)
    Th[:, 0] = 2 * np.pi * np.random.rand(N)

    for t in range(1, T):
        pos = np.column_stack([X[:, t-1], Y[:, t-1]])
        tree = cKDTree(pos, boxsize=L)
        vecinos = tree.query_ball_point(pos, r, return_sorted=False)

        cos_th = np.cos(Th[:, t-1])
        sin_th = np.sin(Th[:, t-1])

        ThAvr = np.array([
            np.arctan2(sin_th[v_i].sum(), cos_th[v_i].sum())
            for v_i in vecinos
        ])

        noise = np.random.uniform(-eta/2, eta/2, N)
        Th[:, t] = ThAvr + noise
        X[:, t] = (X[:, t-1] + v * np.cos(Th[:, t]) * dt) % L
        Y[:, t] = (Y[:, t-1] + v * np.sin(Th[:, t]) * dt) % L

    phi_final = np.mean([promedio_vel(Th[:, t]) for t in range(4*T//5, T)])
    return phi_final

if __name__ == "__main__":
    Ns = [40, 100, 400, 4000]
    Ls = [3.1, 5, 10, 31.6]
    markers = ['.', '+', 'x', 'v', ',']
    T = 2000
    v = 0.03
    r = 1
    dt = 1
    Etas = np.linspace(0, 5, 20)
    n_runs = 10

    plt.figure(figsize=(8, 5))

    for Nindx in range(len(Ns)):
        phis = []
        for eta in Etas:
            N = Ns[Nindx]
            L = Ls[Nindx]
            print(f"N = {N}, eta = {eta:.4f}")

            worker = partial(run_simulation, N=N, T=T, L=L, v=v, r=r, dt=dt, eta=eta)

            with Pool() as pool:
                runsVel = pool.map(worker, range(n_runs))

            runAver = np.mean(runsVel)
            print(f"  phi promedio = {runAver:.4f}")
            phis.append(runAver)

        plt.scatter(Etas, phis, c="black", marker=markers[Nindx], label=f"N={Ns[Nindx]}")

    plt.xlabel(r"$\eta$")
    plt.ylabel(r"$v_a$")
    plt.xlim(0, 5)
    plt.ylim(0, 1)
    plt.legend()
    plt.title(r"Fig 2(a): $v_a$ vs $\eta$, $\rho = 0.4$")
    plt.tight_layout()
    plt.savefig("vicsek_fig2a.png", dpi=150)
    plt.show()