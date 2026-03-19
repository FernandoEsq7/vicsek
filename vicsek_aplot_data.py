"""
Uso:
    python vicsek_run.py <N> <L>

Ejemplo:
    python vicsek_run.py 40 3.1
    python vicsek_run.py 100 5
    python vicsek_run.py 400 10
    python vicsek_run.py 4000 31.6

Genera un archivo  vicsek_N<N>.npy  con un array de shape (n_etas, 2)
donde columna 0 = eta, columna 1 = phi promedio.
"""

import sys
import numpy as np
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
    if len(sys.argv) != 4:
        print("Uso: python vicsek_run.py <N> <L> <Runs>")
        sys.exit(1)

    N = int(sys.argv[1])
    L = float(sys.argv[2])

    # ---------- Parámetros ----------
    T = 4500
    v = 0.03
    r = 1
    dt = 1
    Etas = np.linspace(0, 5, 30)
    n_runs = int(sys.argv[3])
    # --------------------------------

    print(f"=== Vicsek  N = {N}, L = {L}, rho = {N/L**2:.2f} ===")

    phis = []
    for eta in Etas:
        print(f"  eta = {eta:.4f} ...", end=" ", flush=True)

        worker = partial(run_simulation, N=N, T=T, L=L, v=v, r=r, dt=dt, eta=eta)
        with Pool() as pool:
            runsVel = pool.map(worker, range(n_runs))

        phi_avg = np.mean(runsVel)
        print(f"phi = {phi_avg:.4f}")
        phis.append(phi_avg)

    # Guardar: columna 0 = eta, columna 1 = phi
    data = np.column_stack([Etas, phis])
    fname = f"vicsek_N{N}.npy"
    np.save(fname, data)
    print(f"\nDatos guardados en {fname}")
