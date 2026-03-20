#!/usr/bin/env python3
"""
Genera la figura 2b del paper de Vicsek: phi vs rho a eta fijo.
Uso:
    python vicsek_fig_2b_fast.py [n_runs]
Ejemplo:
    python vicsek_fig_2b_fast.py 25

Optimizado con Numba (misma estrategia que vicsek2.py).
"""

import sys
import numpy as np
from multiprocessing import Pool
import numba
from numba import njit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


@njit(cache=True, fastmath=True)
def _simulate(N, T, L, v, r, eta, seed):
    np.random.seed(seed)
    r2 = r * r

    px = np.random.uniform(0.0, L, N)
    py = np.random.uniform(0.0, L, N)
    th = 2.0 * np.pi * np.random.rand(N)

    phi_acc = 0.0
    t_start = 4 * T // 5

    for t in range(1, T):
        cos_th = np.cos(th)
        sin_th = np.sin(th)

        sum_cos = np.empty(N)
        sum_sin = np.empty(N)

        for i in range(N):
            sc = 0.0
            ss = 0.0
            xi = px[i]
            yi = py[i]
            for j in range(N):
                dx = xi - px[j]
                dy = yi - py[j]
                dx -= L * round(dx / L)
                dy -= L * round(dy / L)
                if dx * dx + dy * dy <= r2:
                    sc += cos_th[j]
                    ss += sin_th[j]
            sum_cos[i] = sc
            sum_sin[i] = ss

        for i in range(N):
            noise = eta * (np.random.rand() - 0.5)
            th[i] = np.arctan2(sum_sin[i], sum_cos[i]) + noise

        for i in range(N):
            px[i] = (px[i] + v * np.cos(th[i])) % L
            py[i] = (py[i] + v * np.sin(th[i])) % L

        if t >= t_start:
            sx = 0.0
            sy = 0.0
            for i in range(N):
                sx += np.cos(th[i])
                sy += np.sin(th[i])
            phi_acc += np.sqrt(sx * sx + sy * sy) / N

    n_avg = T - t_start
    return phi_acc / n_avg


def _worker(args):
    n_idx, run_idx, N, T, L, v, r, eta = args
    seed = n_idx * 100_000 + run_idx
    phi = _simulate(N, T, L, v, r, eta, seed)
    return n_idx, phi


if __name__ == "__main__":
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 25

    # ── Parámetros (igual que el original) ──
    Ns = list(range(100, 1200, 100)) + list(range(1200, 4200, 400))
    T = 4500
    L = 20.0
    v = 0.03
    r = 1.0
    eta = 2.1
    n_N = len(Ns)

    print(f"=== Vicsek fig 2b: L={L}, eta={eta}, runs={n_runs} ===")
    print(f"    N values: {Ns}")
    print("Compilando Numba JIT (solo primera vez)...", flush=True)
    _simulate(10, 10, 5.0, 0.03, 1.0, 0.5, 42)
    print("Listo. Lanzando simulaciones...\n", flush=True)

    jobs = [
        (ni, ri, Ns[ni], T, L, v, r, eta)
        for ni in range(n_N)
        for ri in range(n_runs)
    ]

    phi_sums = np.zeros(n_N)
    phi_counts = np.zeros(n_N, dtype=int)

    import os
    n_jobs = max(1, os.cpu_count() - 2)
    print(f"Usando {n_jobs} workers sobre {len(jobs)} trabajos totales\n")

    with Pool(processes=n_jobs) as pool:
        for n_idx, phi in pool.imap_unordered(_worker, jobs, chunksize=2):
            phi_sums[n_idx] += phi
            phi_counts[n_idx] += 1
            done = phi_counts.sum()
            if phi_counts[n_idx] == n_runs:
                avg = phi_sums[n_idx] / n_runs
                print(f"  N={Ns[n_idx]:5d}  rho={Ns[n_idx]/L**2:.4f}  "
                      f"phi={avg:.4f}  [completo]", flush=True)

    phis = phi_sums / phi_counts
    rhos = np.array([N / (L * L) for N in Ns])

    # ── Guardar datos ──
    data = np.column_stack([rhos, phis])
    np.save("vicsek_fig2b.npy", data)
    print(f"\nDatos guardados en vicsek_fig2b.npy")

    # ── Figura ──
    plt.figure(figsize=(8, 5))
    plt.scatter(rhos, phis, c="black", marker="s")
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"$v_a$")
    plt.xlim(0, 10)
    plt.ylim(0, 1)
    plt.title("Modelo de Vicsek: parámetro de orden vs densidad")
    plt.tight_layout()
    plt.savefig("vicsek_fig2b.png", dpi=150)
    print("Figura guardada en vicsek_fig2b.png")
