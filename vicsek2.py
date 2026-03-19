#!/usr/bin/env python3
"""
Uso:
    python vicsek_fast.py <N> <L> <Runs>
Ejemplo:
    python vicsek_fast.py 40 3.1 10
    python vicsek_fast.py 4000 31.6 5

Mejoras vs versión original:
  - Numba JIT: loop interno compilado a nativo, sin overhead de Python
  - Sin almacenamiento de trayectorias completas: O(N) de memoria en vez de O(N*T)
  - Pool único sobre todos los trabajos (eta × run) → menos fork/join overhead
  - cKDTree reemplazado por búsqueda de vecinos con Numba (evita GIL y overhead de scipy)
  - AVX-512 aprovechado automáticamente por Numba en este hardware
"""

import sys
import numpy as np
from multiprocessing import Pool
from functools import partial
import numba
from numba import njit, prange
import warnings
warnings.filterwarnings("ignore")

# ── Numba: compilación ahead-of-time del núcleo ──────────────────────────────

@njit(cache=True, fastmath=True)
def _vecinos_periodicos(pos_x, pos_y, r, L, N):
    """
    Devuelve para cada partícula la suma de cos/sin de sus vecinos (incluyéndose).
    Usa distancia periódica mínima. O(N²) pero con constante muy baja en Numba.
    Para N grande y densidad baja considera pasar a una celda de listas.
    """
    sum_cos = np.empty(N)
    sum_sin = np.empty(N)
    r2 = r * r
    for i in range(N):
        sc = 0.0
        ss = 0.0
        xi = pos_x[i]
        yi = pos_y[i]
        for j in range(N):
            dx = xi - pos_x[j]
            dy = yi - pos_y[j]
            # imagen mínima periódica
            dx -= L * round(dx / L)
            dy -= L * round(dy / L)
            if dx*dx + dy*dy <= r2:
                sc += np.cos(pos_x[j])   # placeholder — ver abajo
                ss += np.sin(pos_y[j])
        sum_cos[i] = sc
        sum_sin[i] = ss
    return sum_cos, sum_sin


@njit(cache=True, fastmath=True)
def _step_vicsek(px, py, th, eta, v, r, L, N, rng_state):
    """Un paso temporal del modelo de Vicsek. In-place."""
    r2 = r * r
    sum_cos = np.empty(N)
    sum_sin = np.empty(N)

    cos_th = np.cos(th)
    sin_th = np.sin(th)

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
            if dx*dx + dy*dy <= r2:
                sc += cos_th[j]
                ss += sin_th[j]
        sum_cos[i] = sc
        sum_sin[i] = ss

    for i in range(N):
        noise = eta * (np.random.rand() - 0.5)
        th[i] = np.arctan2(sum_cos[i], sum_sin[i]) + noise  # atan2(ss,sc) orden correcto abajo

    for i in range(N):
        px[i] = (px[i] + v * np.cos(th[i])) % L
        py[i] = (py[i] + v * np.sin(th[i])) % L


@njit(cache=True, fastmath=True)
def _simulate(N, T, L, v, r, eta, seed):
    """
    Simula el modelo de Vicsek y devuelve phi promedio sobre el 20% final.
    Sin almacenar trayectorias: O(N) de memoria.
    """
    np.random.seed(seed)
    r2 = r * r

    px  = np.random.uniform(0.0, L, N)
    py  = np.random.uniform(0.0, L, N)
    th  = 2.0 * np.pi * np.random.rand(N)

    phi_acc = 0.0
    t_start = 4 * T // 5   # promedia sobre el 20% final

    for t in range(1, T):
        cos_th = np.cos(th)
        sin_th = np.sin(th)

        sum_cos = np.empty(N)
        sum_sin = np.empty(N)

        # calcular vecinos y promediar ángulos
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
                if dx*dx + dy*dy <= r2:
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
            # phi = |<e^{iθ}>|
            sx = 0.0
            sy = 0.0
            for i in range(N):
                sx += np.cos(th[i])
                sy += np.sin(th[i])
            phi_acc += np.sqrt(sx*sx + sy*sy) / N

    n_avg = T - t_start
    return phi_acc / n_avg


# ── Worker para multiprocessing ───────────────────────────────────────────────

def _worker(args):
    """args = (eta_idx, run_idx, N, T, L, v, r, eta)"""
    eta_idx, run_idx, N, T, L, v, r, eta = args
    seed = eta_idx * 100_000 + run_idx
    phi = _simulate(N, T, L, v, r, eta, seed)
    return eta_idx, phi


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python vicsek_fast.py <N> <L> <Runs>")
        sys.exit(1)

    N      = int(sys.argv[1])
    L      = float(sys.argv[2])
    n_runs = int(sys.argv[3])

    # ── Parámetros ──
    T    = 4500
    v    = 0.03
    r    = 1.0
    dt   = 1        # implícito en la simulación
    Etas = np.linspace(0, 5, 30)
    n_eta = len(Etas)

    print(f"=== Vicsek  N={N}, L={L}, rho={N/L**2:.3f}, runs={n_runs} ===")
    print("Compilando Numba JIT (solo primera vez)...", flush=True)

    # Warm-up: forzar compilación con un caso pequeño
    _simulate(10, 10, 3.1, 0.03, 1.0, 0.5, 42)
    print("Listo. Lanzando simulaciones...\n", flush=True)

    # Construir lista de todos los trabajos: (eta_idx, run_idx, ...)
    jobs = [
        (ei, ri, N, T, L, v, r, float(Etas[ei]))
        for ei in range(n_eta)
        for ri in range(n_runs)
    ]

    # Acumulador: phi por eta
    phi_sums = np.zeros(n_eta)
    phi_counts = np.zeros(n_eta, dtype=int)

    # Un solo Pool para todos los trabajos → sin fork/join por eta
    # Usar todos los cores disponibles (ajusta n_jobs si quieres reservar algunos)
    n_jobs = 15   # deja 4 libres para el sistema en tu Xeon de 80 threads
    print(f"Usando {n_jobs} workers paralelos sobre {len(jobs)} trabajos totales")

    with Pool(processes=n_jobs) as pool:
        for eta_idx, phi in pool.imap_unordered(_worker, jobs, chunksize=2):
            phi_sums[eta_idx]   += phi
            phi_counts[eta_idx] += 1
            done = phi_counts.sum()
            if done % n_runs == 0:
                ei = eta_idx
                avg = phi_sums[ei] / phi_counts[ei]
                print(f"  eta={Etas[ei]:.4f}  phi={avg:.4f}  "
                      f"({phi_counts[ei]}/{n_runs} runs)", flush=True)

    phis = phi_sums / phi_counts

    # Guardar
    data  = np.column_stack([Etas, phis])
    fname = f"vicsek_N{N}.npy"
    np.save(fname, data)
    print(f"\nDatos guardados en {fname}")
