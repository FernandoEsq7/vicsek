"""
Uso:
    python vicsek_plot.py

Lee los archivos  vicsek_N40.npy, vicsek_N100.npy, etc.
y genera la gráfica conjunta Fig 2(a).
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ---------- Configuración ----------
cases = [
    (40,   "s"),
    (100,  "+"),
    (400,  "x"),
    (4000, "^"),
    (10000, "D")
]
# -----------------------------------

plt.figure(figsize=(5, 5))

for N, marker in cases:
    fname = f"vicsek_N{N}.npy"
    if not os.path.exists(fname):
        print(f"[!] No se encontró {fname}, saltando N={N}")
        continue

    data = np.load(fname)          # shape (n_etas, 2)
    etas = data[:, 0]
    phis = data[:, 1]

    plt.scatter(etas, phis, c="black", marker=marker, label=f"N={N}")
    print(f"Cargado {fname}  ({len(etas)} puntos)")

plt.xlabel(r"$\eta$")
plt.ylabel(r"$v_a$")
plt.xlim(0, 5)
plt.ylim(0, 1)
plt.legend()
plt.title(r"Fig 2(a): $v_a$ vs $\eta$, $\rho \approx 0.4$")
plt.tight_layout()
plt.savefig("vicsek_fig2a.png", dpi=150)
plt.show()
print("Figura guardada: vicsek_fig2a.png")
