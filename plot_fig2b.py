#!/usr/bin/env python3
"""
Grafica los datos de vicsek_fig2b.npy
Uso:
    python plot_fig2b.py [archivo.npy]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fname = sys.argv[1] if len(sys.argv) > 1 else "vicsek_fig2b.npy"
data = np.load(fname)
rhos, phis = data[:, 0], data[:, 1]

plt.figure(figsize=(5, 5))
plt.scatter(rhos, phis, c="black", marker="s", s=30)
plt.xlabel(r"$\rho$", fontsize=14)
plt.ylabel(r"$v_a$", fontsize=14)
plt.xlim(0, 10)
plt.ylim(0, 0.8)
plt.title(r"Modelo de Vicsek — $v_a$ vs $\rho$ ($\eta = 2.1$, $L = 20$)")
plt.tight_layout()
plt.savefig("vicsek_fig2b.png", dpi=150)
print("Figura guardada en vicsek_fig2b.png")
