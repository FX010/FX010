"""
GAN sobre un anillo de 8 gaussianas (2D): el banco de pruebas clásico para ver MODE COLLAPSE.

   min_G max_D  E_x[ log D(x) ] + E_z[ log(1 - D(G(z))) ]       (juego minimax, Goodfellow 2014)

En la práctica el generador usa la pérdida "no saturante":  max_G E_z[ log D(G(z)) ].
Medimos cuántos de los 8 modos cubre el generador y qué fracción de muestras cae cerca de un modo.

    python gan.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import fijar_semilla, guardar_figura, make_anillo_gaussianas

ZD, K, RADIO = 8, 8, 2.0
CENTROS = np.array([[RADIO * np.cos(2 * np.pi * k / K), RADIO * np.sin(2 * np.pi * k / K)] for k in range(K)])


def mlp(d_in, d_out, h=128):
    return nn.Sequential(nn.Linear(d_in, h), nn.LeakyReLU(0.2), nn.Linear(h, h), nn.LeakyReLU(0.2), nn.Linear(h, d_out))


def cobertura(muestras, umbral=0.5):
    d = np.linalg.norm(muestras[:, None, :] - CENTROS[None], axis=-1)       # [N,K]
    cerca = d.min(1) < umbral
    modos = len(set(d.argmin(1)[cerca]))
    return modos, cerca.mean()


def entrenar(pasos=3000, bs=256, seed=0):
    fijar_semilla(seed)
    datos = torch.tensor(make_anillo_gaussianas(20000, K, RADIO, 0.1))
    G, D = mlp(ZD, 2), mlp(2, 1)
    oG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    for paso in range(pasos + 1):
        real = datos[torch.randint(0, len(datos), (bs,))]
        falso = G(torch.randn(bs, ZD))
        # --- paso del discriminador: real -> 1, falso -> 0
        lD = F.binary_cross_entropy_with_logits(D(real), torch.ones(bs, 1)) + \
             F.binary_cross_entropy_with_logits(D(falso.detach()), torch.zeros(bs, 1))
        oD.zero_grad(); lD.backward(); oD.step()
        # --- paso del generador: engañar a D (pérdida no saturante)
        lG = F.binary_cross_entropy_with_logits(D(G(torch.randn(bs, ZD))), torch.ones(bs, 1))
        oG.zero_grad(); lG.backward(); oG.step()
        if paso % 1000 == 0:
            with torch.no_grad():
                modos, frac = cobertura(G(torch.randn(2000, ZD)).numpy())
            print(f"paso {paso:5d}  loss D={lD.item():.3f}  loss G={lG.item():.3f}  modos cubiertos={modos}/{K}  muestras cerca de un modo={frac*100:4.1f}%")
    return G, datos


if __name__ == "__main__":
    G, datos = entrenar()

    def dibujar(plt):
        with torch.no_grad():
            g = G(torch.randn(1500, ZD)).numpy()
        plt.scatter(datos[:1500, 0], datos[:1500, 1], s=4, alpha=0.3, label="reales")
        plt.scatter(g[:, 0], g[:, 1], s=4, alpha=0.5, label="generadas")
        plt.legend(); plt.title("GAN — 8 gaussianas")

    guardar_figura("semana6_gan", dibujar)
    print("\nSi cubre menos de 8 modos: eso es *mode collapse*. Prueba: más pasos, lr distintos, WGAN-GP (ejercicio).")
