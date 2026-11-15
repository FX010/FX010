"""
¿Por qué existen las CNN? Comparamos un MLP y una CNN con ~el mismo número de parámetros
sobre formas sintéticas que cambian de POSICIÓN y TAMAÑO.

  Sesgos inductivos de la CNN:  localidad + compartición de pesos => equivarianza a traslaciones.

Si tienes torchvision, sustituye `make_formas` por MNIST/FashionMNIST:
    from torchvision import datasets; datasets.MNIST('data', download=True)

    python cnn_vs_mlp.py
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import fijar_semilla, make_formas

fijar_semilla(0)
Xtr, ytr = [torch.tensor(a) for a in make_formas(400, ruido=0.4, seed=1)]
Xte, yte = [torch.tensor(a) for a in make_formas(1000, ruido=0.4, seed=2)]


def n_params(m):
    return sum(p.numel() for p in m.parameters())


mlp = nn.Sequential(nn.Flatten(), nn.Linear(256, 96), nn.ReLU(), nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 4))
cnn = nn.Sequential(                                   # [B,1,16,16]
    nn.Conv2d(1, 12, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),      # [B,12,8,8]
    nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # [B,24,4,4]
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(24, 4),         # [B,24] -> [B,4]
)


def entrenar(nombre, m, epocas=40, bs=32):
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-2)
    t0 = time.time()
    for ep in range(epocas):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), bs):
            ix = perm[i:i + bs]
            loss = F.cross_entropy(m(Xtr[ix]), ytr[ix])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (m(Xte).argmax(1) == yte).float().mean().item()
        acc_tr = (m(Xtr).argmax(1) == ytr).float().mean().item()
    print(f"{nombre:<5} params={n_params(m):>6,}  precisión train={acc_tr*100:5.1f}%  test={acc*100:5.1f}%  ({time.time()-t0:.0f}s)")


entrenar("MLP", mlp)
entrenar("CNN", cnn)
print("\nObserva: la CNN generaliza mejor a posiciones nuevas con menos parámetros: ese es el sesgo inductivo.")
