"""
El MISMO MLP de mlp_numpy.py, ahora en PyTorch, y tres experimentos:

  1. Verificar que el autograd de PyTorch coincide con tu backprop manual.
  2. Comparar optimizadores: SGD, SGD+momentum, Adam, AdamW.
  3. Comparar funciones de activación: ReLU, GELU, SiLU, Tanh.

    python mlp_torch.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import fijar_semilla, make_espirales
from mlp_numpy import MLPNumpy


def crear_mlp(dims, activacion=nn.ReLU):
    capas = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        capas.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            capas.append(activacion())
    return nn.Sequential(*capas)


def experimento_autograd_vs_manual():
    print("\n[1] autograd de PyTorch  vs  backprop manual de NumPy")
    X, y = make_espirales(50, 3)
    manual = MLPNumpy([2, 16, 16, 3], seed=0)
    modelo = crear_mlp([2, 16, 16, 3])
    with torch.no_grad():  # copiar pesos (Linear guarda W como [out,in] => transponer)
        for lin, W, b in zip([m for m in modelo if isinstance(m, nn.Linear)], manual.W, manual.b):
            lin.weight.copy_(torch.tensor(W.T, dtype=torch.float32))
            lin.bias.copy_(torch.tensor(b, dtype=torch.float32))
    logits = manual.forward(X.astype(np.float64))
    _, dl = manual.pérdida_y_dlogits(logits, y)
    gW, _ = manual.backward(dl)

    loss = F.cross_entropy(modelo(torch.tensor(X)), torch.tensor(y))
    loss.backward()
    lins = [m for m in modelo if isinstance(m, nn.Linear)]
    for i, lin in enumerate(lins):
        dif = np.abs(lin.weight.grad.numpy().T - gW[i]).max()
        print(f"   capa {i}: máx |grad_torch - grad_manual| = {dif:.2e}")


def entrenar(nombre, modelo, opt, X, y, pasos=600, batch=64, quiet=False):
    losses = []
    N = len(X)
    g = torch.Generator().manual_seed(0)
    for paso in range(pasos):
        idx = torch.randint(0, N, (batch,), generator=g)
        loss = F.cross_entropy(modelo(X[idx]), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    with torch.no_grad():
        acc = (modelo(X).argmax(1) == y).float().mean().item()
    if not quiet:
        print(f"   {nombre:<22} pérdida final (media últimos 50) = {np.mean(losses[-50:]):.4f}   precisión = {acc*100:5.1f}%")
    return losses


def experimento_optimizadores():
    print("\n[2] optimizadores (misma red, mismos datos, 600 pasos)")
    Xn, yn = make_espirales(100, 3)
    X, y = torch.tensor(Xn), torch.tensor(yn)
    configs = {
        "SGD lr=0.1": lambda p: torch.optim.SGD(p, lr=0.1),
        "SGD+momentum 0.9": lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9),
        "Adam lr=3e-3": lambda p: torch.optim.Adam(p, lr=3e-3),
        "AdamW lr=3e-3 wd=1e-2": lambda p: torch.optim.AdamW(p, lr=3e-3, weight_decay=1e-2),
    }
    for nombre, mk in configs.items():
        fijar_semilla(0)
        m = crear_mlp([2, 64, 64, 3])
        entrenar(nombre, m, mk(m.parameters()), X, y)


def experimento_activaciones():
    print("\n[3] funciones de activación (AdamW, 600 pasos)")
    Xn, yn = make_espirales(100, 3)
    X, y = torch.tensor(Xn), torch.tensor(yn)
    for nombre, act in {"ReLU": nn.ReLU, "GELU": nn.GELU, "SiLU/Swish": nn.SiLU, "Tanh": nn.Tanh}.items():
        fijar_semilla(0)
        m = crear_mlp([2, 64, 64, 3], act)
        entrenar(nombre, m, torch.optim.AdamW(m.parameters(), lr=3e-3), X, y)


if __name__ == "__main__":
    experimento_autograd_vs_manual()
    experimento_optimizadores()
    experimento_activaciones()
