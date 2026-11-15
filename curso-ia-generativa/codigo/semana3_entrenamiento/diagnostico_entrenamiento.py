"""
Receta de Karpathy ("A Recipe for Training Neural Networks") hecha código.
Cinco chequeos que deberías correr ANTES de culpar a la arquitectura:

  1. Pérdida inicial ≈ ln(K) (modelo ignorante = predice uniforme).
  2. Sobreajustar UN solo batch hasta pérdida ≈ 0 (si no puede, hay un bug).
  3. Proporción gradiente/dato (update-to-data ratio) ≈ 1e-3 por parámetro.
  4. Fracción de ReLU 'muertas' por capa.
  5. Efecto de LayerNorm en una red profunda: ¿entrena o se queda en la pérdida inicial?

    python diagnostico_entrenamiento.py
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import fijar_semilla, make_espirales

fijar_semilla(0)
Xn, yn = make_espirales(200, 4, ruido=0.25)
X, y = torch.tensor(Xn), torch.tensor(yn)
K = 4


def red(profundidad=6, ancho=64, norm=False):
    capas, d = [], 2
    for _ in range(profundidad):
        capas += [nn.Linear(d, ancho)] + ([nn.LayerNorm(ancho)] if norm else []) + [nn.ReLU()]
        d = ancho
    return nn.Sequential(*capas, nn.Linear(d, K))


print("— 1. pérdida inicial —")
m = red()
with torch.no_grad():
    l0 = F.cross_entropy(m(X), y).item()
print(f"   pérdida inicial = {l0:.3f}   (esperada ln({K}) = {math.log(K):.3f})")

print("— 2. sobreajustar un batch de 16 —")
xb, yb = X[:16], y[:16]
opt = torch.optim.Adam(m.parameters(), lr=3e-3)
for paso in range(300):
    loss = F.cross_entropy(m(xb), yb)
    opt.zero_grad(); loss.backward(); opt.step()
print(f"   pérdida tras 300 pasos en 16 ejemplos = {loss.item():.5f}   -> {'OK' if loss.item() < 0.01 else 'BUG o lr malo'}")

print("— 3. ratio actualización/dato por parámetro, SGD lr=0.05 (objetivo ≈ 1e-3; si es <<1e-4 sube el lr) —")
m = red()
opt = torch.optim.SGD(m.parameters(), lr=0.05)
loss = F.cross_entropy(m(X), y); opt.zero_grad(); loss.backward()
for n, p in m.named_parameters():
    if p.ndim == 2:
        ratio = (0.05 * p.grad.std() / p.data.std()).item()
        print(f"   {n:<10} ratio = {ratio:.1e}")

print("— 4. ReLU muertas por capa (red profunda sin norm vs con LayerNorm) —")
for etiqueta, norm in [("sin norm", False), ("con LayerNorm", True)]:
    m = red(profundidad=10, norm=norm)
    h, muertas = X, []
    for capa in m:
        h = capa(h)
        if isinstance(capa, nn.ReLU):
            muertas.append(((h <= 0).all(0)).float().mean().item())
    print(f"   {etiqueta:<14} neuronas siempre apagadas por capa: {[round(v,2) for v in muertas]}")

print("— 5. red de 10 capas, SGD lr=0.1, 300 pasos: ¿entrena con y sin LayerNorm? (ln 4 = 1.386 = no aprendió nada) —")
for etiqueta, norm in [("sin norm", False), ("con LayerNorm", True)]:
    fijar_semilla(0)
    m = red(profundidad=10, norm=norm)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    for _ in range(300):
        loss = F.cross_entropy(m(X), y)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"   {etiqueta:<14} pérdida tras 300 pasos = {loss.item():.4f}")
