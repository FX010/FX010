"""
Conexiones residuales (ResNet): por qué permiten entrenar redes MUY profundas.

   Plana:      h_{l+1} = f(h_l)
   Residual:   h_{l+1} = h_l + f(h_l)      => dh_{l+1}/dh_l = I + df/dh   (el "I" es una autopista de gradiente)

Medimos la norma del gradiente en la PRIMERA capa de redes de 5 a 80 capas.

    python residuales_gradientes.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import fijar_semilla, make_espirales


class Bloque(nn.Module):
    def __init__(self, d, residual):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, d), nn.Tanh())
        self.residual = residual

    def forward(self, x):
        return x + self.f(x) if self.residual else self.f(x)


def red(profundidad, residual, d=32):
    return nn.Sequential(nn.Linear(2, d), *[Bloque(d, residual) for _ in range(profundidad)], nn.Linear(d, 3))


def main():
    Xn, yn = make_espirales(100, 3)
    X, y = torch.tensor(Xn), torch.tensor(yn)
    print(f"{'capas':>6} | {'‖grad capa 1‖ plana':>22} | {'‖grad capa 1‖ residual':>24}")
    for L in (5, 10, 20, 40, 80):
        fila = []
        for residual in (False, True):
            fijar_semilla(0)
            m = red(L, residual)
            # inicialización razonable para ambos (Xavier)
            for mod in m.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.xavier_uniform_(mod.weight)
            loss = F.cross_entropy(m(X), y)
            loss.backward()
            fila.append(m[0].weight.grad.norm().item())
        print(f"{L:>6} | {fila[0]:>22.3e} | {fila[1]:>24.3e}")

    print("\nEntrenamiento (80 capas, 300 pasos Adam):")
    for residual in (False, True):
        fijar_semilla(0)
        m = red(80, residual)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(300):
            loss = F.cross_entropy(m(X), y)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"  {'residual' if residual else 'plana':<9} pérdida final = {loss.item():.4f}")


if __name__ == "__main__":
    main()
