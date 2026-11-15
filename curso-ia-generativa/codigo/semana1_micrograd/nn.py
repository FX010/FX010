"""nn.py — Neurona, capa y MLP construidos encima de `engine.Value`."""
from __future__ import annotations

import random

from engine import Value


class Modulo:
    def parametros(self):
        return []

    def cero_grad(self):
        for p in self.parametros():
            p.grad = 0.0


class Neurona(Modulo):
    """y = phi( sum_i w_i x_i + b )"""

    def __init__(self, n_entradas: int, activacion: str = "tanh"):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_entradas)]
        self.b = Value(0.0)
        self.activacion = activacion

    def __call__(self, x):
        z = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.activacion == "tanh":
            return z.tanh()
        if self.activacion == "relu":
            return z.relu()
        return z  # lineal

    def parametros(self):
        return self.w + [self.b]


class Capa(Modulo):
    def __init__(self, n_entradas: int, n_salidas: int, **kw):
        self.neuronas = [Neurona(n_entradas, **kw) for _ in range(n_salidas)]

    def __call__(self, x):
        salidas = [n(x) for n in self.neuronas]
        return salidas[0] if len(salidas) == 1 else salidas

    def parametros(self):
        return [p for n in self.neuronas for p in n.parametros()]


class MLP(Modulo):
    def __init__(self, n_entradas: int, tamanos: list[int]):
        dims = [n_entradas] + tamanos
        self.capas = [
            Capa(dims[i], dims[i + 1], activacion="tanh" if i < len(tamanos) - 1 else "lineal")
            for i in range(len(tamanos))
        ]

    def __call__(self, x):
        for capa in self.capas:
            x = capa(x)
        return x

    def parametros(self):
        return [p for c in self.capas for p in c.parametros()]
