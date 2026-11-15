"""
engine.py — Motor de diferenciación automática en modo inverso (reverse-mode autodiff)
sobre ESCALARES. Inspirado en `micrograd` de Andrej Karpathy.

Idea central: cada operación crea un nodo `Value` que recuerda
  - su valor (data)
  - sus padres (_prev)
  - cómo repartir el gradiente hacia los padres (_backward)   <- regla de la cadena local
`backward()` ordena el grafo topológicamente y aplica la regla de la cadena en sentido inverso.
"""
from __future__ import annotations

import math


class Value:
    def __init__(self, data: float, _hijos=(), _op: str = "", etiqueta: str = ""):
        self.data = float(data)
        self.grad = 0.0                  # dL/d(este nodo)
        self._backward = lambda: None    # función que propaga grad a los padres
        self._prev = set(_hijos)
        self._op = _op
        self.etiqueta = etiqueta

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # ---------------------------------------------------------------- operaciones
    def __add__(self, otro):
        otro = otro if isinstance(otro, Value) else Value(otro)
        out = Value(self.data + otro.data, (self, otro), "+")

        def _backward():            # d(a+b)/da = 1 ; d(a+b)/db = 1
            self.grad += out.grad
            otro.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, otro):
        otro = otro if isinstance(otro, Value) else Value(otro)
        out = Value(self.data * otro.data, (self, otro), "*")

        def _backward():            # d(a*b)/da = b ; d(a*b)/db = a
            self.grad += otro.data * out.grad
            otro.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "solo exponentes constantes"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():            # d(a^k)/da = k a^(k-1)
            self.grad += k * self.data ** (k - 1) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():            # d(e^a)/da = e^a
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():            # d(ln a)/da = 1/a
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():            # d tanh/da = 1 - tanh^2
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():            # 1 si a>0, 0 en otro caso
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    # ---------------------------------------------------------------- backprop
    def backward(self):
        """Regla de la cadena sobre el DAG: orden topológico + recorrido inverso."""
        topo, visitados = [], set()

        def construir(v):
            if v not in visitados:
                visitados.add(v)
                for hijo in v._prev:
                    construir(hijo)
                topo.append(v)

        construir(self)
        self.grad = 1.0                    # dL/dL = 1
        for nodo in reversed(topo):
            nodo._backward()

    # ---------------------------------------------------------------- azúcar sintáctico
    def __neg__(self):
        return self * -1

    def __sub__(self, otro):
        return self + (-otro)

    def __rsub__(self, otro):
        return otro + (-self)

    def __radd__(self, otro):
        return self + otro

    def __rmul__(self, otro):
        return self * otro

    def __truediv__(self, otro):
        return self * otro ** -1

    def __rtruediv__(self, otro):
        return otro * self ** -1
