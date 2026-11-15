"""
MLP vectorizado en NumPy con retropropagación escrita A MANO (sin autograd).

Arquitectura:  X[N,2] -> Linear(2,H) -> ReLU -> Linear(H,H) -> ReLU -> Linear(H,K) -> softmax -> CE

Dimensiones (N = batch, H = ocultas, K = clases):
    X   [N,D]      W1 [D,H]   b1 [H]
    h1  [N,H]      W2 [H,H]   b2 [H]
    h2  [N,H]      W3 [H,K]   b3 [K]
    logits [N,K]

Fórmulas clave de backprop (memorízalas, aparecen en todo el curso):
    Para  Z = A @ W + b :   dA = dZ @ W.T ,   dW = A.T @ dZ ,   db = dZ.sum(0)
    Softmax + entropía cruzada:  d logits = (softmax(logits) - onehot(y)) / N

    python mlp_numpy.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import fijar_semilla, guardar_figura, make_espirales


class MLPNumpy:
    def __init__(self, dims, seed=0):
        rng = np.random.default_rng(seed)
        # Inicialización He (Kaiming) para ReLU: std = sqrt(2 / fan_in)
        self.W = [rng.normal(0, np.sqrt(2 / d_in), (d_in, d_out)) for d_in, d_out in zip(dims[:-1], dims[1:])]
        self.b = [np.zeros(d_out) for d_out in dims[1:]]

    def forward(self, X):
        self.cache = [X]
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            a = z if i == len(self.W) - 1 else np.maximum(0, z)   # ReLU salvo en la última capa
            self.cache.append(a)
        return a  # logits

    @staticmethod
    def pérdida_y_dlogits(logits, y):
        logits = logits - logits.max(1, keepdims=True)             # estabilidad numérica
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        N = len(y)
        loss = -np.log(p[np.arange(N), y] + 1e-12).mean()
        d = p.copy()
        d[np.arange(N), y] -= 1
        return loss, d / N

    def backward(self, dlogits):
        grads_W, grads_b = [None] * len(self.W), [None] * len(self.W)
        d = dlogits
        for i in reversed(range(len(self.W))):
            a_prev = self.cache[i]
            grads_W[i] = a_prev.T @ d
            grads_b[i] = d.sum(0)
            if i > 0:
                d = d @ self.W[i].T
                d = d * (self.cache[i] > 0)                        # derivada de ReLU
        return grads_W, grads_b


def verificar_gradientes(modelo, X, y, h=1e-5):
    """Compara backprop manual con diferencias finitas en unos cuantos parámetros."""
    loss, dl = modelo.pérdida_y_dlogits(modelo.forward(X), y)
    gW, _ = modelo.backward(dl)
    peor = 0.0
    rng = np.random.default_rng(0)
    for i, W in enumerate(modelo.W):
        for _ in range(5):
            r, c = rng.integers(W.shape[0]), rng.integers(W.shape[1])
            viejo = W[r, c]
            W[r, c] = viejo + h
            lp, _ = modelo.pérdida_y_dlogits(modelo.forward(X), y)
            W[r, c] = viejo - h
            lm, _ = modelo.pérdida_y_dlogits(modelo.forward(X), y)
            W[r, c] = viejo
            num = (lp - lm) / (2 * h)
            peor = max(peor, abs(num - gW[i][r, c]) / max(1e-8, abs(num) + abs(gW[i][r, c])))
    print(f"verificación de gradientes: peor error relativo = {peor:.2e}  ({'OK' if peor < 1e-5 else 'REVISA'})")
    return peor


def entrenar():
    fijar_semilla(0)
    X, y = make_espirales(100, 3, ruido=0.2)
    modelo = MLPNumpy([2, 64, 64, 3])
    verificar_gradientes(modelo, X, y)

    lr, pasos = 1.0, 2000
    for paso in range(pasos + 1):
        logits = modelo.forward(X)
        loss, dl = modelo.pérdida_y_dlogits(logits, y)
        gW, gb = modelo.backward(dl)
        for i in range(len(modelo.W)):                             # SGD simple
            modelo.W[i] -= lr * gW[i]
            modelo.b[i] -= lr * gb[i]
        if paso % 400 == 0:
            acc = (logits.argmax(1) == y).mean()
            print(f"paso {paso:5d}  pérdida {loss:.4f}  precisión {acc*100:5.1f}%")

    def dibujar(plt):
        g = np.linspace(-1.2, 1.2, 200)
        xx, yy = np.meshgrid(g, g)
        Z = modelo.forward(np.c_[xx.ravel(), yy.ravel()]).argmax(1).reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", s=15)
        plt.title("MLP en NumPy (backprop a mano)")

    guardar_figura("semana2_espirales_numpy", dibujar)
    return modelo, X, y


if __name__ == "__main__":
    entrenar()
