"""
Entrena un MLP escalar (micrograd) para separar dos medias lunas.
Usa ~100 puntos y Python puro: tarda unos segundos. Observa cómo baja la pérdida.

    python entrenar_moons.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import fijar_semilla, guardar_figura, make_moons
from engine import Value
from nn import MLP

fijar_semilla(1)
X, y = make_moons(100, ruido=0.1, seed=1)
y = y * 2 - 1  # etiquetas en {-1, +1} para la pérdida hinge
modelo = MLP(2, [8, 8, 1])
print(f"parámetros: {len(modelo.parametros())}")


def pérdida():
    """Hinge loss ('max-margin') + regularización L2."""
    salidas = [modelo([Value(a), Value(b)]) for a, b in X]
    hinge = [(1 + -yi * s).relu() for yi, s in zip(y, salidas)]
    datos = sum(hinge) * (1.0 / len(hinge))
    reg = 1e-4 * sum((p * p for p in modelo.parametros()), Value(0.0))
    total = datos + reg
    acc = sum((yi > 0) == (s.data > 0) for yi, s in zip(y, salidas)) / len(y)
    return total, acc


for paso in range(100):
    total, acc = pérdida()
    modelo.cero_grad()
    total.backward()
    lr = 1.0 - 0.9 * paso / 100            # decaimiento lineal de la tasa de aprendizaje
    for p in modelo.parametros():
        p.data -= lr * p.grad              # descenso de gradiente
    if paso % 10 == 0 or paso == 99:
        print(f"paso {paso:3d}  pérdida {total.data:.4f}  precisión {acc*100:5.1f}%")


def dibujar(plt):
    import numpy as np

    g = np.linspace(-1.5, 2.5, 40)
    h = np.linspace(-1.0, 1.5, 40)
    Z = np.array([[modelo([Value(a), Value(b)]).data for a in g] for b in h])
    plt.contourf(g, h, Z > 0, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
    plt.title("Frontera de decisión — micrograd")


guardar_figura("semana1_frontera", dibujar)
