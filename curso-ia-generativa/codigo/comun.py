"""
Utilidades compartidas por todos los scripts del curso.

Todo está generado de forma sintética (NumPy) para que cada script corra SIN
descargar datos y en pocos minutos en CPU. Cuando domines cada idea, cambia estos
datos por MNIST / CIFAR / Tiny Shakespeare (los scripts indican cómo).
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

SALIDAS = Path(__file__).resolve().parent / "salidas"


# ----------------------------------------------------------------------------
# Reproducibilidad
# ----------------------------------------------------------------------------
def fijar_semilla(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch es opcional en las semanas 1-2
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


# ----------------------------------------------------------------------------
# Datos 2D (clasificación y generación)
# ----------------------------------------------------------------------------
def make_moons(n: int = 200, ruido: float = 0.1, seed: int = 0):
    """Dos medias lunas. Devuelve X [n,2] e y [n] en {0,1}."""
    rng = np.random.default_rng(seed)
    n1 = n // 2
    t1 = rng.uniform(0, np.pi, n1)
    t2 = rng.uniform(0, np.pi, n - n1)
    a = np.stack([np.cos(t1), np.sin(t1)], 1)
    b = np.stack([1 - np.cos(t2), 0.5 - np.sin(t2)], 1)
    X = np.vstack([a, b]) + rng.normal(0, ruido, (n, 2))
    y = np.array([0] * n1 + [1] * (n - n1))
    return X.astype(np.float32), y


def make_espirales(n_por_clase: int = 100, clases: int = 3, ruido: float = 0.2, seed: int = 0):
    """Espirales entrelazadas (clásico de CS231n). X [n*clases,2], y [n*clases]."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n_por_clase * clases, 2), dtype=np.float32)
    y = np.zeros(n_por_clase * clases, dtype=np.int64)
    for k in range(clases):
        ix = range(n_por_clase * k, n_por_clase * (k + 1))
        r = np.linspace(0.0, 1, n_por_clase)
        t = np.linspace(k * 4, (k + 1) * 4, n_por_clase) + rng.normal(0, ruido, n_por_clase)
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = k
    return X, y


def make_anillo_gaussianas(n: int = 2048, k: int = 8, radio: float = 2.0, std: float = 0.15, seed: int = 0):
    """Mezcla de k gaussianas sobre un círculo: banco de pruebas de GAN/difusión."""
    rng = np.random.default_rng(seed)
    comp = rng.integers(0, k, n)
    ang = 2 * np.pi * comp / k
    centros = np.stack([radio * np.cos(ang), radio * np.sin(ang)], 1)
    return (centros + rng.normal(0, std, (n, 2))).astype(np.float32)


def make_swiss_2d(n: int = 4096, ruido: float = 0.05, seed: int = 0):
    """Espiral 2D (swiss roll aplanado), normalizada a ~[-1.5, 1.5]."""
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.uniform(0, 1, n))
    X = np.stack([t * np.cos(t), t * np.sin(t)], 1)
    X = X / np.abs(X).max() * 1.5 + rng.normal(0, ruido, (n, 2))
    return X.astype(np.float32)


# ----------------------------------------------------------------------------
# Imágenes sintéticas (sustituto ligero de MNIST)
# ----------------------------------------------------------------------------
FORMAS = ["circulo", "cuadrado", "cruz", "diagonal"]


def make_formas(n: int = 1000, tam: int = 16, ruido: float = 0.1, seed: int = 0, desplazar: bool = True):
    """
    Imágenes tam x tam con 4 formas. Devuelve X [n,1,tam,tam] en [0,1] e y [n].
    Con `desplazar=True` la forma cambia de posición/tamaño: ahí las CNN ganan a los MLP.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 1, tam, tam), dtype=np.float32)
    y = rng.integers(0, 4, n)
    yy, xx = np.mgrid[0:tam, 0:tam]
    for i in range(n):
        r = rng.integers(tam // 5, tam // 3 + 1)
        cx = tam // 2 if not desplazar else rng.integers(r + 1, tam - r - 1)
        cy = tam // 2 if not desplazar else rng.integers(r + 1, tam - r - 1)
        img = np.zeros((tam, tam), dtype=np.float32)
        if y[i] == 0:  # círculo (anillo)
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            img[np.abs(d - r) < 0.9] = 1
        elif y[i] == 1:  # cuadrado (borde)
            m = (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)
            m_in = (np.abs(xx - cx) <= r - 1.5) & (np.abs(yy - cy) <= r - 1.5)
            img[m & ~m_in] = 1
        elif y[i] == 2:  # cruz
            img[(np.abs(xx - cx) < 1) & (np.abs(yy - cy) <= r)] = 1
            img[(np.abs(yy - cy) < 1) & (np.abs(xx - cx) <= r)] = 1
        else:  # diagonal
            img[(np.abs((xx - cx) - (yy - cy)) < 1) & (np.abs(xx - cx) <= r)] = 1
        X[i, 0] = img
    X += rng.normal(0, ruido, X.shape).astype(np.float32)
    return np.clip(X, 0, 1), y.astype(np.int64)


# ----------------------------------------------------------------------------
# Corpus de texto sintético (para GPT, LoRA, DPO sin descargar nada)
# ----------------------------------------------------------------------------
SUJETOS = ["el gato", "la niña", "el ingeniero", "la máquina", "el robot", "mi vecino"]
VERBOS = ["come", "mira", "busca", "construye", "analiza"]
OBJETOS = ["pescado", "el motor", "un puente", "los datos", "la red"]
ADVERBIOS = ["", " rápido", " con calma", " hoy"]


def make_corpus(n_frases: int = 4000, estilo: str = "formal", seed: int = 0) -> str:
    """
    Frases de una gramática sencilla, una por línea.
      formal   : "el gato come pescado."
      informal : "oye, el gato come pescado!"
    Una gramática pequeña permite saber la entropía real del corpus y comprobar
    que el modelo la alcanza (¡el mejor test de que tu GPT funciona!).
    """
    rng = random.Random(seed)
    lineas = []
    for _ in range(n_frases):
        f = f"{rng.choice(SUJETOS)} {rng.choice(VERBOS)} {rng.choice(OBJETOS)}{rng.choice(ADVERBIOS)}"
        lineas.append(f"oye, {f}!" if estilo == "informal" else f"{f}.")
    return "\n".join(lineas) + "\n"


# ----------------------------------------------------------------------------
# Gráficos (opcional)
# ----------------------------------------------------------------------------
def guardar_figura(nombre: str, dibujar) -> None:
    """Llama dibujar(plt) y guarda salidas/<nombre>.png. Si no hay matplotlib, lo omite."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[aviso] matplotlib no instalado: se omite la figura '{nombre}'. (pip install matplotlib)")
        return
    os.makedirs(SALIDAS, exist_ok=True)
    plt.figure(figsize=(6, 5))
    dibujar(plt)
    plt.tight_layout()
    ruta = SALIDAS / f"{nombre}.png"
    plt.savefig(ruta, dpi=120)
    plt.close()
    print(f"[figura] {ruta}")


def entropia_corpus_nats_por_char(corpus: str) -> float:
    """
    Cota inferior teórica de la pérdida (nats/carácter) para el corpus de `make_corpus`.
    Cada frase elige uniformemente sujeto×verbo×objeto×adverbio => H(frase) = ln(combinaciones).
    Ningún modelo puede bajar de este valor: si tu GPT lo alcanza, funciona.
    """
    import math

    combos = len(SUJETOS) * len(VERBOS) * len(OBJETOS) * len(ADVERBIOS)
    n_frases = corpus.count("\n")
    return math.log(combos) / (len(corpus) / n_frases)
