"""
Gradient checking: compara el gradiente de tu autodiff con la derivada numérica
    f'(x) ≈ ( f(x+h) - f(x-h) ) / (2h)
Si coinciden (error relativo < 1e-6), tu backward() es correcto. Es la prueba de
oro que usarás TODO el curso (también con PyTorch y con tus propias capas).
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import Value


def f(a, b, c):
    # función arbitraria que mezcla todas las operaciones
    d = a * b + c ** 2
    e = (d.tanh() + (a - c).relu() + (b * 0.5).exp()) / (1 + a ** 2)
    return (e * e + (d.sigmoid() + 2).log())


def derivada_numerica(idx, vals, h=1e-6):
    mas, menos = list(vals), list(vals)
    mas[idx] += h
    menos[idx] -= h
    return (f(*[Value(v) for v in mas]).data - f(*[Value(v) for v in menos]).data) / (2 * h)


if __name__ == "__main__":
    vals = [0.7, -1.3, 0.4]
    a, b, c = (Value(v) for v in vals)
    f(a, b, c).backward()
    ok = True
    for i, (nombre, v) in enumerate(zip("abc", (a, b, c))):
        num = derivada_numerica(i, vals)
        err = abs(v.grad - num) / max(1e-12, abs(v.grad) + abs(num))
        print(f"d f / d{nombre}:  autodiff={v.grad:+.8f}  numérico={num:+.8f}  error rel={err:.2e}")
        ok &= err < 1e-6
    print("✔ gradientes correctos" if ok else "✘ HAY UN ERROR EN backward()")
    sys.exit(0 if ok else 1)
