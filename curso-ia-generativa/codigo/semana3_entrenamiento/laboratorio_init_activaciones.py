"""
Laboratorio: ¿por qué importa la inicialización de pesos? (solo NumPy)

Apilamos 30 capas (ancho 256) y medimos la desviación estándar de las activaciones
capa por capa con distintas inicializaciones. Si la std se desvanece (→0) o explota (→∞),
el gradiente hará lo mismo y la red NO aprende.

   Xavier/Glorot:  Var(w) = 1/fan_in       (pensada para tanh / sigmoide)
   Kaiming/He   :  Var(w) = 2/fan_in       (pensada para ReLU: compensa que ReLU anula la mitad)
   Ganancia 5/3 :  std = (5/3)/sqrt(fan_in) (ganancia recomendada para tanh: compensa que tanh "aplasta")

    python laboratorio_init_activaciones.py
"""
import numpy as np

rng = np.random.default_rng(0)
N, ANCHO, CAPAS = 512, 256, 30
X = rng.normal(size=(N, ANCHO))

ACTS = {"tanh": np.tanh, "relu": lambda z: np.maximum(0, z)}
INITS = {
    "N(0,1)  (muy grande)": lambda fi: 1.0,
    "N(0,0.01) (muy pequeña)": lambda fi: 0.01,
    "Xavier  sqrt(1/fan_in)": lambda fi: np.sqrt(1 / fi),
    "He      sqrt(2/fan_in)": lambda fi: np.sqrt(2 / fi),
    "tanh gain 5/3 (Karpathy)": lambda fi: (5 / 3) / np.sqrt(fi),
}

for act_nombre, act in ACTS.items():
    print(f"\n=== activación: {act_nombre}  — std de la activación en las capas 1, 5, 10, 20, 30 ===")
    print(f"{'inicialización':<26}" + "".join(f"{'capa '+str(c):>12}" for c in (1, 5, 10, 20, 30)))
    for nombre, f in INITS.items():
        h, stds = X, []
        for _ in range(CAPAS):
            W = rng.normal(size=(ANCHO, ANCHO)) * f(ANCHO)
            h = act(h @ W)
            stds.append(h.std())
        print(f"{nombre:<26}" + "".join(f"{stds[c-1]:>12.2e}" for c in (1, 5, 10, 20, 30)))

print("""
Lectura (compruébalo en las tablas de arriba):
  • N(0,0.01): todo se desvanece (1e-24 en la capa 30) sin importar la activación.
  • N(0,1): con ReLU explota (1e+31); con tanh satura (std≈1, derivada≈0 => gradiente muerto).
  • tanh + Xavier puro se contrae lentamente (0.63 -> 0.14); la ganancia 5/3 o He lo mantiene estable.
  • relu + He se mantiene en el mismo orden de magnitud; relu + Xavier se apaga (2e-5).
Esto es lo que BatchNorm/LayerNorm y las conexiones residuales arreglan 'por construcción'.""")
