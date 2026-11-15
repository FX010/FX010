"""
Modelo de lenguaje a nivel de CARÁCTER con un MLP (estilo Bengio 2003 / makemore de Karpathy).

  contexto de `CTX` caracteres -> embeddings [CTX, D] -> concatenar -> MLP -> logits sobre el vocabulario

Compara tres modelos contra la COTA TEÓRICA de entropía del corpus:
   1) uniforme  (ln |V|)    2) bigrama por conteo    3) este MLP

    python makemore_mlp.py
"""
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import entropia_corpus_nats_por_char, fijar_semilla, make_corpus

fijar_semilla(0)
corpus = make_corpus(6000, "formal", seed=0)
chars = sorted(set(corpus))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
V = len(chars)
data = torch.tensor([stoi[c] for c in corpus])
n_train = int(0.9 * len(data))
CTX, D, H = 8, 16, 200


def construir(datos):
    """X[i] = CTX caracteres previos, Y[i] = siguiente carácter."""
    idx = torch.arange(CTX, len(datos))
    X = torch.stack([datos[idx - CTX + k] for k in range(CTX)], 1)   # [N, CTX]
    return X, datos[idx]


Xtr, Ytr = construir(data[:n_train])
Xva, Yva = construir(data[n_train:])
print(f"vocabulario={V}  ejemplos train={len(Xtr)}  val={len(Xva)}")
piso = entropia_corpus_nats_por_char(corpus)
print(f"cota teórica (entropía del corpus) = {piso:.3f} nats/char | uniforme = {math.log(V):.3f}")

# --- bigrama por conteo (suavizado de Laplace) ---
conteo = torch.ones(V, V)
for a, b in zip(data[:n_train - 1], data[1:n_train]):
    conteo[a, b] += 1
P = conteo / conteo.sum(1, keepdim=True)
loss_bi = -P[Xva[:, -1], Yva].log().mean().item()
print(f"bigrama por conteo: val loss = {loss_bi:.3f}")


class MLPLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)                                    # [V,D]
        self.net = nn.Sequential(nn.Linear(CTX * D, H), nn.LayerNorm(H), nn.GELU(),
                                 nn.Linear(H, H), nn.LayerNorm(H), nn.GELU(), nn.Linear(H, V))

    def forward(self, x):                                                # x: [B,CTX]
        e = self.emb(x)                                                  # [B,CTX,D]
        return self.net(e.flatten(1))                                    # [B,CTX*D] -> [B,V]


m = MLPLM()
opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)
print(f"parámetros: {sum(p.numel() for p in m.parameters()):,}")
t0 = time.time()
for paso in range(2501):
    ix = torch.randint(0, len(Xtr), (256,))
    loss = F.cross_entropy(m(Xtr[ix]), Ytr[ix])
    opt.zero_grad(); loss.backward(); opt.step()
    if paso % 500 == 0:
        with torch.no_grad():
            va = F.cross_entropy(m(Xva), Yva).item()
        print(f"paso {paso:5d}  train {loss.item():.3f}  val {va:.3f}   ({time.time()-t0:.0f}s)")

print("\nmuestras generadas:")
for _ in range(5):
    # Contexto inicial REAL (cola de una frase del corpus): partir de "\n"*CTX estaría fuera de
    # distribución y el modelo diría tonterías (¡distribution shift!).
    inicio = int(torch.randint(CTX + 50, n_train - 100, (1,)))
    while itos[int(data[inicio - 1])] != "\n":
        inicio += 1
    ctx, out = data[inicio - CTX:inicio].tolist(), []
    for _ in range(80):
        with torch.no_grad():
            p = F.softmax(m(torch.tensor([ctx])), -1)
        i = torch.multinomial(p, 1).item()
        if itos[i] == "\n":
            break
        out.append(itos[i]); ctx = ctx[1:] + [i]
    print("  ", "".join(out))
