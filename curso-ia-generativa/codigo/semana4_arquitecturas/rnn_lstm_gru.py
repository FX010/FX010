"""
RNN vs LSTM vs GRU en una tarea de MEMORIA A LARGO PLAZO:
dada una secuencia de T dígitos aleatorios, predecir el PRIMER dígito.
La información debe sobrevivir T pasos de tiempo => la RNN simple sufre el desvanecimiento
del gradiente (BPTT multiplica T veces por la jacobiana recurrente).

    python rnn_lstm_gru.py            # T=30, 3000 pasos por defecto
    python rnn_lstm_gru.py 20 3000    # prueba T=20, 40, 60: ¿quién se rinde primero? (el resultado depende de la semilla)
    python rnn_lstm_gru.py 40
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import fijar_semilla

T = int(sys.argv[1]) if len(sys.argv) > 1 else 30
PASOS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
NCLASES, D, H = 6, 8, 64


class Modelo(nn.Module):
    def __init__(self, tipo):
        super().__init__()
        self.emb = nn.Embedding(NCLASES, D)
        cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[tipo]
        self.rnn = cls(D, H, batch_first=True)
        if tipo == "LSTM":                     # truco clásico: sesgo de la puerta de olvido = 1 (recordar por defecto)
            for nombre, p in self.rnn.named_parameters():
                if "bias_ih" in nombre or "bias_hh" in nombre:
                    p.data[H:2 * H] = 1.0      # orden de puertas en PyTorch: i, f, g, o
        self.out = nn.Linear(H, NCLASES)

    def forward(self, x):                      # x [B,T]
        h, _ = self.rnn(self.emb(x))           # [B,T,H]
        return self.out(h[:, -1])              # usar el último estado oculto


def lote(B):
    x = torch.randint(0, NCLASES, (B, T))
    return x, x[:, 0]


for tipo in ("RNN", "LSTM", "GRU"):
    fijar_semilla(0)
    m = Modelo(tipo)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    t0 = time.time()
    for paso in range(PASOS):
        x, y = lote(64)
        loss = F.cross_entropy(m(x), y)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)       # truco estándar contra gradientes explosivos
        opt.step()
    with torch.no_grad():
        x, y = lote(1000)
        acc = (m(x).argmax(1) == y).float().mean().item()
    print(f"T={T}  {tipo:<5} precisión = {acc*100:5.1f}%  (azar = {100/NCLASES:.1f}%)  pérdida={loss.item():.3f}  ({time.time()-t0:.0f}s)")
