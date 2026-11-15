"""
VAE (Autoencoder Variacional) sobre imágenes sintéticas de formas 16x16.

  Encoder:  x -> (mu, logvar)                       [B,256] -> [B,Z],[B,Z]
  Truco de reparametrización:  z = mu + sigma ⊙ eps,  eps ~ N(0, I)   (hace diferenciable el muestreo)
  Decoder:  z -> x_hat                              [B,Z] -> [B,256]

  ELBO (a maximizar)  =  E_q[ log p(x|z) ]  -  KL( q(z|x) || N(0,I) )
  Pérdida (a minimizar) = reconstrucción (BCE) + beta * KL
  KL cerrada para gaussianas:  -1/2 * sum( 1 + logvar - mu^2 - exp(logvar) )

    python vae.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import fijar_semilla, guardar_figura, make_formas

Z = 4


class VAE(nn.Module):
    def __init__(self, d_in=256, h=128, z=Z):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.mu, self.logvar = nn.Linear(h, z), nn.Linear(h, z)
        self.dec = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, d_in))

    def forward(self, x):                                      # x [B,256]
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)   # reparametrización
        return self.dec(z), mu, logvar                         # logits de x_hat

    @torch.no_grad()
    def muestrear(self, n):
        return torch.sigmoid(self.dec(torch.randn(n, Z)))


def perdida(logits, x, mu, logvar, beta=1.0):
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / len(x)
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)).mean()
    return rec + beta * kl, rec, kl


if __name__ == "__main__":
    fijar_semilla(0)
    X, y = make_formas(4000, ruido=0.0, seed=0)
    X = torch.tensor(X).flatten(1)
    for beta in (0.0, 1.0, 4.0):
        fijar_semilla(0)
        m = VAE()
        opt = torch.optim.Adam(m.parameters(), lr=2e-3)
        for ep in range(30):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), 128):
                xb = X[perm[i:i + 128]]
                logits, mu, lv = m(xb)
                loss, rec, kl = perdida(logits, xb, mu, lv, beta)
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            logits, mu, lv = m(X[:1000])
            _, rec, kl = perdida(logits, X[:1000], mu, lv)
        print(f"beta={beta:<3}  reconstrucción(BCE)={rec.item():7.2f}   KL={kl.item():6.2f}")
        if beta == 1.0:
            modelo_final = m
    print("\nbeta=0 => autoencoder normal (KL enorme: espacio latente irregular, muestrear de N(0,I) da basura).")
    print("beta alto => mejor espacio latente pero reconstrucciones borrosas (compromiso del β-VAE).")

    def dibujar(plt):
        mues = modelo_final.muestrear(16).view(16, 16, 16)
        cuadro = torch.cat([torch.cat(list(mues[r * 4:(r + 1) * 4]), 1) for r in range(4)], 0)
        plt.imshow(cuadro, cmap="gray"); plt.axis("off"); plt.title("Muestras del VAE: z ~ N(0,I)")

    guardar_figura("semana6_vae_muestras", dibujar)
