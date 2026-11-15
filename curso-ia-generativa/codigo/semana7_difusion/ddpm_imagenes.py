"""
Difusión sobre IMÁGENES con una mini U-Net (formas 16x16): la misma receta de ddpm_2d.py,
pero eps_theta es una red convolucional con skip-connections y condicionamiento por tiempo.

   x [B,1,16,16] ──conv──► e1 [B,32,16,16] ──down──► e2 [B,64,8,8] ──mid──► ──up──► [B,32,16,16] ──cat(e1)──► conv ──► eps [B,1,16,16]

Es el esqueleto de Stable Diffusion (que además trabaja en un espacio latente comprimido por un VAE
y se condiciona con texto mediante cross-attention).

    python ddpm_imagenes.py            # ~3-5 min en CPU
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import fijar_semilla, guardar_figura, make_formas
from ddpm_2d import T, ab, alphas, betas, emb_tiempo


class Bloque(nn.Module):
    def __init__(self, c_in, c_out, t_dim=64):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(c_in, c_out, 3, padding=1), nn.Conv2d(c_out, c_out, 3, padding=1)
        self.n1, self.n2 = nn.GroupNorm(4, c_out), nn.GroupNorm(4, c_out)
        self.t = nn.Linear(t_dim, c_out)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, te):
        h = F.silu(self.n1(self.c1(x))) + self.t(te)[:, :, None, None]      # inyectar el tiempo
        h = F.silu(self.n2(self.c2(h)))
        return h + self.skip(x)


class MiniUNet(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        self.inp = nn.Conv2d(1, c, 3, padding=1)
        self.d1 = Bloque(c, c)
        self.down = nn.Conv2d(c, 2 * c, 4, stride=2, padding=1)             # 16 -> 8
        self.mid = Bloque(2 * c, 2 * c)
        self.up = nn.ConvTranspose2d(2 * c, c, 4, stride=2, padding=1)      # 8 -> 16
        self.d2 = Bloque(2 * c, c)                                          # tras concatenar la skip
        self.out = nn.Conv2d(c, 1, 3, padding=1)

    def forward(self, x, t):
        te = emb_tiempo(t)
        e1 = self.d1(self.inp(x), te)                                       # [B,c,16,16]
        m = self.mid(self.down(e1), te)                                     # [B,2c,8,8]
        u = self.up(m)                                                      # [B,c,16,16]
        return self.out(self.d2(torch.cat([u, e1], 1), te))                 # [B,1,16,16]


@torch.no_grad()
def muestrear(m, n):
    x = torch.randn(n, 1, 16, 16)
    for t in reversed(range(T)):
        eps = m(x, torch.full((n,), t))
        x = (x - betas[t] / (1 - ab[t]).sqrt() * eps) / alphas[t].sqrt()
        if t > 0:
            x = x + betas[t].sqrt() * torch.randn_like(x)
    return x.clamp(-1, 1)


if __name__ == "__main__":
    fijar_semilla(0)
    X, _ = make_formas(4000, ruido=0.0, seed=0)
    X = torch.tensor(X) * 2 - 1                                              # a [-1, 1]
    m = MiniUNet()
    print(f"parámetros U-Net: {sum(p.numel() for p in m.parameters()):,}")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    PASOS = 1500
    for paso in range(PASOS + 1):
        x0 = X[torch.randint(0, len(X), (64,))]
        t = torch.randint(0, T, (64,))
        eps = torch.randn_like(x0)
        xt = ab[t].sqrt()[:, None, None, None] * x0 + (1 - ab[t]).sqrt()[:, None, None, None] * eps
        loss = F.mse_loss(m(xt, t), eps)
        opt.zero_grad(); loss.backward(); opt.step()
        for g in opt.param_groups:
            g["lr"] = 1e-3 * 0.5 * (1 + math.cos(math.pi * paso / PASOS))
        if paso % 300 == 0:
            print(f"paso {paso:5d}  pérdida {loss.item():.4f}", flush=True)
    gen = muestrear(m, 16)

    def dibujar(plt):
        g = (gen[:, 0] + 1) / 2
        cuadro = torch.cat([torch.cat(list(g[r * 4:(r + 1) * 4]), 1) for r in range(4)], 0)
        plt.imshow(cuadro, cmap="gray"); plt.axis("off"); plt.title("Muestras de difusión (mini U-Net)")

    guardar_figura("semana7_ddpm_imagenes", dibujar)
