"""
DDPM (Ho et al., 2020) completo sobre datos 2D, con muestreo DDPM y DDIM.

 Proceso directo (cerrado):  x_t = sqrt(ab_t) * x_0 + sqrt(1 - ab_t) * eps,   eps ~ N(0,I)
                             donde  alpha_t = 1 - beta_t ,  ab_t = prod_{s<=t} alpha_s
 Entrenamiento:  elegir t al azar, ruidar x_0, y minimizar  || eps - eps_theta(x_t, t) ||^2
 Muestreo DDPM:  x_{t-1} = 1/sqrt(alpha_t) * ( x_t - beta_t / sqrt(1-ab_t) * eps_theta ) + sigma_t * z
 Muestreo DDIM:  determinista, permite saltar pasos (p. ej. 20 en vez de 200)

Las redes de imágenes reales (U-Net / DiT) hacen EXACTAMENTE esto; solo cambia eps_theta.

    python ddpm_2d.py
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import fijar_semilla, guardar_figura, make_swiss_2d

T = 200
betas = torch.linspace(1e-4, 0.06, T)
alphas = 1 - betas
ab = torch.cumprod(alphas, 0)                       # alpha-barra


def emb_tiempo(t, dim=64):
    """Embedding sinusoidal del paso t (igual que las posiciones del Transformer)."""
    mitad = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(mitad) / mitad)
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


class EpsNet(nn.Module):
    def __init__(self, h=256):
        super().__init__()
        self.t = nn.Sequential(nn.Linear(64, h), nn.SiLU(), nn.Linear(h, h))
        self.inp = nn.Linear(2, h)
        self.bloques = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h)) for _ in range(4)]
        )
        self.out = nn.Linear(h, 2)

    def forward(self, x, t):                         # x [B,2], t [B]
        h = self.inp(x) + self.t(emb_tiempo(t))
        for b in self.bloques:
            h = h + b(h)                             # bloques residuales
        return self.out(h)                           # predice el ruido eps [B,2]


def ruidar(x0, t, eps):
    return ab[t].sqrt()[:, None] * x0 + (1 - ab[t]).sqrt()[:, None] * eps


@torch.no_grad()
def muestrear_ddpm(m, n):
    x = torch.randn(n, 2)
    for t in reversed(range(T)):
        eps = m(x, torch.full((n,), t))
        x = (x - betas[t] / (1 - ab[t]).sqrt() * eps) / alphas[t].sqrt()
        if t > 0:                                    # varianza posterior: sigma^2 = beta_t
            x = x + betas[t].sqrt() * torch.randn_like(x)
    return x


@torch.no_grad()
def muestrear_ddim(m, n, pasos=20):
    ts = torch.linspace(T - 1, 0, pasos).long()
    x = torch.randn(n, 2)
    for i, t in enumerate(ts):
        eps = m(x, torch.full((n,), int(t)))
        x0 = ((x - (1 - ab[t]).sqrt() * eps) / ab[t].sqrt()).clamp(-3, 3)   # estimación de x_0
        ab_prev = ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps               # salto determinista (eta=0)
    return x


def calidad(gen, datos, r=0.08):
    """precisión = % de muestras a menos de r de algún dato real (calidad: ¿caen sobre la espiral?)
    cobertura = % de datos a menos de r de alguna muestra (diversidad: ¿se cubre toda la espiral?)"""
    d = torch.cdist(gen, datos)
    return (d.min(1).values < r).float().mean().item(), (d.min(0).values < r).float().mean().item()


if __name__ == "__main__":
    fijar_semilla(0)
    datos = torch.tensor(make_swiss_2d(4096))
    m = EpsNet()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)
    PASOS = 4000
    for paso in range(PASOS + 1):
        x0 = datos[torch.randint(0, len(datos), (512,))]
        t = torch.randint(0, T, (512,))
        eps = torch.randn_like(x0)
        loss = ((m(ruidar(x0, t, eps), t) - eps) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        for g in opt.param_groups:
            g["lr"] = 1e-3 * 0.5 * (1 + math.cos(math.pi * paso / PASOS))
        if paso % 1000 == 0:
            print(f"paso {paso:5d}  pérdida (MSE del ruido) = {loss.item():.4f}", flush=True)

    ref = torch.randn(2000, 2) * datos.std()
    d0, c0 = calidad(ref, datos)
    g1 = muestrear_ddpm(m, 2000)
    g2 = muestrear_ddim(m, 2000, 20)
    print(f"\n{'':<28}{'precisión (calidad)':>22}{'cobertura (diversidad)':>25}")
    print(f"{'línea base: gaussiana':<28}{d0*100:>21.1f}%{c0*100:>24.1f}%")
    for nombre, g in [("DDPM (200 pasos)", g1), ("DDIM (20 pasos)", g2)]:
        d, c = calidad(g, datos)
        print(f"{nombre:<28}{d*100:>21.1f}%{c*100:>24.1f}%")

    def dibujar(plt):
        plt.gcf().set_size_inches(12, 4)
        for i, (titulo, p) in enumerate([("datos reales", datos[:2000]), ("DDPM 200 pasos", g1), ("DDIM 20 pasos", g2)]):
            plt.subplot(1, 3, i + 1)
            plt.scatter(p[:, 0], p[:, 1], s=3)
            plt.title(titulo)
            plt.xlim(-2.2, 2.2); plt.ylim(-2.2, 2.2)

    guardar_figura("semana7_ddpm_2d", dibujar)
