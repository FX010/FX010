"""
Atención desde cero, con anotación de dimensiones y verificación contra PyTorch.

  Attention(Q,K,V) = softmax( Q K^T / sqrt(d_k) + máscara ) V

  B = batch, T = tokens, C = d_model, H = cabezas, Dh = C/H

Incluye:
  1. atencion_escalada()   — la fórmula tal cual, con máscara causal opcional
  2. MultiHead             — proyecciones Q,K,V, cabezas, concatenación y proyección de salida
  3. RoPE (rotary position embedding) y test de la propiedad relativa
  4. Test: ¿por qué dividir por sqrt(d_k)?  (la varianza de q·k crece con d_k)

    python atencion.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def atencion_escalada(q, k, v, causal=True):
    """q,k,v: [B,H,T,Dh]  ->  salida [B,H,T,Dh], pesos [B,H,T,T]"""
    dk = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(dk)            # [B,H,T,T]
    if causal:
        T = q.size(-2)
        mascara = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
        scores = scores.masked_fill(~mascara, float("-inf"))     # un token no puede mirar al futuro
    pesos = scores.softmax(-1)                                   # cada fila suma 1
    return pesos @ v, pesos


class MultiHead(nn.Module):
    def __init__(self, C, H):
        super().__init__()
        assert C % H == 0
        self.H, self.Dh = H, C // H
        self.qkv = nn.Linear(C, 3 * C, bias=False)
        self.proj = nn.Linear(C, C, bias=False)

    def forward(self, x, causal=True):                           # x [B,T,C]
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)                   # 3 x [B,T,C]
        q, k, v = (t.view(B, T, self.H, self.Dh).transpose(1, 2) for t in (q, k, v))   # [B,H,T,Dh]
        y, self.pesos = atencion_escalada(q, k, v, causal)       # [B,H,T,Dh]
        y = y.transpose(1, 2).contiguous().view(B, T, C)         # [B,T,C]
        return self.proj(y)


# ------------------------------------------------------------------ RoPE
def rope_tablas(T, Dh, base=10000.0):
    """cos/sin [T, Dh]: frecuencias geométricas theta_i = base^(-2i/Dh)."""
    theta = base ** (-torch.arange(0, Dh, 2).float() / Dh)       # [Dh/2]
    ang = torch.outer(torch.arange(T).float(), theta)            # [T, Dh/2]
    ang = torch.cat([ang, ang], -1)                              # [T, Dh]
    return ang.cos(), ang.sin()


def rotar_mitad(x):
    x1, x2 = x.chunk(2, -1)
    return torch.cat([-x2, x1], -1)


def aplicar_rope(x, cos, sin):
    """x [B,H,T,Dh]: rota cada par de dimensiones un ángulo proporcional a la posición."""
    return x * cos + rotar_mitad(x) * sin


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, C, H = 2, 6, 32, 4
    x = torch.randn(B, T, C)

    # 1) comparar con la implementación optimizada de PyTorch
    mh = MultiHead(C, H)
    q, k, v = mh.qkv(x).split(C, -1)
    q, k, v = (t.view(B, T, H, C // H).transpose(1, 2) for t in (q, k, v))
    mio, pesos = atencion_escalada(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"[1] máx |mi atención - F.scaled_dot_product_attention| = {(mio-ref).abs().max():.2e}")
    print(f"    pesos: forma {tuple(pesos.shape)}, filas suman 1: {torch.allclose(pesos.sum(-1), torch.ones(B,H,T))}")
    print(f"    triángulo superior (futuro) = 0: {bool((pesos.triu(1)==0).all())}")
    print(f"    salida MultiHead: {tuple(mh(x).shape)}")

    # 2) RoPE: el producto q·k depende solo de la distancia relativa
    Dh = 16
    cos, sin = rope_tablas(32, Dh)
    q0, k0 = torch.randn(Dh), torch.randn(Dh)
    def punto(i, j):
        qi = aplicar_rope(q0.view(1, 1, 1, Dh), cos[i:i+1], sin[i:i+1])
        kj = aplicar_rope(k0.view(1, 1, 1, Dh), cos[j:j+1], sin[j:j+1])
        return (qi * kj).sum().item()
    print(f"\n[2] RoPE: q·k(pos 3,pos 1)={punto(3,1):.5f}  q·k(pos 20,pos 18)={punto(20,18):.5f}  (misma distancia 2 => iguales)")
    print(f"    q·k(pos 3,pos 1)={punto(3,1):.5f}  vs  q·k(pos 3,pos 2)={punto(3,2):.5f}  (distinta distancia => distintos)")

    # 3) por qué sqrt(d_k)
    print("\n[3] varianza de q·k con q,k ~ N(0,1) (debería ser d_k sin escalar, 1 con escalado)")
    for dk in (16, 64, 256, 1024):
        a, b = torch.randn(20000, dk), torch.randn(20000, dk)
        s = (a * b).sum(-1)
        print(f"    d_k={dk:5d}  var(q·k)={s.var():8.1f}   var(q·k/sqrt(d_k))={(s/math.sqrt(dk)).var():.2f}")
    print("    sin escalar, los scores crecen -> softmax casi one-hot -> gradientes diminutos.")
