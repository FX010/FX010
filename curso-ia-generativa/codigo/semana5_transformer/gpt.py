"""
gpt.py — Un GPT (Transformer decoder-only) moderno y compacto, ~120 líneas, en PyTorch.

Incluye las piezas de los LLM actuales (estilo LLaMA):
  • RMSNorm en vez de LayerNorm (pre-norm)
  • RoPE (posiciones rotatorias) o embeddings posicionales aprendidos (pos="learned")
  • Atención causal con GQA (Grouped-Query Attention): n_kv_head <= n_head
  • MLP con activación SwiGLU
  • Pesos del embedding atados (weight tying) con la cabeza de salida

Mapa de dimensiones (B=batch, T=tokens, C=n_embd, H=n_head, Dh=C/H):
   idx [B,T] -> tok_emb [B,T,C] -> N x ( x + Attn(RMSNorm(x)) ; x + MLP(RMSNorm(x)) ) -> RMSNorm -> logits [B,T,V]
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int | None = None      # None => igual a n_head (MHA). 1 => MQA. Intermedio => GQA
    n_embd: int = 128
    pos: str = "rope"                 # "rope" | "learned"
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.g = eps, nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope_tablas(T, Dh, base=10000.0, device=None):
    theta = base ** (-torch.arange(0, Dh, 2, device=device).float() / Dh)
    ang = torch.outer(torch.arange(T, device=device).float(), theta)
    ang = torch.cat([ang, ang], -1)
    return ang.cos(), ang.sin()


def aplicar_rope(x, cos, sin):                      # x [B,H,T,Dh]
    x1, x2 = x.chunk(2, -1)
    return x * cos + torch.cat([-x2, x1], -1) * sin


class Atencion(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.H, self.Hkv = cfg.n_head, cfg.n_kv_head or cfg.n_head
        assert cfg.n_embd % self.H == 0 and self.H % self.Hkv == 0
        self.Dh = cfg.n_embd // self.H
        self.q_proj = nn.Linear(cfg.n_embd, self.H * self.Dh, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, self.Hkv * self.Dh, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, self.Hkv * self.Dh, bias=False)
        self.o_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.rope = cfg.pos == "rope"
        self.dropout = cfg.dropout

    def forward(self, x):                                              # x [B,T,C]
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.H, self.Dh).transpose(1, 2)      # [B,H,T,Dh]
        k = self.k_proj(x).view(B, T, self.Hkv, self.Dh).transpose(1, 2)    # [B,Hkv,T,Dh]
        v = self.v_proj(x).view(B, T, self.Hkv, self.Dh).transpose(1, 2)
        if self.rope:
            cos, sin = rope_tablas(T, self.Dh, device=x.device)
            q, k = aplicar_rope(q, cos, sin), aplicar_rope(k, cos, sin)
        if self.Hkv != self.H:                                              # GQA: cada K/V sirve a H/Hkv cabezas Q
            rep = self.H // self.Hkv
            k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)   # [B,H,T,Dh]
        return self.o_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU(nn.Module):
    """MLP(x) = W_down( SiLU(W_gate x) * (W_up x) ); hidden ≈ 8/3·C para igualar parámetros del MLP clásico 4C."""

    def __init__(self, C):
        super().__init__()
        h = int(8 * C / 3 / 8) * 8 or 8
        self.gate, self.up, self.down = nn.Linear(C, h, bias=False), nn.Linear(C, h, bias=False), nn.Linear(h, C, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Bloque(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.att = RMSNorm(cfg.n_embd), Atencion(cfg)
        self.n2, self.mlp = RMSNorm(cfg.n_embd), SwiGLU(cfg.n_embd)

    def forward(self, x):
        x = x + self.att(self.n1(x))      # conexión residual 1
        return x + self.mlp(self.n2(x))   # conexión residual 2


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.pos == "learned" else None
        self.bloques = nn.ModuleList(Bloque(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok.weight                         # weight tying
        self.apply(self._init)
        for n, p in self.named_parameters():                          # escalar proyecciones residuales (GPT-2)
            if n.endswith("o_proj.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):                             # idx [B,T]
        B, T = idx.shape
        x = self.tok(idx)                                             # [B,T,C]
        if self.pos is not None:
            x = x + self.pos(torch.arange(T, device=idx.device))
        for b in self.bloques:
            x = b(x)
        logits = self.lm_head(self.norm_f(x))                         # [B,T,V]
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size:])
            logits = logits[:, -1] / max(temperature, 1e-6)           # [B,V] del último token
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            idx = torch.cat([idx, torch.multinomial(logits.softmax(-1), 1)], 1)
        return idx

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
