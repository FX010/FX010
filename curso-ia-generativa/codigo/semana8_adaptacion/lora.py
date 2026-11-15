"""
LoRA (Low-Rank Adaptation, Hu et al. 2021) escrito desde cero.

  W' = W + (alpha/r) * B A         con  A [r, d_in] ,  B [d_out, r] ,  r << d
  - W se CONGELA. Solo se entrenan A y B  (r*(d_in+d_out) parámetros en vez de d_in*d_out).
  - B se inicializa en 0  =>  al empezar el modelo es idéntico al base.
  - Tras entrenar se puede FUSIONAR: W <- W + (alpha/r) B A  (cero latencia extra en inferencia).

Experimento: modelo base mayormente 'formal' -> LoRA con frases 'informal' -> medimos qué % de
frases generadas empieza con "oye," antes y después, y cuántos parámetros entrenamos.

    python lora.py
"""
import copy
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_modelo import cargar_o_entrenar_base, codificar, entrenar, fraccion_informal
from comun import fijar_semilla, make_corpus


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.escala = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.escala

    @torch.no_grad()
    def fusionar(self):
        self.base.weight += self.escala * (self.B @ self.A)
        return self.base


def aplicar_lora(modelo, r=8, alpha=16, objetivos=("q_proj", "k_proj", "v_proj", "o_proj", "gate", "up", "down")):
    """Reemplaza in-place las nn.Linear cuyo nombre esté en `objetivos`."""
    for p in modelo.parameters():
        p.requires_grad_(False)
    for nombre, mod in list(modelo.named_modules()):
        for hijo_nombre, hijo in list(mod.named_children()):
            if isinstance(hijo, nn.Linear) and hijo_nombre in objetivos:
                setattr(mod, hijo_nombre, LoRALinear(hijo, r, alpha))
    return modelo


def fusionar_todo(modelo):
    for mod in list(modelo.modules()):
        for nombre, hijo in list(mod.named_children()):
            if isinstance(hijo, LoRALinear):
                setattr(mod, nombre, hijo.fusionar())
    return modelo


if __name__ == "__main__":
    fijar_semilla(0)
    base, stoi, itos = cargar_o_entrenar_base()
    f0, g0, ej0 = fraccion_informal(base, stoi, itos)
    print(f"\nANTES  de LoRA: {f0*100:5.1f}% informales | {g0*100:5.1f}% gramaticales. Ej: {ej0[:2]}")

    n_base = sum(p.numel() for p in base.parameters())
    modelo = aplicar_lora(copy.deepcopy(base), r=8, alpha=16)
    entrenables = [p for p in modelo.parameters() if p.requires_grad]
    n_ent = sum(p.numel() for p in entrenables)
    print(f"parámetros del modelo base: {n_base:,}")
    print(f"parámetros entrenables con LoRA (r=8): {n_ent:,}  ({100*n_ent/n_base:.1f}% del base)")
    print("(en un LLM real de miles de millones de parámetros con d≈4096 esta fracción es <1%; aquí d=96 es diminuto)")

    datos = codificar(make_corpus(2000, "informal", seed=7), stoi)
    entrenar(modelo, datos, 300, 3e-3, params=entrenables)

    # los pesos congelados NO deben haber cambiado: comparar W base original vs W dentro de LoRALinear
    w_orig = dict(base.named_parameters())
    cambio = 0.0
    for nombre, mod in modelo.named_modules():
        if isinstance(mod, LoRALinear):
            cambio = max(cambio, (mod.base.weight - w_orig[nombre + ".weight"]).abs().max().item())
    print(f"cambio máximo en los pesos congelados W: {cambio:.1e} (debe ser 0)")
    f1, g1, ej1 = fraccion_informal(modelo, stoi, itos)
    print(f"\nDESPUÉS de LoRA: {f1*100:5.1f}% informales | {g1*100:5.1f}% gramaticales. Ej: {ej1[:2]}")

    # fusionar y comprobar equivalencia
    modelo.eval()
    x = codificar("\noye, el gato come", stoi)[None]
    antes, _ = modelo(x)
    fusionar_todo(modelo)
    despues, _ = modelo(x)
    print(f"tras fusionar W + BA: máx |logits_antes - logits_después| = {(antes-despues).abs().max().item():.2e}")
    print(f"parámetros del modelo fusionado = {sum(p.numel() for p in modelo.parameters()):,} (igual que el base: {n_base:,})")
