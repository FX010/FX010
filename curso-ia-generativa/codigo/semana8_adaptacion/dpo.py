"""
DPO (Direct Preference Optimization, Rafailov et al. 2023) desde cero.

RLHF clásico: entrenar un modelo de recompensa + PPO. DPO lo elimina: optimiza la política
directamente con pares (preferido y_w, rechazado y_l):

   L = - log sigma( beta * [ (log pi(y_w|x) - log pi_ref(y_w|x))  -  (log pi(y_l|x) - log pi_ref(y_l|x)) ] )

  pi     = modelo que entrenamos (inicia como copia del base)
  pi_ref = copia CONGELADA del base (ancla: evita que el modelo se aleje demasiado)
  beta   = fuerza de la restricción (típico 0.1)

Experimento: preferencia = estilo informal ("oye, ...!") sobre formal. Medimos el % de frases
informales generadas y el 'margen' de recompensa implícita.

    python dpo.py
"""
import copy
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_modelo import cargar_o_entrenar_base, fraccion_informal
from comun import SUJETOS, VERBOS, OBJETOS, ADVERBIOS, fijar_semilla


def logprob_secuencia(modelo, ids):
    """Suma de log p(token_t | tokens_<t) sobre toda la secuencia. ids: [B,T] (con '\\n' inicial)."""
    logits, _ = modelo(ids[:, :-1])                              # [B,T-1,V]
    lp = F.log_softmax(logits, -1).gather(-1, ids[:, 1:, None]).squeeze(-1)   # [B,T-1]
    return lp.sum(1)


def perdida_dpo(pi, ref, ganador, perdedor, beta=0.1, alpha_sft=0.0):
    """ganador/perdedor: [B,T] enteros (misma longitud: se rellena a la derecha con '\\n', ver pares()).
    alpha_sft > 0 añade un término de máxima verosimilitud sobre la respuesta preferida (variante 'RPO'),
    que evita que DPO baje la probabilidad de AMBAS respuestas (likelihood displacement)."""
    lw, ll = logprob_secuencia(pi, ganador), logprob_secuencia(pi, perdedor)
    with torch.no_grad():
        rw, rl = logprob_secuencia(ref, ganador), logprob_secuencia(ref, perdedor)
    margen = beta * ((lw - rw) - (ll - rl))                      # recompensa implícita relativa
    perdida = -F.logsigmoid(margen).mean()
    if alpha_sft > 0:
        perdida = perdida - alpha_sft * (lw / (ganador.size(1) - 1)).mean()     # NLL por token de la preferida
    return perdida, margen.mean().item(), (margen > 0).float().mean().item()


def pares(stoi, n, rng):
    """Pares (informal preferida, formal rechazada) con la MISMA frase base. Longitud uniforme: se rellena con '\\n'."""
    G, P = [], []
    for _ in range(n):
        f = f"{rng.choice(SUJETOS)} {rng.choice(VERBOS)} {rng.choice(OBJETOS)}{rng.choice(ADVERBIOS)}"
        G.append(f"\noye, {f}!\n")
        P.append(f"\n{f}.\n")
    L = max(len(s) for s in G + P)
    cod = lambda s: [stoi[c] for c in s] + [stoi["\n"]] * (L - len(s))
    return torch.tensor([cod(s) for s in G]), torch.tensor([cod(s) for s in P])


def entrenar_dpo(base, stoi, itos, lr=3e-5, pasos=200, alpha_sft=0.0, beta=0.1, seed=0):
    fijar_semilla(seed)
    rng = random.Random(seed)
    ref = copy.deepcopy(base).eval()                     # política de referencia: CONGELADA
    for p in ref.parameters():
        p.requires_grad_(False)
    pi = copy.deepcopy(base)                             # política a entrenar
    opt = torch.optim.AdamW(pi.parameters(), lr=lr, weight_decay=0.0)
    for paso in range(pasos + 1):
        g, p_ = pares(stoi, 32, rng)
        loss, margen, acc = perdida_dpo(pi, ref, g, p_, beta=beta, alpha_sft=alpha_sft)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0); opt.step()
        if paso % 50 == 0:
            fi, gr, _ = fraccion_informal(pi, stoi, itos, 200)
            print(f"  paso {paso:4d}  pérdida {loss.item():6.3f}  margen {margen:+5.2f}  P(pref>rech)={acc*100:3.0f}%"
                  f"  | informales {fi*100:5.1f}%  gramaticales {gr*100:5.1f}%")
    return fraccion_informal(pi, stoi, itos)


if __name__ == "__main__":
    base, stoi, itos = cargar_o_entrenar_base()
    f0, g0, _ = fraccion_informal(base, stoi, itos)
    print(f"BASE: {f0*100:.1f}% informales | {g0*100:.1f}% gramaticales\n")

    if len(sys.argv) > 1:                                # uso: python dpo.py [lr] [pasos] [alpha_sft]
        configs = [(float(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 200,
                    float(sys.argv[3]) if len(sys.argv) > 3 else 0.0)]
    else:
        configs = [(3e-5, 200, 0.0), (3e-5, 200, 1.0)]   # DPO puro  vs  DPO + término SFT
    resumen = []
    for lr, pasos, alpha in configs:
        print(f"== DPO lr={lr}  pasos={pasos}  alpha_sft={alpha}")
        f, g, ej = entrenar_dpo(base, stoi, itos, lr, pasos, alpha)
        print(f"   ej: {ej[:2]}\n")
        resumen.append((alpha, f, g))
    print(f"{'variante':<24}{'informales':>12}{'gramaticales':>15}")
    print(f"{'modelo base':<24}{f0*100:>11.1f}%{g0*100:>14.1f}%")
    for alpha, f, g in resumen:
        print(f"{'DPO puro' if alpha == 0 else f'DPO + SFT (alpha={alpha})':<24}{f*100:>11.1f}%{g*100:>14.1f}%")
    print("""
Lectura: DPO solo mira la DIFERENCIA entre preferida y rechazada; puede lograr su objetivo bajando la
probabilidad de ambas (el texto 'se sale de la distribución'). Remedios habituales: SFT previo sobre las
preferidas, un término de verosimilitud sobre la preferida (alpha_sft), beta mayor, menos pasos, y
vigilar siempre una métrica de CALIDAD (aquí: % gramatical) además de la de preferencia.""")
