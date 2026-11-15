"""
Modelo base compartido por lora.py y dpo.py.

Preentrena (o carga de caché) un GPT pequeño sobre un corpus MIXTO: 90% frases 'formal'
("el gato come pescado.") y 10% 'informal' ("oye, el gato come pescado!"). Es como un LLM
generalista: sabe hacer de todo pero su estilo por defecto es el formal.
El post-entrenamiento (LoRA / DPO) servirá para empujarlo hacia el estilo informal.
"""
import random
import re
import sys
from pathlib import Path

import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
sys.path.insert(0, str(AQUI.parent / "semana5_transformer"))
from comun import ADVERBIOS, OBJETOS, SALIDAS, SUJETOS, VERBOS, fijar_semilla, make_corpus
from gpt import GPT, GPTConfig

CKPT = SALIDAS / "base_mixto.pt"


def vocab():
    texto = make_corpus(500, "formal") + make_corpus(500, "informal")
    chars = sorted(set(texto))
    return chars, {c: i for i, c in enumerate(chars)}, dict(enumerate(chars))


def corpus_mixto(n=5000, frac_informal=0.1, seed=0):
    rng = random.Random(seed)
    f = make_corpus(int(n * (1 - frac_informal)), "formal", seed).splitlines()
    i = make_corpus(int(n * frac_informal), "informal", seed + 1).splitlines()
    lineas = f + i
    rng.shuffle(lineas)
    return "\n".join(lineas) + "\n"


def codificar(texto, stoi):
    return torch.tensor([stoi[c] for c in texto])


def lote(datos, B=32, T=64):
    ix = torch.randint(0, len(datos) - T - 1, (B,))
    return torch.stack([datos[i:i + T] for i in ix]), torch.stack([datos[i + 1:i + T + 1] for i in ix])


def entrenar(modelo, datos, pasos, lr, params=None, B=32, T=64, log=200):
    opt = torch.optim.AdamW(params or modelo.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    for paso in range(pasos + 1):
        _, loss = modelo(*lote(datos, B, T))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0); opt.step()
        if log and paso % log == 0:
            print(f"   paso {paso:5d}  pérdida {loss.item():.3f}")


def cargar_o_entrenar_base(pasos=800):
    chars, stoi, itos = vocab()
    cfg = GPTConfig(vocab_size=len(chars), block_size=64, n_layer=3, n_head=4, n_embd=96)
    m = GPT(cfg)
    if CKPT.exists():
        m.load_state_dict(torch.load(CKPT)["estado"])
        print(f"[base] cargado {CKPT}")
    else:
        print("[base] preentrenando modelo base (≈1-2 min en CPU)...")
        fijar_semilla(0)
        entrenar(m, codificar(corpus_mixto(), stoi), pasos, 3e-3)
        SALIDAS.mkdir(exist_ok=True)
        torch.save({"estado": m.state_dict()}, CKPT)
    return m, stoi, itos


@torch.no_grad()
def muestrear_lineas(m, stoi, itos, n=200, temperatura=1.0):
    """Genera n frases desde '\\n' y devuelve la lista de frases."""
    m.eval()
    idx = torch.full((n, 1), stoi["\n"])
    salida = m.generate(idx, 60, temperature=temperatura)
    lineas = []
    for fila in salida:
        s = "".join(itos[int(i)] for i in fila).split("\n")
        lineas.append(s[1] if len(s) > 1 else s[0])
    m.train()
    return lineas


_GRAMATICA = re.compile(
    r"^(oye, )?(%s) (%s) (%s)(%s)[.!]$" % (
        "|".join(SUJETOS), "|".join(VERBOS), "|".join(OBJETOS), "|".join(re.escape(a) for a in ADVERBIOS)))


def es_gramatical(linea):
    """¿La línea pertenece EXACTAMENTE a la gramática del corpus? (métrica objetiva de calidad)"""
    m = _GRAMATICA.match(linea)
    return bool(m) and (linea.startswith("oye, ") == linea.endswith("!"))


def fraccion_informal(m, stoi, itos, n=300):
    """Devuelve (fracción que empieza con 'oye,', fracción gramatical, ejemplos)."""
    lineas = muestrear_lineas(m, stoi, itos, n)
    return (sum(l.startswith("oye, ") for l in lineas) / n,
            sum(es_gramatical(l) for l in lineas) / n, lineas)
