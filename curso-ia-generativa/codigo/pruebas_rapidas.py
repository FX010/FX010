"""
pruebas_rapidas.py — Verifica en < 1 minuto (CPU) que TODO el código del curso es correcto.

    python pruebas_rapidas.py

Cada prueba imprime ✔ o ✘. Úsalo (a) al instalar el entorno, (b) cada vez que modifiques
un archivo de las carpetas semanaN_*: si rompes algo, lo sabrás al instante.
Esto es la versión “profesional” de los gradient checks que aprendes en el curso.
"""
import math
import sys
import time
import traceback
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
for sub in ("", "semana1_micrograd", "semana2_mlp", "semana3_entrenamiento", "semana4_arquitecturas",
            "semana5_transformer", "semana6_vae_gan", "semana7_difusion", "semana8_adaptacion"):
    sys.path.insert(0, str(RAIZ / sub))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RESULTADOS = []


def prueba(nombre):
    def deco(f):
        t0 = time.time()
        try:
            f()
            RESULTADOS.append((nombre, True, time.time() - t0, ""))
            print(f"  ✔ {nombre}  ({time.time()-t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            RESULTADOS.append((nombre, False, time.time() - t0, str(e)))
            print(f"  ✘ {nombre}: {e}")
            traceback.print_exc(limit=2)
        return f
    return deco


torch.manual_seed(0)
np.random.seed(0)
print("Semana 1")


@prueba("micrograd: autodiff == derivada numérica")
def _():
    from engine import Value
    from verificar_gradientes import derivada_numerica, f
    vals = [0.7, -1.3, 0.4]
    a, b, c = (Value(v) for v in vals)
    f(a, b, c).backward()
    for i, v in enumerate((a, b, c)):
        num = derivada_numerica(i, vals)
        assert abs(v.grad - num) < 1e-6, (i, v.grad, num)


@prueba("micrograd: ejemplo a mano de la guía (dL/dw = -1.00449)")
def _():
    from engine import Value
    w, x, b = Value(0.5), Value(2.0), Value(-0.3)
    L = ((w * x + b).tanh() - 1) ** 2
    L.backward()
    assert abs(L.data - 0.15652) < 1e-4 and abs(w.grad + 1.00449) < 1e-4 and abs(b.grad + 0.50225) < 1e-4


print("Semana 2")


@prueba("MLP NumPy: backprop manual == diferencias finitas")
def _():
    from comun import make_espirales
    from mlp_numpy import MLPNumpy, verificar_gradientes
    X, y = make_espirales(30, 3)
    assert verificar_gradientes(MLPNumpy([2, 16, 16, 3]), X.astype(np.float64), y) < 1e-5


@prueba("autograd de PyTorch == backprop manual de NumPy")
def _():
    from comun import make_espirales
    from mlp_numpy import MLPNumpy
    from mlp_torch import crear_mlp
    X, y = make_espirales(20, 3)
    man, mod = MLPNumpy([2, 8, 8, 3]), crear_mlp([2, 8, 8, 3])
    lins = [m for m in mod if isinstance(m, nn.Linear)]
    with torch.no_grad():
        for lin, W, b in zip(lins, man.W, man.b):
            lin.weight.copy_(torch.tensor(W.T, dtype=torch.float32))
            lin.bias.copy_(torch.tensor(b, dtype=torch.float32))
    _, dl = man.pérdida_y_dlogits(man.forward(X.astype(np.float64)), y)
    gW, _ = man.backward(dl)
    F.cross_entropy(mod(torch.tensor(X)), torch.tensor(y)).backward()
    for lin, g in zip(lins, gW):
        assert np.abs(lin.weight.grad.numpy().T - g).max() < 1e-6


print("Semana 3")


@prueba("BatchNorm/LayerNorm/RMSNorm propias == torch.nn")
def _():
    from normalizaciones import BatchNorm1dScratch, LayerNormScratch, RMSNorm
    x = torch.randn(32, 64) * 3 + 5
    assert (BatchNorm1dScratch(64)(x) - nn.BatchNorm1d(64)(x)).abs().max() < 1e-4
    assert (LayerNormScratch(64)(x) - nn.LayerNorm(64)(x)).abs().max() < 1e-4
    if hasattr(nn, "RMSNorm"):
        assert (RMSNorm(64)(x) - nn.RMSNorm(64, eps=1e-6)(x)).abs().max() < 1e-4


@prueba("cota de entropía del corpus ≈ 0.20 nats/char")
def _():
    from comun import entropia_corpus_nats_por_char, make_corpus
    h = entropia_corpus_nats_por_char(make_corpus(2000))
    assert 0.18 < h < 0.23, h


print("Semana 4")


@prueba("CNN: nº de parámetros = 2 836 (cálculo a mano de la guía)")
def _():
    cnn = nn.Sequential(nn.Conv2d(1, 12, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(24, 4))
    assert sum(p.numel() for p in cnn.parameters()) == 2836
    assert cnn(torch.randn(2, 1, 16, 16)).shape == (2, 4)


@prueba("residual: gradiente 1.ª capa mayor que en red plana de 40 capas")
def _():
    from comun import make_espirales
    from residuales_gradientes import red
    X, y = make_espirales(50, 3)
    g = []
    for res in (False, True):
        torch.manual_seed(0)
        m = red(40, res)
        for mod in m.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
        F.cross_entropy(m(torch.tensor(X)), torch.tensor(y)).backward()
        g.append(m[0].weight.grad.norm().item())
    assert g[1] > 10 * g[0], g


print("Semana 5")


@prueba("BPE: codificar → decodificar es reversible (con ñ y ✓)")
def _():
    from bpe import BPE
    from comun import make_corpus
    b = BPE()
    b.entrenar(make_corpus(200), 300)
    s = "el ingeniero construye un puente.\nñandú ✓"
    assert b.decodificar(b.codificar(s)) == s


@prueba("atención propia == F.scaled_dot_product_attention (causal)")
def _():
    from atencion import atencion_escalada
    q, k, v = (torch.randn(2, 4, 6, 8) for _ in range(3))
    mio, pesos = atencion_escalada(q, k, v)
    assert (mio - F.scaled_dot_product_attention(q, k, v, is_causal=True)).abs().max() < 1e-5
    assert torch.allclose(pesos.sum(-1), torch.ones(2, 4, 6)) and (pesos.triu(1) == 0).all()


@prueba("RoPE: q·k depende solo de la distancia relativa")
def _():
    from atencion import aplicar_rope, rope_tablas
    Dh = 16
    cos, sin = rope_tablas(32, Dh)
    q0, k0 = torch.randn(Dh), torch.randn(Dh)
    p = lambda i, j: (aplicar_rope(q0.view(1, 1, 1, Dh), cos[i:i+1], sin[i:i+1]) *
                      aplicar_rope(k0.view(1, 1, 1, Dh), cos[j:j+1], sin[j:j+1])).sum().item()
    assert abs(p(3, 1) - p(20, 18)) < 1e-4 and abs(p(3, 1) - p(3, 2)) > 1e-3


@prueba("GPT: forma, pérdida inicial ≈ ln V, causalidad y sobreajuste de 1 batch")
def _():
    from gpt import GPT, GPTConfig
    for pos, kv in (("rope", None), ("learned", None), ("rope", 2)):
        torch.manual_seed(0)
        cfg = GPTConfig(vocab_size=12, block_size=16, n_layer=2, n_head=4, n_kv_head=kv, n_embd=32, pos=pos)
        m = GPT(cfg)
        idx = torch.randint(0, 12, (4, 16))
        tg = torch.randint(0, 12, (4, 16))                         # objetivos independientes de la entrada
        logits, loss = m(idx, tg)
        assert logits.shape == (4, 16, 12) and abs(loss.item() - math.log(12)) < 0.3, loss.item()
        idx2 = idx.clone()
        idx2[:, 8:] = (idx2[:, 8:] + 1) % 12                      # cambiar SOLO el futuro
        assert (m(idx)[0][:, :8] - m(idx2)[0][:, :8]).abs().max() < 1e-5, "¡el pasado ve el futuro!"
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    x, y = idx[:, :-1], idx[:, 1:]
    for _ in range(150):
        _, loss = m(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.1, loss.item()


print("Semana 6")


@prueba("VAE: KL cerrada = 0.5 para mu=[1,0], logvar=0; pérdida es finita")
def _():
    from vae import VAE, perdida
    mu, lv = torch.tensor([[1.0, 0.0]]), torch.zeros(1, 2)
    kl = (-0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1)).item()
    assert abs(kl - 0.5) < 1e-6
    m = VAE()
    x = torch.rand(8, 256)
    logits, mu_, lv_ = m(x)
    l, rec, k = perdida(logits, x, mu_, lv_)
    assert torch.isfinite(l)


print("Semana 7")


@prueba("difusión: x_t cerrado tiene media √ab·x0 y varianza (1-ab)")
def _():
    from ddpm_2d import ab, ruidar
    x0 = torch.full((200000, 2), 0.7)
    t = torch.full((200000,), 99)
    xt = ruidar(x0, t, torch.randn_like(x0))
    assert abs(xt.mean().item() - ab[99].sqrt().item() * 0.7) < 5e-3
    assert abs(xt.var().item() - (1 - ab[99]).item()) < 5e-3
    assert ab[-1] < 0.01 and (ab[1:] < ab[:-1]).all()


@prueba("mini U-Net: forma de salida = forma de entrada")
def _():
    from ddpm_imagenes import MiniUNet
    assert MiniUNet()(torch.randn(3, 1, 16, 16), torch.tensor([0, 50, 199])).shape == (3, 1, 16, 16)


print("Semana 8")


@prueba("LoRA: B=0 ⇒ idéntico al base; congela W; fusión exacta")
def _():
    import copy
    from gpt import GPT, GPTConfig
    from lora import LoRALinear, aplicar_lora, fusionar_todo
    torch.manual_seed(0)
    base = GPT(GPTConfig(vocab_size=12, block_size=16, n_layer=2, n_head=4, n_embd=32))
    x = torch.randint(0, 12, (2, 10))
    m = aplicar_lora(copy.deepcopy(base), r=4, alpha=8)
    assert (m(x)[0] - base(x)[0]).abs().max() < 1e-6, "con B=0 debe ser idéntico"
    assert all(not p.requires_grad for n, p in m.named_parameters() if not (n.endswith(".A") or n.endswith(".B")))
    for mod in m.modules():                                       # perturbar B para que LoRA haga algo
        if isinstance(mod, LoRALinear):
            nn.init.normal_(mod.B, 0, 0.1)
    antes = m(x)[0]
    assert (antes - base(x)[0]).abs().max() > 1e-3
    fusionar_todo(m)
    assert (m(x)[0] - antes).abs().max() < 1e-4, "la fusión debe preservar la salida"


@prueba("DPO: con política == referencia la pérdida es ln 2; si sube la preferida baja la pérdida")
def _():
    import copy
    from dpo import perdida_dpo
    from gpt import GPT, GPTConfig
    torch.manual_seed(0)
    ref = GPT(GPTConfig(vocab_size=12, block_size=16, n_layer=1, n_head=2, n_embd=16))
    pi = copy.deepcopy(ref)
    g, p = torch.randint(0, 12, (6, 12)), torch.randint(0, 12, (6, 12))
    l0, _, _ = perdida_dpo(pi, ref, g, p)
    assert abs(l0.item() - math.log(2)) < 1e-5
    opt = torch.optim.SGD(pi.parameters(), lr=0.05)
    for _ in range(20):
        l, _, _ = perdida_dpo(pi, ref, g, p)
        opt.zero_grad(); l.backward(); opt.step()
    assert l.item() < l0.item() - 0.05


@prueba("RAG: hit@1 ≥ 8/10 en el conjunto de ejemplo")
def _():
    from rag_minimo import DOCS, Indice
    idx = Indice(DOCS)
    qs = [("¿Cómo se calcula el gradiente en una red neuronal?", "backprop"), ("¿Qué hace la máscara causal?", "atencion"),
          ("¿Cómo adaptar un LLM entrenando pocas matrices?", "lora"), ("¿Qué es el colapso de modos?", "gan"),
          ("¿Cómo se alinea un modelo con preferencias sin modelo de recompensa?", "dpo"),
          ("¿Para qué sirve la reparametrización?", "vae"), ("¿Cómo funciona la tokenización por subpalabras?", "bpe"),
          ("¿Qué optimizador usa weight decay desacoplado?", "adam"),
          ("¿Cómo se codifica la posición de forma relativa?", "rope"),
          ("¿Cómo se generan imágenes quitando ruido paso a paso?", "difusion")]
    assert sum(idx.buscar(q, 1)[0][0] == e for q, e in qs) >= 8


@prueba("corpus: es_gramatical reconoce frases válidas y rechaza erróneas")
def _():
    from base_modelo import es_gramatical
    assert es_gramatical("el gato come pescado.") and es_gramatical("oye, la niña mira la red hoy!")
    assert not es_gramatical("ooye, el gato come pescado!") and not es_gramatical("el gato come pescado!")


ok = sum(r[1] for r in RESULTADOS)
print(f"\n{ok}/{len(RESULTADOS)} pruebas correctas")
sys.exit(0 if ok == len(RESULTADOS) else 1)
