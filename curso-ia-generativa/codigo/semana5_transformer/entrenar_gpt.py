"""
Entrena el GPT de gpt.py a nivel de carácter sobre el corpus sintético y compara
la pérdida de validación con la COTA TEÓRICA (entropía del corpus).

    python entrenar_gpt.py                  # RoPE + MHA (por defecto)
    python entrenar_gpt.py --pos learned    # posiciones aprendidas
    python entrenar_gpt.py --kv 1           # MQA (un solo K/V) | --kv 2 => GQA
    python entrenar_gpt.py --archivo mi_texto.txt   # ¡usa TU texto! (p. ej. Tiny Shakespeare)
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import SALIDAS, entropia_corpus_nats_por_char, fijar_semilla, make_corpus
from gpt import GPT, GPTConfig


def cargar_datos(archivo=None, estilo="formal", n_frases=5000, seed=0):
    texto = Path(archivo).read_text(encoding="utf-8") if archivo else make_corpus(n_frases, estilo, seed)
    chars = sorted(set(texto))
    stoi = {c: i for i, c in enumerate(chars)}
    return texto, chars, stoi


def lote(datos, B, T, dispositivo="cpu"):
    ix = torch.randint(0, len(datos) - T - 1, (B,))
    x = torch.stack([datos[i:i + T] for i in ix])
    y = torch.stack([datos[i + 1:i + T + 1] for i in ix])
    return x.to(dispositivo), y.to(dispositivo)


@torch.no_grad()
def evaluar(m, datos, B=64, T=64, iters=10):
    m.eval()
    l = sum(m(*lote(datos, B, T))[1].item() for _ in range(iters)) / iters
    m.train()
    return l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="rope", choices=["rope", "learned"])
    ap.add_argument("--kv", type=int, default=None, help="n_kv_head (None=MHA, 1=MQA, 2=GQA)")
    ap.add_argument("--pasos", type=int, default=800)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--embd", type=int, default=96)
    ap.add_argument("--archivo", default=None)
    ap.add_argument("--guardar", default=None, help="ruta del checkpoint a guardar")
    a = ap.parse_args()

    fijar_semilla(0)
    texto, chars, stoi = cargar_datos(a.archivo)
    itos = {i: c for c, i in stoi.items()}
    datos = torch.tensor([stoi[c] for c in texto])
    n = int(0.9 * len(datos))
    tr, va = datos[:n], datos[n:]

    cfg = GPTConfig(vocab_size=len(chars), block_size=64, n_layer=a.layers, n_head=4, n_kv_head=a.kv,
                    n_embd=a.embd, pos=a.pos)
    m = GPT(cfg)
    print(f"vocab={cfg.vocab_size}  params={m.n_params():,}  pos={a.pos}  kv_heads={cfg.n_kv_head or cfg.n_head}/{cfg.n_head}")
    if not a.archivo:
        print(f"cota teórica de la pérdida = {entropia_corpus_nats_por_char(texto):.3f} nats/char")

    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)
    t0 = time.time()
    for paso in range(a.pasos + 1):
        lr = 3e-3 * min(1.0, (paso + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + __import__("math").cos(3.14159 * paso / a.pasos)))  # warmup + coseno
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = lote(tr, 32, 64)
        _, loss = m(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if paso % 200 == 0:
            print(f"paso {paso:5d}  train {loss.item():.3f}  val {evaluar(m, va):.3f}  ({time.time()-t0:.0f}s)")

    print("\nmuestras (temperatura 0.8):")
    for _ in range(4):
        i0 = torch.tensor([[stoi["\n"]]])
        print("  ", "".join(itos[int(i)] for i in m.generate(i0, 60, temperature=0.8)[0]).strip().split("\n")[0] or "(vacía)")

    if a.guardar:
        Path(a.guardar).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": cfg, "estado": m.state_dict(), "chars": chars}, a.guardar)
        print(f"checkpoint guardado en {a.guardar}")


if __name__ == "__main__":
    main()
