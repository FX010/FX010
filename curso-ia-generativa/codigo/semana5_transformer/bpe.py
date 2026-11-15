"""
Tokenizador BPE (Byte-Pair Encoding) mínimo, a nivel de bytes, desde cero.

Algoritmo de entrenamiento:
   1. Empezar con los 256 bytes como vocabulario base.
   2. Contar todos los pares adyacentes de tokens en el corpus.
   3. Fusionar el par MÁS frecuente en un token nuevo (id 256, 257, ...).
   4. Repetir hasta alcanzar el tamaño de vocabulario deseado.
Codificar = aplicar las fusiones en el MISMO orden en que se aprendieron.

    python bpe.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def contar_pares(ids, conteo=None):
    conteo = {} if conteo is None else conteo
    for par in zip(ids, ids[1:]):
        conteo[par] = conteo.get(par, 0) + 1
    return conteo


def fusionar(ids, par, nuevo_id):
    salida, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == par:
            salida.append(nuevo_id)
            i += 2
        else:
            salida.append(ids[i])
            i += 1
    return salida


class BPE:
    def __init__(self):
        self.fusiones = {}                                   # (a,b) -> nuevo_id (orden de inserción = prioridad)
        self.vocab = {i: bytes([i]) for i in range(256)}     # id -> bytes

    def entrenar(self, texto: str, tam_vocab: int, verbose=False):
        ids = list(texto.encode("utf-8"))
        for nuevo_id in range(256, tam_vocab):
            conteo = contar_pares(ids)
            if not conteo:
                break
            par = max(conteo, key=conteo.get)
            ids = fusionar(ids, par, nuevo_id)
            self.fusiones[par] = nuevo_id
            self.vocab[nuevo_id] = self.vocab[par[0]] + self.vocab[par[1]]
            if verbose:
                print(f"  fusión {nuevo_id}: {self.vocab[par[0]]!r} + {self.vocab[par[1]]!r} -> {self.vocab[nuevo_id]!r}  (x{conteo[par]})")

    def codificar(self, texto: str):
        ids = list(texto.encode("utf-8"))
        while len(ids) >= 2:
            conteo = contar_pares(ids)
            # elegir el par con la fusión más temprana (menor id)
            par = min(conteo, key=lambda p: self.fusiones.get(p, float("inf")))
            if par not in self.fusiones:
                break
            ids = fusionar(ids, par, self.fusiones[par])
        return ids

    def decodificar(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


if __name__ == "__main__":
    from comun import make_corpus

    texto = make_corpus(500, "formal", seed=0)
    bpe = BPE()
    bpe.entrenar(texto, 256 + 40, verbose=False)
    for par, nid in list(bpe.fusiones.items())[:12]:
        print(f"  {nid}: {bpe.vocab[par[0]]!r} + {bpe.vocab[par[1]]!r} -> {bpe.vocab[nid]!r}")
    prueba = "el ingeniero construye un puente con calma.\nñandú ✓"
    ids = bpe.codificar(prueba)
    assert bpe.decodificar(ids) == prueba, "¡la codificación debe ser reversible!"
    print(f"\ntexto: {len(prueba.encode('utf-8'))} bytes -> {len(ids)} tokens (compresión x{len(prueba.encode())/len(ids):.2f})")
    print("ida y vuelta OK (incluye ñ y ✓ gracias al nivel de bytes)")
