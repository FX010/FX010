"""
RAG (Retrieval-Augmented Generation) mínimo y SIN dependencias externas.

Pipeline:  documentos -> trozos (chunks) -> embeddings -> índice vectorial
           pregunta   -> embedding -> top-k por similitud coseno -> prompt = contexto + pregunta -> LLM

Aquí el 'embedding' es un hashing de palabras y bigramas con TF (suficiente para entender la
mecánica y probar la recuperación). En producción lo cambias por embeddings neuronales
(sentence-transformers, API de embeddings) y un índice vectorial (FAISS, pgvector, Chroma...):
SOLO cambia la función `embed`.

    python rag_minimo.py
"""
import re
import unicodedata

import numpy as np

DOCS = {
    "backprop": "La retropropagación aplica la regla de la cadena sobre el grafo computacional para obtener el gradiente de la pérdida respecto a cada parámetro.",
    "adam": "AdamW mantiene promedios móviles del gradiente y de su cuadrado, y aplica el weight decay desacoplado de la actualización adaptativa.",
    "atencion": "La atención escalada calcula softmax de Q por K transpuesta dividido por la raíz de d_k, y pondera los valores V. La máscara causal impide mirar tokens futuros.",
    "rope": "RoPE rota pares de dimensiones de las consultas y claves según la posición, de modo que el producto punto depende solo de la distancia relativa entre tokens.",
    "lora": "LoRA congela los pesos originales y entrena dos matrices de bajo rango A y B cuya multiplicación se suma a W, reduciendo muchísimo los parámetros entrenables.",
    "dpo": "DPO optimiza directamente la política con pares de respuestas preferida y rechazada usando un modelo de referencia congelado, sin entrenar un modelo de recompensa.",
    "vae": "El autoencoder variacional aprende una distribución latente gaussiana y usa el truco de reparametrización para poder retropropagar a través del muestreo.",
    "difusion": "Los modelos de difusión añaden ruido gaussiano en T pasos y entrenan una red para predecir ese ruido, que luego se revierte paso a paso para generar muestras.",
    "gan": "En una GAN el generador intenta engañar al discriminador; el entrenamiento adversarial puede sufrir colapso de modos e inestabilidad.",
    "bpe": "BPE fusiona iterativamente el par de tokens más frecuente para construir un vocabulario de subpalabras que equilibra longitud de secuencia y tamaño del vocabulario.",
}


def normalizar(t):
    t = unicodedata.normalize("NFD", t.lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


PARADA = set("a al algo como con cual cuales de del el ella en es esta este la las lo los me mi para por que se si sin su sus un una uno unos y o u e ni que hace hay son ser sobre entre cada solo".split())


def tokens(t):
    """Palabras sin stopwords y 'stemming' bruto (prefijo de 5 letras) + bigramas."""
    w = [x[:5] for x in re.findall(r"[a-z0-9_]+", normalizar(t)) if x not in PARADA]
    return w + [a + "_" + b for a, b in zip(w, w[1:])]


def embed(texto, dim=2048):
    v = np.zeros(dim)
    for tok in tokens(texto):
        v[hash_estable(tok) % dim] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


def hash_estable(s):                       # hash() de Python cambia entre ejecuciones; este no
    h = 5381
    for c in s:
        h = (h * 33 + ord(c)) & 0xFFFFFFFF
    return h


class Indice:
    def __init__(self, docs):
        self.ids = list(docs)
        self.textos = [docs[i] for i in self.ids]
        self.M = np.stack([embed(t) for t in self.textos])            # [N, dim]

    def buscar(self, pregunta, k=2):
        sim = self.M @ embed(pregunta)                                # coseno (vectores normalizados)
        orden = np.argsort(-sim)[:k]
        return [(self.ids[i], float(sim[i]), self.textos[i]) for i in orden]


def construir_prompt(pregunta, resultados):
    contexto = "\n".join(f"[{i}] {t}" for i, _, t in resultados)
    return (f"Responde SOLO con la información del contexto. Si no está, di 'no lo sé'.\n\n"
            f"Contexto:\n{contexto}\n\nPregunta: {pregunta}\nRespuesta:")


if __name__ == "__main__":
    idx = Indice(DOCS)
    evaluacion = [
        ("¿Cómo se calcula el gradiente en una red neuronal?", "backprop"),
        ("¿Qué hace la máscara causal?", "atencion"),
        ("¿Cómo adaptar un LLM entrenando pocas matrices?", "lora"),
        ("¿Qué es el colapso de modos?", "gan"),
        ("¿Cómo se alinea un modelo con preferencias sin modelo de recompensa?", "dpo"),
        ("¿Para qué sirve la reparametrización?", "vae"),
        ("¿Cómo funciona la tokenización por subpalabras?", "bpe"),
        ("¿Qué optimizador usa weight decay desacoplado?", "adam"),
        ("¿Cómo se codifica la posición de forma relativa?", "rope"),
        ("¿Cómo se generan imágenes quitando ruido paso a paso?", "difusion"),
    ]
    aciertos = 0
    for q, esperado in evaluacion:
        top = idx.buscar(q, 1)[0]
        aciertos += top[0] == esperado
        print(f"{'✔' if top[0]==esperado else '✘'} {q:<70} -> {top[0]:<9} (sim={top[1]:.2f})")
    print(f"\nhit@1 = {aciertos}/{len(evaluacion)}")
    print("\n--- prompt que enviarías al LLM ---")
    q = "¿Qué diferencia hay entre DPO y RLHF clásico?"
    print(construir_prompt(q, idx.buscar(q, 2)))
