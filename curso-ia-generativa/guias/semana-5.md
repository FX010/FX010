# Semana 5 · La revolución Transformer: tokenización, atención y tu propio GPT

> **14 – 20 dic 2026** · ~14 h (semana central: dedícale más tiempo) · Módulo V de la hoja de ruta
> **Entregable:** un GPT *decoder-only* escrito por ti (RMSNorm, RoPE, atención causal, SwiGLU) que alcanza la cota teórica de entropía de un corpus y genera texto, más un tokenizador BPE propio.

## 1. Objetivos

- [ ] Programar un tokenizador **BPE** y explicar por qué se tokeniza por subpalabras.
- [ ] Derivar y programar la **atención escalada** y la máscara causal; justificar el factor $1/\sqrt{d_k}$.
- [ ] Explicar **multi-head**, **GQA**, **codificación posicional** (sinusoidal, aprendida, RoPE).
- [ ] Armar un bloque Transformer *pre-norm* con residuales y un MLP **SwiGLU**.
- [ ] Entrenar un GPT pequeño, controlar el muestreo (temperatura, top-k) y medir **perplejidad**.
- [ ] Mapear dimensiones de punta a punta: `[B,T] → [B,T,C] → … → [B,T,V]`.

## 2. Plan día a día (≈14 h)

| Día | Tiempo | Qué hacer |
|---|---|---|
| **Lun** | 2 h | **3Blue1Brown** caps. *But what is a GPT?* y *Attention in transformers, step-by-step*. Lee *The Illustrated Transformer* (Jay Alammar). Dibuja el diagrama de dimensiones (§3.6) **de memoria**. |
| **Mar** | 2 h | **Tokenización:** Karpathy – *Let's build the GPT Tokenizer* (primera parte) + `bpe.py`. Entrena un BPE sobre un texto tuyo y mide la compresión. |
| **Mié** | 2 h | **Atención:** `atencion.py`. Implementa `atencion_escalada` sin mirar; compárala con `F.scaled_dot_product_attention` (< 1e-6). Reproduce el experimento de $\sqrt{d_k}$. |
| **Jue** | 3 h | Karpathy – *Let's build GPT: from scratch, in code, spelled out* (≈2 h). Sigue escribiendo código tú. |
| **Vie** | 2 h | Lee `gpt.py` línea por línea. Anota el *shape* de cada tensor. Compara con tu versión de ayer. Lee el paper *Attention Is All You Need* (secciones 3.1–3.5). |
| **Sáb** | 4 h | **Proyecto:** entrena tu GPT (`entrenar_gpt.py`) con (a) el corpus sintético y (b) un texto real tuyo (`--archivo`). Prueba `--pos learned`, `--kv 2`, distintas temperaturas. |
| **Dom** | 1 h | Autoevaluación + resumen. Mira los resultados de la semana 6–7 para anticipar. |

## 3. Teoría esencial

### 3.1 Tokenización por subpalabras (BPE)

Trabajar con **caracteres** da secuencias demasiado largas; con **palabras** el vocabulario explota y no maneja palabras nuevas. BPE es el compromiso:

1. Vocabulario inicial = los 256 **bytes** (cualquier texto UTF-8 es representable: sin “desconocidos”).
2. Contar pares adyacentes; **fusionar el más frecuente** en un token nuevo.
3. Repetir hasta el tamaño de vocabulario deseado (GPT-2: 50 257; modelos recientes: 100 k–250 k).

Codificar = aplicar las fusiones en el orden en que se aprendieron. Consecuencias prácticas: los números, el código y los idiomas poco representados gastan más tokens (más caro y a veces peor rendimiento); muchas rarezas de los LLM (contar letras, aritmética) nacen de la tokenización.

### 3.2 Embeddings

Tabla `E ∈ ℝ^{V×C}`; el token $i$ se convierte en la fila $E_i$. Se aprende junto con el resto. Con ***weight tying*** la misma matriz se usa (traspuesta) para producir los logits de salida, ahorrando $V\cdot C$ parámetros.

### 3.3 Atención escalada por producto punto

Cada token genera tres vectores por proyección lineal: **consulta** $q$ (“qué busco”), **clave** $k$ (“qué ofrezco”) y **valor** $v$ (“qué contenido aporto”).

$$\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V,\qquad M_{ij}=\begin{cases}0 & j\le i\\-\infty & j>i\end{cases}$$

- $QK^\top\in\mathbb{R}^{T\times T}$: afinidad de cada par de posiciones. La **máscara causal** $M$ impide mirar al futuro (necesaria para generar texto token a token).
- La fila $i$ del softmax es una **distribución** sobre los tokens que $i$ puede ver; la salida es un **promedio ponderado de valores**.
- **¿Por qué $\sqrt{d_k}$?** Si $q,k$ tienen componentes de varianza 1, $\mathrm{Var}(q\cdot k)=d_k$. Sin escalar, los logits crecen, el softmax satura (casi *one-hot*) y los gradientes se anulan. Lo medimos en `atencion.py`: con $d_k=1024$ la varianza sin escalar es ≈ 1007 y con escalado ≈ 1.
- Coste: $O(T^2\cdot d)$ en tiempo y $O(T^2)$ en memoria de atención → el límite de contexto largo (FlashAttention reduce la memoria sin cambiar el resultado).

### 3.4 Multi-head, MQA y GQA

- **MHA:** $H$ cabezas, cada una con su $Q,K,V$ de dimensión $d_h=C/H$; se concatenan y se proyectan con $W_O$. Cada cabeza puede especializarse (sintaxis, posición, copia…).
- **MQA / GQA:** muchas cabezas de $Q$ comparten pocas de $K,V$ (1 en MQA; $G$ grupos en GQA). Reduce el **KV-cache** (la memoria que crece con el contexto durante la inferencia) casi sin perder calidad. `gpt.py` lo implementa con `n_kv_head`.

### 3.5 Posiciones

La atención por sí sola es **invariante al orden** → hay que inyectar la posición:

| Método | Idea | Notas |
|---|---|---|
| Sinusoidal (2017) | sumar senos/cosenos de frecuencias geométricas | sin parámetros |
| Aprendida (GPT-2) | tabla `[block_size, C]` | no extrapola más allá de `block_size` |
| **RoPE** (2021) | **rotar** cada par de dimensiones de $q$ y $k$ en un ángulo $\propto$ posición | $q_i\cdot k_j$ depende solo de $i-j$; estándar en LLaMA y otros |

Verificado en `atencion.py`: el producto $q\cdot k$ da exactamente lo mismo para (pos 3, pos 1) que para (pos 20, pos 18).

### 3.6 El bloque Transformer decoder-only y su mapa de dimensiones

```
idx            [B,T]                      enteros (ids de tokens)
 │ Embedding(V,C)
x              [B,T,C]
 │  ┌─────────────── repetir N veces ───────────────┐
 │  │ h = RMSNorm(x)                [B,T,C]          │
 │  │ q,k,v = proyecciones           [B,H,T,Dh]      │   (k,v: [B,Hkv,T,Dh] con GQA)
 │  │ RoPE(q,k) ; att = softmax(qkᵀ/√Dh + M) v       │   scores [B,H,T,T]  → [B,H,T,Dh]
 │  │ x = x + Wo · concat(cabezas)  [B,T,C]          │   ← residual 1
 │  │ h = RMSNorm(x)                [B,T,C]          │
 │  │ x = x + W_down( SiLU(W_gate h) ⊙ (W_up h) )    │   ← residual 2 (SwiGLU, hidden≈8C/3)
 │  └────────────────────────────────────────────────┘
 │ RMSNorm
 │ lm_head = Embedding.T  (weight tying)
logits         [B,T,V]                    → cross_entropy con idx desplazado una posición
```

**Entrenamiento autorregresivo:** la entrada es `idx[:, :-1]` y el objetivo `idx[:, 1:]`: en **una pasada** se entrena a predecir el siguiente token en las $T$ posiciones a la vez (la máscara causal lo hace legítimo).

### 3.7 Generación y muestreo

Se toma `logits[:, -1]`, se divide por la **temperatura** $\tau$ ($\tau\to0$ determinista; $\tau>1$ más aleatorio), opcionalmente se recorta a los **top-k** (o top-p/núcleo) y se muestrea con `multinomial`. Se agrega el token y se repite. En producción se guarda el **KV-cache** para no recomputar el pasado.

### 3.8 Escalado, a vista de pájaro

Parámetros del Transformer ≈ $12\,N_{capas}\,C^2$ (sin embeddings). La pérdida decrece como ley de potencias en parámetros, datos y cómputo (*scaling laws*, Kaplan 2020; Chinchilla 2022 mostró que conviene ≈ 20 tokens de entrenamiento por parámetro para un cómputo dado). Tu modelo de 335 k parámetros hace lo mismo que uno de 100 B, solo que más pequeño.

## 4. Código de la semana

Carpeta: [`codigo/semana5_transformer/`](../codigo/semana5_transformer)

```bash
cd curso-ia-generativa/codigo/semana5_transformer
python bpe.py                                          # tokenizador desde cero
python atencion.py                                     # atención, RoPE, √d_k: con verificaciones
python entrenar_gpt.py                                 # GPT (RoPE, MHA)  ≈ 1–3 min en CPU
python entrenar_gpt.py --pos learned --kv 2            # variante: posiciones aprendidas + GQA
python entrenar_gpt.py --archivo mi_texto.txt          # ¡con tu propio texto!
```

| Archivo | Contenido |
|---|---|
| `bpe.py` | `contar_pares`, `fusionar`, `BPE.entrenar/codificar/decodificar` (a nivel de bytes) |
| `atencion.py` | atención escalada, `MultiHead`, RoPE y tres verificaciones |
| `gpt.py` | `GPT`: RMSNorm, RoPE, GQA, SwiGLU, *weight tying*, `generate(temperature, top_k)` |
| `entrenar_gpt.py` | bucle con AdamW (β₂=0,95), *warmup* + coseno, recorte de gradiente, evaluación, muestreo |

**Resultados de referencia**

- **BPE:** 40 fusiones sobre el corpus; texto de 55 bytes (con `ñ` y `✓`) → 35 tokens; la **codificación es reversible** (assert pasa).
- **Atención:** diferencia con `F.scaled_dot_product_attention` ≈ $9\times10^{-8}$; filas de pesos suman 1; el triángulo superior es 0 ✔. RoPE: $q\cdot k$ idéntico (2,73378) a la misma distancia ✔.
- **GPT** (3 capas, $C=96$, 4 cabezas, **334 944 parámetros**, 800 pasos, batch 32×64):

| | Valor |
|---|---|
| Pérdida inicial | 3,36 (≈ $\ln 26=3,26$ ✔) |
| Pérdida de validación final | **0,232** nats/carácter |
| **Cota teórica** (entropía del corpus) | **0,203** |
| Muestras | *“el robot come pescado rápido.”*, *“la niña analiza los datos.”* · de 500 frases generadas: **100 % gramaticales a T=0,8** y **96,4 % a T=1,0** (la temperatura alta introduce errores) |

Se alcanza ~0,03 nats de la cota con 800 pasos; con más pasos se acerca más (ejercicio 4). Si tu GPT **no** baja de ~1,0 o se queda en $\ln V$, hay un bug: ese es el valor de tener una cota exacta.

## 5. Ejercicios

**Básicos**
1. Entrena un BPE sobre un texto tuyo con vocabulario 300, 500, 1000. Grafica **bytes/token** vs tamaño de vocabulario.
2. Calcula a mano $QK^\top/\sqrt{d_k}$ y el softmax para $T=3,d_k=2$ con un ejemplo pequeño; aplica la máscara causal.
3. Cuenta los parámetros de `gpt.py` con `n_params()` y **deriva la fórmula** para confirmar el número. (Pista: con $V=26$, $C=96$, 3 capas y *hidden* de SwiGLU = 256: $334\,944 = 26\cdot96 + 3\,(4\cdot96^2+3\cdot96\cdot256+2\cdot96)+96$.)

**Intermedios**
4. Entrena 3 000 pasos y grafica pérdida vs cota. ¿A qué paso cruza 0,21?
5. Compara `--pos rope` vs `learned` y `--kv` ∈ {4,2,1} (MHA, GQA, MQA) en **pérdida final y velocidad**. Usa ≥ 3 semillas.
6. Implementa **KV-cache** en `generate` y mide la aceleración de generar 500 tokens.

**Reto**
7. Reemplaza el tokenizador de caracteres por tu BPE y entrena sobre un texto real (p. ej. *Tiny Shakespeare* o textos en español de dominio público). Reporta pérdida **por byte** para poder comparar con el modelo de caracteres.
8. Visualiza los mapas de atención por cabeza (`pesos` de `MultiHead`; o BertViz con un modelo de HuggingFace) e identifica una cabeza con patrón claro (p. ej. “mirar el token anterior”).

## 6. Errores comunes

| Síntoma | Causa |
|---|---|
| El modelo “hace trampa” (pérdida ≈ 0) | máscara causal ausente o desplazamiento mal hecho: ve el token que debe predecir |
| Pérdida inicial ≫ $\ln V$ | embeddings/cabeza con std grande; usa `N(0, 0.02)` y escala proyecciones residuales |
| `view` falla tras `transpose` | falta `.contiguous()` |
| Salidas repetitivas | temperatura demasiado baja o sin top-k/top-p |
| Memoria explota con contexto largo | atención $O(T^2)$: reduce `block_size` o usa SDPA/FlashAttention |
| Softmax con `nan` | fila totalmente enmascarada (todo $-\infty$) |

## 7. Autoevaluación

1. ¿Qué *shape* tienen `scores` y `out` en la atención multi-cabeza (`B,H,T,Dh`)?
2. ¿Por qué $\sqrt{d_k}$ y no $d_k$?
3. ¿Qué hace la máscara causal y por qué es necesaria en entrenamiento?
4. ¿Qué ventaja tiene RoPE sobre posiciones aprendidas?
5. Con $C=4096$, 32 capas: ¿cuántos parámetros hay aprox. (sin embeddings)?
6. ¿Qué es el KV-cache y qué problema resuelve GQA?

<details><summary>Respuestas</summary>

1. `scores` `[B,H,T,T]`; `out` `[B,H,T,Dh]` (luego `[B,T,C]`).
2. Para que la varianza de $q\cdot k$ sea ≈ 1: $\mathrm{Var}(q\cdot k)=d_k$, así que dividir por $\sqrt{d_k}$ la normaliza.
3. Pone $-\infty$ en el futuro para que la probabilidad sea 0. Sin ella, al entrenar en paralelo, la posición $i$ vería el token $i+1$ que debe predecir.
4. Codifica distancia **relativa** y extrapola/extiende mejor; no consume parámetros.
5. $\approx12\cdot32\cdot4096^2\approx6.4\times10^9$.
6. Guarda $K,V$ de tokens previos para no recalcularlos al generar. GQA comparte $K,V$ entre cabezas de $Q$ → el cache es $H/G$ veces menor.
</details>

## 8. Checklist de salida

- [ ] Dibujé de memoria el mapa de dimensiones del GPT.
- [ ] Mi atención coincide con `F.scaled_dot_product_attention` (< 1e-6).
- [ ] Mi GPT alcanza una pérdida cercana a la cota (o val estable en datos reales).
- [ ] Puedo explicar $\sqrt{d_k}$, máscara causal y RoPE sin notas.
- [ ] Generé texto con distintas temperaturas y describí el efecto.

## 9. Recursos de la semana

- 3Blue1Brown: capítulos de GPT, atención y MLPs en Transformers.
- Karpathy, *Zero to Hero*: *Let's build GPT* y *Let's build the GPT Tokenizer*; (extra) *Let's reproduce GPT-2 (124M)*.
- Jay Alammar: *The Illustrated Transformer* y *The Illustrated GPT-2*.
- Lilian Weng (Lil'Log): *Attention? Attention!* y *The Transformer Family*.
- Raschka, *Build a Large Language Model (From Scratch)* (caps. 2–5) · Stanford **CS336** (primeras clases) y **CS224n**.
- Papers: Vaswani et al. (2017) *Attention Is All You Need* (arXiv 1706.03762) · Su et al. (2021) *RoFormer / RoPE* (arXiv 2104.09864) · Ainslie et al. (2023) *GQA* (arXiv 2305.13245) · Sennrich et al. (2015) *BPE* (arXiv 1508.07909) · Kaplan et al. (2020) *Scaling Laws* (arXiv 2001.08361) · Hoffmann et al. (2022) *Chinchilla* (arXiv 2203.15556).

[← Semana 4](semana-4.md) · **Siguiente:** [Semana 6 →](semana-6.md)
