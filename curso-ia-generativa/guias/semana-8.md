# Semana 8 · Adaptación, alineamiento, interpretabilidad y producción

> **4 – 10 ene 2027** · ~14 h · Módulo VII de la hoja de ruta (Paso 5) + **cierre del proyecto final**
> **Entregable:** tu proyecto integrador terminado, documentado y presentado (ver [`proyecto/proyecto-final.md`](../proyecto/proyecto-final.md)), usando al menos una técnica de esta semana (LoRA, DPO, RAG o interpretabilidad).

## 1. Objetivos

- [ ] Explicar el *pipeline* completo: **preentrenamiento → SFT → alineamiento → despliegue**.
- [ ] Implementar **LoRA** desde cero y comprobar que (a) congela el modelo base, (b) se puede **fusionar** sin pérdida.
- [ ] Derivar **DPO** a partir del objetivo de RLHF, e implementarlo; reconocer **cuándo falla**.
- [ ] Construir un sistema **RAG** y medir la calidad de la recuperación.
- [ ] Describir un **flujo agéntico** (llamada a herramientas) y sus riesgos.
- [ ] Usar al menos una técnica de **interpretabilidad** para “abrir la caja negra”.

## 2. Plan día a día (≈14 h)

| Día | Tiempo | Qué hacer |
|---|---|---|
| **Lun** | 2 h | **SFT y LoRA/QLoRA:** Raschka *Build an LLM from Scratch* (caps. de *fine-tuning*) y/o *Hugging Face LLM Course*. Lee el paper LoRA (§1–4). |
| **Mar** | 2 h | `lora.py`: lee `LoRALinear`; reprográmalo sin mirar; verifica congelación y fusión. |
| **Mié** | 2,5 h | **Alineamiento:** deriva DPO (§3.3). Ejecuta `dpo.py` (DPO puro vs DPO + SFT) y explica **por qué** falla el puro. |
| **Jue** | 2,5 h | **RAG + agentes:** `rag_minimo.py`; construye un RAG sobre **tus propios documentos** y mide hit@1/hit@3. Esboza un bucle de *tool calling* (§3.5). |
| **Vie** | 2 h | **Interpretabilidad:** *Transformer Circuits Thread* (“A Mathematical Framework…”), BertViz sobre un modelo de HF, *logit lens* sobre tu GPT. |
| **Sáb** | 4 h | **Proyecto final:** integración, evaluación, README y demo. |
| **Dom** | 2 h | **Presentación final** (usa el material de [`presentacion/`](../presentacion)), autoevaluación del curso, plan de los próximos 3 meses. |

## 3. Teoría esencial

### 3.1 El *pipeline* moderno de un LLM

```
Preentrenamiento      →   SFT (ajuste supervisado)   →   Alineamiento por preferencias   →   Despliegue
predecir siguiente        pares instrucción→respuesta     RLHF (PPO) / DPO / variantes        cuantización, KV-cache,
token en billones         (solo se calcula pérdida         pares (preferida, rechazada)       batching, RAG, herramientas,
de tokens                  sobre los tokens de respuesta)                                      evaluación y monitoreo
```

**SFT:** misma pérdida de entropía cruzada, pero con datos de instrucciones formateados con una *plantilla de chat* y **enmascarando** (ignorando) los tokens del *prompt* para entrenar solo la respuesta (`labels = -100` en el prompt).

### 3.2 LoRA y QLoRA

Ajustar todos los pesos es caro (memoria de gradientes y estados de Adam). **LoRA** congela $W\in\mathbb{R}^{d_{out}\times d_{in}}$ y aprende una actualización de **rango bajo**:

$$W'=W+\frac{\alpha}{r}BA,\qquad A\in\mathbb{R}^{r\times d_{in}},\ B\in\mathbb{R}^{d_{out}\times r},\ r\ll d$$

- $B$ se inicializa en 0 (el modelo parte idéntico al base) y $A$ aleatorio.
- Parámetros entrenables por matriz: $r(d_{in}+d_{out})$ en vez de $d_{in}d_{out}$. Para $d=4096$, $r=16$: **0,8 %**.
- **Fusión:** $W\leftarrow W+\frac\alpha rBA$ → cero latencia extra; también puedes guardar **adaptadores** pequeños y cambiarlos por tarea.
- Hiperparámetros: rango $r$ (4–64), $\alpha$ (a menudo $2r$), qué módulos adaptar (`q,k,v,o` y/o MLP), tasa de aprendizaje (≈ 1e-4–3e-4, mayor que en *full fine-tuning*).
- **QLoRA:** el modelo base se carga **cuantizado a 4 bits** (NF4, doble cuantización) y se entrenan adaptadores LoRA en bf16 sobre él → permite afinar modelos grandes en una sola GPU.
- Idea de fondo (conecta con la SVD de la Semana 1): el cambio de pesos al ajustar una tarea tiene **bajo rango intrínseco**.

### 3.3 RLHF y DPO

**RLHF clásico:** (1) SFT; (2) entrenar un **modelo de recompensa** $r_\phi(x,y)$ con pares $(y_w\succ y_l)$ bajo el modelo de Bradley–Terry: $P(y_w\succ y_l)=\sigma\big(r(x,y_w)-r(x,y_l)\big)$; (3) optimizar la política con RL (PPO) maximizando $\mathbb{E}[r]-\beta\,D_{\mathrm{KL}}(\pi_\theta\|\pi_{ref})$.

**DPO (Rafailov et al., 2023).** La solución óptima del paso (3) es $\pi^*(y|x)\propto\pi_{ref}(y|x)\exp\big(r(x,y)/\beta\big)$. Despejando, la recompensa es $r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)}+\beta\log Z(x)$. Sustituyéndola en Bradley–Terry, **$Z$ se cancela** y queda una pérdida directa sobre la política:

$$\mathcal{L}_{\text{DPO}}=-\,\mathbb{E}\Big[\log\sigma\Big(\beta\Big[\log\tfrac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-\log\tfrac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\Big]\Big)\Big]$$

Sin modelo de recompensa explícito, sin RL, con una copia **congelada** $\pi_{ref}$. Variantes y vecinos: IPO, KTO, ORPO, SimPO; y RL con **recompensas verificables** (GRPO, razonamiento matemático/código) que hoy es muy usado.

**Cuándo falla DPO (lo medimos en `dpo.py`):** la pérdida solo mira la **diferencia** entre preferida y rechazada, así que puede reducirse bajando la probabilidad de **ambas** (*likelihood displacement*). El texto se “sale de la distribución”, aunque el margen suba. Remedios: SFT previo sobre las preferidas, término de verosimilitud sobre la preferida, $\beta$ mayor, menos pasos/lr menor, y **vigilar siempre una métrica de calidad** además de la de preferencia.

### 3.4 RAG: Generación Aumentada por Recuperación

```
documentos → trozos (chunks) → embeddings → índice vectorial
pregunta → embedding → top-k por coseno → [reranking] → prompt (contexto + pregunta) → LLM → respuesta con citas
```

Decisiones de diseño: **tamaño y solapamiento de chunks** (200–500 tokens típicos), modelo de embeddings, búsqueda **híbrida** (BM25 + vectorial), *reranker*, número $k$, plantilla de prompt (“responde solo con el contexto; si no está, di que no lo sabes”), citas. **Evalúa por separado** recuperación (hit@k, MRR) y generación (fidelidad al contexto, “alucinación”).

Fallos típicos: la respuesta no está en los *chunks* (problema de recuperación); está pero el modelo la ignora (problema de generación/prompt); los *chunks* contradicen o son obsoletos; **inyección de instrucciones** desde documentos recuperados.

### 3.5 Agentes y *tool calling*

Un LLM con **herramientas** funciona en bucle: el modelo propone una llamada (nombre + argumentos JSON) → **tu código** la ejecuta → devuelves el resultado al modelo → repite hasta que responde. Esqueleto con el SDK de Anthropic (**no se ha ejecutado en este entorno**: necesita una clave de API; revísalo contra la documentación vigente):

```python
import anthropic
client = anthropic.Anthropic()                       # lee ANTHROPIC_API_KEY del entorno
tools = [{"name": "buscar_docs",
          "description": "Busca pasajes relevantes en la base de conocimiento.",
          "input_schema": {"type": "object",
                           "properties": {"consulta": {"type": "string"}},
                           "required": ["consulta"]}}]
mensajes = [{"role": "user", "content": "¿Qué diferencia hay entre DPO y RLHF?"}]
while True:
    r = client.messages.create(model="claude-sonnet-5-5", max_tokens=1024, tools=tools, messages=mensajes)
    if r.stop_reason != "tool_use":
        break
    mensajes.append({"role": "assistant", "content": r.content})
    resultados = [{"type": "tool_result", "tool_use_id": b.id,
                   "content": str(idx.buscar(b.input["consulta"], 2))}     # idx = tu Indice de rag_minimo.py
                  for b in r.content if b.type == "tool_use"]
    mensajes.append({"role": "user", "content": resultados})
print(r.content[0].text)
```

**Riesgos y buenas prácticas:** valida y limita los argumentos (nunca ejecutes texto del modelo como código/SQL sin sandbox); permisos mínimos; confirmación humana para acciones irreversibles; registra todo; tope de iteraciones y de coste; trata el contenido recuperado/herramientas como **no confiable** (inyección de instrucciones).

### 3.6 Interpretabilidad: abrir la caja negra

| Técnica | Qué muestra | Herramienta |
|---|---|---|
| Mapas de atención | a qué tokens “mira” cada cabeza | **BertViz**, los `pesos` de `atencion.py` |
| Atribución de entrada | qué píxeles/tokens influyen en la salida | **Captum** (Integrated Gradients, Grad-CAM) |
| *Logit lens* | proyectar el estado oculto de cada capa al vocabulario: ¿cuándo “decide” el modelo? | unas pocas líneas sobre tu GPT |
| *Probing* | ¿una capa codifica linealmente una propiedad (p. ej. número, tiempo verbal)? | regresión logística sobre activaciones |
| **Interpretabilidad mecanística** | circuitos concretos (*induction heads*), superposición de características | *Transformer Circuits Thread* |
| **Sparse Autoencoders (SAE)** | descomponer activaciones en características dispersas e interpretables: $\min\ \lVert x-D\,f(x)\rVert^2+\lambda\lVert f(x)\rVert_1$ con $f(x)=\text{ReLU}(Ex+b)$ | Anthropic, OpenAI, Neuronpedia |

Cuidado: un mapa de atención **no** es una explicación causal; contrasta con intervenciones (ablación/parcheo de activaciones).

### 3.7 Producción

| Tema | Idea clave |
|---|---|
| **Cuantización** | pesos en int8/int4 (GPTQ, AWQ, GGUF) → 2–4× menos memoria; mide la pérdida de calidad |
| **Inferencia** | KV-cache, *batching* continuo, FlashAttention, decodificación especulativa; motores como vLLM o llama.cpp |
| **Exportación** | `safetensors` (pesos), ONNX (grafo; Netron para inspeccionar) |
| **Evaluación continua** | conjuntos de prueba propios, *LLM-as-judge* (con sus sesgos), pruebas de regresión por cada cambio |
| **Seguridad y ética** | sesgos, privacidad, licencias de datos/modelos, uso responsable, *red teaming* |
| **Costos** | tokens de entrada/salida, latencia, caché de *prompts*, modelos pequeños para tareas fáciles |

## 4. Código de la semana

Carpeta: [`codigo/semana8_adaptacion/`](../codigo/semana8_adaptacion)

```bash
cd curso-ia-generativa/codigo/semana8_adaptacion
python lora.py          # preentrena un modelo base (1ª vez ≈ 1–2 min, luego en caché) y lo adapta con LoRA
python dpo.py           # DPO puro vs DPO + término SFT
python rag_minimo.py    # RAG sin dependencias
```

| Archivo | Contenido |
|---|---|
| `base_modelo.py` | preentrena/carga un GPT base con corpus mixto (90 % formal / 10 % informal) y mide **% informales** y **% gramaticales** |
| `lora.py` | `LoRALinear`, `aplicar_lora`, `fusionar_todo` y el experimento |
| `dpo.py` | `logprob_secuencia`, `perdida_dpo` (con `alpha_sft`), `entrenar_dpo` |
| `rag_minimo.py` | normalización, “embedding” por *hashing*, `Indice.buscar`, `construir_prompt`, evaluación hit@1 |

**Resultados de referencia** (varían unos puntos entre ejecuciones porque el muestreo no tiene semilla fija)

*Modelo base:* ≈ 4–7 % de frases informales y ≈ 90 % gramaticales (la gramática exacta del corpus).

*LoRA* ($r=8$, $\alpha=16$ sobre q, k, v, o y MLP; 300 pasos):

| | Valor |
|---|---|
| Parámetros del base / entrenables con LoRA | 335 136 / 43 776 (**13,1 %**; en un LLM real con $d\approx4096$ es < 1 %) |
| Cambio en los pesos congelados | **0,0** ✔ |
| Frases informales tras LoRA | **100 %** (gramaticales ≈ 97 %) |
| Fusión $W+\tfrac\alpha rBA$: diferencia de *logits* | ≈ $5\times10^{-6}$ ✔, mismo nº de parámetros que el base |

*DPO* (preferencia: informal ≻ formal; lr 3e-5, 200 pasos):

| Variante | Informales | Gramaticales |
|---|---|---|
| Modelo base | 4,0 % | 90,3 % |
| **DPO puro** | 15,7 % | **31,7 %** ← degradación |
| **DPO + término SFT** ($\alpha=1$) | **95,7 %** | 87,0 % |

El DPO puro “gana” el juego del margen (la probabilidad de preferida > rechazada es 100 % y el margen sube a +7), pero **destruye la gramática** y casi no cambia el estilo; añadir SFT sobre la preferida lo arregla. (Además, durante el entrenamiento con SFT se observó un bache transitorio en el paso 150 —informales 8 %, gramaticales 5,5 %— que se recuperó: la dinámica de DPO **es inestable**, vigila las curvas.) Compara con LoRA, que logró el cambio de estilo sin degradar la gramática: son herramientas distintas (LoRA = *cómo* se entrena; DPO = *con qué señal*).

*RAG:* `hit@1 = 10/10` sobre 10 preguntas y 10 pasajes. **Cuidado al interpretarlo:** con la tokenización inicial (sin *stopwords* ni raíces) el resultado fue 7/10 y ajusté el preprocesamiento viendo las 3 fallas; así que es una **demostración de la mecánica**, no una evaluación independiente. Los fallos originales (*“tokenización”* vs *“tokens”*, palabras vacías que dominan) ilustran por qué en producción se usan **embeddings neuronales** y un conjunto de evaluación separado.

## 5. Ejercicios

**Básicos**
1. Cambia $r\in\{1,2,4,8,16\}$ en `lora.py` y grafica **parámetros entrenables vs % informales**. ¿Cuál es el $r$ mínimo que funciona?
2. Aplica LoRA **solo** a `q_proj` y `v_proj`. ¿Qué pierdes?
3. Añade 10 pasajes propios a `DOCS` en `rag_minimo.py` y escribe 10 preguntas de evaluación **antes** de ajustar nada.

**Intermedios**
4. Sustituye el “embedding” por **hashing** por uno neuronal (`sentence-transformers`) y compara hit@1/hit@3 en un conjunto de preguntas parafraseadas.
5. En `dpo.py` barre $\beta\in\{0.05,0.1,0.5\}$ y $\alpha_{sft}\in\{0,0.1,1\}$; reporta la **frontera** informal-vs-gramatical.
6. Implementa *logit lens* sobre tu GPT de la Semana 5: proyecta el estado tras cada bloque con `norm_f` + `lm_head` y observa en qué capa aparece la predicción correcta.

**Reto**
7. Usa **`peft`** (Hugging Face) para aplicar QLoRA a un modelo abierto pequeño sobre un conjunto de instrucciones de tu dominio; compara con tu `LoRALinear`.
8. Entrena un **Sparse Autoencoder** sobre las activaciones de una capa de tu GPT y busca características que se activen con “sujeto”, “verbo” o “adverbio”.

## 6. Errores comunes

| Síntoma | Causa |
|---|---|
| LoRA no cambia nada | $B$ inicializado en 0 y lr demasiado baja, o los parámetros del adaptador no están en el optimizador |
| Los pesos base cambian | olvidaste `requires_grad_(False)` o pasaste todos los parámetros al optimizador |
| Tras fusionar, el modelo difiere | fusión con escala equivocada ($\alpha/r$) o con dtype distinto |
| DPO mejora la “preferencia” pero el texto empeora | *likelihood displacement*: añade SFT/regularización y monitorea calidad |
| RAG “alucina” aunque recupera bien | prompt sin instrucción de ceñirse al contexto; $k$ demasiado alto trae ruido |
| Agente en bucle infinito / gasta de más | falta tope de iteraciones y de coste |

## 7. Autoevaluación

1. ¿Cuántos parámetros entrena LoRA en una matriz de $4096\times4096$ con $r=16$? ¿Qué % es?
2. ¿Por qué $B=0$ al inicio?
3. ¿Qué se cancela en la derivación de DPO y por qué importa?
4. ¿Por qué DPO puede degradar la calidad del texto?
5. ¿Cómo evalúas por separado un sistema RAG?
6. Da dos razones por las que un mapa de atención no prueba causalidad.

<details><summary>Respuestas</summary>

1. $16(4096+4096)=131\,072$ frente a $16\,777\,216$ → **0,78 %**.
2. Para que $W'=W$ al empezar: el ajuste parte exactamente del modelo base y el entrenamiento es estable.
3. La función de partición $Z(x)$: permite expresar la pérdida solo con razones de probabilidades de la política y la referencia, sin modelo de recompensa ni RL.
4. Solo optimiza la diferencia relativa; puede bajar la probabilidad de las dos respuestas y mover masa a texto fuera de distribución.
5. Recuperación (hit@k, MRR sobre preguntas con pasaje conocido) y generación (fidelidad al contexto/alucinación), por separado.
6. La atención es solo una de las vías de información (residuales, MLP); y los valores $V$ pueden tener norma pequeña, así que un peso alto no implica gran efecto. Hay que intervenir (ablación) para afirmar causalidad.
</details>

## 8. Checklist de salida (fin del curso)

- [ ] Mi proyecto final corre de punta a punta y tiene README y evaluación **honesta** (con semillas y limitaciones).
- [ ] Implementé LoRA y comprobé congelación y fusión.
- [ ] Puedo derivar DPO y explicar cuándo falla.
- [ ] Construí un RAG con **mis** documentos y medí recuperación.
- [ ] Usé al menos una herramienta de interpretabilidad/mapeo y la documenté.
- [ ] Presenté el proyecto (aunque sea a una sola persona) en 10 minutos.

## 9. Recursos de la semana

- Raschka, *Build a Large Language Model (From Scratch)*: *fine-tuning* y alineamiento · *Ahead of AI*.
- Hugging Face: *LLM Course*, documentación de `peft`, `trl` (DPO/SFT) y `transformers`.
- DeepLearning.AI *Short Courses*: *Finetuning LLMs*, RAG, evaluación de sistemas.
- Anthropic **Transformer Circuits Thread** · Distill.pub · BertViz · Captum.
- Papers: Hu et al. (2021) *LoRA* (arXiv 2106.09685) · Dettmers et al. (2023) *QLoRA* (arXiv 2305.14314) · Ouyang et al. (2022) *InstructGPT* (arXiv 2203.02155) · Rafailov et al. (2023) *DPO* (arXiv 2305.18290) · Lewis et al. (2020) *RAG* (arXiv 2005.11401) · Elhage et al. (2021) *A Mathematical Framework for Transformer Circuits*.

[← Semana 7](semana-7.md) · **Final:** [Proyecto final](../proyecto/proyecto-final.md) · [Volver al inicio](../README.md)
