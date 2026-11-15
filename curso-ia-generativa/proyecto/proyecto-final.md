# Proyecto final integrador

> Se desarrolla de forma incremental entre las semanas 5 y 8 y se presenta el **10 de enero de 2027**. Su objetivo es demostrar que **construyes, mides y explicas** un sistema generativo; no que compres cómputo.

## 1. Reglas del proyecto

1. **Construido por ti:** el núcleo del modelo (modelo, bucle de entrenamiento, evaluación) lo escribes tú. Puedes usar librerías (`torch`, `peft`, `transformers`, `diffusers`) para lo periférico y **debes poder explicar cada línea** del núcleo.
2. **Evaluación honesta:** ≥ 3 semillas, métricas de **calidad y diversidad**, línea base trivial y comparación justa. Reporta también lo que **no** funcionó.
3. **Reproducible:** un repositorio con `README`, `requirements.txt`, semilla fija y un comando que reproduce el resultado principal.
4. **Presentación de 10 minutos** con una demo (ver §5).
5. **Mínimo técnico:** usa al menos **una** técnica de cada bloque: (a) una arquitectura de las semanas 4–7, (b) una técnica de la semana 8 (LoRA, DPO, RAG o interpretabilidad) o una variante propia justificada.

## 2. Cuatro rutas (elige una)

### Ruta A · “Mi propio mini-LLM”
Preentrena un GPT pequeño sobre un corpus **tuyo** (textos de tu dominio, 1–20 MB), con BPE propio, y adáptalo con LoRA a un estilo o tarea.
- **Núcleo:** `gpt.py` extendido (tu tokenizador BPE, KV-cache), evaluación por **perplejidad por byte** frente a bigramas y a un modelo de caracteres.
- **Extensión:** LoRA para dos “estilos” e intercambio de adaptadores; medición de olvido (perplejidad en el dominio original).
- **Riesgo típico:** corpus demasiado pequeño → sobreajuste. Mitigación: validación estricta + parada temprana.

### Ruta B · “Datos sintéticos con VAE/difusión para un proceso”
Genera **series o tablas sintéticas** para un proceso (demanda, tiempos de ciclo, sensores) con un VAE o un modelo de difusión tabular/1D, y valídalas.
- **Núcleo:** modelo generativo propio; **pruebas de utilidad**: ¿un modelo entrenado con datos sintéticos rinde casi igual en datos reales (TSTR: *train on synthetic, test on real*)?
- **Extensión:** detección de anomalías con el error de reconstrucción del VAE; fidelidad estadística (distribuciones, correlaciones), y revisión de **privacidad**.
- **Riesgo:** métricas que no capturan estructura temporal; añade pruebas de autocorrelación.

### Ruta C · “RAG con evaluación seria sobre documentos reales”
Sistema de preguntas y respuestas sobre **tus documentos** (procedimientos, manuales, apuntes, normativa).
- **Núcleo:** pipeline propio de chunking + embeddings + índice + prompt; **conjunto de evaluación de ≥ 50 preguntas** con pasaje correcto etiquetado (hit@k, MRR) **y** evaluación de la respuesta (fidelidad al contexto).
- **Extensión:** búsqueda híbrida (BM25 + vectorial), *reranking*, bucle de herramientas con tope de iteraciones, y análisis de fallos (¿recuperación o generación?).
- **Riesgo:** evaluar con las mismas preguntas con las que ajustaste el sistema (en `rag_minimo.py` ya ocurrió): separa **desarrollo** y **prueba**.

### Ruta D · “Difusión desde cero sobre imágenes propias”
Modelo de difusión sobre un conjunto pequeño de imágenes (formas, dígitos, símbolos de tu dominio).
- **Núcleo:** `MiniUNet` mejorada (atención, EMA), DDPM + DDIM, **classifier-free guidance** condicional.
- **Extensión:** difusión en espacio latente con tu VAE; métricas: precisión/recall de distribuciones y un clasificador “juez” entrenado aparte.
- **Riesgo:** cómputo. Usa Colab con GPU, imágenes de 16–32 px y planifica 1–2 h de entrenamiento.

## 3. Cronograma

| Cuándo | Hito | Entregable |
|---|---|---|
| Fin S5 (20 dic) | Elegir ruta y definir **pregunta, datos, línea base y métricas** | 1 página (`PLAN.md`) |
| Fin S6 (27 dic) | Datos listos + línea base trivial funcionando | métrica de la línea base |
| Fin S7 (3 ene) | **Primer prototipo** del modelo principal entrenando | curvas + primeras muestras |
| S8 (4–9 ene) | Técnica de la S8, evaluación final con semillas, README, demo | repo completo |
| 10 ene | **Presentación** (10 min + 5 min de preguntas) | diapositivas + demo |

## 4. Rúbrica (100 puntos)

| Criterio | Pts | Excelente |
|---|---|---|
| **Corrección técnica** | 25 | Núcleo propio correcto; pasa verificaciones (gradient checks, shapes, comparación con referencia) |
| **Evaluación** | 25 | ≥ 3 semillas, línea base, calidad **y** diversidad, ablación de al menos un componente, análisis de fallos |
| **Comprensión** | 20 | Explica cualquier línea del núcleo y las decisiones (por qué esta pérdida, esta arquitectura) |
| **Reproducibilidad y claridad** | 15 | `README`, entorno, semilla, un comando; código con *shapes* comentados |
| **Mapeo e interpretabilidad** | 10 | ≥ 2 herramientas de mapeo (Netron/torchviz/hooks/BertViz…) y un hallazgo |
| **Comunicación** | 5 | Presentación clara, demo funcional, limitaciones reconocidas |

**Penalizaciones:** resultados sin semillas (−10), conclusiones no sustentadas (−5), copiar código sin entenderlo (−15), no reportar fallos (−5).

## 5. Plantilla de presentación (10 minutos)

1. **Problema y por qué importa** (1 min).
2. **Datos y línea base** (1 min).
3. **Modelo:** diagrama con *shapes* (2 min).
4. **Resultados:** tabla con media ± desviación, curvas, muestras (3 min).
5. **Qué falló y qué aprendí** (1,5 min).
6. **Demo en vivo** o video corto (1,5 min).

## 6. Estructura sugerida del repositorio

```
mi-proyecto/
├── README.md            # problema, resultados (tabla), cómo reproducir
├── PLAN.md              # pregunta, datos, línea base, métricas
├── requirements.txt
├── src/                 # modelo.py, entrenar.py, evaluar.py
├── tests/               # pruebas (gradient checks, shapes, regresión)
├── notebooks/           # exploración
├── resultados/          # tablas y figuras (con la semilla en el nombre)
└── docs/                # diagramas de arquitectura, informe de mapeo
```

## 7. Lista de verificación final

- [ ] Hay una **línea base** y mi modelo la supera (o explico por qué no).
- [ ] Reporto media ± desviación en ≥ 3 semillas.
- [ ] Mido calidad **y** diversidad (o precisión y cobertura).
- [ ] Hay una prueba automática que falla si rompo el núcleo.
- [ ] Documenté limitaciones, datos y licencias, y riesgos de uso.
- [ ] Puedo ejecutar el resultado principal con **un solo comando** en otra máquina.
