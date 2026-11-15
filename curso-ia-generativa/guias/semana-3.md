# Semana 3 · Ingeniería de entrenamiento y mapeo de redes

> **30 nov – 6 dic 2026** · ~12 h · Módulo III de la hoja de ruta
> **Entregable:** un modelo de lenguaje a nivel de carácter (MLP) que alcanza la **cota teórica de entropía** de un corpus, más un “informe de diagnóstico” con las herramientas de mapeo de la sección 3.

## 1. Objetivos

- [ ] Diagnosticar un entrenamiento: pérdida inicial, sobreajuste de un batch, *update-to-data ratio*, neuronas muertas, estadísticas de activaciones.
- [ ] Implementar y comparar **BatchNorm, LayerNorm y RMSNorm**; saber cuándo se usa cada una y por qué.
- [ ] Aplicar **regularización** (dropout, weight decay, aumento de datos, parada temprana).
- [ ] Construir un **modelo de lenguaje de caracteres** y evaluarlo contra una cota teórica.
- [ ] Dominar el **mapeo de redes** en sus 4 niveles: tensorial, de grafo, topológico e interno.

## 2. Plan día a día (≈12 h)

| Día | Tiempo | Qué hacer |
|---|---|---|
| **Lun** | 2 h | Karpathy – *makemore, parte 3* (activaciones, gradientes y BatchNorm). Ejecuta `laboratorio_init_activaciones.py` y **predice** cada tabla antes de verla. |
| **Mar** | 1,5 h | Lee *A Recipe for Training Neural Networks* (Karpathy). Ejecuta `diagnostico_entrenamiento.py` y relaciona cada chequeo con un paso de la receta. |
| **Mié** | 1,5 h | `normalizaciones.py`: implementa BatchNorm/LayerNorm/RMSNorm desde cero y verifica contra `torch.nn`. Dibuja **sobre qué ejes** normaliza cada una. |
| **Jue** | 2 h | Karpathy – *makemore, parte 4* (backprop manual “ninja”) — al menos la primera mitad. Mapa de dimensiones del MLP de Bengio. |
| **Vie** | 1,5 h | **Mapeo:** `shape_tracking.py`, `torchinfo.summary`, `torchviz`, abre un modelo exportado a ONNX en **Netron** (sección 3). |
| **Sáb** | 3–4 h | **Proyecto:** `makemore_mlp.py` → cámbialo para (a) usar tu propio texto (nombres, frases de tu área), (b) añadir `RMSNorm`, (c) graficar pérdida vs. cota. |
| **Dom** | 1 h | Autoevaluación + resumen de la semana. |

## 3. Teoría esencial

### 3.1 Normalización: ¿sobre qué se promedia?

Para un tensor `[B, D]` (o `[B, T, C]` en secuencias):

| Capa | Normaliza sobre | Estadísticas en inferencia | Típica en |
|---|---|---|---|
| **BatchNorm** | el **batch** (eje 0) para cada feature | promedios móviles guardados | CNN, MLP |
| **LayerNorm** | las **features** de cada muestra (último eje) | las de la propia muestra | Transformers (GPT-2, BERT) |
| **RMSNorm** | features, **sin restar la media** | las de la propia muestra | LLaMA, Mistral, Qwen… |

$$\text{BN/LN: } \hat x=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}\gamma+\beta\qquad\qquad \text{RMSNorm: } \hat x=\frac{x}{\sqrt{\tfrac1d\sum_i x_i^2+\epsilon}}\,g$$

Por qué LayerNorm/RMSNorm en Transformers: no dependen del tamaño de batch ni de la longitud de la secuencia, funcionan igual en entrenamiento e inferencia y son más baratas (RMSNorm omite la media y $\beta$). **Pre-norm** (normalizar *antes* de cada subcapa y sumar el residual) es más estable que post-norm en redes profundas.

### 3.2 Diagnóstico: la receta de Karpathy convertida en checklist

1. **Mira tus datos** antes de tocar el modelo.
2. **Esqueleto simple y verificable:** una sola semilla, sin aumento de datos.
3. **Pérdida inicial = $\ln K$** (con $K$ clases / tamaño de vocabulario).
4. **Sobreajusta un batch pequeño** hasta pérdida ≈ 0. Si no puedes, hay un bug.
5. **Verifica que el gradiente fluye** a todas las capas (que depende de *este* ejemplo y no de otros: bug clásico de mezclar el eje batch).
6. **Añade complejidad de a una cosa por vez** (más datos → más capacidad → regularización).
7. **Ajusta hiperparámetros** (búsqueda aleatoria > cuadrícula), y **solo al final** exprime el último 1 %.

Métrica útil: el **update-to-data ratio** $\;\dfrac{\eta\cdot\mathrm{std}(g)}{\mathrm{std}(\theta)}\;$; ronda $10^{-3}$ en redes que entrenan bien. Muy por debajo → lr demasiado bajo; muy por encima → inestable.

### 3.3 Regularización

| Técnica | Idea | Detalle |
|---|---|---|
| **Weight decay** | penaliza pesos grandes | con AdamW va desacoplado; no se aplica a sesgos ni normalizaciones |
| **Dropout** | apaga activaciones al azar con prob. $p$ | *inverted dropout*: escala por $1/(1-p)$ en entrenamiento |
| **Aumento de datos** | transformaciones que preservan la etiqueta | giros, recortes, ruido |
| **Parada temprana** | detener cuando sube la pérdida de validación | guarda el mejor checkpoint |
| **Más datos** | lo más potente | casi siempre |

### 3.4 Modelo de lenguaje y entropía: cómo saber que “ya terminaste”

Un modelo de lenguaje minimiza la **entropía cruzada** (NLL) por token. La pérdida **mínima posible** es la entropía de la distribución real de los datos. En el corpus sintético del curso (frases de una gramática pequeña) esa entropía se **calcula exactamente**:

$$H_{\text{frase}}=\ln(6\cdot5\cdot5\cdot4)=\ln 600\approx 6.40\ \text{nats},\qquad H_{\text{carácter}}=\frac{H_{\text{frase}}}{\text{largo medio de la frase}}\approx0.20\ \text{nats/carácter}$$

Un modelo que se acerque a ese valor ha aprendido la gramática completa. Es un **test de corrección** de tu código (algo que no puedes hacer con datos reales, donde la cota se desconoce). La perplejidad es $e^{\text{pérdida}}$.

### 3.5 Mapeo de redes neuronales (4 niveles)

| Nivel | Pregunta que responde | Herramienta | Cuándo usarla |
|---|---|---|---|
| **1. Tensorial** | ¿Qué *shape* tiene cada tensor? | comentarios `# [B,T,C]`, *hooks*, `torchinfo.summary()`, `einops` | siempre, en **cada** línea nueva |
| **2. Grafo** | ¿Cómo fluyen los gradientes? | `torchviz.make_dot(loss)`, `torch.fx.symbolic_trace`, TensorBoard Graph | depurar `backward`, ver ops fusionables |
| **3. Topológico** | ¿Cuál es la arquitectura? ¿Cómo la documento? | **Netron** (abre `.onnx`, `.pt`, `.safetensors`), `torchview`, **PlotNeuralNet**, **NN-SVG** | documentación y *code review* |
| **4. Interno** | ¿Qué “ve” la red? | **BertViz**, **Captum** (atribución, Grad-CAM), *Sparse Autoencoders* | interpretabilidad (Semana 8) |

**Nivel 1 en la práctica**, con *hooks* (ver `shape_tracking.py`):

```python
def registrar_formas(modelo):
    def hook(mod, entrada, salida):
        print(mod.__class__.__name__, tuple(entrada[0].shape), "->", tuple(salida.shape))
    return [m.register_forward_hook(hook) for m in modelo.modules() if not list(m.children())]
```

Salida real para la CNN del ejemplo (`B=2`): `Conv2d (2,1,16,16)→(2,8,16,16)` · `MaxPool2d →(2,8,8,8)` · `Conv2d →(2,16,8,8)` · `MaxPool2d →(2,16,4,4)` · `Flatten →(2,256)` · `Linear →(2,4)`.

**Nivel 2:**

```python
# pip install torchviz   (requiere Graphviz instalado en el sistema)
from torchviz import make_dot
make_dot(loss, params=dict(modelo.named_parameters())).render("grafo", format="png")
```

**Nivel 3 (Netron):** exporta y abre.

```python
torch.onnx.export(modelo, (ejemplo_entrada,), "modelo.onnx")   # requiere: pip install onnx onnxscript
# luego: pip install netron; netron modelo.onnx
```

## 4. Código de la semana

Carpeta: [`codigo/semana3_entrenamiento/`](../codigo/semana3_entrenamiento)

```bash
cd curso-ia-generativa/codigo/semana3_entrenamiento
python laboratorio_init_activaciones.py   # NumPy puro
python normalizaciones.py                 # tus normalizaciones vs torch.nn
python diagnostico_entrenamiento.py       # la receta, convertida en chequeos
python shape_tracking.py                  # hooks + reshape de multi-head attention
python makemore_mlp.py                    # LM de caracteres vs cota de entropía
```

**Resultados de referencia**

*Laboratorio de inicialización* (30 capas × 256 neuronas; std de la activación en la capa 30):

| Inicialización | tanh | ReLU |
|---|---|---|
| $\mathcal{N}(0,1)$ | 0,97 (satura) | $2\times10^{31}$ (explota) |
| $\mathcal{N}(0,0.01)$ | $10^{-24}$ (se desvanece) | $10^{-29}$ |
| Xavier $\sqrt{1/d}$ | 0,14 (se contrae despacio) | $2\times10^{-5}$ |
| He $\sqrt{2/d}$ | 0,55 (estable) | 2,2 (mismo orden) |
| tanh con ganancia 5/3 | 0,65 (estable) | — |

*Normalizaciones:* diferencias con `torch.nn` ≈ $10^{-7}$–$10^{-6}$ ✔ (BatchNorm en train y eval, LayerNorm, RMSNorm).

*Diagnóstico:* pérdida inicial 1,390 (esperada $\ln 4=1,386$); sobreajusta 16 ejemplos a 0,00000; en una red de 10 capas con SGD lr 0,1, **sin norm la pérdida se queda en 1,3863 (no aprende nada)** mientras que **con LayerNorm baja a 0,646** en los mismos 300 pasos.

*makemore (corpus de 6000 frases, vocabulario 26, 72 442 parámetros):*

| Modelo | Pérdida de validación (nats/carácter) |
|---|---|
| Uniforme ($\ln 26$) | 3,258 |
| Bigrama por conteo | 1,591 |
| MLP (contexto de 8 caracteres) | **≈ 0,205** |
| **Cota teórica (entropía del corpus)** | **0,202** |

El MLP queda a ~0,003 nats de la cota y genera frases como *“el robot mira el motor hoy.”*, *“la niña busca la red.”*.

> Detalle pedagógico en `makemore_mlp.py`: para **generar**, el contexto inicial debe ser una cola real del corpus; empezar con `"\n"*8` (nunca visto en entrenamiento) produce basura (*distribution shift*). Lo comprobé: así salían `mrctmun puente con calma.`

## 5. Ejercicios

**Básicos**
1. Añade `BatchNorm1d` al MLP de espirales de la Semana 2 y compara velocidad de convergencia.
2. En `laboratorio_init_activaciones.py` agrega **LeakyReLU** y **GELU**; ¿qué ganancia mantiene estable la std?
3. Calcula a mano el *update-to-data ratio* de tu mejor configuración.

**Intermedios**
4. Sustituye el corpus sintético por **tus propios datos** (p. ej. 2 000 nombres de productos, calles o frases de tu carrera) y compara con bigrama/uniforme. Ahora **no hay cota**: ¿cómo sabrás cuándo parar? (Pista: validación + parada temprana.)
5. Convierte `makemore_mlp.py` a **RMSNorm** (de `normalizaciones.py`) y a **residuales**. ¿Mejora?
6. Haz que `diagnostico_entrenamiento.py` grafique (histograma) las activaciones de cada capa y los gradientes por capa.

**Reto**
7. Implementa *backprop manual* de `makemore_mlp.py` (Karpathy parte 4) y compara cada gradiente con `.grad` de PyTorch usando una función `cmp(nombre, mi_grad, tensor)`.
8. Exporta tu modelo a ONNX, ábrelo en Netron y **compara** el diagrama con tu dibujo hecho a mano.

## 6. Errores comunes

| Síntoma | Causa |
|---|---|
| Pérdida desciende pero las muestras son basura | distribution shift en la generación (contexto inicial irreal) |
| BatchNorm da distinto en `train()` y `eval()` | normal: usa estadísticas del batch vs. promedios móviles; revisa momentum |
| Con batch=1 BatchNorm falla | no hay varianza que estimar: usa LayerNorm/GroupNorm |
| Gradientes “mezclan” ejemplos | operar sobre el eje del batch por error (p. ej., `softmax(dim=0)`) |
| Pérdida de val < pérdida de train | dropout activo solo en train: es normal al principio |

## 7. Autoevaluación

1. ¿Sobre qué eje normaliza BatchNorm en `[B,D]`? ¿Y LayerNorm?
2. ¿Por qué un Transformer usa LayerNorm/RMSNorm y no BatchNorm?
3. Si tu pérdida inicial en un LM con vocabulario 65 es 12, ¿qué sospechas?
4. ¿Cómo saber que tu modelo de lenguaje ya no puede mejorar?
5. ¿Por qué no se aplica weight decay a sesgos y normalizaciones?

<details><summary>Respuestas</summary>

1. BatchNorm: eje 0 (batch) por feature; LayerNorm: eje −1 (features) por muestra.
2. BatchNorm depende del batch y de la longitud de secuencia (padding), y se comporta distinto en entrenamiento/inferencia; LayerNorm es por muestra.
3. Debería ser $\ln 65\approx4.17$: salidas iniciales demasiado grandes (inicialización de la última capa, falta de escalado).
4. Cuando se acerca a la entropía de los datos (si se conoce) o cuando la validación se estanca/sube (sobreajuste).
5. No son “complejidad” del modelo; decaerlos no regulariza y puede perjudicar la escala de las capas normalizadas.
</details>

## 8. Checklist de salida

- [ ] Predije correctamente los resultados del laboratorio de inicialización.
- [ ] Mis tres normalizaciones coinciden con `torch.nn` (< 1e-5).
- [ ] Sé sobreajustar un batch y verificar $\ln K$ al inicio.
- [ ] Mi LM de caracteres se acerca a la cota (o a un mínimo de validación estable con datos reales).
- [ ] Generé un diagrama de mi red con al menos dos herramientas de mapeo.

## 9. Recursos de la semana

- Karpathy, *Zero to Hero*: *makemore* partes 3, 4 y (opcional) 5 (WaveNet).
- Karpathy, ensayo *A Recipe for Training Neural Networks* (karpathy.github.io/2019/04/25/recipe).
- Papers: Ioffe & Szegedy (2015) *Batch Normalization* (arXiv 1502.03167) · Ba et al. (2016) *Layer Normalization* (arXiv 1607.06450) · Zhang & Sennrich (2019) *RMSNorm* (arXiv 1910.07467).
- CS231n notes: *Neural Networks Part 2* (inicialización, batch norm) y *Part 3* (diagnóstico de aprendizaje).
- Distill.pub: *Why Momentum Really Works* (intuición visual de optimización).

[← Semana 2](semana-2.md) · **Siguiente:** [Semana 4 →](semana-4.md)
