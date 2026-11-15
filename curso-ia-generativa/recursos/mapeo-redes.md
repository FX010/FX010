# Cómo mapear redes neuronales: guía práctica de los 4 niveles

“Mapear” una red significa cosas distintas según la pregunta. Esta guía te da, para cada nivel, **cuándo usarlo, el código mínimo y qué buscar**. Se practica sobre todo en las semanas 3 y 8.

| Nivel | Pregunta | Herramientas |
|---|---|---|
| 1. **Tensorial** | ¿Qué *shape* tiene cada tensor? | comentarios, *hooks*, `torchinfo`, `einops` |
| 2. **Grafo computacional** | ¿Cómo fluyen datos y gradientes? | `torchviz`, `torch.fx`, TensorBoard |
| 3. **Topológico / diagramas** | ¿Cómo documento la arquitectura? | Netron, `torchview`, PlotNeuralNet, NN-SVG |
| 4. **Interno / interpretabilidad** | ¿Qué hace cada capa/cabeza/neurona? | BertViz, Captum, *logit lens*, SAEs |

## Nivel 1 · Mapeo tensorial (*shape tracking*)

**Regla:** cada línea que crea un tensor lleva su forma en un comentario.

```python
x = self.tok(idx)                                   # [B,T] -> [B,T,C]
q = self.q_proj(x).view(B,T,H,Dh).transpose(1,2)    # [B,T,C] -> [B,H,T,Dh]
att = q @ k.transpose(-2,-1) / Dh**0.5              # [B,H,T,Dh]@[B,H,Dh,T] -> [B,H,T,T]
y = (att.softmax(-1) @ v).transpose(1,2).reshape(B,T,C)   # -> [B,T,C]
```

**Hooks** (en `codigo/semana3_entrenamiento/shape_tracking.py`):

```python
def registrar_formas(modelo):
    def hook(mod, entrada, salida):
        print(mod.__class__.__name__, tuple(entrada[0].shape), "->", tuple(salida.shape))
    return [m.register_forward_hook(hook) for m in modelo.modules() if not list(m.children())]
```

**`torchinfo`** (resumen tipo Keras): `pip install torchinfo` →

```python
from torchinfo import summary
summary(modelo, input_size=(8, 1, 16, 16))        # capas, shapes de salida, nº de parámetros
```

**`einops`**: operaciones con nombres de ejes, autodocumentadas.

```python
from einops import rearrange
q = rearrange(q, "b t (h d) -> b h t d", h=H)
y = rearrange(y, "b h t d -> b t (h d)")
```

**Qué buscar:** que la última dimensión de cada `@` coincida; que el *batch* nunca se mezcle; el recuento de parámetros por capa (¿dónde está el 80 %? en el Transformer: MLP y atención ≈ $12C^2$ por capa).

## Nivel 2 · Grafo computacional

```python
# pip install torchviz   + Graphviz en el sistema (apt install graphviz / brew install graphviz)
from torchviz import make_dot
loss = F.cross_entropy(modelo(x), y)
make_dot(loss, params=dict(modelo.named_parameters())).render("grafo_autograd", format="png")
```

Cada nodo `…Backward` es una operación del *backward pass*; las ramas que se unen son los lugares donde los gradientes se **suman** (Semana 1). Con `torch.fx`:

```python
import torch.fx
gm = torch.fx.symbolic_trace(modelo)
print(gm.graph)            # lista de nodos: placeholder, call_module, call_function, output
```

Útil para ver qué operaciones hay, fusionarlas o transformarlas (cuantización, compilación). **TensorBoard**: `SummaryWriter().add_graph(modelo, ejemplo)` y `tensorboard --logdir runs`.

**Qué buscar:** ramas desconectadas (parámetros sin gradiente), operaciones inesperadas (`copy`, `view` costosos), el camino del residual.

## Nivel 3 · Topología y diagramas para documentación

**Netron** (visor interactivo): exporta y abre.

```python
torch.onnx.export(modelo, (ejemplo_entrada,), "modelo.onnx")    # luego: pip install netron && netron modelo.onnx
```

> En PyTorch reciente el exportador necesita paquetes extra: `pip install onnx onnxscript` (sin ellos falla con `ModuleNotFoundError: onnxscript`, error que comprobé al preparar el curso).

También abre `.pt`, `.safetensors`, `.tflite`, etc. Haz clic en cada nodo para ver atributos y *shapes*.

| Herramienta | Salida | Úsala para |
|---|---|---|
| **Netron** | visor web/escritorio | inspeccionar modelos exportados y de terceros |
| **`torchview`** | gráfico del *forward* con módulos agrupados (más legible que `torchviz`) | diagramas de arquitecturas jerárquicas |
| **PlotNeuralNet** | LaTeX/TikZ: bloques 3D de CNN | figuras de publicación |
| **NN-SVG** | SVG de redes tipo FCNN/LeNet/AlexNet | diapositivas y *posters* |
| **Mermaid / draw.io** | diagramas de bloques a mano | arquitecturas propias (como en las guías de este curso) |

**Buenas prácticas de diagramación:** un bloque = un módulo con nombre; anota `[B,T,C]` en cada flecha; muestra explícitamente los residuales; indica qué se repite ×N; no mezcles niveles de detalle.

## Nivel 4 · Mapeo interno e interpretabilidad

| Técnica | Código/herramienta | Pregunta |
|---|---|---|
| Mapas de atención | **BertViz** (`head_view`, `model_view`) o los `pesos` que guarda `atencion.py` | ¿a qué tokens mira cada cabeza? |
| Atribución | **Captum**: `IntegratedGradients`, `LayerGradCam` | ¿qué entradas empujan esta salida? |
| Activaciones | *hooks* de forward + histogramas | ¿hay neuronas muertas/saturadas? (Semana 3) |
| ***Logit lens*** | `norm_f` + `lm_head` aplicados al estado de cada capa | ¿en qué capa “aparece” la predicción? |
| ***Probing*** | regresión logística sobre activaciones | ¿se codifica linealmente X en esta capa? |
| **Ablación / activation patching** | poner a cero o intercambiar activaciones | ¿es **causal** este componente? |
| **SAE** | $\min\lVert x-D\,\text{ReLU}(Ex)\rVert^2+\lambda\lVert\cdot\rVert_1$ | ¿qué características interpretables hay en las activaciones? |

**Logit lens** sobre tu GPT (esqueleto):

```python
@torch.no_grad()
def logit_lens(m, idx):
    x = m.tok(idx)
    for i, b in enumerate(m.bloques):
        x = b(x)
        pred = m.lm_head(m.norm_f(x))[:, -1].argmax(-1)       # ¿qué predeciría si el modelo terminara aquí?
        print(f"tras el bloque {i}: {pred.tolist()}")
```

**Regla de oro:** correlación ≠ causa. Un patrón de atención o una atribución es una **hipótesis**; confírmala interviniendo (ablación) y midiendo el efecto en la salida.

## Plantilla de “informe de mapeo” (entregable de la Semana 3)

1. Diagrama de bloques con *shapes* (Nivel 3).
2. Tabla de parámetros por capa (Nivel 1, `torchinfo`).
3. Grafo de autograd de una pérdida de ejemplo (Nivel 2).
4. Estadísticas de activaciones y gradientes por capa (Nivel 4, hooks).
5. Un hallazgo: “esta capa tiene X % de neuronas muertas / esta cabeza hace Y”.
