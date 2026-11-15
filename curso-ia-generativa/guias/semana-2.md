# Semana 2 · El MLP vectorizado y la dinámica de entrenamiento

> **23 – 29 nov 2026** · ~12 h · Módulo II de la hoja de ruta (Paso 2)
> **Entregable:** un MLP en NumPy con backprop escrita a mano que pasa el gradient check, y el mismo modelo en PyTorch con tu propio bucle de entrenamiento.

## 1. Objetivos

- [ ] Pasar de escalares a **tensores**: derivar y programar el forward/backward de una capa densa con *shapes* correctos.
- [ ] Derivar el gradiente de **softmax + entropía cruzada** ($\partial L/\partial z = p - \text{onehot}$).
- [ ] Entender y justificar la **inicialización** Xavier/Glorot y Kaiming/He.
- [ ] Comparar **activaciones** (ReLU, LeakyReLU, GELU, SiLU) y **optimizadores** (SGD, momentum, Adam, AdamW).
- [ ] Escribir el **bucle de entrenamiento estándar** en PyTorch sin ayuda.
- [ ] Separar *train/validación* y reconocer sobreajuste.

## 2. Plan día a día (≈12 h)

| Día | Tiempo | Qué hacer |
|---|---|---|
| **Lun** | 1,5 h | Karpathy – *makemore, parte 1* (bigramas): tensores de PyTorch, broadcasting, entropía cruzada/NLL como pérdida. Repasa [`recursos/chuleta-formulas.md`](../recursos/chuleta-formulas.md). |
| **Mar** | 1,5 h | **A mano, en papel:** derivar $dW, db, dA$ para $Z=AW+b$ y $\partial L/\partial z$ para softmax+CE. Lee `mlp_numpy.py` (docstring + `backward`). |
| **Mié** | 2 h | Programa tu propio `MLPNumpy` **sin mirar**, con `verificar_gradientes`. Compara con `mlp_numpy.py`. |
| **Jue** | 1,5 h | Karpathy – *makemore, parte 2* (MLP de Bengio): embeddings, *mini-batches*, *learning rate* por búsqueda, *train/dev/test*. |
| **Vie** | 1,5 h | `mlp_torch.py`: autograd vs tu backprop manual; optimizadores y activaciones. Lee Prince cap. 5–7 (pérdidas, ajuste, gradiente). |
| **Sáb** | 3–4 h | **Proyecto:** clasificador de espirales en PyTorch con tu bucle, *grid* de lr × optimizador × activación, y un gráfico de curvas de pérdida train/val. |
| **Dom** | 1 h | Autoevaluación, tarjetas de repaso (Anki o papel), plan de la Semana 3. |

## 3. Teoría esencial

### 3.1 Capa densa vectorizada

Para un lote de $N$ ejemplos:

$$Z = A\,W + b,\qquad A\in\mathbb{R}^{N\times d_{in}},\; W\in\mathbb{R}^{d_{in}\times d_{out}}$$

**Backward** (dado $dZ=\partial L/\partial Z$, de shape `[N,d_out]`):

$$dW = A^\top dZ\;\;[d_{in},d_{out}],\qquad db=\sum_{n} dZ_{n}\;\;[d_{out}],\qquad dA = dZ\,W^\top\;\;[N,d_{in}]$$

> Truco para no equivocarte: cada gradiente tiene **el mismo shape que su variable**. Si no cuadra, trasponer y reordenar hasta que cuadre es casi siempre la solución correcta.

Con ReLU después: $dZ = dA_{\text{sig}}\odot\mathbb{1}[Z>0]$.

### 3.2 Softmax + entropía cruzada

$$p_i=\frac{e^{z_i}}{\sum_j e^{z_j}},\qquad L=-\log p_y\quad\Rightarrow\quad \frac{\partial L}{\partial z_i}=p_i-\mathbb{1}[i=y]$$

Con un lote se divide por $N$ (pérdida media). **Estabilidad numérica:** restar $\max_j z_j$ antes de exponenciar (no cambia el resultado). Pérdida inicial de un modelo “ignorante” con $K$ clases: $\ln K$. Si tu pérdida inicial es muy distinta, hay un bug o una inicialización mala.

### 3.3 Inicialización (por qué importa)

Si $w$ y $x$ son independientes, de media cero: $\mathrm{Var}(z)=d_{in}\,\mathrm{Var}(w)\,\mathrm{Var}(x)$. Para que la varianza **no crezca ni decaiga** capa tras capa:

| Esquema | $\mathrm{Var}(w)$ | Úsalo con |
|---|---|---|
| Xavier/Glorot | $\dfrac{2}{d_{in}+d_{out}}$ (o $1/d_{in}$) | tanh, sigmoide (simétricas) |
| Kaiming/He | $\dfrac{2}{d_{in}}$ | ReLU (anula la mitad → se compensa con el 2) |

Lo verás con números en la Semana 3 (`laboratorio_init_activaciones.py`). Los **sesgos** se inicializan en 0.

### 3.4 Activaciones

| Activación | Fórmula | Nota |
|---|---|---|
| ReLU | $\max(0,z)$ | simple y rápida; neuronas “muertas” si $z<0$ siempre |
| LeakyReLU | $\max(\alpha z, z)$, $\alpha\approx0.01$ | evita neuronas muertas |
| GELU | $z\,\Phi(z)$ | estándar en BERT/GPT-2 |
| SiLU/Swish | $z\,\sigma(z)$ | base de **SwiGLU** (LLaMA y similares) |
| tanh | $\tanh z$ | acotada, centrada en 0; satura |

### 3.5 Optimizadores

- **SGD:** $\theta\leftarrow\theta-\eta g$.
- **Momentum:** $v\leftarrow\mu v+g,\;\theta\leftarrow\theta-\eta v$ (suaviza el ruido y acelera por valles).
- **Adam:** estima el primer y segundo momento del gradiente con corrección de sesgo:

$$m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2,\quad \hat m=\frac{m_t}{1-\beta_1^t},\ \hat v=\frac{v_t}{1-\beta_2^t}$$
$$\theta\leftarrow\theta-\eta\,\frac{\hat m}{\sqrt{\hat v}+\epsilon}$$

- **AdamW:** igual, pero el *weight decay* se aplica **desacoplado**: $\theta\leftarrow\theta-\eta\big(\tfrac{\hat m}{\sqrt{\hat v}+\epsilon}+\lambda\theta\big)$. Es el optimizador por defecto de casi todos los LLM (típico $\beta_2=0.95$ en preentrenamiento, $\lambda\in[0.01,0.1]$).
- **Planificación de $\eta$:** *warmup* lineal (primeros cientos de pasos) + decaimiento coseno. Lo usarás en el GPT de la Semana 5.

### 3.6 Plantilla del bucle de entrenamiento (memorízala)

```python
modelo = MiModelo()
opt = torch.optim.AdamW(modelo.parameters(), lr=3e-3, weight_decay=0.01)
for paso in range(N_PASOS):
    xb, yb = obtener_lote()                       # 1. datos
    logits = modelo(xb)                           # 2. forward
    loss = F.cross_entropy(logits, yb)            # 3. pérdida
    opt.zero_grad(set_to_none=True)               # 4. limpiar gradientes viejos
    loss.backward()                               # 5. backprop (autograd)
    torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)   # 6. (opcional) recortar
    opt.step()                                    # 7. actualizar parámetros
```

Evaluación: `modelo.eval()` + `with torch.no_grad():` (desactiva dropout/BN-entrenamiento y no construye el grafo).

## 4. Código de la semana

Carpeta: [`codigo/semana2_mlp/`](../codigo/semana2_mlp)

```bash
cd curso-ia-generativa/codigo/semana2_mlp
python mlp_numpy.py     # backprop manual + gradient check (solo NumPy)
python mlp_torch.py     # autograd vs manual · optimizadores · activaciones
```

**Resultados de referencia** (espirales de 3 clases, semilla 0):

| Verificación | Resultado |
|---|---|
| Gradient check NumPy | peor error relativo ≈ $10^{-9}$ ✔ |
| Autograd (PyTorch) vs backprop manual | diferencia máx. ≈ $10^{-8}$ por capa ✔ |
| MLP NumPy, 2000 pasos de SGD (lr 1.0) | pérdida 1,12 → **0,021**, precisión **99,3 %** |

Optimizadores (misma red 2-64-64-3, 600 pasos, batch 64):

| Optimizador | Pérdida (media últimos 50) | Precisión |
|---|---|---|
| SGD lr=0.1 | 0,262 | 90,0 % |
| SGD + momentum 0.9 | 0,056 | 99,0 % |
| Adam lr=3e-3 | 0,035 | 99,3 % |
| AdamW lr=3e-3, wd=1e-2 | 0,035 | 99,3 % |

Activaciones (AdamW): ReLU 0,035 · GELU 0,039 · tanh 0,057 · SiLU 0,061. **Cuidado:** diferencias tan pequeñas, con **una sola semilla**, no son concluyentes; para sacar conclusiones repite con ≥5 semillas (ejercicio 5).

## 5. Ejercicios

**Básicos**
1. A mano: $\partial L/\partial z$ de softmax+CE para $z=[2,1,0]$, $y=0$. Verifica con NumPy.
2. En `mlp_numpy.py` cambia la inicialización a $\mathcal{N}(0,1)$ y a $\mathcal{N}(0,0.01)$. ¿Qué ocurre con la pérdida inicial y por qué?
3. Añade LeakyReLU al MLP de NumPy (forward y backward) y pasa el gradient check.

**Intermedios**
4. Implementa **momentum** y **Adam** en NumPy (clase `Optimizador` con `.step()`); compara con los de PyTorch en 10 pasos (deben coincidir a 1e-6).
5. Repite el experimento de activaciones con 5 semillas y reporta media ± desviación. ¿Sigue ganando ReLU?
6. Añade **Dropout** (inverted dropout) a la versión PyTorch, y **weight decay**. Observa la brecha train/val con pocos datos (n=30 por clase).

**Reto**
7. Implementa *learning-rate finder* (sube lr exponencialmente, grafica pérdida vs lr, elige 1/10 del mínimo).
8. Entrena sobre **MNIST** (`torchvision.datasets.MNIST`) con un MLP de 2 capas ocultas y alcanza ≥ 97 % en test.

## 6. Errores comunes

| Síntoma | Causa |
|---|---|
| `RuntimeError: ... shapes cannot be multiplied` | Olvidaste que `nn.Linear` guarda $W$ como `[out,in]` y calcula $xW^\top$ |
| Pérdida `nan` | lr alto, `log(0)` sin estabilizar, o no restaste el máximo en softmax |
| Pérdida inicial ≠ $\ln K$ | inicialización de la última capa demasiado grande |
| Entrena en train y falla en val | sobreajuste: más datos, dropout, weight decay, parada temprana |
| Olvidar `opt.zero_grad()` | los gradientes se **acumulan** entre pasos |
| `model.eval()` olvidado | dropout/BatchNorm siguen en modo entrenamiento al validar |

## 7. Autoevaluación

1. ¿Qué shape tiene $dW$ y por qué se calcula como $A^\top dZ$?
2. Si $K=10$ clases, ¿qué pérdida inicial esperas?
3. ¿Por qué He usa un 2 y Xavier no?
4. ¿Qué diferencia hay entre Adam con L2 y AdamW?
5. ¿Para qué sirve `torch.no_grad()` al evaluar?

<details><summary>Respuestas</summary>

1. `[d_in,d_out]`; cada elemento $W_{ij}$ acumula (sobre el lote) activación de entrada $i$ × error de salida $j$.
2. $\ln 10\approx2.30$.
3. ReLU pone a cero la mitad de las pre-activaciones, reduciendo a la mitad la varianza de la salida; el 2 lo compensa.
4. En Adam+L2 el término de decaimiento entra en el gradiente y es reescalado por la adaptación de Adam; en AdamW se resta directamente del peso, desacoplado.
5. No construye el grafo de autograd: ahorra memoria y tiempo.
</details>

## 8. Checklist de salida

- [ ] Mi MLP de NumPy pasa el gradient check y alcanza ≥ 95 % en espirales.
- [ ] Sé escribir el bucle de entrenamiento de memoria.
- [ ] Puedo explicar por qué $dZ=p-y$ en softmax+CE.
- [ ] Comparé dos optimizadores/activaciones con más de una semilla.
- [ ] Tengo curvas de pérdida train vs val guardadas.

## 9. Recursos de la semana

- Karpathy, *Zero to Hero*: *makemore* partes 1 y 2.
- Prince, *Understanding Deep Learning*, cap. 5–8.
- *Dive into Deep Learning* (d2l.ai): capítulos *Multilayer Perceptrons* y *Optimization Algorithms*.
- CS231n notes: *Neural Networks Part 1–3* (activaciones, inicialización, optimización).
- Papers: He et al. (2015) *Delving Deep into Rectifiers* (Kaiming init, arXiv 1502.01852) · Kingma & Ba (2014) *Adam* (arXiv 1412.6980) · Loshchilov & Hutter (2017) *Decoupled Weight Decay Regularization* (arXiv 1711.05101).

[← Semana 1](semana-1.md) · **Siguiente:** [Semana 3 →](semana-3.md)
