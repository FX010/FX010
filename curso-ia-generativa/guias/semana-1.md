# Semana 1 · Cimientos: tensores, cálculo y tu primer motor de autograd

> **16 – 22 nov 2026** · ~12 h · Módulo I y II de la hoja de ruta
> **Entregable:** un motor de diferenciación automática (`micrograd`) escrito por ti, con gradient check y una red que separa dos medias lunas.

## 1. Objetivos de la semana

Al terminar podrás:

- [ ] Explicar qué es un tensor de rango $N$, calcular su *shape* tras una multiplicación matricial y razonar sobre *broadcasting*.
- [ ] Calcular a mano el gradiente de una neurona y de una pérdida usando la **regla de la cadena sobre un grafo**.
- [ ] Explicar por qué la retropropagación es *diferenciación en modo inverso* y por qué es tan barata.
- [ ] Programar desde cero (Python puro) un `Value` con `+`, `*`, `**`, `exp`, `tanh`, `relu` y `backward()`.
- [ ] Verificar tu implementación con **gradient checking** (derivada numérica).

**Por qué empezar aquí:** todo el curso (CNN, Transformers, difusión, LoRA) es el mismo bucle: *forward → pérdida → backward → actualizar*. Si dominas el átomo escalar, el resto es “lo mismo, pero con tensores”.

## 2. Plan día a día (≈12 h)

| Día | Tiempo | Qué hacer | Resultado |
|---|---|---|---|
| **Lun** | 1,5 h | Ver **3Blue1Brown – serie *Neural Networks*** (cap. 1–4: qué es una red, descenso de gradiente, intuición y cálculo de backprop). Lee la [portada del curso](../README.md) y el [plan maestro](00-plan-maestro.md). | Cuaderno de notas con 5 ideas clave en tus palabras |
| **Mar** | 1,5 h | **Álgebra lineal**: *Mathematics for ML* cap. 2 (secciones de matrices y producto) o notas de CS231n. En NumPy: crea tensores de rango 0–4, practica `@`, `.T`, `reshape`, broadcasting. | Hoja de 10 ejercicios de *shapes* resueltos |
| **Mié** | 1,5 h | **Cálculo**: derivada parcial, gradiente, jacobiana, regla de la cadena. Resuelve a mano el ejemplo de §3.3. | Gradiente de una neurona derivado en papel |
| **Jue** | 1,5 h | **Karpathy – *micrograd*** (≈2,5 h en total; hoy la primera mitad): derivadas numéricas, grafo, backward manual. | Entiendes `Value` y `_backward` |
| **Vie** | 1,5 h | Termina micrograd: orden topológico, neuronas, capas, MLP, bucle de entrenamiento. Lee `codigo/semana1_micrograd/engine.py`. | Tu versión corre y entrena |
| **Sáb** | 3–4 h | **Proyecto de la semana:** reescribe `engine.py` y `nn.py` **sin mirar** (solo con tus notas), pasa `verificar_gradientes.py` y `entrenar_moons.py`. Añade `sigmoid` y `log` por tu cuenta. | Repo propio con tests |
| **Dom** | 1 h | Autoevaluación (§7), repaso espaciado de la semana, planifica la Semana 2. | Checklist de salida (§8) |

## 3. Teoría esencial

### 3.1 Tensores y *shapes*

| Objeto | Rango | Ejemplo | Shape |
|---|---|---|---|
| escalar | 0 | pérdida | `[]` |
| vector | 1 | un *embedding* | `[d]` |
| matriz | 2 | pesos de una capa | `[d_in, d_out]` |
| tensor 3D | 3 | batch de secuencias | `[B, T, C]` |
| tensor 4D | 4 | batch de imágenes | `[B, C, H, W]` |

**Regla de oro de la multiplicación matricial:** `[a, b] @ [b, c] → [a, c]`. La dimensión interna debe coincidir y *desaparece*. Todo bloque de atención y toda capa densa es esto:

$$Z = XW + b,\qquad X\in\mathbb{R}^{N\times d_{in}},\; W\in\mathbb{R}^{d_{in}\times d_{out}},\; b\in\mathbb{R}^{d_{out}}\ (\text{se transmite por broadcasting})$$

**Producto punto como similitud:** $\;a\cdot b = \lVert a\rVert\lVert b\rVert\cos\theta$. Si los vectores están normalizados, el producto punto *es* el coseno del ángulo. Esa es la intuición geométrica detrás de la atención ($q\cdot k$) y de la recuperación en RAG.

**Broadcasting (NumPy/PyTorch):** se alinean las formas por la derecha; dos dimensiones son compatibles si son iguales o una es 1. `[32,64] + [64] → [32,64]`; `[32,1] + [1,64] → [32,64]`.

### 3.2 La neurona

$$y=\phi\Big(\sum_i w_i x_i + b\Big)=\phi(w\cdot x+b)$$

Sin la no linealidad $\phi$, apilar capas sigue siendo una sola transformación lineal. Derivadas que usarás:

| $\phi$ | $\phi'$ |
|---|---|
| $\tanh(z)$ | $1-\tanh^2(z)$ |
| $\sigma(z)=1/(1+e^{-z})$ | $\sigma(z)(1-\sigma(z))$ |
| $\text{ReLU}(z)=\max(0,z)$ | $1$ si $z>0$, $0$ si no |

### 3.3 Backprop a mano en un grafo (ejemplo numérico completo)

Neurona con una entrada: $x=2,\; w=0.5,\; b=-0.3,\; y_{obj}=1$.

```
z = w·x + b        →  z = 0.7
a = tanh(z)        →  a = 0.60437
L = (a − y)²       →  L = 0.15652
```

**Backward** (de la derecha a la izquierda, multiplicando derivadas locales):

| Paso | Cálculo | Valor |
|---|---|---|
| $\partial L/\partial a$ | $2(a-y)$ | $-0.79126$ |
| $\partial a/\partial z$ | $1-a^2$ | $0.63474$ |
| $\partial L/\partial z$ | $(-0.79126)(0.63474)$ | $-0.50225$ |
| $\partial L/\partial w$ | $\partial L/\partial z\cdot x$ | $-1.00449$ |
| $\partial L/\partial b$ | $\partial L/\partial z\cdot 1$ | $-0.50225$ |
| $\partial L/\partial x$ | $\partial L/\partial z\cdot w$ | $-0.25112$ |

Comprueba con tu código: `Value(0.5)`, etc. deben dar esos mismos números. **Cuando un nodo se usa varias veces, los gradientes se *suman*** (por eso en `engine.py` se usa `+=`, nunca `=`).

### 3.4 Regla de la cadena y modo inverso

Para $L=f_n(\dots f_2(f_1(x)))$:

$$\frac{\partial L}{\partial x}=\frac{\partial f_n}{\partial f_{n-1}}\cdots\frac{\partial f_2}{\partial f_1}\frac{\partial f_1}{\partial x}$$

- **Modo directo (*forward*)**: propaga derivadas de las entradas hacia la salida; un pase por *cada* parámetro.
- **Modo inverso (*reverse*, = backprop)**: propaga desde la pérdida hacia atrás; **un solo pase da el gradiente de *todos* los parámetros**. Con millones de parámetros y una pérdida escalar, es la única opción práctica.
- Con tensores, cada derivada local es una **jacobiana**; en la práctica nunca se construye entera, se aplica el producto vector-jacobiana (VJP). Para una capa lineal $Z=AW$: $\;dA=dZ\,W^\top,\; dW=A^\top dZ$ (lo derivarás en la Semana 2).

### 3.5 Descenso de gradiente

$$\theta\leftarrow\theta-\eta\,\nabla_\theta\mathcal{L}$$

El gradiente apunta hacia donde la pérdida **sube** más rápido; restamos para bajar. $\eta$ (tasa de aprendizaje) demasiado grande → diverge; demasiado pequeña → no avanza.

## 4. Código de la semana

Carpeta: [`codigo/semana1_micrograd/`](../codigo/semana1_micrograd)

```bash
cd curso-ia-generativa/codigo/semana1_micrograd
python verificar_gradientes.py     # gradient check: autodiff vs derivada numérica
python entrenar_moons.py           # MLP escalar (105 parámetros) sobre dos medias lunas
```

| Archivo | Qué contiene |
|---|---|
| `engine.py` | Clase `Value`: operaciones, `_backward` local y `backward()` con orden topológico |
| `nn.py` | `Neurona`, `Capa`, `MLP` construidas sobre `Value` |
| `verificar_gradientes.py` | Compara autodiff contra $\frac{f(x+h)-f(x-h)}{2h}$ |
| `entrenar_moons.py` | Pérdida *hinge* + L2, SGD con decaimiento de $\eta$ |

**Resultados de referencia** (obtenidos al probar los scripts, semilla fija):

- Gradient check: error relativo del orden de $10^{-10}$ en `a`, `b`, `c` → ✔.
- `entrenar_moons.py`: 105 parámetros; precisión 84 % en el paso 0 → **98 %** en el paso 99 (la pérdida baja de 0,347 a 0,067). Tarda ≈ 20 s: es Python puro sobre escalares, precisamente por eso en la Semana 2 pasamos a tensores.

## 5. Ejercicios

**Básicos**
1. Calcula a mano $\partial L/\partial w$ para $L=(\sigma(wx+b)-y)^2$ con $x=1,w=0.3,b=0,y=0$ y confirma con `Value`.
2. Implementa `Value.sqrt()` reutilizando `__pow__` (exponente 0,5) y comprueba su gradiente con `verificar_gradientes.py`.
3. Dibuja el grafo de $L=(a\cdot b + c)^2$ y anota los `grad` tras `backward()`.

**Intermedios**
4. Implementa la pérdida de **entropía cruzada binaria** con `sigmoid` y `log` y entrena con ella en lugar de *hinge*.
5. Añade **momentum** al bucle de entrenamiento (guarda una velocidad por parámetro).
6. Haz que `entrenar_moons.py` imprima la **norma del gradiente** por capa. ¿Qué capa tiene el gradiente mayor?

**Reto**
7. Escribe `softmax` sobre una lista de `Value` (resta el máximo para estabilidad numérica) y úsalo con entropía cruzada multiclase. Verifica con gradient check.
8. Implementa **modo directo (forward-mode)** con *dual numbers* para una función de 2 variables y compara coste con el modo inverso.

## 6. Errores comunes

| Síntoma | Causa probable |
|---|---|
| Gradientes la mitad (o doble) de lo esperado | Usar `=` en vez de `+=` cuando un nodo aparece dos veces |
| `backward()` da cero en nodos intermedios | Olvidaste `self.grad = 1.0` en la raíz o el orden topológico está mal |
| La pérdida sube | $\eta$ demasiado alto, signo del gradiente invertido, o no pones a cero los `grad` entre iteraciones |
| Gradient check falla ≈1e-2 en ReLU | Evaluar justo en $z=0$ (no diferenciable); usa otro punto |

## 7. Autoevaluación (responde sin mirar)

1. ¿Cuál es el *shape* de `[B,T,C] @ [C,D]`? ¿Y de `[B,H,T,Dh] @ [B,H,Dh,T]`?
2. ¿Por qué se acumulan (`+=`) los gradientes?
3. ¿Por qué backprop cuesta aproximadamente lo mismo que un forward y no “un forward por parámetro”?
4. ¿Qué pasa con una red de solo capas lineales sin activación?
5. ¿Por qué se usa $h\approx10^{-6}$ y no $10^{-12}$ en la derivada numérica?

<details><summary>Respuestas</summary>

1. `[B,T,D]` y `[B,H,T,T]`.
2. Regla de la cadena multivariable: si $x$ influye en $L$ por varios caminos, las contribuciones se suman.
3. El modo inverso reutiliza los productos parciales ya calculados al recorrer el grafo una sola vez de atrás hacia delante.
4. Equivale a una única capa lineal (composición de lineales = lineal).
5. Con $h$ minúsculo domina el error de redondeo de coma flotante (cancelación catastrófica); $10^{-5}$–$10^{-6}$ equilibra truncamiento y redondeo.
</details>

## 8. Checklist de salida

- [ ] Mi `engine.py` pasa el gradient check (error < 1e-6).
- [ ] Reproduzco a mano el ejemplo de §3.3.
- [ ] Entrené un MLP escalar y vi bajar la pérdida.
- [ ] Puedo explicar a otra persona, en 3 minutos y sin diapositivas, qué hace `backward()`.
- [ ] Escribí mis notas de la semana (5–10 líneas) en mi cuaderno de aprendizaje.

## 9. Recursos de la semana

- 3Blue1Brown, *Neural Networks* (YouTube): cap. 1–4.
- Karpathy, *Neural Networks: Zero to Hero* → *The spelled-out intro to neural networks and backpropagation: building micrograd*.
- Deisenroth et al., *Mathematics for Machine Learning*, cap. 2 (álgebra lineal) y 5 (cálculo vectorial).
- Prince, *Understanding Deep Learning*, cap. 1–4 y apéndice de matemáticas.
- Stanford CS231n, notas de *Backpropagation, Intuitions* y *Optimization*.
- Paper fundacional: Rumelhart, Hinton & Williams (1986), *Learning representations by back-propagating errors*.

**Siguiente:** [Semana 2 →](semana-2.md)
