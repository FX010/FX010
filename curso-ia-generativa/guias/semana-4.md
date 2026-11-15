# Semana 4 · Arquitecturas estructurales: CNN, residuales y recurrentes

> **7 – 13 dic 2026** · ~12 h · Módulo IV de la hoja de ruta (Paso 3)
> **Entregable:** una CNN y una red residual propias, entrenadas sobre imágenes, con una comparación **medida** contra un MLP, y un informe de por qué los residuales permiten entrenar redes profundas.

## 1. Objetivos

- [ ] Calcular a mano el *shape* de salida y el número de parámetros de una capa convolucional, de *pooling* y de una red completa.
- [ ] Explicar los **sesgos inductivos** (localidad, compartición de pesos, equivarianza) y por qué las CNN generalizan mejor que un MLP en imágenes.
- [ ] Explicar por qué las **conexiones residuales** (ResNet) resuelven el desvanecimiento del gradiente, con una medición.
- [ ] Escribir las ecuaciones de **RNN, LSTM y GRU**, y explicar por qué la RNN simple no recuerda a largo plazo y qué limitaciones (secuencial, no paralelizable) motivaron la atención.
- [ ] Preparar el terreno para la Semana 5: ver la atención como “la solución” a las limitaciones recurrentes.

## 2. Plan día a día (≈12 h)

| Día | Tiempo | Qué hacer |
|---|---|---|
| **Lun** | 2 h | **CS231n** (notas *Convolutional Networks*): convolución, *stride*, *padding*, *pooling*, cálculo de dimensiones. Resuelve 6 ejercicios de *shapes* a mano (sección 3.1). |
| **Mar** | 1,5 h | Ejecuta `cnn_vs_mlp.py`. Implementa una **convolución 2D ingenua con bucles** en NumPy y compárala con `F.conv2d` (error < 1e-5). |
| **Mié** | 1,5 h | **ResNet:** lee el paper (He et al., 2015) solo secciones 1–3. Ejecuta `residuales_gradientes.py` y explica la tabla. |
| **Jue** | 1,5 h | **RNN/LSTM/GRU:** *Understanding LSTM Networks* (Chris Olah) + CS224n (clase de RNN). Escribe las ecuaciones de la LSTM de memoria. |
| **Vie** | 1,5 h | Ejecuta `rnn_lstm_gru.py` con T=10, 20, 30, 40. Anota qué pasa; luego lee sobre BPTT y *gradient clipping*. |
| **Sáb** | 3–4 h | **Proyecto:** CNN con bloques residuales sobre **FashionMNIST/MNIST** (`torchvision`) o sobre tus propias imágenes; meta ≥ 90 % en test y curvas de pérdida. |
| **Dom** | 1 h | Autoevaluación + puente a Transformers: lee 3Blue1Brown cap. 5 (*But what is a GPT?*). |

## 3. Teoría esencial

### 3.1 Convolución 2D: dimensiones y parámetros

Entrada `[B, C_in, H, W]`, filtros de tamaño $k\times k$, `C_out` filtros, *stride* $s$, *padding* $p$:

$$H_{out}=\left\lfloor\frac{H+2p-k}{s}+1\right\rfloor,\qquad \text{parámetros}=C_{out}\,(C_{in}\,k^2+1)$$

Ejemplo de la red de `cnn_vs_mlp.py` (entrada `[B,1,16,16]`):

| Capa | Salida | Parámetros |
|---|---|---|
| `Conv2d(1→12, k=3, p=1)` | `[B,12,16,16]` | $12(1\cdot9+1)=120$ |
| `MaxPool2d(2)` | `[B,12,8,8]` | 0 |
| `Conv2d(12→24, k=3, p=1)` | `[B,24,8,8]` | $24(12\cdot9+1)=2616$ |
| `MaxPool2d(2)` | `[B,24,4,4]` | 0 |
| `AdaptiveAvgPool2d(1)` + `Flatten` | `[B,24]` | 0 |
| `Linear(24→4)` | `[B,4]` | $24\cdot4+4=100$ |
| **Total** | | **2 836** ✔ (coincide con `sum(p.numel())`) |

**Campo receptivo:** cada capa 3×3 con stride 1 suma 2 píxeles de alcance (multiplicado por el *salto* acumulado si hubo pooling). En esta red: tras la 1.ª conv ve 3×3; tras el 1.er pool, 4×4; tras la 2.ª conv, 8×8; tras el 2.º pool, 10×10 píxeles de la imagen original.

**Sesgos inductivos de la CNN**
- **Localidad:** cada neurona mira un vecindario pequeño.
- **Compartición de pesos:** el mismo filtro se aplica en todas las posiciones → menos parámetros.
- **Equivarianza a traslaciones:** si desplazas la entrada, el mapa de activaciones se desplaza igual. (Con *pooling*/promedio global se obtiene además cierta **invarianza**.)

### 3.2 Conexiones residuales

$$h_{l+1}=h_l+F(h_l)\quad\Longrightarrow\quad \frac{\partial h_{l+1}}{\partial h_l}=I+\frac{\partial F}{\partial h_l}$$

El término $I$ crea una “autopista” por la que el gradiente llega intacto a las primeras capas. Sin él, el gradiente es un producto de $L$ jacobianas y se desvanece (o explota). Además, aprender $F\approx0$ (identidad) es fácil, así que añadir capas **no puede empeorar** demasiado el modelo. Los Transformers usan exactamente esto (`x = x + Attn(Norm(x))`).

### 3.3 Redes recurrentes

**RNN simple:** $\;h_t=\tanh(W_{hh}h_{t-1}+W_{xh}x_t+b)$. Se entrena con **BPTT** (retropropagación a través del tiempo): el gradiente hacia el paso $t-k$ contiene el producto $\prod_{i}W_{hh}^\top\,\mathrm{diag}(\tanh')$ de $k$ términos → se desvanece (si el radio espectral < 1) o explota (> 1).

**LSTM** (Hochreiter & Schmidhuber, 1997): introduce una **celda de memoria** $c_t$ con *puertas*:

$$\begin{aligned}
f_t&=\sigma(W_f[h_{t-1},x_t]+b_f)&&\text{(olvido)}\\
i_t&=\sigma(W_i[h_{t-1},x_t]+b_i)&&\text{(entrada)}\\
\tilde c_t&=\tanh(W_c[h_{t-1},x_t]+b_c)\\
c_t&=f_t\odot c_{t-1}+i_t\odot\tilde c_t&&\text{(actualización aditiva ≈ residual)}\\
o_t&=\sigma(W_o[h_{t-1},x_t]+b_o),\quad h_t=o_t\odot\tanh(c_t)
\end{aligned}$$

La actualización **aditiva** de $c_t$ es la misma idea que el residual: un camino casi lineal para el gradiente. Truco: inicializar el sesgo de la puerta de olvido en 1 (se “recuerda” por defecto).

**GRU:** versión simplificada con dos puertas (actualización y reinicio) y sin celda separada; menos parámetros.

**Limitaciones que motivan la Semana 5:** (1) cómputo **secuencial** (no se paraleliza en el tiempo); (2) toda la historia debe comprimirse en un vector de tamaño fijo; (3) distancias largas siguen siendo difíciles. La **atención** conecta directamente cualquier par de posiciones y se calcula en paralelo.

## 4. Código de la semana

Carpeta: [`codigo/semana4_arquitecturas/`](../codigo/semana4_arquitecturas)

```bash
cd curso-ia-generativa/codigo/semana4_arquitecturas
python cnn_vs_mlp.py             # CNN vs MLP sobre formas que cambian de posición
python residuales_gradientes.py  # norma del gradiente en la 1.ª capa: plana vs residual
python rnn_lstm_gru.py 30 3000   # memoria a largo plazo (T, pasos)
```

**Resultados de referencia** (CPU, una semilla)

*CNN vs MLP* (400 imágenes de entrenamiento, ruido 0,4, formas en posición/tamaño aleatorios):

| Modelo | Parámetros | Precisión train | Precisión test |
|---|---|---|---|
| MLP | 31 140 | 100 % | **55,2 %** |
| CNN | 2 836 | 96,2 % | **92,4 %** |

El MLP memoriza (train 100 %) pero no generaliza; la CNN, con **11× menos parámetros**, sí. (Con 3 000 imágenes sin ruido los dos llegaban al ~100 %: la ventaja de la CNN aparece cuando hay pocos datos o cambios de posición.)

*Residuales* (norma del gradiente de la primera capa, redes de tanh con Xavier):

| Capas | Plana | Residual |
|---|---|---|
| 5 | 1,1e-1 | 6,3e-1 |
| 20 | 1,9e-2 | 4,0 |
| 40 | 1,4e-3 | 4,9 |
| 80 | 1,3e-3 | 28,3 |

Tras 300 pasos de Adam con 80 capas: la plana **no aprende** (pérdida 1,0986 = $\ln 3$) y la residual llega a **0,044**. Ojo: el gradiente residual *crece* con la profundidad; por eso los Transformers profundos escalan las proyecciones residuales y usan normalización (ver `gpt.py`).

*RNN/LSTM/GRU* (recordar el primer dígito de una secuencia de longitud T; 3 000 pasos, **una semilla**):

| T | RNN | LSTM | GRU |
|---|---|---|---|
| 20 | 100 % | 100 % | 100 % |
| 30 | 50,5 % | 100 % | 16,4 % (azar) |
| 40 | 16,2 % | 34,6 % | 16,7 % |

> **Lee esto con espíritu crítico.** El patrón general (todos aprenden con T pequeño; al crecer T cada vez cuesta más, con “mesetas” largas donde la pérdida no baja) es el esperado. Pero **el orden entre RNN/LSTM/GRU en un solo experimento depende de la semilla y del número de pasos**: aquí la GRU falló en T=30 y la RNN simple resolvió T=20, algo que la teoría ingenua no predice. Para concluir algo necesitas ≥ 5 semillas y curvas de aprendizaje (ejercicio 6). Es una lección real de investigación: *un experimento no es evidencia*.

## 5. Ejercicios

**Básicos**
1. Calcula a mano el *shape* y los parámetros de: `Conv2d(3→16,k=5,s=2,p=2)` sobre `[B,3,64,64]`; luego `MaxPool2d(2)`.
2. Dibuja el campo receptivo de 3 capas conv 3×3 (stride 1) sin pooling. ¿Cuántos píxeles ve la última neurona?
3. Implementa un **bloque residual** `x + F(x)` con `Conv-BN-ReLU-Conv-BN` y verifica que con $F\equiv0$ es la identidad.

**Intermedios**
4. Convolución ingenua con bucles en NumPy vs `F.conv2d` (misma salida), y mide cuánto más lenta es.
5. Entrena `cnn_vs_mlp.py` con **aumento de datos** (desplazamientos aleatorios) y sin él. ¿Cuánto mejora el MLP?
6. Repite `rnn_lstm_gru.py` con 5 semillas para T=30 y reporta media ± desviación por modelo. Grafica la curva de pérdida de una corrida que “se queda en la meseta”.

**Reto**
7. Implementa una **LSTM desde cero** (con `nn.Linear` y sus 4 puertas) y compárala con `nn.LSTM`: misma salida con los mismos pesos.
8. Construye un mini-ResNet (3 etapas, ~100 k parámetros) y entrénalo en CIFAR-10 hasta ≥ 80 %. Visualiza los filtros de la 1.ª capa y mapas de activación (nivel 4 de mapeo, con `Captum`/Grad-CAM).

## 6. Errores comunes

| Síntoma | Causa |
|---|---|
| `size mismatch` entre `Flatten` y `Linear` | calculaste mal el *shape* tras conv/pool → imprime con *hooks* (`shape_tracking.py`) |
| CNN no mejora | falta normalizar entradas, lr demasiado alto, o `padding` que hace perder bordes |
| LSTM no aprende en secuencias largas | sin *gradient clipping*, o lr mal; prueba sesgo de olvido=1 y recorte a 1.0 |
| Resultado de comparación “no reproducible” | una sola semilla; fija semillas y repite |
| `batch_first` mal usado | `[B,T,F]` vs `[T,B,F]`: lee la doc de `nn.LSTM` |

## 7. Autoevaluación

1. ¿Cuántos parámetros tiene `Conv2d(64→128, k=3)` con sesgo?
2. ¿Por qué `MaxPool` da cierta invarianza a traslaciones pequeñas?
3. Explica en dos frases por qué un residual ayuda al gradiente.
4. ¿Qué papel cumple la puerta de olvido de la LSTM?
5. Nombra tres limitaciones de las RNN que resuelve la atención.

<details><summary>Respuestas</summary>

1. $128(64\cdot9+1)=73\,856$.
2. Toma el máximo de una ventana: si la activación se mueve dentro de la ventana, la salida no cambia.
3. $\partial h_{l+1}/\partial h_l=I+\partial F/\partial h_l$, así que el gradiente tiene un camino directo con factor 1 hacia capas anteriores; no depende de multiplicar muchas jacobianas pequeñas.
4. Decide cuánto de la memoria previa $c_{t-1}$ se conserva; con valores cercanos a 1 la información persiste muchos pasos.
5. Procesamiento secuencial (sin paralelismo), cuello de botella de un único vector de estado, dificultad con dependencias largas.
</details>

## 8. Checklist de salida

- [ ] Calculo *shapes* y parámetros de convoluciones sin ejecutar código.
- [ ] Mi comparación CNN vs MLP usa los **mismos datos** y reporto train y test.
- [ ] Puedo explicar con una medición (no solo con teoría) el efecto del residual.
- [ ] Escribí de memoria las ecuaciones de la LSTM.
- [ ] Hice al menos un experimento con ≥ 3 semillas y reporté la dispersión.

## 9. Recursos de la semana

- Stanford **CS231n**: notas *Convolutional Neural Networks* y clases de arquitecturas.
- Stanford **CS224n**: clases de RNN/LSTM y de la transición a atención.
- Chris Olah, *Understanding LSTM Networks* (colah.github.io).
- Prince, *Understanding Deep Learning*: cap. 10 (convolucionales) y 11 (residuales).
- *Dive into Deep Learning*: capítulos de CNN modernas (ResNet) y RNN.
- Papers: He et al. (2015) *Deep Residual Learning for Image Recognition* (arXiv 1512.03385) · Hochreiter & Schmidhuber (1997) *Long Short-Term Memory* · Cho et al. (2014) *GRU* (arXiv 1406.1078).

[← Semana 3](semana-3.md) · **Siguiente:** [Semana 5 →](semana-5.md)
