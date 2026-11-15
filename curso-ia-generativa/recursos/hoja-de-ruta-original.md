# Hoja de Ruta y Biblioteca Maestra: Redes Neuronales e Inteligencia Artificial Generativa

Este documento reúne los fundamentos matemáticos, la secuencia paso a paso de construcción, las técnicas de mapeo arquitectónico y las fuentes de referencia indispensables para estudiar y redactar una guía completa sobre redes neuronales e inteligencia artificial generativa.

---

## 1. Prerrequisitos Técnicos: Qué dominar antes de construir

Antes de implementar o explicar una arquitectura neuronal, conviene asegurar cuatro pilares conceptuales que sostienen todo el aprendizaje profundo moderno:

* **Álgebra Lineal aplicada a Tensores:**
  * Comprensión de escalares, vectores, matrices y tensores de rango $N$.
  * Producto punto como medida de similitud geométrica.
  * Multiplicación matricial por bloques (el núcleo computacional de toda capa densa y bloque de atención).
  * Descomposición en valores singulares (SVD) y factorización de bajo rango, indispensables para entender *embeddings* y técnicas de adaptación eficiente como LoRA.
* **Cálculo Diferencial Multivariado y Grafos:**
  * Derivadas parciales, vector gradiente ($\nabla_\theta \mathcal{L}$) y matrices jacobianas.
  * **Regla de la cadena** aplicada sobre grafos dirigidos acíclicos (DAG), que constituye el mecanismo matemático exacto detrás de la retropropagación (*backpropagation*).
* **Probabilidad y Teoría de la Información:**
  * Distribuciones de probabilidad (gaussiana multivariada y categórica).
  * Estimación de máxima verosimilitud (MLE).
  * Entropía de Shannon, entropía cruzada (*Cross-Entropy Loss*) y divergencia de Kullback-Leibler ($D_{\text{KL}}$), fundamentales tanto para el entrenamiento de modelos de lenguaje como para regularizar el espacio latente de los *Autoencoders* Variacionales (VAE).
* **Paradigma Discriminativo vs. Generativo:**
  * Un modelo **discriminativo** aprende fronteras de decisión o distribuciones condicionales $P(y \mid x)$ para clasificar o predecir una etiqueta a partir de una entrada.
  * Un modelo **generativo** aprende la distribución subyacente de los datos $P(x)$ o la distribución conjunta $P(x, y)$ para poder realizar **muestreo (*sampling*)** y sintetizar instancias nuevas estadísticamente coherentes con los datos de entrenamiento.

---

## 2. Hoja de Ruta Paso a Paso: Cómo armar redes neuronales

La secuencia pedagógica más efectiva para aprender —y estructurar una guía progresiva— sigue cinco etapas acumulativas:

### Paso 1: El átomo neuronal y *Autodiff* escalar (Python puro)
* Implementar una neurona artificial básica:
  $$y = \phi\left(\sum_{i} w_i x_i + b\right)$$
* Construir un grafo computacional dinámico que registre cada operación aritmética elemental (suma, multiplicación, exponenciación, no linealidades).
* Programar un motor de diferenciación automática en modo inverso (*reverse-mode autodiff*) que propague los gradientes desde la pérdida hasta cada parámetro sin utilizar librerías de *deep learning*.

### Paso 2: Perceptrón Multicapa (MLP) vectorizado y dinámica de entrenamiento
* Escalar de operaciones escalares a tensores multidimensionales en NumPy y PyTorch.
* Dominar los componentes que evitan que una red colapse o diverja durante el entrenamiento:
  * **Inicialización de pesos:** Xavier/Glorot (para funciones simétricas como $\tanh$ y sigmoide) y Kaiming/He (para funciones asimétricas como $\text{ReLU}$).
  * **Funciones de activación:** $\text{ReLU}$, $\text{LeakyReLU}$, $\text{GELU}$ y $\text{SiLU}$ / $\text{Swish}$.
  * **Capas de normalización:** *Batch Normalization* (`BatchNorm`), *Layer Normalization* (`LayerNorm`) y *Root Mean Square Normalization* (`RMSNorm`).
  * **Regularización y optimización:** *Dropout*, *Weight Decay*, descenso de gradiente estocástico con momento (`SGD + Momentum`), `Adam` y `AdamW`.

### Paso 3: Arquitecturas base según la estructura de los datos
* **Datos espaciales e imágenes:** Redes Neuronales Convolucionales (CNN), operaciones de *pooling* y conexiones residuales (*ResNet*) para mitigar el desvanecimiento del gradiente en redes muy profundas.
* **Datos secuenciales y temporales:** Redes Neuronales Recurrentes (RNN), LSTM (*Long Short-Term Memory*) y GRU (*Gated Recurrent Unit*), comprendiendo sus limitaciones de paralelización y memoria a largo plazo.

### Paso 4: Las cuatro familias de la IA Generativa
1. **Autoencoders Variacionales (VAE):** Un codificador proyecta los datos hacia una distribución latente continua parametrizada ($\mu, \sigma$), aplica el truco de reparametrización ($z = \mu + \sigma \odot \epsilon$) y un decodificador reconstruye la muestra original.
2. **Redes Generativas Antagónicas (GAN):** Entrenamiento adversarial minimax entre un **Generador** (que mapea ruido aleatorio a muestras sintéticas) y un **Discriminador** (que evalúa si una muestra es real o generada).
3. **Modelos de Difusión (DDPM y *Latent Diffusion*):** Un proceso directo (*forward*) destruye gradualmente los datos añadiendo ruido gaussiano en $T$ pasos, mientras una red neuronal (típicamente U-Net o *Diffusion Transformer*) aprende a predecir y revertir ese ruido paso a paso.
4. **Transformers y Modelos Autorregresivos (LLMs):**
   * Tokenización por subpalabras (BPE, *Byte-Pair Encoding*).
   * Tablas de *embeddings* y codificación posicional (Sinusoidal o *RoPE*).
   * Mecanismo de atención escalada por producto punto (*Scaled Dot-Product Attention*):
     $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
   * *Multi-Head Attention* (MHA), *Grouped-Query Attention* (GQA), máscaras causales y bloques *Decoder-only*.

### Paso 5: Post-entrenamiento y Sistemas Compuestos
* **Ajuste fino supervisado (SFT)** y adaptación eficiente en parámetros (**LoRA / QLoRA**).
* **Alineamiento por preferencias:** Aprendizaje por refuerzo a partir de retroalimentación humana (**RLHF**) y optimización directa de preferencias (**DPO**).
* **Arquitecturas aumentadas:** Generación aumentada por recuperación (**RAG**) mediante *embeddings* y bases de datos vectoriales, junto con flujos agénticos con uso de herramientas (*tool calling*).

---

## 3. Cómo mapear redes neuronales (Diseño, rastreo e inspección)

En ingeniería de *deep learning*, "mapear" una red neuronal abarca cuatro prácticas técnicas distintas que conviene diferenciar explícitamente en tu guía:

| Nivel de Mapeo | Qué resuelve en la práctica | Herramientas y técnicas estándar |
| :--- | :--- | :--- |
| **1. Mapeo Tensorial (*Shape Tracking*)** | Verificar cómo cambian las dimensiones de los tensores capa por capa (por ejemplo, pasar de $[B, T, d_{\text{model}}]$ a $[B, H, T, d_{\text{head}}]$) para evitar errores de álgebra matricial. | Anotación dimensional explícita en comentarios de código, librería `einops` y `torchinfo.summary()`. |
| **2. Mapeo de Grafos Computacionales** | Visualizar el grafo dirigido acíclico (DAG) de operaciones y el flujo inverso de gradientes durante la llamada a `backward()`. | `torchviz` (genera grafos de *autograd* con Graphviz), `torch.fx` y TensorBoard Graphs. |
| **3. Mapeo Topológico y Diagramación** | Inspeccionar modelos compilados o exportados y generar diagramas arquitectónicos claros para documentación técnica. | **Netron** (visor interactivo para formatos `.onnx`, `.pt`, `.safetensors`), `torchview`, **PlotNeuralNet** (código LaTeX para diagramas 3D de capas) y **NN-SVG**. |
| **4. Mapeo Interno e Interpretabilidad** | Abrir la "caja negra" para observar qué patrones activa cada capa, neurona o cabeza de atención. | **BertViz** (visualización de pesos de atención en Transformers), **Captum** (atribución de características y *Grad-CAM*) y *Sparse Autoencoders* (interpretabilidad mecanística). |

---

## 4. Biblioteca Maestra de Fuentes Recomendadas

### A. Cursos y Series Magistrales (De código puro a frontera)
* **Neural Networks: Zero to Hero — Andrej Karpathy (YouTube y GitHub):**
  La referencia paso a paso más sólida disponible. Comienza construyendo `micrograd` (un motor de *autograd* y red neuronal en ~100 líneas de Python), continúa con la serie `makemore` (implementando MLP, análisis estadístico de activaciones y gradientes, `BatchNorm`, retropropagación manual tensorial y `WaveNet`) y culmina programando un modelo GPT y un tokenizador BPE desde cero.
* **3Blue1Brown — Serie *Neural Networks* y *Deep Learning* (Grant Sanderson, YouTube):**
  Indispensable para la intuición geométrica. Sus capítulos sobre descenso de gradiente, *backpropagation* y el flujo interno de los Transformers (atención, matrices $Q, K, V$ y capas MLP como memoria asociativa) son el estándar visual para explicar conceptos abstractos.
* **Stanford CS231n (*Deep Learning for Computer Vision*) y CS224n (*NLP with Deep Learning*):**
  Cursos universitarios canónicos. Sus notas de clase (*course notes*) explican con rigor quirúrgico el cálculo de gradientes vectorizados, inicialización, optimización y arquitecturas secuenciales/atencionales.
* **Stanford CS336 (*Language Modeling from Scratch*):**
  Orientado a la ingeniería moderna de LLMs a escala: diseño de arquitecturas contemporáneas (SwiGLU, RoPE, RMSNorm), leyes de escalado (*Scaling Laws*), paralelismo en GPU, curaduría de datos y post-entrenamiento.
* **Practical Deep Learning for Coders — Fast.ai (Jeremy Howard):**
  Aplica el método *top-down*: entrenar primero un modelo funcional de alto rendimiento y luego desarmar sus capas progresivamente hasta llegar a la matemática subyacente.
* **DeepLearning.AI — Short Courses (Andrew Ng junto a Anthropic, OpenAI y Hugging Face):**
  Módulos prácticos enfocados en técnicas aplicadas específicas como modelos de difusión, evaluación de sistemas, *fine-tuning* y RAG.

### B. Libros de Referencia (Teoría y Código)
* ***Understanding Deep Learning* — Simon J.D. Prince (MIT Press, disponible gratis en línea):**
  Probablemente el libro moderno más claro y mejor ilustrado del campo. Cubre desde redes superficiales hasta Transformers, VAEs, GANs y modelos de difusión con una notación y diagramación matemática uniforme.
* ***Dive into Deep Learning (`d2l.ai`)* — Aston Zhang, Zachary C. Lipton, Mu Li y Alexander J. Smola:**
  Libro interactivo gratuito donde cada ecuación viene acompañada de su implementación ejecutable en PyTorch y JAX.
* ***Build a Large Language Model (From Scratch)* — Sebastian Raschka (Manning):**
  Guía paso a paso en PyTorch que construye un modelo estilo GPT desde la tokenización y el mecanismo de atención hasta el preentrenamiento y el ajuste fino para clasificación e instrucciones.
* ***Deep Learning* — Ian Goodfellow, Yoshua Bengio y Aaron Courville (MIT Press):**
  El texto fundacional de referencia teórica para rigor matemático, optimización y modelos generativos clásicos.
* ***Mathematics for Machine Learning* — Marc Peter Deisenroth, A. Aldo Faisal y Cheng Soon Ong (PDF gratuito oficial):**
  Referencia directa para cubrir exactamente el álgebra lineal, cálculo vectorial y probabilidad necesarios.

### C. Blogs Técnicos, Guías Visuales y Documentación
* **Lil'Log (Lilian Weng):** Artículos técnicos enciclopédicos con derivaciones matemáticas limpias sobre *Attention*, *The Transformer Family*, *What are Diffusion Models?* y *LLM Powered Autonomous Agents*.
* **The Illustrated Transformer / GPT-2 / Stable Diffusion (Jay Alammar):** El estándar de referencia para aprender a **diagramar visualmente** el paso de tensores y vectores a través de cada bloque de una arquitectura generativa.
* **Transformer Circuits Thread (Chris Olah y equipo de Anthropic) y Distill.pub:** Las fuentes más profundas para entender cómo mapear circuitos internos dentro de redes neuronales y visualizar qué representan las activaciones en capas ocultas.
* **Andrej Karpathy — *"A Recipe for Training Neural Networks"*:** Ensayo práctico obligatorio sobre cómo diagnosticar, depurar y escalar redes neuronales paso a paso sin caer en errores silenciosos.
* **Ahead of AI (Sebastian Raschka) y Hugging Face Courses (NLP, Audio, Diffusion, LLM):** Excelentes para contrastar variantes arquitectónicas modernas (MoE, GQA, RoPE) y revisar implementaciones estandarizadas en código abierto.

### D. Papers Fundacionales (Bibliografía primaria para tu guía)
* **Retropropagación:** Rumelhart, Hinton & Williams (1986), *Learning representations by back-propagating errors*.
* **Estabilidad en redes profundas:** Ioffe & Szegedy (2015), *Batch Normalization*; He et al. (2015), *Deep Residual Learning for Image Recognition (ResNet)*.
* **VAEs y GANs:** Kingma & Welling (2014), *Auto-Encoding Variational Bayes*; Goodfellow et al. (2014), *Generative Adversarial Nets*.
* **Transformers:** Vaswani et al. (2017), *Attention Is All You Need*.
* **Modelos de Difusión:** Ho et al. (2020), *Denoising Diffusion Probabilistic Models (DDPM)*; Rombach et al. (2022), *High-Resolution Image Synthesis with Latent Diffusion Models*.
* **Adaptación y Alineamiento:** Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models*; Rafailov et al. (2023), *Direct Preference Optimization (DPO)*.

---

## 5. Propuesta de Índice Maestro para tu Guía

Para que tu guía tenga coherencia pedagógica de principio a fin, puedes estructurarla en siete módulos progresivos donde cada capítulo integre **intuición geométrica + formulación matemática + diagrama de dimensiones + código mínimo reproducible**:

1. **Módulo I — Cimientos:** De la neurona biológica al perceptrón matemático; álgebra de tensores y regla de la cadena sobre grafos.
2. **Módulo II — El motor de aprendizaje:** Construcción de un sistema de *autograd* desde cero; descenso de gradiente, funciones de pérdida e inicialización de pesos.
3. **Módulo III — Ingeniería de entrenamiento y mapeo:** Normalización, regularización, diagnóstico de gradientes/activaciones y rastreo de dimensiones (*shape tracking*) con herramientas de visualización (Netron, TensorBoard, `torchviz`).
4. **Módulo IV — Arquitecturas estructurales:** Redes convolucionales (visión), redes recurrentes (secuencias) y conexiones residuales.
5. **Módulo V — La revolución Transformer:** Tokenización, *embeddings*, atención escalada por producto punto, cabezas múltiples y codificación posicional.
6. **Módulo VI — Ecosistema Generativo:** Modelos autorregresivos de lenguaje (LLMs), Autoencoders Variacionales (VAE), GANs y Modelos de Difusión.
7. **Módulo VII — Adaptación, Interpretabilidad y Producción:** *Fine-tuning* eficiente (LoRA), alineamiento (DPO), sistemas RAG y técnicas para mapear e interpretar el comportamiento interno del modelo.