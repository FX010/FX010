# Semana 7 · Ecosistema generativo (II): difusión y LLM a escala

> **28 dic 2026 – 3 ene 2027** · ~12 h (semana de Año Nuevo: **Plan B** de 8 h disponible) · Módulo VI de la hoja de ruta (cierre)
> **Entregable:** un modelo de difusión (DDPM + DDIM) propio sobre datos 2D y sobre imágenes, y un documento de “cómo se entrena un LLM de verdad” con cifras.

## 1. Objetivos

- [ ] Derivar el **proceso directo** en forma cerrada y la **pérdida simplificada** $\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert^2$.
- [ ] Programar entrenamiento y muestreo **DDPM** y **DDIM**, y explicar el compromiso pasos/calidad.
- [ ] Explicar **U-Net / DiT**, **condicionamiento temporal**, **classifier-free guidance** y **Latent Diffusion** (Stable Diffusion).
- [ ] Describir el *pipeline* de **entrenamiento de un LLM a escala**: datos, precisión mixta, paralelismo, *scaling laws*, estabilidad.
- [ ] Decidir, con criterio, entre VAE / GAN / difusión / autorregresivo para un problema dado.

## 2. Plan día a día (≈12 h)

| Día | Tiempo | Qué hacer |
|---|---|---|
| **Lun** | 2 h | Lil’Log *What are Diffusion Models?* + Prince cap. 18. Mira el paper DDPM (Ho et al., 2020) §2–3. |
| **Mar** | 2 h | **Papel:** deriva $q(x_t\mid x_0)$ (§3.1) y la media posterior (§3.2). Comprueba con los números de `ab` en `ddpm_2d.py`. |
| **Mié** | 2 h | `ddpm_2d.py`: reprograma el entrenamiento y el muestreo **sin mirar**; compara DDPM (200 pasos) y DDIM (20 pasos). |
| **Jue** | 2 h | `ddpm_imagenes.py`: lee la mini U-Net (mapa de dimensiones en el docstring); entrena y observa las muestras; cambia el nº de pasos. |
| **Vie** | 2 h | **LLM a escala:** Karpathy *Let’s reproduce GPT-2 (124M)* (primera mitad) **o** Stanford CS336 (clases de arquitectura, escalado y sistemas). Completa la tabla §3.6. |
| **Sáb** | 3–4 h | **Proyecto:** (a) *classifier-free guidance* en 2D condicional (ejercicio 6) **o** (b) difusión sobre tus propios datos; **y** avanza el proyecto final. |
| **Dom** | 1 h | Autoevaluación y planificación de la semana 8 (cierre). |

## 3. Teoría esencial

### 3.1 Proceso directo (añadir ruido)

Con un calendario de varianzas $\beta_1,\dots,\beta_T$ pequeñas y $\alpha_t=1-\beta_t$, $\bar\alpha_t=\prod_{s\le t}\alpha_s$:

$$q(x_t\mid x_{t-1})=\mathcal{N}\big(\sqrt{1-\beta_t}\,x_{t-1},\,\beta_t I\big)\ \Longrightarrow\ \boxed{x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon},\quad\epsilon\sim\mathcal{N}(0,I)$$

La fórmula cerrada (suma de gaussianas independientes) permite saltar **directamente** a cualquier $t$ sin simular los pasos intermedios: por eso el entrenamiento es barato. Con el calendario de `ddpm_2d.py` ($T=200$, $\beta$ de $10^{-4}$ a $0.06$): $\bar\alpha_T\approx0.0022$, es decir $x_T$ contiene solo un 4,7 % de señal ($\sqrt{\bar\alpha_T}$): prácticamente ruido puro, que es lo que necesita el muestreo.

### 3.2 Proceso inverso y pérdida

El inverso verdadero $q(x_{t-1}\mid x_t,x_0)$ es gaussiano con media

$$\tilde\mu_t=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\Big).$$

Una red $\epsilon_\theta(x_t,t)$ aprende a **predecir el ruido** $\epsilon$ que se añadió; sustituyéndolo en $\tilde\mu_t$ se obtiene el paso de muestreo. El ELBO se reduce, con pesos ignorados (Ho et al.), a:

$$\mathcal{L}_{\text{simple}}=\mathbb{E}_{x_0,\,t\sim U\{1..T\},\,\epsilon}\Big[\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert^2\Big]$$

**Algoritmo de entrenamiento (5 líneas):** muestrea $x_0$; muestrea $t$ y $\epsilon$; calcula $x_t$; mide MSE entre $\epsilon$ y la predicción; descenso de gradiente.
**Muestreo DDPM:** parte de $x_T\sim\mathcal{N}(0,I)$ y para $t=T..1$: $\;x_{t-1}=\tilde\mu_t(x_t,\epsilon_\theta)+\sigma_t z$.

### 3.3 Equivalencias que conviene conocer

- Predecir $\epsilon$, predecir $x_0$ o predecir la *velocidad* $v$ son reparametrizaciones equivalentes; difieren en pesos de la pérdida y estabilidad.
- **Score matching:** $\epsilon_\theta\propto-\nabla_{x_t}\log p_t(x_t)$ (el *score*): difusión = aprender el gradiente de la log-densidad a distintos niveles de ruido. Sirve de puente a las formulaciones SDE y a **flow matching** (trayectorias más rectas, menos pasos).
- **DDIM:** muestreo **determinista** (sin ruido inyectado) con la misma red; permite saltarse pasos (p. ej. 20 de 200). En `ddpm_2d.py`: $x_{t_{prev}}=\sqrt{\bar\alpha_{prev}}\hat x_0+\sqrt{1-\bar\alpha_{prev}}\,\epsilon_\theta$.

### 3.4 Arquitecturas y condicionamiento

- **U-Net:** codificador (reduce resolución) → cuello → decodificador (la restaura) con **conexiones de salto**. Mantiene detalle fino (skips) y contexto global (cuello). Bloques residuales + normalización + (en versiones grandes) atención.
- **Condicionamiento temporal:** $t$ se convierte en un *embedding sinusoidal* (idéntico a las posiciones del Transformer) y se **inyecta** en cada bloque (suma o modulación tipo AdaLN/FiLM).
- **DiT** (*Diffusion Transformer*): sustituye la U-Net por un Transformer sobre *parches* de la imagen; escala mejor.
- **Condicionamiento por texto:** un codificador de texto (CLIP/T5) produce embeddings que entran por **atención cruzada** (Q de la imagen, K/V del texto).
- **Classifier-free guidance (CFG):** se entrena con la etiqueta/texto a veces **omitido** (p. ej. 10 %); al muestrear se extrapola
 $\;\tilde\epsilon=\epsilon_\theta(x_t,\varnothing)+w\big(\epsilon_\theta(x_t,c)-\epsilon_\theta(x_t,\varnothing)\big)$, con $w>1$ → más fidelidad al condicionante, menos diversidad.

### 3.5 Latent Diffusion (Stable Diffusion)

Difundir directamente en píxeles es caro. **Latent Diffusion** (Rombach et al., 2022): (1) un **autoencoder** (tipo VAE; Semana 6) comprime la imagen (p. ej. $512^2\times3\to64^2\times4$, ~48× menos elementos); (2) la difusión ocurre en ese **espacio latente**; (3) el decodificador devuelve a píxeles. Es la síntesis de lo que viste: **VAE + difusión + atención + CFG**.

### 3.6 Entrenar un LLM de verdad (de tu GPT de 335 k a uno de miles de millones)

| Aspecto | Qué cambia a escala |
|---|---|
| **Datos** | web filtrada y deduplicada, código, libros; mezclas por calidad; tokenizador BPE grande. Cantidad ≈ 20 tokens/parámetro (óptimo Chinchilla), a menudo mucho más para modelos de inferencia barata |
| **Precisión** | `bf16`/`fp16` (precisión mixta) con *master weights* en fp32; `torch.compile`; FlashAttention |
| **Memoria** | pesos + gradientes + estados de Adam ≈ 16 B/parámetro en precisión mixta (7 B → ~112 GB) → no cabe en una GPU |
| **Paralelismo** | datos (DP/DDP) · **FSDP/ZeRO** (partir estados) · tensorial (TP) · pipeline (PP) · expertos (MoE) |
| **Lote efectivo** | acumulación de gradiente; lotes de ~0,5–4 M tokens |
| **Optimización** | AdamW ($\beta_2\approx0.95$), *warmup* + decaimiento coseno, recorte de gradiente 1.0, *weight decay* 0.1 |
| **Estabilidad** | picos de pérdida, *z-loss*, QK-norm, reinicio desde checkpoint, monitoreo continuo |
| **Evaluación** | pérdida en validación, *benchmarks* (MMLU, etc.), evaluación humana y de seguridad |
| **Post-entrenamiento** | SFT → alineamiento (RLHF/DPO) → (Semana 8) |

Regla mental: tu `entrenar_gpt.py` **es** el bucle de preentrenamiento; todo lo demás es ingeniería para que quepa y no se rompa.

## 4. Código de la semana

Carpeta: [`codigo/semana7_difusion/`](../codigo/semana7_difusion)

```bash
cd curso-ia-generativa/codigo/semana7_difusion
python ddpm_2d.py          # ≈1 min: DDPM y DDIM sobre una espiral 2D (+ figura)
python ddpm_imagenes.py    # ≈3 min: mini U-Net sobre formas 16×16 (+ figura)
```

| Archivo | Contenido |
|---|---|
| `ddpm_2d.py` | calendario de $\beta$, `ruidar`, red $\epsilon_\theta$ con bloques residuales + embedding de tiempo, muestreo DDPM y DDIM, métricas |
| `ddpm_imagenes.py` | `MiniUNet` (197 217 parámetros) con skip, GroupNorm y tiempo inyectado; muestreo DDPM |

**Resultados de referencia**

*Espiral 2D* (4 096 puntos; red de 4 bloques residuales; 4 000 pasos). Precisión = % de muestras a < 0,08 de algún dato real; cobertura = % de datos con una muestra a < 0,08:

| Muestras | Precisión (calidad) | Cobertura (diversidad) |
|---|---|---|
| Línea base: gaussiana ancha | 39,9 % | 95,2 % |
| **DDPM (200 pasos)** | **99,3 %** | **99,9 %** |
| **DDIM (20 pasos)** | 95,0 % | 98,9 % |

DDIM con **10× menos pasos** pierde ~4 puntos de precisión (en la figura se ven puntos sueltos entre los brazos de la espiral). La pérdida (MSE del ruido) baja de 1,37 a ≈ 0,24 y se estanca: **es normal**, porque predecir ruido en niveles muy altos de $t$ es intrínsecamente impredecible; la calidad real se mide por las muestras, no por esa pérdida.

*Imágenes de formas* (4 000 imágenes, 1 500 pasos, batch 64, ≈ 3 min en CPU): la pérdida baja de 1,13 a ≈ 0,024 y las muestras muestran **cuadrados, anillos, diagonales y cruces reconocibles pero con defectos** (varias formas incompletas o fragmentadas). Es lo esperable de un entrenamiento tan corto con una red de 197 k parámetros; con 10–20× más pasos (ejercicio 5) mejora claramente. Es un ejemplo honesto: la difusión funciona, pero necesita cómputo.

## 5. Ejercicios

**Básicos**
1. Calcula $\bar\alpha_t$ para $t=1,50,100,200$ con el calendario del script y grafica $\sqrt{\bar\alpha_t}$ y $\sqrt{1-\bar\alpha_t}$.
2. Visualiza el proceso directo: toma 500 puntos de la espiral y dibuja $x_t$ para $t\in\{0,25,50,100,200\}$.
3. Cambia el calendario lineal por uno **coseno** ($\bar\alpha_t=\cos^2(\tfrac{t/T+s}{1+s}\tfrac\pi2)$) y compara la precisión de DDPM.

**Intermedios**
4. Varía los pasos de DDIM (5, 10, 20, 50, 100) y grafica **precisión vs pasos**.
5. Entrena `ddpm_imagenes.py` con 10 000+ pasos (o `c=64`) y compara las muestras. Calcula una métrica simple (p. ej. fracción de muestras que un clasificador entrenado en las formas reconoce como “válidas”).
6. **Classifier-free guidance en 2D:** genera dos nubes de puntos (clases) y entrena $\epsilon_\theta(x_t,t,c)$ con la etiqueta omitida el 10 % de las veces. Muestrea con $w\in\{0,1,3,7\}$ y mide pureza de clase vs diversidad.

**Reto**
7. Implementa **flow matching** en 2D (regresión de la velocidad entre un punto de ruido y uno de datos con interpolación lineal) y compáralo con DDIM en pasos necesarios.
8. Usa **Latent Diffusion** de la librería `diffusers` (Hugging Face) para generar imágenes con un modelo abierto y localiza en el código la U-Net, el VAE y el codificador de texto.

## 6. Errores comunes

| Síntoma | Causa |
|---|---|
| Muestras = ruido | calendario con $\bar\alpha_T$ demasiado alto (no llega a ruido puro), o falta inyectar $t$ |
| Muestras “quemadas” / valores enormes | falta `clamp` de $\hat x_0$ en DDIM o escala de datos fuera de $[-1,1]$ |
| La pérdida no baja de ≈ 0,2–0,25 en 2D | **normal** (ver §4); evalúa con muestras |
| Entrenar muy lento | muestrear $t$ siempre grande; usa $t\sim U\{1..T\}$ |
| DDPM y DDIM dan resultados incompatibles | la red debe ser **la misma**; solo cambia el muestreador |
| Memoria explota en difusión de imágenes | difunde en latentes o reduce resolución/ancho |

## 7. Autoevaluación

1. Escribe $x_t$ en función de $x_0$ y $\epsilon$. ¿Por qué esto hace barato el entrenamiento?
2. ¿Qué predice la red y qué minimiza?
3. ¿Qué diferencia hay entre DDPM y DDIM?
4. ¿Cómo funciona classifier-free guidance?
5. ¿Qué aporta el VAE en Stable Diffusion?
6. Un modelo de 7 000 M de parámetros con AdamW en bf16: ¿cuánta memoria de pesos+gradientes+estados necesitas aprox.?

<details><summary>Respuestas</summary>

1. $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$; permite ir a cualquier $t$ en un solo paso, sin simular la cadena.
2. El ruido $\epsilon$ añadido; minimiza el MSE $\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert^2$.
3. DDPM es estocástico (inyecta ruido en cada paso, ~$T$ pasos); DDIM es determinista y admite saltar pasos con la misma red entrenada.
4. Se entrena con y sin condicionante; al muestrear se extrapola desde la predicción incondicional hacia la condicional con un peso $w>1$.
5. Comprime la imagen a un latente pequeño donde ocurre la difusión (mucho más barato) y la reconstruye después.
6. ≈ 16 B × 7 000 M ≈ **112 GB** (por eso se necesita paralelismo/FSDP o *offloading*).
</details>

## 8. Checklist de salida

- [ ] Derivé $q(x_t\mid x_0)$ y entiendo de dónde sale la pérdida de ruido.
- [ ] Mi DDPM/DDIM sobre 2D alcanza > 95 % de precisión.
- [ ] Sé explicar U-Net, condicionamiento temporal, CFG y Latent Diffusion.
- [ ] Tengo la tabla “LLM a escala” completa con cifras que entiendo.
- [ ] Mi proyecto final tiene un primer prototipo funcionando.

## 9. Recursos de la semana

- Lil’Log: *What are Diffusion Models?* · Jay Alammar: *The Illustrated Stable Diffusion*.
- Prince, *Understanding Deep Learning*, cap. 18 (modelos de difusión).
- Hugging Face *Diffusion Models Course* y la librería `diffusers`.
- DeepLearning.AI *Short Courses*: *How Diffusion Models Work*.
- Karpathy *Let’s reproduce GPT-2 (124M)* · Stanford **CS336** (arquitectura, escalado, GPUs/paralelismo, datos).
- Papers: Ho et al. (2020) *DDPM* (arXiv 2006.11239) · Song et al. (2020) *DDIM* (arXiv 2010.02502) · Nichol & Dhariwal (2021) *Improved DDPM* (arXiv 2102.09672) · Ho & Salimans (2022) *Classifier-Free Guidance* (arXiv 2207.12598) · Rombach et al. (2022) *Latent Diffusion* (arXiv 2112.10752) · Peebles & Xie (2022) *DiT* (arXiv 2212.09748) · Lipman et al. (2022) *Flow Matching* (arXiv 2210.02747).

[← Semana 6](semana-6.md) · **Siguiente:** [Semana 8 →](semana-8.md)
