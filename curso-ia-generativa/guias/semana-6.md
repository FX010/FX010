# Semana 6 · Ecosistema generativo (I): VAE y GAN

> **21 – 27 dic 2026** · ~12 h (semana de Navidad: usa el **Plan B** de 8 h si lo necesitas, ver [plan maestro](00-plan-maestro.md)) · Módulo VI de la hoja de ruta
> **Entregable:** un VAE y una GAN propios, con medición del compromiso reconstrucción/KL y de la cobertura de modos, y una comparación escrita de las cuatro familias generativas.

## 1. Objetivos

- [ ] Distinguir **discriminativo** ($P(y\mid x)$) de **generativo** ($P(x)$): qué significa “muestrear”.
- [ ] Derivar el **ELBO** del VAE, la KL cerrada entre gaussianas y el **truco de reparametrización**.
- [ ] Explicar el papel de $\beta$ (β-VAE) y el fenómeno de *posterior collapse*.
- [ ] Explicar el juego **minimax** de la GAN, el discriminador óptimo y por qué hay *mode collapse* e inestabilidad.
- [ ] Medir calidad **y** diversidad de un modelo generativo con métricas honestas.

## 2. Plan día a día (≈12 h)

| Día | Tiempo | Qué hacer |
|---|---|---|
| **Lun** | 2 h | Lee Prince cap. 14 (aprendizaje no supervisado) y 17 (VAE); Lil’Log *From Autoencoder to Beta-VAE*. Responde: ¿por qué un autoencoder normal **no** sirve para generar? |
| **Mar** | 2 h | **Derivación en papel** del ELBO (§3.2) y de la KL cerrada (§3.3). Lee el paper de Kingma & Welling (2014), secciones 2–3. |
| **Mié** | 2 h | `vae.py`: programa tu VAE sin mirar; barrido de $\beta\in\{0,1,4\}$; mira la figura de muestras. |
| **Jue** | 2 h | Goodfellow et al. (2014) *Generative Adversarial Nets* + Prince cap. 15; Lil’Log *From GAN to WGAN*. |
| **Vie** | 2 h | `gan.py`: ejecuta, mide cobertura de modos; cambia lr, batch, ruido y observa. |
| **Sáb** | 3–4 h | **Proyecto:** (a) VAE condicional o interpolación en el espacio latente de formas; **o** (b) VAE para **detección de anomalías** (error de reconstrucción) sobre datos de proceso. |
| **Dom** | 1 h | Autoevaluación y tabla comparativa de las 4 familias (§3.6). |

## 3. Teoría esencial

### 3.1 Discriminativo vs generativo

| | Aprende | Sirve para |
|---|---|---|
| **Discriminativo** | $P(y\mid x)$ o una frontera de decisión | clasificar, predecir |
| **Generativo** | $P(x)$ o $P(x,y)$ | **muestrear** datos nuevos, estimar densidades, imputar, detectar anomalías, comprimir |

Un autoencoder determinista $x\to z\to\hat x$ comprime bien, pero su espacio latente tiene “huecos”: decodificar un $z$ al azar produce basura. El VAE **obliga** al espacio latente a parecerse a una distribución conocida, $\mathcal{N}(0,I)$, de la que sí sabemos muestrear.

### 3.2 ELBO

Modelo: $z\sim p(z)=\mathcal{N}(0,I)$, $x\sim p_\theta(x\mid z)$. Queremos maximizar $\log p_\theta(x)$, intratable. Introducimos un **codificador** aproximado $q_\phi(z\mid x)$:

$$\log p_\theta(x)=\underbrace{\mathbb{E}_{q_\phi(z|x)}\big[\log p_\theta(x\mid z)\big]-D_{\mathrm{KL}}\big(q_\phi(z\mid x)\,\|\,p(z)\big)}_{\text{ELBO}}+D_{\mathrm{KL}}\big(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\big)\;\ge\;\text{ELBO}$$

(la desigualdad sale de que la última KL es ≥ 0). Maximizar el ELBO = **reconstruir bien** ($\mathbb{E}\log p(x|z)$) **manteniendo el codificador cerca del prior** (KL). Pérdida a minimizar: $\;-\text{ELBO}$.

Con datos en $[0,1]$: reconstrucción = entropía cruzada binaria (BCE) por píxel; con datos reales/gaussianos: MSE.

### 3.3 KL cerrada y reparametrización

Si $q=\mathcal{N}(\mu,\mathrm{diag}(\sigma^2))$ y $p=\mathcal{N}(0,I)$:

$$D_{\mathrm{KL}}=-\frac12\sum_{j=1}^{d}\left(1+\log\sigma_j^2-\mu_j^2-\sigma_j^2\right)$$

**Truco de reparametrización:** muestrear $z\sim\mathcal{N}(\mu,\sigma^2)$ no es diferenciable respecto a $\mu,\sigma$. Se reescribe $\;z=\mu+\sigma\odot\epsilon,\ \epsilon\sim\mathcal{N}(0,I)$: la aleatoriedad queda en $\epsilon$ (que no depende de parámetros) y el gradiente fluye por $\mu$ y $\sigma$. En código se predice `logvar` ($=\log\sigma^2$) para que $\sigma=\exp(\tfrac12\text{logvar})>0$ siempre.

### 3.4 β-VAE y *posterior collapse*

$\;\mathcal{L}=\text{Reconstrucción}+\beta\cdot\mathrm{KL}$.

- $\beta=0$: autoencoder normal (KL sin freno, espacio latente irregular).
- $\beta$ alto: latente muy regular / “desenredado”, pero reconstrucciones borrosas.
- *Posterior collapse*: con decodificadores muy potentes (p. ej. autorregresivos) el modelo ignora $z$ ($q\approx p$, KL≈0). Remedios: *KL annealing* (subir $\beta$ gradualmente), *free bits*, decodificadores más débiles.

### 3.5 GAN

Dos redes: **Generador** $G(z)$ (ruido → muestra) y **Discriminador** $D(x)$ (probabilidad de ser real):

$$\min_G\max_D\;\mathbb{E}_{x\sim p_{data}}[\log D(x)]+\mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]$$

- Para $G$ fijo, el discriminador óptimo es $D^*(x)=\dfrac{p_{data}(x)}{p_{data}(x)+p_g(x)}$ y el objetivo se reduce a $2\,\mathrm{JSD}(p_{data}\|p_g)-\log4$. El equilibrio es $p_g=p_{data}$, $D\equiv\tfrac12$.
- **Pérdida no saturante** para $G$: maximizar $\log D(G(z))$ en vez de minimizar $\log(1-D(G(z)))$ (gradientes fuertes al inicio, cuando $D$ gana fácilmente). Es la que usa `gan.py`.
- **Problemas:** *mode collapse* (el generador produce pocas variantes), oscilaciones, $D$ que gana demasiado (gradiente de $G$ nulo). **Remedios:** WGAN / WGAN-GP (distancia de Wasserstein + penalización de gradiente), normalización espectral, DCGAN (conv + BN), *two-timescale* de tasas de aprendizaje, StyleGAN para calidad fotográfica.

### 3.6 Las cuatro familias, lado a lado

| | VAE | GAN | Difusión | Autorregresivo (LLM) |
|---|---|---|---|---|
| Qué aprende | $p(x\mid z)$ + codificador | $G$ y $D$ (juego) | a **quitar ruido** paso a paso | $p(x_t\mid x_{<t})$ |
| Verosimilitud | cota (ELBO) | no hay | cota (ELBO) | **exacta** |
| Calidad de muestra | media (borrosa) | **alta** (nítida) | **muy alta** | muy alta (texto/código) |
| Diversidad | buena | riesgo de *mode collapse* | **muy buena** | muy buena |
| Estabilidad de entrenamiento | **estable** | **inestable** | estable | estable |
| Velocidad de muestreo | **1 pasada** | **1 pasada** | lenta (decenas–cientos de pasos) | token a token |
| Hoy se usa para | espacio latente (en *Latent Diffusion*), anomalías | superresolución, *GANs* rápidas, adversarial | imagen/video/audio SOTA | texto, código, multimodal |

### 3.7 Cómo se evalúa un modelo generativo (y por qué es difícil)

- **Verosimilitud / NLL / bits por dimensión**: solo en modelos con densidad (VAE→cota, AR→exacta).
- **FID** (Fréchet Inception Distance) y **IS**: distancia entre estadísticas de características de imágenes reales y generadas.
- **Precisión y recall** (calidad vs cobertura): en `gan.py` medimos *modos cubiertos* y *% de muestras cerca de un modo*.
- **Evaluación humana** y pruebas de utilidad (¿mejora un clasificador entrenado con datos sintéticos?).
- **Regla:** mide **siempre calidad y diversidad por separado**; una sola métrica se puede “hackear”.

## 4. Código de la semana

Carpeta: [`codigo/semana6_vae_gan/`](../codigo/semana6_vae_gan)

```bash
cd curso-ia-generativa/codigo/semana6_vae_gan
python vae.py      # barrido de beta + figura de muestras en codigo/salidas/
python gan.py      # GAN sobre 8 gaussianas en anillo + cobertura de modos
```

**Resultados de referencia**

*VAE* (4 000 imágenes 16×16 de 4 formas, $z\in\mathbb{R}^4$, 30 épocas):

| $\beta$ | Reconstrucción (BCE por imagen) | KL | Lectura |
|---|---|---|---|
| 0 | **5,68** | **1 465** | casi perfecto al reconstruir, pero el latente es irregular (muestrear $z\sim\mathcal{N}(0,I)$ da basura) |
| 1 | 17,39 | 9,17 | equilibrio: muestras reconocibles (cuadrados, anillos, cruces, diagonales, algo borrosas) |
| 4 | 39,20 | 3,65 | latente muy comprimido, reconstrucción borrosa |

*GAN* (8 gaussianas en un anillo, 3 000 pasos, MLP 128):

| Paso | Modos cubiertos | Muestras cerca de un modo |
|---|---|---|
| 1 000 | 8/8 | 58 % |
| 2 000 | 8/8 | 73 % |
| 3 000 | 8/8 | **80 %** |

En esta semilla no hay *mode collapse*; **no es una garantía**: cambia semilla, lr o arquitectura y aparecerá. Esa fragilidad es parte de lo que debes aprender a medir.

## 5. Ejercicios

**Básicos**
1. Calcula a mano la KL cerrada para $\mu=[1,0],\ \log\sigma^2=[0,0]$ ($=0.5$). Verifica con `perdida()`.
2. Cambia $z$ de 4 a 2 dimensiones y **grafica el espacio latente** coloreado por clase de forma.
3. Muestrea $z$ en una rejilla 2D y decodifica: ¿ves transiciones suaves entre formas?

**Intermedios**
4. **Interpolación:** codifica dos imágenes, interpola $\mu$ en 8 pasos y decodifica. Compara con interpolar píxeles directamente.
5. Implementa *KL annealing* (β de 0 a 1 en las primeras 10 épocas). ¿Cambia la KL final?
6. En `gan.py` mide *mode collapse* con 5 semillas × 3 tasas de aprendizaje y reporta una tabla.

**Reto**
7. Implementa **WGAN-GP** (crítico sin sigmoide + penalización de gradiente) sobre las 8 gaussianas y compara estabilidad.
8. **VAE para anomalías:** entrena con datos “normales” (p. ej. mediciones de un proceso simulado) y marca anomalía si el error de reconstrucción supera el percentil 99. Reporta precisión/recall con anomalías inyectadas.

## 6. Errores comunes

| Síntoma | Causa |
|---|---|
| KL ≈ 0 y reconstrucciones iguales para todo | *posterior collapse*: $\beta$ demasiado alto o decodificador demasiado potente |
| Reconstrucción perfecta pero muestras basura | $\beta$ muy bajo: el latente no se parece al prior |
| `nan` en la KL | `logvar.exp()` explota: limita `logvar` o baja lr |
| GAN: $D$ pierde 0 y $G$ no aprende | $D$ demasiado fuerte: baja su lr, añade ruido a las entradas o usa WGAN-GP |
| GAN: todas las muestras iguales | *mode collapse*: más ruido latente, *minibatch discrimination*, WGAN-GP |
| Comparar GAN “a ojo” | usa una métrica (cobertura, FID) en varias semillas |

## 7. Autoevaluación

1. ¿Por qué se necesita el truco de reparametrización?
2. ¿Qué dos términos forman la pérdida del VAE y qué fuerza opone cada uno?
3. ¿Qué es $D^*$ y a qué converge el juego?
4. ¿Por qué la pérdida “no saturante” del generador?
5. Da una ventaja y una desventaja de VAE vs GAN.

<details><summary>Respuestas</summary>

1. Para que el gradiente atraviese el muestreo: $z=\mu+\sigma\epsilon$ mueve la aleatoriedad a $\epsilon$.
2. Reconstrucción (fidelidad a los datos) vs KL (regularidad del latente / cercanía al prior).
3. $D^*=p_{data}/(p_{data}+p_g)$; a $p_g=p_{data}$ con $D\equiv1/2$.
4. $\log(1-D(G(z)))$ da gradientes casi nulos cuando $D$ rechaza con seguridad las muestras al principio; $-\log D(G(z))$ da gradientes fuertes.
5. VAE: entrenamiento estable y latente útil, pero muestras borrosas; GAN: muestras nítidas, pero inestable y con *mode collapse*.
</details>

## 8. Checklist de salida

- [ ] Derivé el ELBO y la KL cerrada en papel.
- [ ] Mi VAE genera muestras reconocibles y reporto reconstrucción **y** KL.
- [ ] Mido la cobertura de modos de mi GAN en varias semillas.
- [ ] Puedo llenar la tabla de las 4 familias sin mirarla.
- [ ] Definí qué proyecto final (§ [proyecto](../proyecto/proyecto-final.md)) voy a hacer.

## 9. Recursos de la semana

- Prince, *Understanding Deep Learning*: cap. 14 (no supervisado), 15 (GAN), 17 (VAE).
- Lil’Log: *From Autoencoder to Beta-VAE*, *From GAN to WGAN*.
- Kingma & Welling, *An Introduction to Variational Autoencoders* (arXiv 1906.02691).
- Papers: Kingma & Welling (2014) *Auto-Encoding Variational Bayes* (arXiv 1312.6114) · Goodfellow et al. (2014) *Generative Adversarial Nets* (arXiv 1406.2661) · Radford et al. (2015) *DCGAN* (arXiv 1511.06434) · Arjovsky et al. (2017) *WGAN* (arXiv 1701.07875) · Gulrajani et al. (2017) *WGAN-GP* (arXiv 1704.00028) · Heusel et al. (2017) *FID* (arXiv 1706.08500).
- CS231n: clase de *Generative Models*.

[← Semana 5](semana-5.md) · **Siguiente:** [Semana 7 →](semana-7.md)
