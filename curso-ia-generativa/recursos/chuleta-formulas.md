# Chuleta de fórmulas y *shapes*

Una página para imprimir. Notación: $B$ batch, $T$ tokens, $C$ dimensión del modelo, $H$ cabezas, $D_h=C/H$, $V$ vocabulario, $K$ clases.

## Álgebra y cálculo

| | |
|---|---|
| Producto matricial | `[a,b]@[b,c]→[a,c]` |
| Producto punto | $a\cdot b=\lVert a\rVert\lVert b\rVert\cos\theta$ |
| Regla de la cadena | $\dfrac{\partial L}{\partial x}=\dfrac{\partial L}{\partial y}\dfrac{\partial y}{\partial x}$ (los caminos **se suman**) |
| Capa densa $Z=AW+b$ | $dW=A^\top dZ,\ db=\sum_n dZ,\ dA=dZ\,W^\top$ |
| ReLU / tanh / σ | $\mathbb{1}[z>0]$ · $1-\tanh^2$ · $\sigma(1-\sigma)$ |
| Descenso de gradiente | $\theta\leftarrow\theta-\eta\nabla_\theta L$ |

## Pérdidas y probabilidad

| | |
|---|---|
| Softmax | $p_i=\dfrac{e^{z_i}}{\sum_j e^{z_j}}$ (resta $\max z$) |
| Entropía cruzada | $L=-\log p_y$ ; $\ \partial L/\partial z=p-\text{onehot}(y)$ |
| Pérdida inicial esperada | $\ln K$ |
| Perplejidad | $e^{L}$ |
| KL entre gaussianas con $\mathcal N(0,I)$ | $-\tfrac12\sum(1+\log\sigma^2-\mu^2-\sigma^2)$ |
| MLE ≡ minimizar NLL | $\max\sum\log p_\theta(x_i)$ |

## Inicialización y normalización

| | |
|---|---|
| Xavier | $\mathrm{Var}(w)=\dfrac{2}{d_{in}+d_{out}}$ (tanh, σ) |
| He/Kaiming | $\mathrm{Var}(w)=\dfrac{2}{d_{in}}$ (ReLU) |
| LayerNorm | $\gamma\dfrac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$ (por muestra) |
| RMSNorm | $g\,\dfrac{x}{\sqrt{\frac1d\sum x_i^2+\epsilon}}$ |
| BatchNorm | como LayerNorm pero $\mu,\sigma$ sobre el **batch** |

## Optimizadores

| | |
|---|---|
| Momentum | $v\leftarrow\mu v+g;\ \theta\leftarrow\theta-\eta v$ |
| Adam | $m,v$ con $\beta_1=.9,\beta_2=.999$; $\theta\leftarrow\theta-\eta\hat m/(\sqrt{\hat v}+\epsilon)$ |
| AdamW | Adam $+\ \eta\lambda\theta$ **desacoplado** ($\beta_2\!=\!.95$ en LLM) |
| Programación de lr | *warmup* lineal + decaimiento coseno |
| Update/data ratio | $\eta\,\mathrm{std}(g)/\mathrm{std}(\theta)\approx10^{-3}$ |

## Convoluciones

| | |
|---|---|
| Salida | $\lfloor (H+2p-k)/s\rfloor+1$ |
| Parámetros | $C_{out}(C_{in}k^2+1)$ |
| Residual | $h_{l+1}=h_l+F(h_l)$ |

## Recurrentes

$h_t=\tanh(W_{hh}h_{t-1}+W_{xh}x_t+b)$ · LSTM: $c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t,\ h_t=o_t\odot\tanh c_t$

## Transformer

| | |
|---|---|
| Atención | $\text{softmax}\!\big(QK^\top/\sqrt{d_k}+M\big)V$ ; scores `[B,H,T,T]` |
| $\mathrm{Var}(q\cdot k)$ | $d_k$ → por eso $/\sqrt{d_k}$ |
| Multi-head | `[B,T,C]→[B,H,T,D_h]→[B,T,C]` |
| KV-cache por token | $2\cdot N_{capas}\cdot H_{kv}\cdot D_h\cdot$ bytes |
| Bloque pre-norm | $x\leftarrow x+\text{Attn}(\text{Norm}\,x);\ x\leftarrow x+\text{MLP}(\text{Norm}\,x)$ |
| SwiGLU | $W_{down}\big(\text{SiLU}(W_g x)\odot W_u x\big)$ , hidden $\approx\tfrac83C$ |
| Parámetros | $\approx12\,N_{capas}\,C^2$ + embeddings $V\cdot C$ |
| RoPE | rota pares de dims por ángulo $m\theta_i$, $\theta_i=10000^{-2i/D_h}$ |
| Muestreo | $\text{softmax}(\text{logits}/\tau)$, top-k / top-p |
| Chinchilla | ≈ 20 tokens de entrenamiento por parámetro |
| Memoria de entrenamiento (AdamW, mixta) | ≈ 16 bytes/parámetro |

## Modelos generativos

| | |
|---|---|
| **VAE** | $\mathcal{L}=\text{Rec}+\beta\,\mathrm{KL}$ ; $z=\mu+\sigma\odot\epsilon$ |
| **GAN** | $\min_G\max_D\ \mathbb{E}\log D(x)+\mathbb{E}\log(1-D(G(z)))$ ; $D^*=\dfrac{p_{data}}{p_{data}+p_g}$ |
| **Difusión** directa | $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$ |
| Pérdida | $\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert^2$ |
| Paso DDPM | $x_{t-1}=\dfrac1{\sqrt{\alpha_t}}\Big(x_t-\dfrac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta\Big)+\sigma_tz$ |
| DDIM ($\eta=0$) | $x_{prev}=\sqrt{\bar\alpha_{prev}}\hat x_0+\sqrt{1-\bar\alpha_{prev}}\,\epsilon_\theta$ |
| CFG | $\tilde\epsilon=\epsilon_\varnothing+w(\epsilon_c-\epsilon_\varnothing)$ |

## Adaptación y alineamiento

| | |
|---|---|
| **LoRA** | $W'=W+\tfrac\alpha rBA$ ; entrenables $r(d_{in}+d_{out})$ ; $B_0=0$ |
| **DPO** | $-\log\sigma\Big(\beta\big[\log\tfrac{\pi(y_w)}{\pi_{ref}(y_w)}-\log\tfrac{\pi(y_l)}{\pi_{ref}(y_l)}\big]\Big)$ |
| **Bradley–Terry** | $P(y_w\succ y_l)=\sigma(r_w-r_l)$ |
| **RAG** | top-k por coseno → prompt (contexto+pregunta) → LLM |
| **SAE** | $\min\lVert x-D\,\text{ReLU}(Ex)\rVert^2+\lambda\lVert\text{ReLU}(Ex)\rVert_1$ |

## Diagnóstico exprés

1. Pérdida inicial $=\ln K$ → 2. sobreajusta un batch → 3. ratio update/data $\approx10^{-3}$ → 4. neuronas muertas / saturación → 5. más datos antes que más capacidad → 6. ≥ 3 semillas antes de concluir.
