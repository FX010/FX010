# Biblioteca maestra de fuentes

Organizada **por semana** (qué consultar cuándo) y luego por tipo. Está basada en tu hoja de ruta original y ampliada con los capítulos concretos.

> ⚠️ **Aviso sobre los enlaces:** el entorno donde se preparó este curso no tenía acceso a internet general, así que **no pude verificar que cada URL siga vigente**. Los dominios son los canónicos conocidos de cada recurso; si alguno falla, busca el título exacto. Los identificadores de arXiv sí son los que corresponden a cada paper.

## 1. Mapa rápido: qué usar en cada semana

| Sem. | Video / curso | Lectura / libro | Paper / blog |
|---|---|---|---|
| **1** | 3Blue1Brown cap. 1–4 · Karpathy *micrograd* | MML cap. 2 y 5 · Prince cap. 1–4 | Rumelhart, Hinton & Williams (1986) |
| **2** | Karpathy *makemore* 1–2 | Prince cap. 5–8 · d2l MLP y optimización · CS231n NN 1–3 | He (init) · Adam · AdamW |
| **3** | Karpathy *makemore* 3–4 | Karpathy *Recipe* · CS231n NN 2–3 | BatchNorm · LayerNorm · RMSNorm |
| **4** | CS231n (CNN) · CS224n (RNN) | Prince cap. 10–11 · Olah *LSTM* | ResNet · LSTM · GRU |
| **5** | 3B1B cap. 5–7 · Karpathy *GPT* y *Tokenizer* | Alammar · Raschka caps. 2–5 · Lil’Log *Attention* | *Attention Is All You Need* · RoPE · GQA · BPE |
| **6** | CS231n *Generative Models* | Prince cap. 14, 15, 17 · Lil’Log VAE y GAN | VAE · GAN · WGAN(-GP) · FID |
| **7** | DeepLearning.AI *How Diffusion Models Work* · Karpathy *GPT-2* · CS336 | Lil’Log *Diffusion* · Prince cap. 18 · Alammar *Stable Diffusion* | DDPM · DDIM · CFG · Latent Diffusion · DiT · Flow Matching |
| **8** | HF *LLM Course* · DeepLearning.AI short courses | Raschka (fine-tuning) · Transformer Circuits | LoRA · QLoRA · InstructGPT · DPO · RAG |

## 2. Cursos y series (de código puro a frontera)

| Recurso | Qué aporta | Dónde |
|---|---|---|
| **Neural Networks: Zero to Hero** — Andrej Karpathy | La referencia paso a paso: `micrograd` → `makemore` (MLP, BatchNorm, backprop manual, WaveNet) → GPT → tokenizador BPE → reproducir GPT-2 | karpathy.ai/zero-to-hero.html · YouTube y GitHub (`micrograd`, `makemore`, `nanoGPT`, `minbpe`) |
| **Neural Networks** (serie) — 3Blue1Brown | Intuición geométrica: gradiente, backprop, GPT, atención, MLP como memoria | YouTube, canal 3Blue1Brown |
| **CS231n** *Deep Learning for Computer Vision* | Notas de clase con rigor: gradientes vectorizados, init, optimización, CNN | cs231n.github.io |
| **CS224n** *NLP with Deep Learning* | RNN, atención, Transformers, LLM (clases y notas) | web.stanford.edu/class/cs224n |
| **CS336** *Language Modeling from Scratch* | LLM a escala: arquitectura moderna, escalado, GPU/paralelismo, datos, alineamiento | sitio del curso en Stanford (busca “Stanford CS336”) |
| **Practical Deep Learning for Coders** — fast.ai | Método *top-down*: primero un modelo que funciona, luego se abre | course.fast.ai |
| **DeepLearning.AI Short Courses** | Prácticos y cortos: difusión, fine-tuning, RAG, evaluación | deeplearning.ai/short-courses |
| **Hugging Face Courses** | LLM, NLP, difusión, audio; librerías `transformers`, `diffusers`, `peft`, `trl` | huggingface.co/learn |

## 3. Libros

| Libro | Para qué |
|---|---|
| ***Understanding Deep Learning*** — Simon J.D. Prince (MIT Press, gratis en línea) | **Tu libro principal.** Cap. 1–4 fundamentos · 5–8 entrenamiento · 10 conv · 11 residuales · 12 Transformers · 14–18 modelos generativos. Notación y figuras uniformes. udlbook.github.io/udlbook |
| ***Dive into Deep Learning*** — Zhang, Lipton, Li, Smola | Cada ecuación con código ejecutable (PyTorch, JAX…). d2l.ai |
| ***Build a Large Language Model (From Scratch)*** — Sebastian Raschka (Manning) | De tokenización y atención a preentrenamiento y *fine-tuning*; repo `LLMs-from-scratch` |
| ***Deep Learning*** — Goodfellow, Bengio, Courville (MIT Press) | Rigor teórico, optimización, modelos generativos clásicos. deeplearningbook.org |
| ***Mathematics for Machine Learning*** — Deisenroth, Faisal, Ong (PDF gratis) | Álgebra lineal, cálculo vectorial, probabilidad justo para este curso. mml-book.com |

## 4. Blogs, guías visuales y documentación

| Recurso | Cuándo |
|---|---|
| **Lil’Log** (Lilian Weng): *Attention? Attention!*, *The Transformer Family*, *From Autoencoder to Beta-VAE*, *From GAN to WGAN*, *What are Diffusion Models?*, *LLM Powered Autonomous Agents* — lilianweng.github.io | S5–S8 |
| **Jay Alammar**: *The Illustrated Transformer / GPT-2 / Stable Diffusion* — jalammar.github.io | S5, S7 |
| **Chris Olah**: *Understanding LSTM Networks* — colah.github.io | S4 |
| **Transformer Circuits Thread** (Anthropic) y **Distill.pub** — transformer-circuits.pub · distill.pub | S8 |
| **Karpathy**: *A Recipe for Training Neural Networks* — karpathy.github.io/2019/04/25/recipe | S3 |
| **Ahead of AI** (Raschka) — magazine.sebastianraschka.com | variantes modernas (MoE, GQA, RoPE) |

## 5. Herramientas de mapeo (ver [`mapeo-redes.md`](mapeo-redes.md))

Netron (netron.app) · `torchinfo` · `torchviz` · `torchview` · PlotNeuralNet · NN-SVG (alexlenail.me/NN-SVG) · BertViz · Captum (captum.ai) · TensorBoard.

## 6. Papers fundacionales (con arXiv)

| Tema | Paper | arXiv |
|---|---|---|
| Retropropagación | Rumelhart, Hinton & Williams (1986), *Learning representations by back-propagating errors* | — (Nature) |
| Inicialización | He et al. (2015), *Delving Deep into Rectifiers* | 1502.01852 |
| Optimización | Kingma & Ba (2014), *Adam* · Loshchilov & Hutter (2017), *Decoupled Weight Decay* | 1412.6980 · 1711.05101 |
| Normalización | Ioffe & Szegedy (2015), *Batch Normalization* · Ba et al. (2016), *Layer Norm* · Zhang & Sennrich (2019), *RMSNorm* | 1502.03167 · 1607.06450 · 1910.07467 |
| Residuales | He et al. (2015), *Deep Residual Learning* | 1512.03385 |
| Recurrentes | Hochreiter & Schmidhuber (1997), *LSTM* · Cho et al. (2014), *GRU* | — · 1406.1078 |
| Transformers | Vaswani et al. (2017), *Attention Is All You Need* | 1706.03762 |
| Posiciones y atención | Su et al. (2021), *RoFormer (RoPE)* · Ainslie et al. (2023), *GQA* | 2104.09864 · 2305.13245 |
| Tokenización | Sennrich et al. (2015), *Neural Machine Translation of Rare Words with Subword Units (BPE)* | 1508.07909 |
| Escalado | Kaplan et al. (2020), *Scaling Laws* · Hoffmann et al. (2022), *Chinchilla* | 2001.08361 · 2203.15556 |
| VAE | Kingma & Welling (2014), *Auto-Encoding Variational Bayes* · *An Introduction to VAEs* | 1312.6114 · 1906.02691 |
| GAN | Goodfellow et al. (2014) · Radford et al. (2015), *DCGAN* · Arjovsky et al. (2017), *WGAN* · Gulrajani et al. (2017), *WGAN-GP* | 1406.2661 · 1511.06434 · 1701.07875 · 1704.00028 |
| Evaluación generativa | Heusel et al. (2017), *FID* | 1706.08500 |
| Difusión | Ho et al. (2020), *DDPM* · Song et al. (2020), *DDIM* · Nichol & Dhariwal (2021), *Improved DDPM* | 2006.11239 · 2010.02502 · 2102.09672 |
| Guía y latentes | Ho & Salimans (2022), *Classifier-Free Guidance* · Rombach et al. (2022), *Latent Diffusion* · Peebles & Xie (2022), *DiT* · Lipman et al. (2022), *Flow Matching* | 2207.12598 · 2112.10752 · 2212.09748 · 2210.02747 |
| Adaptación | Hu et al. (2021), *LoRA* · Dettmers et al. (2023), *QLoRA* | 2106.09685 · 2305.14314 |
| Alineamiento | Ouyang et al. (2022), *InstructGPT* · Rafailov et al. (2023), *DPO* | 2203.02155 · 2305.18290 |
| RAG | Lewis et al. (2020), *Retrieval-Augmented Generation* | 2005.11401 |
| Interpretabilidad | Elhage et al. (2021), *A Mathematical Framework for Transformer Circuits* (transformer-circuits.pub) | — |

## 7. Cómo leer un paper en 30 minutos

1. **Título + abstract + figura 1** (5 min): ¿qué problema y qué resultado?
2. **Introducción y conclusiones** (5 min): la motivación y las limitaciones que dicen los autores.
3. **Método** (10 min): reescribe la ecuación central con tus palabras; dibuja el diagrama de dimensiones.
4. **Experimentos** (5 min): ¿contra qué comparan? ¿con cuántas semillas? ¿qué falta?
5. **Una pregunta propia** (5 min): ¿cómo lo reproducirías a pequeña escala? (En este curso casi siempre se puede.)
