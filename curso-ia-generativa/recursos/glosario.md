# Glosario (español ↔ inglés)

Los términos están en orden alfabético por la forma en español. Úsalo al leer papers y documentación, casi siempre en inglés.

| Español | English | Definición breve |
|---|---|---|
| **Activación** (función de) | *activation function* | No linealidad aplicada tras una capa lineal (ReLU, GELU, tanh…) |
| **Adaptador** | *adapter* | Módulo pequeño entrenable sobre un modelo congelado (p. ej. LoRA) |
| **Agente** | *agent* | LLM que decide acciones y llama herramientas en bucle |
| **Alineamiento** | *alignment* | Ajustar un modelo a preferencias/valores humanos (RLHF, DPO) |
| **Atención** | *attention* | Promedio ponderado de valores según la afinidad consulta–clave |
| **Autodiff / autograd** | *automatic differentiation* | Cálculo automático de derivadas con la regla de la cadena |
| **Autoencoder variacional (VAE)** | *variational autoencoder* | Codificador-decodificador con latente gaussiana regularizada por KL |
| **Autorregresivo** | *autoregressive* | Modela $p(x_t\mid x_{<t})$ y genera token a token |
| **Backpropagation** | *backpropagation* | Autodiff en modo inverso aplicado a redes |
| **Batch / lote** | *batch* | Conjunto de ejemplos procesados a la vez |
| **Cabeza (de atención)** | *attention head* | Un subespacio de atención con sus propios $Q,K,V$ |
| **Campo receptivo** | *receptive field* | Región de la entrada que influye en una neurona |
| **Colapso de modos** | *mode collapse* | La GAN genera pocas variantes de los datos |
| **Cuantización** | *quantization* | Representar pesos con menos bits (int8/int4) |
| **Descenso de gradiente** | *gradient descent* | Actualizar parámetros en contra del gradiente |
| **Difusión** | *diffusion* | Generar quitando ruido paso a paso |
| **Discriminativo / generativo** | *discriminative / generative* | $P(y\mid x)$ vs $P(x)$ |
| **Embedding** | *embedding* | Vector denso que representa un token, imagen o texto |
| **Entropía cruzada** | *cross-entropy* | $-\sum p\log q$; pérdida estándar de clasificación y LM |
| **Espacio latente** | *latent space* | Representación comprimida donde “vive” la muestra |
| **Fine-tuning / ajuste fino** | *fine-tuning* | Continuar entrenando un modelo preentrenado |
| **Gradient checking** | *gradient checking* | Validar gradientes contra diferencias finitas |
| **Grafo computacional** | *computational graph* | DAG de operaciones; base del autodiff |
| **Inicialización** | *initialization* | Valores iniciales de los pesos (Xavier, He) |
| **KV-cache** | *KV cache* | Claves/valores previos guardados para generar más rápido |
| **Logit** | *logit* | Salida sin normalizar antes del softmax |
| **LoRA** | *low-rank adaptation* | $W+\frac\alpha rBA$ con $W$ congelado |
| **Máscara causal** | *causal mask* | Impide que un token vea tokens futuros |
| **Normalización de capa** | *layer normalization* | Normaliza las features de cada muestra |
| **Pérdida / función de pérdida** | *loss* | Número que el entrenamiento minimiza |
| **Perplejidad** | *perplexity* | $e^{\text{NLL}}$; “cuántas opciones equiprobables” |
| **Preentrenamiento** | *pretraining* | Entrenamiento masivo inicial sobre datos generales |
| **Prompt** | *prompt* | Entrada de texto que condiciona al modelo |
| **RAG** | *retrieval-augmented generation* | Recuperar documentos y dárselos al LLM como contexto |
| **Regularización** | *regularization* | Técnicas contra el sobreajuste (dropout, weight decay) |
| **Reparametrización** (truco de) | *reparameterization trick* | $z=\mu+\sigma\epsilon$: hace diferenciable el muestreo |
| **Residual** (conexión) | *residual / skip connection* | $x+F(x)$ |
| **RLHF** | *RL from human feedback* | Alinear con recompensa aprendida de preferencias |
| **RoPE** | *rotary position embedding* | Posición por rotación de pares de dimensiones |
| **Sesgo inductivo** | *inductive bias* | Suposiciones de la arquitectura (localidad en CNN) |
| **Sobreajuste** | *overfitting* | Buen rendimiento en entrenamiento, malo en datos nuevos |
| **Temperatura** | *temperature* | Escala los logits antes de muestrear |
| **Tensor** | *tensor* | Arreglo $N$-dimensional |
| **Token / tokenizador** | *token / tokenizer* | Unidad de texto / función texto→ids (BPE) |
| **Transformer** | *transformer* | Arquitectura basada en atención y residuales |
| **U-Net** | *U-Net* | Codificador-decodificador con saltos; típico en difusión |
| **Vector gradiente / jacobiana** | *gradient / Jacobian* | Derivadas parciales de escalar / de vector a vector |
| **Weight decay** | *weight decay* | Penalización de pesos grandes (desacoplada en AdamW) |
