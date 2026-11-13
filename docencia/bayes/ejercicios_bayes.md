---
title: "Teorema de Bayes: 4 ejercicios de desarrollo con pauta"
subtitle: "Referencia: Lind, Marchal y Wathen, *Estadística aplicada a los negocios y la economía*, cap. 5, sección 5.7 (pp. 167-171)"
---

# Notación y fórmula (como en el libro)

Si $A_1, A_2, \dots, A_n$ son eventos **mutuamente excluyentes y colectivamente exhaustivos** y $B$ es la información observada:

$$P(A_i \mid B) = \frac{P(A_i)\,P(B \mid A_i)}{P(A_1)P(B \mid A_1) + P(A_2)P(B \mid A_2) + \dots + P(A_n)P(B \mid A_n)}$$

- $P(A_i)$: probabilidad **a priori** (antes de observar los datos).
- $P(B \mid A_i)$: probabilidad **condicional** (verosimilitud).
- $P(A_i \text{ y } B) = P(A_i)P(B \mid A_i)$: probabilidad **conjunta**.
- $P(A_i \mid B)$: probabilidad **a posteriori** (revisada con la nueva información).

Los ejercicios avanzan en dificultad:

| Ej. | Tema | Estructura | Referencia del libro |
|---|---|---|---|
| 1 | Detección de fraude | 2 eventos, prueba imperfecta | Ejemplo de Umen (p. 167) |
| 2 | Couriers y retrasos | 3 eventos, árbol y tabla | Ejemplo chips LS-24 (pp. 168-170), ej. 37 |
| 3 | Control de calidad | Actualización secuencial | Extensión del ej. 33-34 |
| 4 | Lanzamiento de producto | Bayes + decisión (valor esperado) | Integra 5.7 con valor esperado |

\newpage

# ENUNCIADOS

## Ejercicio 1 — Alertas de fraude en una billetera digital (16 puntos)

Una fintech chilena procesa pagos con su billetera digital. Según su historial, el **2%** de las transacciones son fraudulentas. Para detectarlas, la empresa usa un sistema automático de alertas con el siguiente desempeño:

- Si la transacción **es fraudulenta**, el sistema emite una alerta el **95%** de las veces.
- Si la transacción **es legítima**, el sistema igualmente emite una alerta (falso positivo) el **4%** de las veces.

Se selecciona al azar una transacción.

a) Defina los eventos y escriba con notación de probabilidad los datos del enunciado. Identifique cuál es la probabilidad a priori y cuáles son las condicionales. **(3 pts)**

b) Construya el diagrama de árbol y calcule la probabilidad de que el sistema emita una alerta. **(4 pts)**

c) El sistema emitió una alerta. ¿Cuál es la probabilidad de que la transacción sea realmente fraudulenta? Resuma sus cálculos en una tabla de Bayes (a priori, condicional, conjunta, a posteriori). **(4 pts)**

d) El sistema **no** emitió alerta. ¿Cuál es la probabilidad de que la transacción sea fraudulenta igualmente? **(2 pts)**

e) El gerente de riesgo propone invertir en mejorar el sistema para bajar los falsos positivos de 4% a 1% (manteniendo el 95% de detección). Recalcule la probabilidad de la pregunta c) y comente, en no más de cuatro líneas, por qué la mejora tiene un efecto tan grande. **(3 pts)**

\newpage

## Ejercicio 2 — ¿Qué courier se atrasó? (18 puntos)

Una tienda online de artículos deportivos despacha sus pedidos con tres empresas de courier. Los paquetes se entregan al cliente sin que el área de servicio al cliente sepa de inmediato qué courier los transportó.

| Courier | % de pedidos despachados | % de pedidos que llegan con retraso |
|---|---|---|
| RápidoSur | 45% | 8% |
| EnvíaYa | 35% | 12% |
| CargoNorte | 20% | 5% |

Un cliente reclama porque su pedido llegó **con retraso**.

a) Defina los eventos $A_1, A_2, A_3$ y $B$ (retraso). Justifique por qué los $A_i$ cumplen las condiciones para aplicar el teorema de Bayes. **(2 pts)**

b) Construya el diagrama de árbol completo (incluyendo las ramas de "llega a tiempo") con todas las probabilidades conjuntas. Verifique que las conjuntas suman 1. **(4 pts)**

c) Calcule la probabilidad de que un pedido cualquiera llegue con retraso. **(2 pts)**

d) Dado que el pedido llegó con retraso, calcule la probabilidad a posteriori de que lo haya transportado cada courier. Presente los resultados en una tabla de Bayes. **(5 pts)**

e) Otro cliente comenta que su pedido llegó **a tiempo**. ¿Cuál es la probabilidad de que lo haya transportado CargoNorte? **(2 pts)**

f) El gerente de operaciones afirma: *"EnvíaYa despacha solo un tercio de los pedidos, así que no puede ser el principal responsable de los reclamos por retraso"*. Evalúe esta afirmación usando sus resultados. **(3 pts)**

\newpage

## Ejercicio 3 — Inspección secuencial de una máquina (18 puntos)

En una planta de envases plásticos, una máquina inyectora puede estar **ajustada** o **desajustada**. Por los registros de mantenimiento se sabe que, al inicio de un turno cualquiera, la máquina está ajustada con probabilidad **0,90** y desajustada con probabilidad **0,10**.

- Si está ajustada, cada envase producido es defectuoso con probabilidad **0,02**.
- Si está desajustada, cada envase producido es defectuoso con probabilidad **0,20**.

Suponga que, dado el estado de la máquina, los envases salen defectuosos de forma independiente unos de otros.

La política de la planta es **detener la máquina para recalibrarla** si la probabilidad de que esté desajustada supera **0,80**.

a) Al iniciar el turno, el supervisor inspecciona un envase y resulta **defectuoso**. Calcule la probabilidad a posteriori de que la máquina esté desajustada. ¿Debe detenerse según la política? **(5 pts)**

b) Usando como nueva probabilidad a priori el resultado de a), el supervisor inspecciona un **segundo** envase y también resulta **defectuoso**. Actualice la probabilidad de que la máquina esté desajustada. ¿Debe detenerse ahora? **(5 pts)**

c) Verifique el resultado de b) resolviendo el problema **en un solo paso**: calcule directamente $P(\text{desajustada} \mid \text{2 defectuosos de 2})$ usando las probabilidades a priori originales. **(4 pts)**

d) Suponga, en cambio, que el primer envase fue defectuoso y el segundo **no defectuoso**. ¿Cuál sería la probabilidad de que la máquina esté desajustada? Interprete qué le ocurre a la "creencia" del supervisor con cada nueva observación. **(4 pts)**

\newpage

## Ejercicio 4 — ¿Vale la pena el estudio de mercado? (24 puntos)

Una empresa de alimentos evalúa lanzar una nueva bebida isotónica. El equipo comercial estima la demanda que tendría el producto y la utilidad asociada (en millones de pesos):

| Nivel de demanda | Probabilidad a priori | Utilidad si se lanza (MM\$) |
|---|---|---|
| Alta ($A_1$) | 0,25 | 500 |
| Media ($A_2$) | 0,45 | 120 |
| Baja ($A_3$) | 0,30 | −300 |

Si la empresa **no lanza** el producto, su utilidad es 0.

Una consultora ofrece un estudio de mercado cuyo resultado puede ser **favorable (F)** o **desfavorable (D)**. Según el historial de la consultora:

- $P(F \mid \text{demanda alta}) = 0{,}85$
- $P(F \mid \text{demanda media}) = 0{,}50$
- $P(F \mid \text{demanda baja}) = 0{,}10$

a) **Sin estudio**: calcule la utilidad esperada de lanzar el producto. ¿Qué decisión conviene? **(3 pts)**

b) Calcule la probabilidad de que el estudio resulte favorable y la probabilidad de que resulte desfavorable. **(4 pts)**

c) Calcule las probabilidades a posteriori de cada nivel de demanda **si el estudio es favorable** y **si es desfavorable** (dos tablas de Bayes). **(8 pts)**

d) Con las probabilidades a posteriori, calcule la utilidad esperada de lanzar el producto en cada caso. Indique qué decisión tomaría la empresa según el resultado del estudio. **(4 pts)**

e) Calcule la utilidad esperada de la estrategia "contratar el estudio y decidir según su resultado" (sin considerar el costo del estudio). Si la consultora cobra **MM\$ 20**, ¿conviene contratarla? ¿Cuál es el máximo que la empresa debería estar dispuesta a pagar? **(5 pts)**

\newpage

# PAUTAS DE CORRECCIÓN

## Pauta Ejercicio 1 (16 pts)

**a) (3 pts)** Eventos: $F$ = "transacción fraudulenta", $L$ = "transacción legítima" (complementos), $B$ = "el sistema emite alerta".

- A priori: $P(F) = 0{,}02$; $P(L) = 0{,}98$. **(1 pt)**
- Condicionales: $P(B \mid F) = 0{,}95$; $P(B \mid L) = 0{,}04$. **(1 pt)**
- Reconoce que $F$ y $L$ son mutuamente excluyentes y colectivamente exhaustivos. **(1 pt)**

**b) (4 pts)** Árbol: primera rama $F$ / $L$; segunda rama alerta / no alerta. **(2 pts)**

$$P(B) = (0{,}02)(0{,}95) + (0{,}98)(0{,}04) = 0{,}0190 + 0{,}0392 = 0{,}0582$$ **(2 pts)**

**c) (4 pts)**

| Evento | A priori | Condicional $P(B \mid \cdot)$ | Conjunta | A posteriori |
|---|---|---|---|---|
| Fraude | 0,02 | 0,95 | 0,0190 | 0,0190 / 0,0582 = **0,3265** |
| Legítima | 0,98 | 0,04 | 0,0392 | 0,0392 / 0,0582 = 0,6735 |
| | | | $P(B) = 0{,}0582$ | 1,0000 |

$P(F \mid B) \approx 0{,}33$. Aunque hubo alerta, solo una de cada tres alertas corresponde realmente a fraude. **(3 pts cálculo, 1 pt interpretación)**

**d) (2 pts)** $P(\bar B) = 1 - 0{,}0582 = 0{,}9418$; $P(F \text{ y } \bar B) = (0{,}02)(0{,}05) = 0{,}001$.

$$P(F \mid \bar B) = \frac{0{,}001}{0{,}9418} \approx 0{,}0011$$

Sin alerta, la probabilidad de fraude baja de 2% a aproximadamente 0,1%.

**e) (3 pts)** $P(B) = 0{,}0190 + (0{,}98)(0{,}01) = 0{,}0190 + 0{,}0098 = 0{,}0288$

$$P(F \mid B) = \frac{0{,}0190}{0{,}0288} \approx 0{,}6597$$ **(2 pts)**

Comentario esperado **(1 pt)**: como el fraude es poco frecuente (a priori 2%), las transacciones legítimas son tantas que incluso un 4% de falsos positivos genera más alertas falsas (0,0392) que verdaderas (0,0190). Reducir los falsos positivos baja mucho ese número y la probabilidad a posteriori se duplica (de 0,33 a 0,66). Es el mismo fenómeno del ejemplo de Umen en el libro: con una probabilidad a priori baja, un resultado positivo no garantiza que el evento haya ocurrido.

\newpage

## Pauta Ejercicio 2 (18 pts)

**a) (2 pts)** $A_1$: RápidoSur, $A_2$: EnvíaYa, $A_3$: CargoNorte; $B$: el pedido llega con retraso. Cada pedido lo transporta un solo courier (mutuamente excluyentes) y entre los tres cubren el 100% de los pedidos (colectivamente exhaustivos: 0,45 + 0,35 + 0,20 = 1).

**b) (4 pts)**

| Rama | Conjunta |
|---|---|
| RápidoSur y retraso | (0,45)(0,08) = 0,036 |
| RápidoSur y a tiempo | (0,45)(0,92) = 0,414 |
| EnvíaYa y retraso | (0,35)(0,12) = 0,042 |
| EnvíaYa y a tiempo | (0,35)(0,88) = 0,308 |
| CargoNorte y retraso | (0,20)(0,05) = 0,010 |
| CargoNorte y a tiempo | (0,20)(0,95) = 0,190 |
| **Total** | **1,000** |

(3 pts árbol y conjuntas correctas; 1 pt verificación de suma 1)

**c) (2 pts)** $P(B) = 0{,}036 + 0{,}042 + 0{,}010 = 0{,}088$. El 8,8% de los pedidos llega con retraso.

**d) (5 pts)**

| Courier | A priori | $P(B \mid A_i)$ | Conjunta | A posteriori $P(A_i \mid B)$ |
|---|---|---|---|---|
| RápidoSur | 0,45 | 0,08 | 0,036 | 0,036/0,088 = **0,4091** |
| EnvíaYa | 0,35 | 0,12 | 0,042 | 0,042/0,088 = **0,4773** |
| CargoNorte | 0,20 | 0,05 | 0,010 | 0,010/0,088 = **0,1136** |
| | | | $P(B)=0{,}088$ | 1,0000 |

(1 pt por cada posterior correcta; 2 pts por tabla bien construida y suma 1)

**e) (2 pts)** $P(\text{a tiempo}) = 1 - 0{,}088 = 0{,}912$.

$$P(A_3 \mid \bar B) = \frac{0{,}190}{0{,}912} \approx 0{,}2083$$

(Sube levemente desde 0,20, porque CargoNorte es el courier más puntual.)

**f) (3 pts)** La afirmación es **incorrecta**. Aunque EnvíaYa despacha solo el 35% de los pedidos, dado un retraso es el courier **más probable** (47,7%), por sobre RápidoSur (40,9%). Su tasa de retraso (12%) es la más alta y compensa su menor participación. La probabilidad a posteriori combina participación (a priori) y desempeño (condicional); mirar solo una de las dos lleva a conclusiones erradas.

\newpage

## Pauta Ejercicio 3 (18 pts)

Notación: $D$ = desajustada, $J$ = ajustada, $d$ = envase defectuoso, $b$ = envase bueno.

**a) (5 pts)**

| Estado | A priori | $P(d \mid \cdot)$ | Conjunta | A posteriori |
|---|---|---|---|---|
| Ajustada | 0,90 | 0,02 | 0,018 | 0,018/0,038 = 0,4737 |
| Desajustada | 0,10 | 0,20 | 0,020 | 0,020/0,038 = **0,5263** |
| | | | 0,038 | 1,0000 |

(4 pts cálculo.) $0{,}5263 < 0{,}80$: **no se detiene** todavía. La probabilidad de desajuste pasó de 10% a 52,6% con una sola observación. **(1 pt)**

**b) (5 pts)** Nueva a priori: $P(J) = 0{,}4737$, $P(D) = 0{,}5263$.

| Estado | A priori | $P(d \mid \cdot)$ | Conjunta | A posteriori |
|---|---|---|---|---|
| Ajustada | 0,4737 | 0,02 | 0,009474 | 0,0826 |
| Desajustada | 0,5263 | 0,20 | 0,105263 | **0,9174** |
| | | | 0,114737 | 1,0000 |

(4 pts cálculo.) $0{,}9174 > 0{,}80$: **se debe detener la máquina**. **(1 pt)**

**c) (4 pts)** Por la independencia condicional: $P(dd \mid J) = 0{,}02^2 = 0{,}0004$; $P(dd \mid D) = 0{,}20^2 = 0{,}04$.

$$P(D \mid dd) = \frac{(0{,}10)(0{,}04)}{(0{,}90)(0{,}0004) + (0{,}10)(0{,}04)} = \frac{0{,}004}{0{,}00036 + 0{,}004} = \frac{0{,}004}{0{,}00436} \approx 0{,}9174$$

Coincide con b): actualizar paso a paso o en un solo paso entrega el mismo resultado. **(3 pts cálculo, 1 pt conclusión)**

**d) (4 pts)** A priori: 0,4737 / 0,5263 (resultado de a)). $P(b \mid J) = 0{,}98$; $P(b \mid D) = 0{,}80$.

- Conjuntas: $0{,}4737 \times 0{,}98 = 0{,}4642$; $0{,}5263 \times 0{,}80 = 0{,}4211$; suma = 0,8853.
- $P(D \mid d, b) = 0{,}4211 / 0{,}8853 \approx 0{,}4756$. **(3 pts)**

Interpretación **(1 pt)**: cada observación revisa la creencia anterior. Un defectuoso la sube fuerte (10% → 52,6%) porque es 10 veces más probable con la máquina desajustada; un envase bueno la baja solo un poco (52,6% → 47,6%), porque un envase bueno es frecuente en ambos estados (98% vs 80%) y por lo tanto aporta poca información.

\newpage

## Pauta Ejercicio 4 (24 pts)

**a) (3 pts)** $VE(\text{lanzar}) = 0{,}25(500) + 0{,}45(120) + 0{,}30(-300) = 125 + 54 - 90 = \textbf{MM\$ 89}$.

Como 89 > 0, sin estudio conviene **lanzar**.

**b) (4 pts)**

$$P(F) = 0{,}25(0{,}85) + 0{,}45(0{,}50) + 0{,}30(0{,}10) = 0{,}2125 + 0{,}2250 + 0{,}0300 = 0{,}4675$$

$$P(D) = 0{,}25(0{,}15) + 0{,}45(0{,}50) + 0{,}30(0{,}90) = 0{,}0375 + 0{,}2250 + 0{,}2700 = 0{,}5325$$

Verificación: $0{,}4675 + 0{,}5325 = 1$.

**c) (8 pts, 4 por tabla)**

*Estudio favorable:*

| Demanda | A priori | $P(F \mid A_i)$ | Conjunta | A posteriori |
|---|---|---|---|---|
| Alta | 0,25 | 0,85 | 0,2125 | **0,4545** |
| Media | 0,45 | 0,50 | 0,2250 | **0,4813** |
| Baja | 0,30 | 0,10 | 0,0300 | **0,0642** |
| | | | 0,4675 | 1,0000 |

*Estudio desfavorable:*

| Demanda | A priori | $P(D \mid A_i)$ | Conjunta | A posteriori |
|---|---|---|---|---|
| Alta | 0,25 | 0,15 | 0,0375 | **0,0704** |
| Media | 0,45 | 0,50 | 0,2250 | **0,4225** |
| Baja | 0,30 | 0,90 | 0,2700 | **0,5070** |
| | | | 0,5325 | 1,0000 |

**d) (4 pts)**

- Si F: $VE = 0{,}4545(500) + 0{,}4813(120) + 0{,}0642(-300) \approx \textbf{MM\$ 265,8}$ → **lanzar**.
- Si D: $VE = 0{,}0704(500) + 0{,}4225(120) + 0{,}5070(-300) \approx \textbf{MM\$ −66,2}$ → **no lanzar** (utilidad 0).

(Se acepta un pequeño margen por redondeo; los valores exactos son 124,25/0,4675 y −35,25/0,5325.)

**e) (5 pts)**

$$VE(\text{con estudio}) = P(F)(265{,}8) + P(D)(0) = 0{,}4675 \times 265{,}8 \approx \textbf{MM\$ 124,25}$$ **(2 pts)**

Valor de la información del estudio $= 124{,}25 - 89 = \textbf{MM\$ 35,25}$. **(1 pt)**

Si el estudio cuesta MM\$ 20: $124{,}25 - 20 = 104{,}25 > 89$, **conviene contratarlo**. El máximo que debería pagar es **MM\$ 35,25**. **(2 pts)**

Idea clave: el estudio genera valor porque, cuando sale desfavorable, permite **evitar** lanzar un producto cuya demanda probablemente es baja (posterior de 50,7%). La información vale en la medida en que puede cambiar la decisión.

\newpage

# Errores frecuentes a observar al corregir

1. **Confundir $P(B \mid A)$ con $P(A \mid B)$**: por ejemplo, responder 0,95 en el Ej. 1c.
2. **Olvidar el denominador** (probabilidad total) o sumar solo las ramas del evento de interés.
3. **Usar las a priori originales** en el segundo paso del Ej. 3b, en lugar de las a posteriori del paso anterior.
4. **No verificar** que las probabilidades a posteriori sumen 1.
5. **Usar $P(B)$ en vez de $P(\bar B)$** al condicionar en "no alerta" o "a tiempo" (Ej. 1d, 2e).
6. **Ej. 4e**: calcular el VE con estudio como si siempre se lanzara (promediando 265,8 y −66,2), en lugar de tomar la mejor decisión en cada rama.
