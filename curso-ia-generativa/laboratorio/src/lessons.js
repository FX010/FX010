/* Contenido del laboratorio. Para añadir una lección nueva, agrega un objeto a LESSONS.
   Cada reto `check` se ejecuta dentro del kernel, con acceso a lo que el alumno definió. */
const LESSONS = [
  {
    id: "neurona", num: 1, title: "La neurona", week: "Semana 1", minutes: 10, widget: "neurona",
    intro: `<p>Una neurona artificial hace tres cosas: multiplica cada entrada por un <b>peso</b>, suma todo junto con un <b>sesgo</b> y pasa el resultado por una <b>activación</b> no lineal.</p>
      <p class="formula">y = φ( w₁·x₁ + w₂·x₂ + b )</p>
      <p>Los pesos giran la frontera de decisión, el sesgo la desplaza y la activación decide si la transición es brusca o suave. Sin la activación, apilar capas no serviría de nada, porque una composición de funciones lineales sigue siendo lineal.</p>`,
    widgetTitle: "Mueve los pesos y mira la frontera",
    widgetNote: "El color muestra la salida de la neurona en cada punto (x₁, x₂). Haz clic en el mapa para evaluar un punto concreto.",
    taskIntro: `<p>Escribe la neurona en código. Después ajusta pesos a mano para que una neurona sigmoide se comporte como la puerta lógica <b>AND</b>: salida alta solo si <code>x₁ = 1</code> y <code>x₂ = 1</code>.</p>`,
    code: `// y = tanh(w·x + b)
function neurona(w, x, b) {
  // suma los productos w[i] * x[i], añade b y aplica Math.tanh
  return 0;
}
print(neurona([1, 2], [0.5, -1], 0.1)); // debería ser ≈ -0.8854

// Reto 2: elige w y b para que sigmoide(w1*x1 + w2*x2 + b) sea una puerta AND
const AND = { w: [0, 0], b: 0 };
const sig = (z) => 1 / (1 + Math.exp(-z));
for (const [a, c] of [[0, 0], [0, 1], [1, 0], [1, 1]]) {
  print(a, c, '->', sig(AND.w[0] * a + AND.w[1] * c + AND.b));
}`,
    solution: `function neurona(w, x, b) {
  let s = b;
  for (let i = 0; i < w.length; i++) s += w[i] * x[i];
  return Math.tanh(s);
}
print(neurona([1, 2], [0.5, -1], 0.1));

const AND = { w: [10, 10], b: -15 };
const sig = (z) => 1 / (1 + Math.exp(-z));
for (const [a, c] of [[0, 0], [0, 1], [1, 0], [1, 1]]) {
  print(a, c, '->', sig(AND.w[0] * a + AND.w[1] * c + AND.b));
}`,
    hints: ["Recorre w con un bucle for y acumula w[i] * x[i] en una variable que empiece en b.", "Para AND necesitas que w₁ + w₂ + b sea claramente positivo, pero que w₁ + b, w₂ + b y b sean claramente negativos.", "Prueba con pesos grandes (10 y 10) y un sesgo que deje (1,1) en +5 y los demás por debajo de −5."],
    check: function () {
      const t = (n, ok, m) => ({ name: n, ok: !!ok, msg: ok ? "" : m });
      const r = [];
      if (typeof neurona !== "function") return [t("Existe la función neurona(w, x, b)", false, "Todavía no está definida. Ejecuta la celda con function neurona(w, x, b) { ... }.")];
      const y = neurona([1, 2], [0.5, -1], 0.1);
      r.push(t("neurona([1,2], [0.5,-1], 0.1) ≈ −0.8854", Math.abs(y - Math.tanh(-1.4)) < 1e-6, "Devolvió " + fmt(y) + ". Calcula tanh(w·x + b)."));
      r.push(t("Con pesos 0 la salida es tanh(b)", Math.abs(neurona([0, 0], [3, 4], 0.5) - Math.tanh(0.5)) < 1e-9, "Con w = [0,0] solo debe contar el sesgo."));
      r.push(t("Funciona con una sola entrada", Math.abs(neurona([3], [1], 0.5) - Math.tanh(3.5)) < 1e-9, "No asumas que w tiene exactamente 2 elementos."));
      if (typeof AND === "undefined" || !AND || !AND.w) r.push(t("Existe AND = { w: [..], b: .. }", false, "Define la constante AND."));
      else {
        const s = (a, c) => 1 / (1 + Math.exp(-(AND.w[0] * a + AND.w[1] * c + AND.b)));
        const o = [s(0, 0), s(0, 1), s(1, 0), s(1, 1)];
        r.push(t("La neurona implementa AND", o[3] > 0.9 && o[0] < 0.1 && o[1] < 0.1 && o[2] < 0.1, "Salidas para (0,0) (0,1) (1,0) (1,1): " + fmt(o) + ". Debe verse algo como [≈0, ≈0, ≈0, ≈1]."));
      }
      return r;
    },
    outro: `Una neurona traza una frontera recta. Para dibujar curvas hacen falta varias capas, que verás en la lección 4. Esto es la <b>Semana 1</b> de la guía.`,
  },
  {
    id: "gradiente", num: 2, title: "Bajar por la pérdida", week: "Semana 1", minutes: 10, widget: "gradiente",
    intro: `<p>Entrenar una red significa encontrar parámetros que hagan pequeña una <b>pérdida</b>. El gradiente apunta hacia donde la pérdida sube más rápido, así que damos un paso en sentido contrario:</p>
      <p class="formula">x ← x − η · f′(x)</p>
      <p>η es la <b>tasa de aprendizaje</b>. Si es muy pequeña, avanzas con lentitud. Si es muy grande, te pasas del mínimo y puedes divergir. Además, con varios valles, el punto de partida decide en cuál terminas.</p>`,
    widgetTitle: "Un paso a la vez",
    widgetNote: "La función tiene dos valles. Cambia la tasa de aprendizaje y el punto de partida, y pulsa «Paso» o «Ejecutar».",
    taskIntro: `<p>Implementa el descenso de gradiente sobre <code>f(x) = (x − 3)² + 1</code>, cuyo gradiente es <code>2(x − 3)</code>. Luego elige la tasa <code>LR</code> que llega al mínimo en solo 5 pasos.</p>`,
    code: `// Aplica  x ← x − lr · grad(x)  durante 'pasos' veces y devuelve el x final
function descenso(grad, x0, lr, pasos) {
  let x = x0;
  // tu bucle aquí
  return x;
}
const g = (x) => 2 * (x - 3);
print('lr 0.1, 50 pasos ->', descenso(g, 0, 0.1, 50));   // cerca de 3
print('lr 1.1, 50 pasos ->', descenso(g, 0, 1.1, 50));   // diverge

// Reto 2: ¿qué lr llega a ≈3 (error < 0.01) en solo 5 pasos desde x = 0?
const LR = 0.1;
print('con LR =', LR, '->', descenso(g, 0, LR, 5));`,
    solution: `function descenso(grad, x0, lr, pasos) {
  let x = x0;
  for (let i = 0; i < pasos; i++) x = x - lr * grad(x);
  return x;
}
const g = (x) => 2 * (x - 3);
print('lr 0.1, 50 pasos ->', descenso(g, 0, 0.1, 50));
print('lr 1.1, 50 pasos ->', descenso(g, 0, 1.1, 50));

const LR = 0.5;
print('con LR =', LR, '->', descenso(g, 0, LR, 5));`,
    hints: ["Dentro del bucle: x = x - lr * grad(x).", "Cada paso multiplica el error por (1 − 2·lr). ¿Qué lr lo hace 0?", "Con lr = 0.5 el error se anula en un paso. Cualquier valor entre 0.4 y 0.6 sirve."],
    check: function () {
      const t = (n, ok, m) => ({ name: n, ok: !!ok, msg: ok ? "" : m });
      if (typeof descenso !== "function") return [t("Existe descenso(grad, x0, lr, pasos)", false, "Todavía no está definida.")];
      const g = (x) => 2 * (x - 3), r = [];
      r.push(t("Un paso desde 0 con lr 0.1 da 0.6", Math.abs(descenso(g, 0, 0.1, 1) - 0.6) < 1e-9, "Con x = 0: 0 − 0.1·2·(0 − 3) = 0.6. Obtuviste " + fmt(descenso(g, 0, 0.1, 1)) + "."));
      r.push(t("Con 50 pasos y lr 0.1 llega a 3 (error < 0.001)", Math.abs(descenso(g, 0, 0.1, 50) - 3) < 1e-3, "Obtuviste " + fmt(descenso(g, 0, 0.1, 50)) + "."));
      r.push(t("Con lr 1.1 diverge (|x| > 1000)", Math.abs(descenso(g, 0, 1.1, 50)) > 1000, "Con lr = 1.1 cada paso multiplica el error por −1.2, así que debe crecer."));
      if (typeof LR !== "number") r.push(t("Define LR", false, "Declara const LR = ..."));
      else r.push(t("LR llega a 3 en 5 pasos (error < 0.01)", Math.abs(descenso(g, 0, LR, 5) - 3) < 0.01, "Con LR = " + LR + " obtienes " + fmt(descenso(g, 0, LR, 5)) + ". Cada paso multiplica el error por (1 − 2·LR)."));
      return r;
    },
    outro: `Todo el entrenamiento de redes, desde esta parábola hasta un GPT, repite este paso. Lo que cambia es cómo se calcula el gradiente: eso es la retropropagación, la siguiente lección.`,
  },
  {
    id: "backprop", num: 3, title: "Retropropagación", week: "Semana 1", minutes: 15, widget: "grafo",
    intro: `<p>Una red es un grafo de operaciones simples. Para obtener el gradiente de la pérdida respecto a cada parámetro, se recorre el grafo <b>hacia atrás</b> multiplicando derivadas locales (la regla de la cadena).</p>
      <p class="formula">L = ( tanh(w·x + b) − 1 )²</p>
      <p>Dos reglas lo resumen: cada nodo reparte el gradiente que recibe según su derivada local, y si un nodo se usa varias veces, <b>los gradientes se suman</b>.</p>`,
    widgetTitle: "El grafo de L con los números de la guía",
    widgetNote: "Pulsa «Hacia delante» para calcular valores y «Hacia atrás» para propagar gradientes. Son los números del ejemplo de la Semana 1.",
    taskIntro: `<p>La clase <code>Value</code> ya está en el kernel (es el <code>micrograd</code> de la guía). Construye el grafo de arriba con sus métodos y llama a <code>backward()</code>.</p>`,
    code: `const w = new Value(0.5), x = new Value(2), b = new Value(-0.3);
// L = (tanh(w*x + b) - 1)^2   → usa .mul  .add  .tanh  .sub  .pow
const L = w;   // ← reemplaza por la expresión
L.backward();
print('dL/dw =', w.grad, ' dL/db =', b.grad, ' dL/dx =', x.grad);

// Reto 2: un nodo usado dos veces. q = y*y + y con y = 3  → dq/dy = 2y + 1 = 7
const y = new Value(3);
const q = y;   // ← reemplaza
q.backward();
print('dq/dy =', y.grad, '(debería ser 7)');`,
    solution: `const w = new Value(0.5), x = new Value(2), b = new Value(-0.3);
const L = w.mul(x).add(b).tanh().sub(1).pow(2);
L.backward();
print('dL/dw =', w.grad, ' dL/db =', b.grad, ' dL/dx =', x.grad);

const y = new Value(3);
const q = y.mul(y).add(y);
q.backward();
print('dq/dy =', y.grad, '(debería ser 7)');`,
    hints: ["w.mul(x) crea el nodo w·x. Encadena .add(b), luego .tanh().", "(a − 1)² se escribe a.sub(1).pow(2).", "Para q usa y.mul(y).add(y). Verás que y recibe gradiente por tres caminos y los suma."],
    check: function () {
      const t = (n, ok, m) => ({ name: n, ok: !!ok, msg: ok ? "" : m });
      const r = [];
      if (typeof w === "undefined" || typeof L === "undefined") return [t("Existen w, x, b y L", false, "Ejecuta la celda para crear los valores.")];
      r.push(t("L ≈ 0.15652", L instanceof Value && Math.abs(L.data - 0.15652) < 1e-4, "L vale " + (L && L.data !== undefined ? fmt(L.data) : "?") + ". Revisa la expresión: (tanh(w·x + b) − 1)²."));
      r.push(t("dL/dw ≈ −1.00449", Math.abs(w.grad + 1.00449) < 1e-3, "w.grad = " + fmt(w.grad) + ". ¿Llamaste a L.backward()?"));
      r.push(t("dL/db ≈ −0.50225", Math.abs(b.grad + 0.50225) < 1e-3, "b.grad = " + fmt(b.grad)));
      r.push(t("dL/dx ≈ −0.25112", Math.abs(x.grad + 0.25112) < 1e-3, "x.grad = " + fmt(x.grad)));
      r.push(t("Tu grafo pasa el gradient check", (() => { try { return gradcheck((a, c, d) => a.mul(d).add(c).tanh().sub(1).pow(2), [0.5, -0.3, 2]) < 1e-6; } catch (e) { return false; } })(), "La función de referencia no pasó el chequeo numérico."));
      if (typeof q === "undefined" || typeof y === "undefined") r.push(t("Existen y y q", false, "Define y y q."));
      else r.push(t("dq/dy = 7 (los gradientes se suman)", Math.abs(y.grad - 7) < 1e-9, "y.grad = " + fmt(y.grad) + ". Con q = y·y + y, y aparece tres veces."));
      return r;
    },
    outro: `Esto es lo que hace <code>loss.backward()</code> en PyTorch, sobre tensores en lugar de escalares. En la guía de la Semana 1 lo escribes completo y lo verificas con derivadas numéricas.`,
  },
  {
    id: "red", num: 4, title: "Tu primera red", week: "Semana 2", minutes: 15, widget: "playground",
    intro: `<p>Con una sola neurona la frontera es una recta. Una red con <b>capas ocultas</b> compone muchas neuronas y puede dibujar fronteras curvas, como estas espirales entrelazadas.</p>
      <p>Cada paso de entrenamiento es el ciclo de siempre: <b>forward</b> (predecir), <b>pérdida</b> (medir el error), <b>backward</b> (gradientes) y <b>actualizar</b> los pesos.</p>`,
    widgetTitle: "Entrena una red en vivo",
    widgetNote: "Prueba una red muy pequeña sobre espirales y luego agrándala. Compara SGD con Adam. Todo corre en tu navegador.",
    taskIntro: `<p>El código de abajo entrena una red diminuta y se queda corto. Cámbialo hasta superar el <b>95 %</b> de precisión en las espirales, y que además generalice a puntos nuevos (el reto los comprueba con otra semilla).</p>`,
    code: `const data = spirals(100, 3, 0.2, 0);          // 300 puntos, 3 clases
const red = new MLP([2, 4, 3], { act: 'tanh', seed: 1 });
for (let paso = 0; paso <= 200; paso++) {
  const r = red.trainStep(data.X, data.y, 0.1); // lote completo, lr 0.1
  if (paso % 50 === 0) print('paso', paso, ' pérdida', r.loss, ' precisión', r.acc);
}
plotBoundary(red, data);
// Prueba: más neuronas, más capas, más pasos, otra tasa de aprendizaje, optim: 'adam'`,
    solution: `const data = spirals(100, 3, 0.2, 0);
const red = new MLP([2, 16, 16, 3], { act: 'tanh', seed: 1 });
for (let paso = 0; paso <= 800; paso++) {
  const r = red.trainStep(data.X, data.y, 0.5);
  if (paso % 200 === 0) print('paso', paso, ' pérdida', r.loss, ' precisión', r.acc);
}
plotBoundary(red, data);`,
    hints: ["Con 4 neuronas ocultas no alcanza la capacidad para las espirales. Prueba [2, 16, 16, 3].", "200 pasos con lr 0.1 apenas empiezan a aprender. Sube el lr a 0.3–0.5 o usa más pasos.", "Con optim: 'adam' y lr 0.02 suele converger más rápido que con SGD."],
    check: function () {
      const t = (n, ok, m) => ({ name: n, ok: !!ok, msg: ok ? "" : m });
      if (typeof red === "undefined" || typeof data === "undefined" || !(red instanceof MLP)) return [t("Existen data y red (un MLP)", false, "Ejecuta la celda para crearlos.")];
      const tr = red.evaluate(data.X, data.y), nuevo = spirals(100, 3, 0.2, 99), te = red.evaluate(nuevo.X, nuevo.y);
      return [
        t("Precisión en entrenamiento ≥ 95 %", tr.acc >= 0.95, "Ahora tienes " + (tr.acc * 100).toFixed(1) + " %. Agranda la red o entrena más."),
        t("Precisión en datos nuevos ≥ 88 %", te.acc >= 0.88, "En datos nuevos: " + (te.acc * 100).toFixed(1) + " %. Si entrenamiento es alto y esto bajo, hay sobreajuste."),
        t("La pérdida es menor que ln 3 ≈ 1.099", tr.loss < Math.log(3), "Una red que no aprendió tiene pérdida ≈ ln(3) = 1.099."),
      ];
    },
    outro: `Ahora sabes cómo se ve entrenar una red de verdad. En la <b>Semana 2</b> escribes este mismo MLP con NumPy y backprop manual, y en la <b>Semana 3</b> aprendes a diagnosticar cuando no converge.`,
  },
  {
    id: "init", num: 5, title: "Inicializar bien", week: "Semana 3", minutes: 15, widget: "init",
    intro: `<p>Si los pesos iniciales son demasiado pequeños, las señales se apagan capa tras capa. Si son demasiado grandes, explotan o saturan. En ambos casos el gradiente no llega y la red no aprende.</p>
      <p class="formula">Xavier: Var(w) = 1/fan_in &nbsp;·&nbsp; He: Var(w) = 2/fan_in</p>
      <p>La idea: que la <b>desviación estándar</b> de las activaciones se mantenga más o menos igual a través de las capas. ReLU anula la mitad de las señales, por eso He lleva un 2.</p>`,
    widgetTitle: "30 capas, cuatro inicializaciones",
    widgetNote: "Muestra la desviación estándar de las activaciones en cada capa (escala logarítmica). Cambia la activación y compara.",
    taskIntro: `<p>Escribe las funciones que devuelven la desviación estándar de los pesos para cada activación. El reto simula 30 capas de ancho 128 y exige que la señal sobreviva sin desvanecerse ni explotar.</p>`,
    code: `function desvPorCapa(escala, act, capas = 30, ancho = 128) {
  let h = randn(64, ancho, 1);
  const res = [];
  for (let l = 0; l < capas; l++) {
    const W = randn(ancho, ancho, 100 + l).map(f => f.map(v => v * escala(ancho)));
    h = matmul(h, W).map(f => f.map(act));
    res.push(std(h.flat()));
  }
  return res;
}
// Devuelven la desviación estándar de los pesos para una capa con fanIn entradas
function escalaRelu(fanIn) { return 1; }   // ← cámbiala
function escalaTanh(fanIn) { return 1; }   // ← cámbiala

const relu = (z) => Math.max(0, z);
const sr = desvPorCapa(escalaRelu, relu), st = desvPorCapa(escalaTanh, Math.tanh);
print('ReLU, capa 30:', sr[29], ' tanh, capa 30:', st[29]);
plot({ type: 'line', title: 'std de activaciones por capa', ylog: true,
       series: [{ name: 'ReLU', y: sr }, { name: 'tanh', y: st }] });`,
    solution: `function desvPorCapa(escala, act, capas = 30, ancho = 128) {
  let h = randn(64, ancho, 1);
  const res = [];
  for (let l = 0; l < capas; l++) {
    const W = randn(ancho, ancho, 100 + l).map(f => f.map(v => v * escala(ancho)));
    h = matmul(h, W).map(f => f.map(act));
    res.push(std(h.flat()));
  }
  return res;
}
function escalaRelu(fanIn) { return Math.sqrt(2 / fanIn); }
function escalaTanh(fanIn) { return (5 / 3) / Math.sqrt(fanIn); }

const relu = (z) => Math.max(0, z);
const sr = desvPorCapa(escalaRelu, relu), st = desvPorCapa(escalaTanh, Math.tanh);
print('ReLU, capa 30:', sr[29], ' tanh, capa 30:', st[29]);
plot({ type: 'line', title: 'std de activaciones por capa', ylog: true,
       series: [{ name: 'ReLU', y: sr }, { name: 'tanh', y: st }] });`,
    hints: ["Para ReLU usa He: Math.sqrt(2 / fanIn).", "Para tanh, Xavier (Math.sqrt(1 / fanIn)) se contrae despacio. Prueba la ganancia 5/3: (5/3) / Math.sqrt(fanIn), o simplemente He.", "Mira la gráfica: la línea debe quedarse aproximadamente horizontal."],
    check: function () {
      const t = (n, ok, m) => ({ name: n, ok: !!ok, msg: ok ? "" : m });
      if (typeof escalaRelu !== "function" || typeof escalaTanh !== "function") return [t("Existen escalaRelu y escalaTanh", false, "Define ambas funciones.")];
      const sim = (escala, act) => { let h = randn(64, 128, 1), s = 0; for (let l = 0; l < 30; l++) { const sc = escala(128); h = matmul(h, randn(128, 128, 100 + l).map((f) => f.map((v) => v * sc))).map((f) => f.map(act)); s = std(h.flat()); } return s; };
      const a = sim(escalaRelu, (z) => Math.max(0, z)), b = sim(escalaTanh, Math.tanh);
      return [
        t("ReLU: std en la capa 30 entre 0.3 y 3", a > 0.3 && a < 3, "Tu std final es " + fmt(a) + ". Si es casi 0 la señal se apagó; si es enorme, explotó."),
        t("tanh: std en la capa 30 entre 0.3 y 1", b > 0.3 && b < 1, "Tu std final es " + fmt(b) + ". Con tanh, std ≈ 1 significa saturación; casi 0, desvanecimiento."),
      ];
    },
    outro: `Hoy las normalizaciones (LayerNorm, RMSNorm) y las conexiones residuales hacen este problema mucho menos frágil, pero la inicialización sigue importando. Lo verás completo en la <b>Semana 3</b>.`,
  },
  {
    id: "atencion", num: 6, title: "Atención", week: "Semana 5", minutes: 20, widget: "atencion",
    intro: `<p>La atención deja que cada token mire a los demás y tome de ellos una mezcla ponderada de información. Cada token produce una consulta <b>Q</b>, una clave <b>K</b> y un valor <b>V</b>:</p>
      <p class="formula">Atención(Q, K, V) = softmax( Q·Kᵀ / √d + M ) · V</p>
      <p>La máscara causal <b>M</b> impide mirar al futuro (necesaria para generar texto). Dividir por √d evita que los productos punto crezcan con la dimensión y que el softmax se vuelva casi un «uno-caliente» con gradientes diminutos.</p>`,
    widgetTitle: "El mapa de atención de una frase",
    widgetNote: "Cada fila es un token y cuánto atiende a cada columna. Sube la dimensión d y quita el escalado: las filas se vuelven casi 0/1.",
    taskIntro: `<p>Implementa la función de atención completa con las utilidades del kernel (<code>matmul</code>, <code>transpose</code>, <code>softmax</code>). Debe devolver la salida y los pesos.</p>`,
    code: `// Q, K, V: matrices T×d (arreglos de arreglos). Devuelve { salida, pesos }
function atencion(Q, K, V, causal = true) {
  const T = Q.length, d = Q[0].length;
  // 1) scores = Q·Kᵀ / √d     2) máscara causal: pon -Infinity cuando j > i
  // 3) pesos = softmax por fila  4) salida = pesos · V
  return { salida: V, pesos: zeros(T, T) };
}
const Q = randn(4, 3, 1), K = randn(4, 3, 2), V = randn(4, 3, 3);
const r = atencion(Q, K, V);
print('pesos (fila 0):', r.pesos[0]);
print('pesos (fila 3):', r.pesos[3]);
plot({ type: 'heat', title: 'pesos de atención', matrix: r.pesos });`,
    solution: `function atencion(Q, K, V, causal = true) {
  const T = Q.length, d = Q[0].length;
  const scores = matmul(Q, transpose(K)).map(f => f.map(s => s / Math.sqrt(d)));
  const pesos = scores.map((fila, i) => softmax(fila.map((s, j) => (causal && j > i ? -Infinity : s))));
  return { salida: matmul(pesos, V), pesos };
}
const Q = randn(4, 3, 1), K = randn(4, 3, 2), V = randn(4, 3, 3);
const r = atencion(Q, K, V);
print('pesos (fila 0):', r.pesos[0]);
print('pesos (fila 3):', r.pesos[3]);
plot({ type: 'heat', title: 'pesos de atención', matrix: r.pesos });`,
    hints: ["scores = matmul(Q, transpose(K)) y luego divide cada elemento por Math.sqrt(d).", "Para la máscara: recorre cada fila i y sustituye por -Infinity los elementos con j > i, antes del softmax.", "salida = matmul(pesos, V). Cada fila de pesos suma 1."],
    check: function () {
      const t = (n, ok, m) => ({ name: n, ok: !!ok, msg: ok ? "" : m });
      if (typeof atencion !== "function") return [t("Existe atencion(Q, K, V, causal)", false, "Todavía no está definida.")];
      const Q = randn(5, 4, 11), K = randn(5, 4, 12), V = randn(5, 4, 13), r = [];
      let res;
      try { res = atencion(Q, K, V, true); } catch (e) { return [t("La función se ejecuta sin errores", false, String(e.message))]; }
      if (!res || !res.pesos || !res.salida) return [t("Devuelve { salida, pesos }", false, "Debe devolver un objeto con esas dos propiedades.")];
      const ref = (causal) => { const S = matmul(Q, transpose(K)).map((f) => f.map((s) => s / 2)); const P = S.map((f, i) => softmax(f.map((s, j) => (causal && j > i ? -Infinity : s)))); return { P, O: matmul(P, V) }; };
      const e1 = ref(true), maxd = (A, B) => Math.max(...A.flat().map((x, i) => Math.abs(x - B.flat()[i])));
      r.push(t("Las filas de pesos suman 1", res.pesos.every((f) => Math.abs(sum(f) - 1) < 1e-9), "Aplica softmax a cada fila."));
      r.push(t("Máscara causal: el futuro tiene peso 0", res.pesos.every((f, i) => f.every((p, j) => j <= i || Math.abs(p) < 1e-12)), "Hay peso en posiciones j > i."));
      r.push(t("Los pesos coinciden con la referencia", maxd(res.pesos, e1.P) < 1e-6, "Revisa que dividas por √d (d = " + Q[0].length + ") antes del softmax."));
      r.push(t("La salida coincide con la referencia", maxd(res.salida, e1.O) < 1e-6, "salida = pesos · V."));
      const sin = atencion(Q, K, V, false), e2 = ref(false);
      r.push(t("Con causal = false mira todo el contexto", maxd(sin.pesos, e2.P) < 1e-6, "Cuando causal es false no debes enmascarar."));
      return r;
    },
    outro: `Esta función, repetida en varias cabezas con proyecciones aprendidas, más una MLP y conexiones residuales, es un bloque Transformer. La <b>Semana 5</b> lo implementa completo, con RoPE y GQA.`,
  },
  {
    id: "lenguaje", num: 7, title: "Modelos de lenguaje", week: "Semana 3 y 5", minutes: 15, widget: "lenguaje",
    intro: `<p>Un modelo de lenguaje asigna probabilidades al siguiente carácter (o palabra) dado el contexto. Se entrena minimizando la <b>entropía cruzada</b> y la métrica clave es esa pérdida media, en nats por carácter.</p>
      <p>Aquí el texto sale de una gramática diminuta cuya entropía conocemos: <b>ningún modelo puede bajar de ese valor</b>. Eso permite saber exactamente cuánto falta.</p>
      <p>Para generar, se elige el siguiente carácter al azar según las probabilidades. La <b>temperatura</b> τ las reescala: τ pequeña vuelve el muestreo casi determinista, τ grande lo acerca a un sorteo uniforme.</p>`,
    widgetTitle: "Más contexto, menos pérdida",
    widgetNote: "Un modelo de conteo con contexto de n caracteres. Mira cómo la pérdida baja hacia la cota teórica y genera frases con distintas temperaturas.",
    taskIntro: `<p>Implementa el muestreo con temperatura. Recibe un arreglo de probabilidades, la temperatura y un número aleatorio <code>u</code> en [0, 1) (así el reto es determinista), y devuelve el índice elegido.</p>`,
    code: `// probs: probabilidades que suman 1.  T: temperatura.  u: número en [0,1)
// 1) reescala: q ∝ probs^(1/T)  (equivale a softmax de log(probs)/T)
// 2) recorre la suma acumulada y devuelve el primer índice con u < acumulado
function muestrear(probs, T, u) {
  return 0;
}
const p = [0.1, 0.6, 0.3];
print('T=1, u=0.05 ->', muestrear(p, 1, 0.05), ' u=0.5 ->', muestrear(p, 1, 0.5), ' u=0.9 ->', muestrear(p, 1, 0.9));
print('T=0.05 (casi argmax), u=0.01 ->', muestrear(p, 0.05, 0.01));

// ¿Cuántas veces sale cada índice con T = 0.5? (1000 sorteos)
const g = rng(3), cuenta = [0, 0, 0];
for (let i = 0; i < 1000; i++) cuenta[muestrear(p, 0.5, g.next())]++;
print('T=0.5 ->', cuenta);`,
    solution: `function muestrear(probs, T, u) {
  const q = softmax(probs.map(p => Math.log(p + 1e-12) / T));
  let acum = 0;
  for (let i = 0; i < q.length; i++) {
    acum += q[i];
    if (u < acum) return i;
  }
  return q.length - 1;
}
const p = [0.1, 0.6, 0.3];
print('T=1, u=0.05 ->', muestrear(p, 1, 0.05), ' u=0.5 ->', muestrear(p, 1, 0.5), ' u=0.9 ->', muestrear(p, 1, 0.9));
print('T=0.05 (casi argmax), u=0.01 ->', muestrear(p, 0.05, 0.01));
const g = rng(3), cuenta = [0, 0, 0];
for (let i = 0; i < 1000; i++) cuenta[muestrear(p, 0.5, g.next())]++;
print('T=0.5 ->', cuenta);`,
    hints: ["softmax(probs.map(p => Math.log(p) / T)) hace el reescalado. Suma 1 por construcción.", "Con T pequeña, el cociente log(p)/T separa mucho los valores y el softmax se vuelve casi un 1 en el más probable.", "Acumula q[0], q[0]+q[1], … y devuelve el primer i con u < acumulado."],
    check: function () {
      const t = (n, ok, m) => ({ name: n, ok: !!ok, msg: ok ? "" : m });
      if (typeof muestrear !== "function") return [t("Existe muestrear(probs, T, u)", false, "Todavía no está definida.")];
      const p = [0.1, 0.6, 0.3], r = [];
      r.push(t("T = 1 respeta las probabilidades (u = 0.05, 0.5, 0.9 → 0, 1, 2)", muestrear(p, 1, 0.05) === 0 && muestrear(p, 1, 0.5) === 1 && muestrear(p, 1, 0.9) === 2, "Obtuviste " + fmt([muestrear(p, 1, 0.05), muestrear(p, 1, 0.5), muestrear(p, 1, 0.9)]) + ". Con T = 1 los umbrales acumulados son 0.1 y 0.7."));
      r.push(t("T muy pequeña elige siempre el más probable", [0.01, 0.4, 0.99].every((u) => muestrear(p, 0.05, u) === 1), "Con T = 0.05 debería salir siempre el índice 1."));
      r.push(t("T muy grande se acerca a un sorteo uniforme", muestrear(p, 100, 0.2) === 0 && muestrear(p, 100, 0.5) === 1 && muestrear(p, 100, 0.9) === 2, "Con T = 100 las tres opciones son casi equiprobables (tercios)."));
      const g = rng(5), c = [0, 0, 0];
      for (let i = 0; i < 4000; i++) c[muestrear(p, 1, g.next())]++;
      r.push(t("Con T = 1, las frecuencias se parecen a [0.1, 0.6, 0.3]", Math.abs(c[0] / 4000 - 0.1) < 0.03 && Math.abs(c[1] / 4000 - 0.6) < 0.03, "Frecuencias observadas: " + fmt(c.map((x) => x / 4000))));
      return r;
    },
    outro: `Esta es literalmente la última línea de <code>GPT.generate</code> en la guía de la Semana 5. La pérdida contra la cota es la prueba de corrección de la Semana 3: tu GPT de 335 mil parámetros llega a 0,232 frente a 0,203.`,
  },
  {
    id: "difusion", num: 8, title: "Difusión", week: "Semana 7", minutes: 20, widget: "difusion",
    intro: `<p>Un modelo de difusión aprende a <b>quitar ruido</b>. Primero se define un proceso que destruye los datos añadiendo ruido gaussiano poco a poco. Gracias a una fórmula cerrada, se salta a cualquier paso t directamente:</p>
      <p class="formula">xₜ = √ᾱₜ · x₀ + √(1 − ᾱₜ) · ε &nbsp;&nbsp; con ᾱₜ = ∏ (1 − βₛ)</p>
      <p>Una red aprende a predecir el ruido ε en cada nivel. Para generar, se parte de ruido puro y se retrocede paso a paso.</p>`,
    widgetTitle: "Destruir y reconstruir una espiral",
    widgetNote: "Arrastra t para ver cómo el ruido borra la forma. Con «Generar» el proceso inverso parte de ruido y lo ordena. Aquí usamos el denoiser ideal (exacto para este conjunto de puntos); una red neuronal lo aproxima y generaliza.",
    taskIntro: `<p>Implementa las dos piezas matemáticas del proceso directo: el calendario acumulado <code>ᾱ</code> y el paso de ruidado. El reto usa el mismo calendario que <code>ddpm_2d.py</code> en la guía.</p>`,
    code: `// betas: arreglo de varianzas. Devuelve ᾱ_t = producto de (1 − β_s) hasta t
function alphaBarra(betas) {
  return betas.map(() => 1);
}
// x_t = √ᾱ · x0 + √(1 − ᾱ) · ε
function ruidar(x0, abT, eps) {
  return x0;
}
const betas = linspace(1e-4, 0.06, 200);
const ab = alphaBarra(betas);
print('ᾱ al final:', ab[199], '(en la guía ≈ 0.0022)');
print('ruidar(1, 0.25, 2) =', ruidar(1, 0.25, 2), '(esperado ≈ 2.2320)');
plot({ type: 'line', title: 'ᾱ_t y 1 − ᾱ_t', series: [{ name: 'ᾱ', y: ab }, { name: '1 − ᾱ', y: ab.map(a => 1 - a) }] });`,
    solution: `function alphaBarra(betas) {
  let p = 1;
  return betas.map((b) => (p *= 1 - b));
}
function ruidar(x0, abT, eps) {
  return Math.sqrt(abT) * x0 + Math.sqrt(1 - abT) * eps;
}
const betas = linspace(1e-4, 0.06, 200);
const ab = alphaBarra(betas);
print('ᾱ al final:', ab[199], '(en la guía ≈ 0.0022)');
print('ruidar(1, 0.25, 2) =', ruidar(1, 0.25, 2), '(esperado ≈ 2.2320)');
plot({ type: 'line', title: 'ᾱ_t y 1 − ᾱ_t', series: [{ name: 'ᾱ', y: ab }, { name: '1 − ᾱ', y: ab.map(a => 1 - a) }] });`,
    hints: ["Mantén un acumulador p que empieza en 1 y se multiplica por (1 − β) en cada elemento.", "ruidar: Math.sqrt(abT) * x0 + Math.sqrt(1 - abT) * eps.", "ᾱ_T debe quedar muy cerca de 0: al final solo queda ruido."],
    check: function () {
      const t = (n, ok, m) => ({ name: n, ok: !!ok, msg: ok ? "" : m });
      if (typeof alphaBarra !== "function" || typeof ruidar !== "function") return [t("Existen alphaBarra y ruidar", false, "Define ambas funciones.")];
      const betas = linspace(1e-4, 0.06, 200), ab = alphaBarra(betas), r = [];
      r.push(t("ᾱ₁ = 1 − β₁", Math.abs(ab[0] - (1 - 1e-4)) < 1e-12, "ab[0] = " + fmt(ab[0])));
      r.push(t("ᾱ decrece y ᾱ_T ≈ 0.00217", ab.every((a, i) => i === 0 || a < ab[i - 1]) && Math.abs(ab[199] - 0.002167) < 2e-5, "ab[199] = " + fmt(ab[199])));
      r.push(t("ruidar(1, 0.25, 2) ≈ 2.23205", Math.abs(ruidar(1, 0.25, 2) - 2.2320508) < 1e-6, "Obtuviste " + fmt(ruidar(1, 0.25, 2))));
      const g = rng(7), n = 20000, a = ab[99], xs = [];
      for (let i = 0; i < n; i++) xs.push(ruidar(0.7, a, g.normal()));
      r.push(t("Media = √ᾱ·x₀ y varianza = 1 − ᾱ", Math.abs(mean(xs) - Math.sqrt(a) * 0.7) < 0.02 && Math.abs(std(xs) ** 2 - (1 - a)) < 0.03, "Media " + fmt(mean(xs)) + ", varianza " + fmt(std(xs) ** 2) + "; esperadas " + fmt(Math.sqrt(a) * 0.7) + " y " + fmt(1 - a)));
      return r;
    },
    outro: `En la guía de la <b>Semana 7</b> entrenas una red que predice ε y muestreas con DDPM y DDIM sobre esta misma espiral y sobre imágenes. Lo que ves aquí con el denoiser ideal es lo que esa red aprende a aproximar.`,
  },
];

/* Lecciones previstas: aparecen como «próximamente» hasta que se publiquen. */
const PROXIMAS = [
  { title: "Un GPT completo", week: "Semana 5", note: "BPE, bloque Transformer, entrenamiento y generación en el navegador" },
  { title: "VAE y GAN", week: "Semana 6", note: "Espacio latente, ELBO y colapso de modos" },
  { title: "LoRA y DPO", week: "Semana 8", note: "Adaptar un modelo congelado y alinearlo con preferencias" },
];
if (typeof module !== "undefined") module.exports = { LESSONS, PROXIMAS };
