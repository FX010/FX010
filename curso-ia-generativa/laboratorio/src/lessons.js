/* Contenido del laboratorio (versión para quien empieza con Python).
   Cada lección: idea corta → ejemplos para ejecutar → reto con huecos ___ que se comprueba solo.
   El código de los retos es Python real. Para añadir una lección, agrega un objeto a LESSONS. */
const MODULES = [
  { id: "python", title: "Python básico", blurb: "Variables, listas, funciones y decisiones: solo lo necesario." },
  { id: "redes", title: "Redes neuronales", blurb: "Una neurona, cómo aprende y qué pasa al juntar muchas." },
  { id: "generativa", title: "IA generativa", blurb: "Cómo un modelo predice, elige y crea a partir de ruido." },
];

const LESSONS = [
  // ============================================================ MÓDULO 1 · PYTHON
  {
    id: "hola", module: "python", num: 1, title: "Tu primer programa", minutes: 3,
    goal: "Guardar datos en variables y mostrarlos con print.",
    intro: `<p>Un programa es una lista de instrucciones. Con <code>print()</code> muestras algo en pantalla. Con el signo <code>=</code> guardas un valor en una <b>variable</b>, que es una cajita con nombre.</p>
      <p>Python hace cuentas con <code>+ - * /</code>. Las potencias se escriben con <code>**</code> y el texto va entre comillas.</p>`,
    examples: [{ note: "Pulsa Ejecutar. Luego cambia los valores y vuelve a ejecutar.", code: `nombre = "Ana"
edad = 30
print("Hola", nombre)
print("En 5 años tendrás", edad + 5)` }],
    taskIntro: `<p>Completa el hueco <code>___</code> para calcular el total de una compra: el precio por la cantidad.</p>`,
    code: `precio = 20
cantidad = 3
total = ___      # multiplica precio por cantidad
print("Total:", total)`,
    solution: `precio = 20
cantidad = 3
total = precio * cantidad
print("Total:", total)`,
    hints: ["Multiplicar se escribe con el signo *.", "Escribe: total = precio * cantidad"],
    check: `def probar():
    r = []
    ok = all(n in U for n in ("precio", "cantidad", "total"))
    _t(r, "Existen precio, cantidad y total", ok, "Crea las tres variables.")
    if ok:
        _t(r, "total es precio por cantidad", U["total"] == U["precio"] * U["cantidad"], "Ahora total vale " + repr(U["total"]) + ". Debe ser precio * cantidad.")
    return r`,
    outro: `Ya sabes crear variables y mostrar resultados. En las redes neuronales todo son números guardados en variables.`,
  },
  {
    id: "listas", module: "python", num: 2, title: "Listas y bucles", minutes: 5,
    goal: "Recorrer una lista con for y calcular una media.",
    intro: `<p>Una <b>lista</b> guarda varios valores entre corchetes: <code>[6, 8, 7]</code>. El primero está en la posición <b>0</b>.</p>
      <p>Con <code>for</code> repites algo para cada elemento. Lo que va dentro del bucle lleva <b>sangría</b>: 4 espacios al inicio de la línea.</p>`,
    examples: [{ note: "Ejecútalo y prueba a cambiar las notas.", code: `notas = [6, 8, 7]
print(notas[0], len(notas))        # el primero (posición 0) y cuántos hay
for n in notas:
    print("Nota:", n, "doble:", n * 2)
notas.append(10)                   # añadir al final
print(notas)` }],
    taskIntro: `<p>Calcula la media de las notas. Dentro del bucle, ve acumulando la suma.</p>`,
    code: `notas = [6, 8, 7, 9]
suma = 0
for n in notas:
    suma = ___        # lo que ya llevas más la nota n
media = suma / len(notas)
print("Media:", media)`,
    solution: `notas = [6, 8, 7, 9]
suma = 0
for n in notas:
    suma = suma + n
media = suma / len(notas)
print("Media:", media)`,
    hints: ["En cada vuelta, n es una nota distinta. Hay que sumarla a lo que ya había.", "Escribe: suma = suma + n"],
    check: `def probar():
    r = []
    ok = all(n in U for n in ("notas", "suma", "media"))
    _t(r, "Existen notas, suma y media", ok, "Ejecuta la celda completa.")
    if ok:
        _t(r, "suma es la suma de todas las notas", _cerca(U["suma"], sum(U["notas"])), "suma vale " + repr(U["suma"]) + " y debería ser " + repr(sum(U["notas"])) + ".")
        _t(r, "media = suma / cantidad de notas", _cerca(U["media"], sum(U["notas"]) / len(U["notas"])), "media vale " + repr(U["media"]) + ".")
    return r`,
    outro: `Los bucles son el motor del aprendizaje automático: una red repite miles de veces los mismos pasos.`,
  },
  {
    id: "funciones", module: "python", num: 3, title: "Funciones", minutes: 5,
    goal: "Crear tus propias funciones con def y return.",
    intro: `<p>Una <b>función</b> es una receta con nombre: recibe datos, hace algo y devuelve un resultado con <code>return</code>.</p>
      <p>Se define con <code>def</code> y se usa escribiendo su nombre con paréntesis. Ojo con los dos puntos <code>:</code> y la sangría.</p>`,
    examples: [{ note: "Define la función una vez y úsala las veces que quieras.", code: `def doble(x):
    return x * 2

print(doble(4))
print(doble(10) + 1)` }],
    taskIntro: `<p>Completa las dos funciones. Cuando las ejecutes, se imprimirán sus resultados de prueba.</p>`,
    code: `def cuadrado(x):
    return ___        # x multiplicado por x

def area_rectangulo(base, altura):
    return ___

print(cuadrado(5), area_rectangulo(3, 4))`,
    solution: `def cuadrado(x):
    return x * x

def area_rectangulo(base, altura):
    return base * altura

print(cuadrado(5), area_rectangulo(3, 4))`,
    hints: ["El cuadrado de un número es ese número por sí mismo.", "cuadrado: return x * x. Área: return base * altura."],
    check: `def probar():
    r = []
    ok1 = callable(U.get("cuadrado"))
    ok2 = callable(U.get("area_rectangulo"))
    _t(r, "Existen las dos funciones", ok1 and ok2, "Define cuadrado(x) y area_rectangulo(base, altura).")
    if ok1:
        _t(r, "cuadrado(5) = 25", _try(U["cuadrado"], 5) == 25, "Obtuviste " + repr(_try(U["cuadrado"], 5)) + ".")
        _t(r, "cuadrado(-3) = 9", _try(U["cuadrado"], -3) == 9, "Obtuviste " + repr(_try(U["cuadrado"], -3)) + ".")
    if ok2:
        _t(r, "area_rectangulo(3, 4) = 12", _try(U["area_rectangulo"], 3, 4) == 12, "Obtuviste " + repr(_try(U["area_rectangulo"], 3, 4)) + ".")
        _t(r, "area_rectangulo(2.5, 2) = 5.0", _cerca(_try(U["area_rectangulo"], 2.5, 2), 5.0), "Obtuviste " + repr(_try(U["area_rectangulo"], 2.5, 2)) + ".")
    return r`,
    outro: `Una red neuronal es, en el fondo, una función enorme con muchos números ajustables. Ahora sabes escribir funciones.`,
  },
  {
    id: "decisiones", module: "python", num: 4, title: "Decisiones con if", minutes: 4,
    goal: "Elegir qué hacer según una condición.",
    intro: `<p><code>if</code> ejecuta algo solo si una condición es verdadera. Compara con <code>==</code> (igual), <code>!=</code> (distinto), <code>&gt;</code>, <code>&lt;</code>, <code>&gt;=</code> y <code>&lt;=</code>.</p>
      <p>Con <code>elif</code> y <code>else</code> añades más casos. Python prueba las condiciones en orden y se queda con la primera que se cumple.</p>`,
    examples: [{ note: "Cambia la temperatura y mira qué rama se ejecuta.", code: `temperatura = 28
if temperatura > 30:
    print("Hace calor")
elif temperatura > 15:
    print("Clima agradable")
else:
    print("Hace frío")` }],
    taskIntro: `<p>Completa la función que clasifica una nota: 9 o más es sobresaliente, 5 o más es aprobado y menos de 5 es suspenso.</p>`,
    code: `def clasificar(nota):
    if nota >= 9:
        return "sobresaliente"
    elif ___:
        return "aprobado"
    else:
        return ___

print(clasificar(9.5), clasificar(6), clasificar(3))`,
    solution: `def clasificar(nota):
    if nota >= 9:
        return "sobresaliente"
    elif nota >= 5:
        return "aprobado"
    else:
        return "suspenso"

print(clasificar(9.5), clasificar(6), clasificar(3))`,
    hints: ["La segunda condición es para notas de 5 o más.", "elif nota >= 5: …  y al final return \"suspenso\""],
    check: `def probar():
    r = []
    if not callable(U.get("clasificar")):
        return [["Existe la función clasificar(nota)", False, "Define def clasificar(nota)."]]
    f = U["clasificar"]
    for nota, esperado in [(9, "sobresaliente"), (9.5, "sobresaliente"), (5, "aprobado"), (6.9, "aprobado"), (4.9, "suspenso"), (0, "suspenso")]:
        _t(r, "clasificar(" + str(nota) + ") = " + esperado, _try(f, nota) == esperado, "Obtuviste " + repr(_try(f, nota)) + ".")
    return r`,
    outro: `Con variables, bucles, funciones y decisiones ya puedes leer casi cualquier código de IA. Vamos con las redes neuronales.`,
  },

  // ============================================================ MÓDULO 2 · REDES NEURONALES
  {
    id: "neurona", module: "redes", num: 5, title: "La neurona", minutes: 6, widget: "neurona",
    goal: "Programar una neurona artificial y ver qué decide.",
    intro: `<p>Una neurona artificial es una función muy simple. Multiplica cada entrada por un <b>peso</b>, suma todo junto con un <b>sesgo</b> y «aplasta» el resultado con una función como <code>math.tanh</code>, que devuelve un número entre −1 y 1.</p>
      <p class="formula">salida = tanh( peso₁·x₁ + peso₂·x₂ + sesgo )</p>
      <p>Las redes gigantes son millones de estas piezas conectadas.</p>`,
    examples: [{ note: "Una neurona con dos entradas, paso a paso.", code: `import math
entradas = [1.0, 2.0]
pesos = [0.5, -0.3]
sesgo = 0.1
suma = sesgo
for x, w in zip(entradas, pesos):
    suma = suma + x * w
print("suma:", suma)
print("salida:", math.tanh(suma))` }],
    widgetTitle: "Mueve los pesos y mira qué decide la neurona",
    widgetNote: "El color es la salida en cada punto (x₁, x₂). La línea discontinua es donde la neurona cambia de opinión. Haz clic en el mapa para probar un punto.",
    taskIntro: `<p>Convierte la neurona en una función. Después elige pesos y sesgo para que una neurona se comporte como la puerta lógica <b>AND</b>: salida alta solo si las dos entradas valen 1.</p>`,
    code: `import math

def neurona(pesos, entradas, sesgo):
    # 1) suma cada peso por su entrada   2) añade el sesgo   3) aplica math.tanh
    suma = sesgo
    for p, x in zip(pesos, entradas):
        suma = ___
    return math.tanh(suma)

print(neurona([1, 2], [0.5, -1], 0.1))     # debería dar ≈ -0.8854

# Reto 2: elige w1, w2 y b para que la neurona sea una puerta AND
w1 = 0
w2 = 0
b = 0
def sigmoide(z):
    return 1 / (1 + math.exp(-z))          # devuelve un número entre 0 y 1
for a in [0, 1]:
    for c in [0, 1]:
        print(a, c, "->", round(sigmoide(w1 * a + w2 * c + b), 2))`,
    solution: `import math

def neurona(pesos, entradas, sesgo):
    suma = sesgo
    for p, x in zip(pesos, entradas):
        suma = suma + p * x
    return math.tanh(suma)

print(neurona([1, 2], [0.5, -1], 0.1))

w1 = 10
w2 = 10
b = -15
def sigmoide(z):
    return 1 / (1 + math.exp(-z))
for a in [0, 1]:
    for c in [0, 1]:
        print(a, c, "->", round(sigmoide(w1 * a + w2 * c + b), 2))`,
    hints: ["Cada vuelta suma peso por entrada: suma = suma + p * x.", "Para AND necesitas que w1 + w2 + b sea claramente positivo, pero que w1 + b, w2 + b y b sean negativos.", "Prueba w1 = 10, w2 = 10 y un sesgo b = -15."],
    check: `def probar():
    r = []
    import math
    f = U.get("neurona")
    if not callable(f):
        r.append(["Existe la función neurona(pesos, entradas, sesgo)", False, "Defínela con def neurona(...)."])
    else:
        y = _try(f, [1, 2], [0.5, -1], 0.1)
        _t(r, "neurona([1,2], [0.5,-1], 0.1) ≈ -0.8854", _cerca(y, math.tanh(-1.4)), "Obtuviste " + repr(y) + ". Debe ser tanh(1·0.5 + 2·(-1) + 0.1).")
        _t(r, "Con pesos 0 solo cuenta el sesgo", _cerca(_try(f, [0, 0], [3, 4], 0.5), math.tanh(0.5)), "Con pesos 0 la salida es tanh(sesgo).")
        _t(r, "Funciona con una sola entrada", _cerca(_try(f, [3], [1], 0.5), math.tanh(3.5)), "No supongas que siempre hay dos entradas.")
    if all(n in U for n in ("w1", "w2", "b")):
        s = lambda a, c: 1 / (1 + math.exp(-(U["w1"] * a + U["w2"] * c + U["b"])))
        o = [s(0, 0), s(0, 1), s(1, 0), s(1, 1)]
        _t(r, "La neurona implementa AND", o[3] > 0.9 and o[0] < 0.1 and o[1] < 0.1 and o[2] < 0.1, "Salidas (0,0) (0,1) (1,0) (1,1): " + str([round(v, 2) for v in o]) + ". Debe ser casi [0, 0, 0, 1].")
    else:
        r.append(["Existen w1, w2 y b", False, "Define las tres variables."])
    return r`,
    outro: `Una neurona traza una frontera recta. Para curvas hace falta juntar varias, pero antes hay que ver cómo se eligen los pesos solos.`,
  },
  {
    id: "descenso", module: "redes", num: 6, title: "Aprender es bajar el error", minutes: 6, widget: "gradiente",
    goal: "Entender el descenso de gradiente y programarlo.",
    intro: `<p>Una red aprende midiendo su <b>error</b> y ajustando sus números para que sea menor. Imagina una pelota en una montaña: la <b>pendiente</b> te dice hacia dónde baja el terreno y das un paso pequeño en esa dirección.</p>
      <p class="formula">x = x − tasa · pendiente</p>
      <p>La <b>tasa de aprendizaje</b> es el tamaño del paso. Si es muy pequeña, avanzas lento. Si es muy grande, te pasas del mínimo y el error se dispara.</p>`,
    examples: [{ note: "El error es (x − 3)². Su pendiente es 2·(x − 3) y el mínimo está en x = 3.", code: `x = 0.0
tasa = 0.1
for paso in range(6):
    pendiente = 2 * (x - 3)
    x = x - tasa * pendiente
    print(paso, round(x, 3))` }],
    widgetTitle: "Un paso a la vez",
    widgetNote: "Esta curva tiene dos valles. Cambia la tasa y el punto de partida, y pulsa Paso o Ejecutar.",
    taskIntro: `<p>Programa el descenso de gradiente como una función. Después elige la tasa <code>TASA</code> que llega al mínimo (x = 3) en solo 5 pasos.</p>`,
    code: `def descenso(pendiente, x0, tasa, pasos):
    x = x0
    for _ in range(pasos):
        x = ___           # x menos tasa por la pendiente en x
    return x

def mi_pendiente(x):
    return 2 * (x - 3)

print(descenso(mi_pendiente, 0, 0.1, 50))   # cerca de 3
print(descenso(mi_pendiente, 0, 1.1, 50))   # se dispara

# Reto 2: ¿qué tasa llega a ≈ 3 en solo 5 pasos?
TASA = 0.1
print(descenso(mi_pendiente, 0, TASA, 5))`,
    solution: `def descenso(pendiente, x0, tasa, pasos):
    x = x0
    for _ in range(pasos):
        x = x - tasa * pendiente(x)
    return x

def mi_pendiente(x):
    return 2 * (x - 3)

print(descenso(mi_pendiente, 0, 0.1, 50))
print(descenso(mi_pendiente, 0, 1.1, 50))

TASA = 0.5
print(descenso(mi_pendiente, 0, TASA, 5))`,
    hints: ["Dentro del bucle: la pendiente en x se calcula llamando a pendiente(x).", "x = x - tasa * pendiente(x)", "Con tasa 0.5 el error desaparece en un paso. Sirve cualquier valor entre 0.4 y 0.6."],
    check: `def probar():
    r = []
    d = U.get("descenso")
    if not callable(d):
        return [["Existe descenso(pendiente, x0, tasa, pasos)", False, "Defínela con def."]]
    g = lambda x: 2 * (x - 3)
    v1 = _try(d, g, 0, 0.1, 1)
    _t(r, "Un paso desde 0 con tasa 0.1 da 0.6", _cerca(v1, 0.6, 1e-9), "Obtuviste " + repr(v1) + ". Cuenta: 0 - 0.1 * 2 * (0 - 3).")
    v2 = _try(d, g, 0, 0.1, 50)
    _t(r, "Con 50 pasos llega a 3", _cerca(v2, 3, 1e-3), "Obtuviste " + repr(v2) + ".")
    v3 = _try(d, g, 0, 1.1, 50)
    _t(r, "Con tasa 1.1 se dispara", (not isinstance(v3, str)) and abs(v3) > 1000, "Con tasa 1.1 cada paso aumenta el error, así que el valor debe crecer mucho.")
    if "TASA" in U:
        v4 = _try(d, g, 0, U["TASA"], 5)
        _t(r, "Tu TASA llega a 3 en 5 pasos (error < 0.01)", _cerca(v4, 3, 0.01), "Con TASA = " + repr(U["TASA"]) + " obtienes " + repr(v4) + ".")
    else:
        r.append(["Existe TASA", False, "Define la variable TASA."])
    return r`,
    outro: `Todo el entrenamiento de las redes, de esta parábola a un modelo como ChatGPT, repite este paso. Cambia cómo se calcula la pendiente, no la idea.`,
  },
  {
    id: "aprende", module: "redes", num: 7, title: "Tu neurona aprende sola", minutes: 8, widget: "playground",
    goal: "Entrenar una neurona para que aprenda la puerta OR.",
    intro: `<p>Hasta ahora elegiste los pesos tú. Aprender es dejar que el programa los ajuste solo: prueba un ejemplo, mide el error y corrige un poquito. Se repite miles de veces.</p>
      <p>La corrección usa la idea del descenso: cada peso se mueve en proporción a <b>error × su entrada</b>, y el sesgo en proporción al <b>error</b>.</p>`,
    examples: [{ note: "Una sola corrección, a cámara lenta. Mira cómo cambia el peso.", code: `import math
def sigmoide(z):
    return 1 / (1 + math.exp(-z))

w, b = 0.0, 0.0
x, y = 1, 1                        # entrada 1, resultado deseado 1
pred = sigmoide(w * x + b)
print("predicción:", pred, " error:", pred - y)
w = w - 0.5 * (pred - y) * x       # corregimos el peso
b = b - 0.5 * (pred - y)           # y el sesgo
print("nuevo peso:", w, " nuevo sesgo:", b)` }],
    widgetTitle: "Con varias capas la red dibuja curvas",
    widgetNote: "Una sola neurona solo traza rectas. Pulsa Entrenar y mira cómo una red con capas aprende las espirales. Prueba menos neuronas, o Adam como optimizador.",
    taskIntro: `<p>Completa el entrenamiento. La neurona debe aprender la puerta <b>OR</b>: da 1 si al menos una entrada es 1.</p>`,
    code: `import math

def sigmoide(z):
    return 1 / (1 + math.exp(-z))

datos = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]   # (x1, x2, resultado): puerta OR
w1, w2, b = 0.0, 0.0, 0.0
tasa = 0.5

for vuelta in range(2000):
    for x1, x2, y in datos:
        pred = sigmoide(w1 * x1 + w2 * x2 + b)
        error = pred - y
        w1 = w1 - tasa * error * ___       # peso 1: error por su entrada
        w2 = w2 - tasa * error * ___       # peso 2: error por su entrada
        b = b - tasa * ___                 # sesgo: solo el error

for x1, x2, y in datos:
    print(x1, x2, "->", round(sigmoide(w1 * x1 + w2 * x2 + b), 2))`,
    solution: `import math

def sigmoide(z):
    return 1 / (1 + math.exp(-z))

datos = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
w1, w2, b = 0.0, 0.0, 0.0
tasa = 0.5

for vuelta in range(2000):
    for x1, x2, y in datos:
        pred = sigmoide(w1 * x1 + w2 * x2 + b)
        error = pred - y
        w1 = w1 - tasa * error * x1
        w2 = w2 - tasa * error * x2
        b = b - tasa * error

for x1, x2, y in datos:
    print(x1, x2, "->", round(sigmoide(w1 * x1 + w2 * x2 + b), 2))`,
    hints: ["El peso 1 se corrige con el error por x1; el peso 2, con el error por x2.", "El sesgo no multiplica por ninguna entrada: solo usa el error.", "w1 = w1 - tasa * error * x1  ·  w2 = w2 - tasa * error * x2  ·  b = b - tasa * error"],
    check: `def probar():
    r = []
    import math
    if not all(n in U for n in ("w1", "w2", "b")):
        return [["Existen w1, w2 y b", False, "Ejecuta la celda completa para entrenar."]]
    w1, w2, b = U["w1"], U["w2"], U["b"]
    p = lambda x1, x2: 1 / (1 + math.exp(-(w1 * x1 + w2 * x2 + b)))
    _t(r, "Los pesos cambiaron (la neurona aprendió algo)", abs(w1) + abs(w2) > 0.5, "Los pesos siguen casi en 0. ¿Completaste los tres huecos?")
    _t(r, "(0, 0) → casi 0", p(0, 0) < 0.25, "Da " + str(round(p(0, 0), 2)) + ". Debería ser menor que 0.25.")
    for x1, x2 in [(0, 1), (1, 0), (1, 1)]:
        _t(r, "(" + str(x1) + ", " + str(x2) + ") → casi 1", p(x1, x2) > 0.75, "Da " + str(round(p(x1, x2), 2)) + ". Debería ser mayor que 0.75.")
    return r`,
    outro: `Eso es entrenar: predecir, medir el error y corregir. Una red de millones de neuronas hace lo mismo, con la ayuda de la retropropagación para calcular todas las correcciones.`,
  },

  // ============================================================ MÓDULO 3 · IA GENERATIVA
  {
    id: "siguiente", module: "generativa", num: 8, title: "Predecir la siguiente letra", minutes: 7, widget: "lenguaje",
    goal: "Construir un mini modelo de lenguaje por conteo.",
    intro: `<p>Un modelo de lenguaje, como el de un chat, hace una sola cosa: <b>predecir qué viene después</b>. La versión más simple es contar: en un texto, ¿qué letra aparece después de la «e»?</p>
      <p>Guardamos los conteos en un <b>diccionario</b> (pares clave → valor). Los modelos grandes hacen lo mismo con redes neuronales y muchísimo más contexto.</p>`,
    examples: [{ note: "Cuenta qué letras vienen después de «e» en una frase.", code: `texto = "el gato come pescado. el gato mira el pez."
conteo = {}
for i in range(len(texto) - 1):
    if texto[i] == "e":
        sig = texto[i + 1]
        conteo[sig] = conteo.get(sig, 0) + 1
print(conteo)` }],
    widgetTitle: "Más contexto, menos error",
    widgetNote: "Un modelo de conteo con contexto de n letras. La línea verde es el mínimo posible para este texto. Genera frases y cambia la temperatura.",
    taskIntro: `<p>Escribe <code>siguiente(texto, letra)</code>, que devuelve la letra que más veces aparece justo después de <code>letra</code>. El conteo ya está hecho; completa la parte que elige la mejor.</p>`,
    code: `def siguiente(texto, letra):
    conteo = {}
    for i in range(len(texto) - 1):
        if texto[i] == letra:
            sig = texto[i + 1]
            conteo[sig] = conteo.get(sig, 0) + 1
    mejor = ""
    mejor_n = 0
    for sig in conteo:
        if conteo[sig] > mejor_n:
            mejor = ___         # la letra que va ganando
            mejor_n = ___       # cuántas veces apareció
    return mejor

print(siguiente("abababac", "a"))
print(siguiente("el gato come pescado", "e"))`,
    solution: `def siguiente(texto, letra):
    conteo = {}
    for i in range(len(texto) - 1):
        if texto[i] == letra:
            sig = texto[i + 1]
            conteo[sig] = conteo.get(sig, 0) + 1
    mejor = ""
    mejor_n = 0
    for sig in conteo:
        if conteo[sig] > mejor_n:
            mejor = sig
            mejor_n = conteo[sig]
    return mejor

print(siguiente("abababac", "a"))
print(siguiente("el gato come pescado", "e"))`,
    hints: ["Cuando una letra supera el mejor conteo, pasa a ser la nueva mejor.", "mejor = sig  y  mejor_n = conteo[sig]"],
    check: `def probar():
    r = []
    f = U.get("siguiente")
    if not callable(f):
        return [["Existe siguiente(texto, letra)", False, "Define def siguiente(texto, letra)."]]
    for texto, letra, esperado in [("abababac", "a", "b"), ("hola hola", "h", "o"), ("el gato come", "c", "o"), ("xyz", "q", "")]:
        _t(r, "siguiente(\\"" + texto + "\\", \\"" + letra + "\\") = \\"" + esperado + "\\"", _try(f, texto, letra) == esperado, "Obtuviste " + repr(_try(f, texto, letra)) + ".")
    return r`,
    outro: `Esto es un modelo de lenguaje de verdad, el más pequeño posible. ChatGPT y similares cambian los conteos por una red neuronal enorme que mira miles de palabras de contexto.`,
  },
  {
    id: "temperatura", module: "generativa", num: 9, title: "Temperatura: orden o creatividad", minutes: 6, widget: "temperatura",
    goal: "Controlar cuánto se arriesga un modelo al elegir.",
    intro: `<p>El modelo no elige siempre la opción más probable: <b>sortea</b> según las probabilidades. La <b>temperatura</b> controla ese sorteo. Con temperatura baja casi siempre gana la más probable (respuestas previsibles). Con temperatura alta hay más variedad, y también más errores.</p>
      <p>Se consigue elevando cada probabilidad a <code>1 / T</code> y volviendo a repartir para que sumen 1.</p>`,
    examples: [{ note: "Mira cómo cambian las probabilidades con la temperatura.", code: `probs = [0.1, 0.6, 0.3]
for T in [0.2, 1, 5]:
    pesos = [p ** (1 / T) for p in probs]     # forma corta de hacer un for
    total = sum(pesos)
    print("T =", T, [round(p / total, 2) for p in pesos])` }],
    widgetTitle: "Mueve la temperatura",
    widgetNote: "Cada barra es la probabilidad de esa opción después de aplicar la temperatura. Al lado de cada letra ves cuánto valía antes.",
    taskIntro: `<p>Completa <code>muestrear</code>: recibe las probabilidades, la temperatura y un número al azar <code>u</code> entre 0 y 1, y devuelve la posición elegida.</p>`,
    code: `def muestrear(probs, T, u):
    pesos = []
    for p in probs:
        pesos.append(___)            # p elevado a 1 / T
    total = sum(pesos)
    acumulado = 0
    for i in range(len(pesos)):
        acumulado = acumulado + pesos[i] / total
        if u < acumulado:
            return ___               # la posición elegida
    return len(pesos) - 1

p = [0.1, 0.6, 0.3]
print(muestrear(p, 1, 0.05), muestrear(p, 1, 0.5), muestrear(p, 1, 0.9))   # 0 1 2
print(muestrear(p, 0.05, 0.01))                                           # casi siempre la mejor: 1`,
    solution: `def muestrear(probs, T, u):
    pesos = []
    for p in probs:
        pesos.append(p ** (1 / T))
    total = sum(pesos)
    acumulado = 0
    for i in range(len(pesos)):
        acumulado = acumulado + pesos[i] / total
        if u < acumulado:
            return i
    return len(pesos) - 1

p = [0.1, 0.6, 0.3]
print(muestrear(p, 1, 0.05), muestrear(p, 1, 0.5), muestrear(p, 1, 0.9))
print(muestrear(p, 0.05, 0.01))`,
    hints: ["El primer hueco es p ** (1 / T).", "Cuando u cae dentro del tramo acumulado, devuelve la posición i.", "pesos.append(p ** (1 / T))  y  return i"],
    check: `def probar():
    r = []
    f = U.get("muestrear")
    if not callable(f):
        return [["Existe muestrear(probs, T, u)", False, "Define def muestrear(probs, T, u)."]]
    p = [0.1, 0.6, 0.3]
    a = [_try(f, p, 1, u) for u in (0.05, 0.5, 0.9)]
    _t(r, "T = 1 respeta las probabilidades (u = 0.05, 0.5, 0.9 → 0, 1, 2)", a == [0, 1, 2], "Obtuviste " + str(a) + ". Con T = 1 los tramos acumulados son 0.1 y 0.7.")
    b = [_try(f, p, 0.05, u) for u in (0.01, 0.4, 0.99)]
    _t(r, "Con T muy baja gana siempre la más probable", b == [1, 1, 1], "Obtuviste " + str(b) + ". Con T = 0.05 debe salir siempre la posición 1.")
    c = [_try(f, p, 100, u) for u in (0.2, 0.5, 0.9)]
    _t(r, "Con T muy alta casi todo da igual (tercios)", c == [0, 1, 2], "Obtuviste " + str(c) + ". Con T = 100 las tres opciones son casi equiprobables.")
    return r`,
    outro: `Esa es la perilla de «creatividad» de los chats: temperatura baja para datos y código, más alta para ideas y poesía.`,
  },
  {
    id: "difusion", module: "generativa", num: 10, title: "De ruido a imagen", minutes: 6, widget: "difusion",
    goal: "Entender y programar el truco de los generadores de imágenes.",
    intro: `<p>Generadores como Stable Diffusion aprenden a <b>quitar ruido</b>. Para entrenarlos, primero se destruye la imagen mezclándola con ruido aleatorio. Después, una red aprende a deshacer cada paso. Para crear algo nuevo, se parte de ruido puro y se va limpiando.</p>
      <p class="formula">mezcla = √ᾱ · imagen + √(1 − ᾱ) · ruido</p>
      <p><code>ᾱ</code> («alfa barra») es la fracción de imagen que queda: empieza en 1 y baja casi a 0.</p>`,
    examples: [{ note: "Un solo píxel con valor 1 que se va llenando de ruido.", code: `import math, random
imagen = 1.0
for queda in [1.0, 0.5, 0.1, 0.0]:
    ruido = random.random() * 2 - 1
    mezcla = math.sqrt(queda) * imagen + math.sqrt(1 - queda) * ruido
    print("queda", queda, "->", round(mezcla, 2))` }],
    widgetTitle: "Destruir y reconstruir una espiral",
    widgetNote: "Arrastra t para llenar de ruido los puntos azules. «Generar» parte de ruido y lo ordena paso a paso (aquí con un limpiador ideal; una red lo aprende).",
    taskIntro: `<p>Programa las dos piezas: cuánta imagen queda en cada paso (<code>alpha_barra</code>) y la mezcla con ruido (<code>ruidar</code>).</p>`,
    code: `import math

betas = [0.0001 + (0.06 - 0.0001) * i / 199 for i in range(200)]   # ruido añadido en cada paso

def alpha_barra(betas):
    resultado = []
    p = 1
    for b in betas:
        p = ___            # lo que llevas por (1 - b)
        resultado.append(p)
    return resultado

def ruidar(x0, ab, ruido):
    return ___             # raiz(ab) * x0 + raiz(1 - ab) * ruido

ab = alpha_barra(betas)
print("al final queda:", ab[199])      # casi 0
print(ruidar(1, 0.25, 2))              # ≈ 2.2320
grafica(ab, "fracción de imagen que queda")`,
    solution: `import math

betas = [0.0001 + (0.06 - 0.0001) * i / 199 for i in range(200)]

def alpha_barra(betas):
    resultado = []
    p = 1
    for b in betas:
        p = p * (1 - b)
        resultado.append(p)
    return resultado

def ruidar(x0, ab, ruido):
    return math.sqrt(ab) * x0 + math.sqrt(1 - ab) * ruido

ab = alpha_barra(betas)
print("al final queda:", ab[199])
print(ruidar(1, 0.25, 2))
grafica(ab, "fracción de imagen que queda")`,
    hints: ["Cada paso multiplica lo acumulado por (1 - b).", "Para la mezcla usa math.sqrt dos veces.", "p = p * (1 - b)  ·  return math.sqrt(ab) * x0 + math.sqrt(1 - ab) * ruido"],
    check: `def probar():
    r = []
    import math, random
    if not (callable(U.get("alpha_barra")) and callable(U.get("ruidar"))):
        return [["Existen alpha_barra y ruidar", False, "Define las dos funciones."]]
    betas = [0.0001 + (0.06 - 0.0001) * i / 199 for i in range(200)]
    ab = _try(U["alpha_barra"], betas)
    if isinstance(ab, str):
        return [["alpha_barra se ejecuta sin errores", False, ab]]
    _t(r, "alpha_barra(betas)[0] = 1 - beta0", _cerca(ab[0], 1 - betas[0], 1e-12), "ab[0] vale " + repr(ab[0]) + ".")
    _t(r, "La fracción baja y al final es ≈ 0.00217", all(ab[i] < ab[i - 1] for i in range(1, 200)) and _cerca(ab[199], 0.002167, 2e-5), "ab[199] vale " + repr(ab[199]) + ".")
    v = _try(U["ruidar"], 1, 0.25, 2)
    _t(r, "ruidar(1, 0.25, 2) ≈ 2.23205", _cerca(v, 2.2320508, 1e-6), "Obtuviste " + repr(v) + ".")
    random.seed(7)
    a = ab[99]
    xs = []
    for _ in range(4000):
        g = math.sqrt(-2 * math.log(1 - random.random())) * math.cos(2 * math.pi * random.random())
        xs.append(_try(U["ruidar"], 0.7, a, g))
    if any(isinstance(x, str) for x in xs):
        r.append(["ruidar funciona con muchos valores", False, str(xs[0])])
    else:
        m = sum(xs) / len(xs)
        var = sum((x - m) ** 2 for x in xs) / len(xs)
        _t(r, "La media es raiz(ab)·x0 y la varianza es 1 - ab", _cerca(m, math.sqrt(a) * 0.7, 0.05) and _cerca(var, 1 - a, 0.06), "Media " + str(round(m, 3)) + ", varianza " + str(round(var, 3)) + "; esperadas " + str(round(math.sqrt(a) * 0.7, 3)) + " y " + str(round(1 - a, 3)) + ".")
    return r`,
    outro: `Con esto ya conoces la idea detrás de los generadores de imágenes modernos. En el curso completo entrenas la red que quita el ruido.`,
  },
];

/* Próximamente: aparecen en la barra lateral hasta que se publiquen. */
const PROXIMAS = [
  { title: "Nivel 2: retropropagación", week: "Python avanzado", note: "Calcular gradientes solo, con una mini versión de autograd" },
  { title: "Nivel 2: atención y GPT", week: "Python avanzado", note: "Cómo un Transformer lee el contexto" },
  { title: "Nivel 2: LoRA, DPO y RAG", week: "Python avanzado", note: "Adaptar y alinear modelos; buscar en documentos" },
];
if (typeof module !== "undefined") module.exports = { LESSONS, MODULES, PROXIMAS };
