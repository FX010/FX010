# Laboratorio Neuronal

Página interactiva para **aprender Python e IA en una hora**, pensada para quien sabe poco Python y tiene poco tiempo.

- **10 lecciones de 3 a 8 minutos** (≈ 56 min): Python básico → redes neuronales → IA generativa.
- Cada lección: una idea corta, ejemplos para ejecutar, un widget visual y un **reto con huecos `___`** que se comprueba solo (con pistas progresivas y solución).
- **Consola de Python** que comparte variables con las lecciones.
- Errores traducidos a español sencillo, y botón «Ya sé Python» para saltar el primer módulo.
- Funciona **sin instalar nada y sin conexión** una vez cargada la página.

Abre [`laboratorio-neuronal.html`](laboratorio-neuronal.html) en cualquier navegador.

## Cómo está hecho
| Archivo (`src/`) | Qué es |
|---|---|
| `pyinterp.js` | Intérprete de un subconjunto de Python 3, escrito desde cero. No usa `eval`, así que funciona incluso donde el navegador bloquea código generado. Detiene bucles infinitos a los 6 s sin perder las variables. |
| `lessons.js` | Contenido: módulos, lecciones, retos y sus comprobaciones (en Python). Para añadir una lección, agrega un objeto a `LESSONS`. |
| `app.js` | Interfaz: navegación, editor con resaltado, celdas, consola y widgets. |
| `kernel.js` | Biblioteca numérica en JavaScript que usan solo los widgets (red neuronal en vivo, etc.). |
| `styles.css`, `template.html`, `build.py` | Estilos, plantilla y el script que las une: `python3 src/build.py` regenera `laboratorio-neuronal.html` y `dist/artifact.html`. |
| `legacy-js/` | Primera versión del laboratorio (código en JavaScript, con temas avanzados), por si se quiere recuperar. |

## Qué Python ejecuta
Soporta: variables, `int`/`float` exactos, texto, listas, tuplas, diccionarios, `if/elif/else`, `for`, `while`, `def`, `lambda`, comprensiones, f-strings, `import math` / `import random` y las funciones integradas más usadas.
No soporta (avisa con un mensaje claro): clases, `try/except`, `with`, `yield`, conjuntos, `*args/**kwargs`, números complejos y librerías externas como `numpy`.

## Verificación
`tests/` compara el intérprete con **CPython real** (≈ 130 programas y 4 000 expresiones aleatorias) y comprueba que cada reto falla con los huecos y pasa con su solución. Ver [`tests/README.md`](tests/README.md) para los resultados y los límites conocidos (por ejemplo, la última cifra de `exp`/`log`/`sin`/`cos` puede diferir).

## Próximas lecciones
Retropropagación con autograd, atención y GPT, LoRA/DPO/RAG (nivel 2, en Python).
