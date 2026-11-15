# Pruebas del laboratorio

El laboratorio ejecuta Python con un intérprete propio (`src/pyinterp.js`). Estas pruebas comprueban que se comporta como **CPython** (Python real).

```bash
cd curso-ia-generativa/laboratorio/tests
python3 generar_esperados.py     # corre ~130 programas en CPython y guarda su salida
node comparar_con_cpython.js     # los corre en el intérprete y compara carácter por carácter
python3 fuzz_aritmetica.py       # 4 000 expresiones aleatorias evaluadas en CPython
node fuzz_aritmetica.js          # idem en el intérprete
node lecciones.js                # cada lección: falla con los huecos, pasa con la solución
```

Resultado de referencia (CPython 3.13): ~130 programas con 3 diferencias a propósito (clases, `try` y `numpy`, no soportados) y 1 de último decimal en una suma de 1 000 números gaussianos; 4 000 expresiones con 10 diferencias, todas números complejos (la raíz de un negativo), que el intérprete rechaza con un aviso.

## Límites conocidos
- Funciones `exp`, `log`, `sin`, `cos`, `atan`: V8 y la biblioteca C de Python difieren en la última cifra en un 3–9 % de los valores. Las potencias enteras, la división entera de floats, `round`, `sum` (con compensación, como Python 3.12+) y `random` (mismo generador MT19937) sí coinciden.
- No hay clases, `try/except`, `with`, `yield`, conjuntos `{a, b}`, `*args/**kwargs` ni números complejos: avisan con un mensaje claro.

`cpython_shim.py` es la versión en Python de las ayudas de comprobación (`_t`, `_try`, `_cerca`) por si se quiere probar los retos directamente en CPython.
