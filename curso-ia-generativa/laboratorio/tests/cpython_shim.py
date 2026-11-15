# Entorno de Python del laboratorio. Se ejecuta una vez al cargar Brython.
# Todo lo que el alumno escribe corre en el diccionario G, así las variables se conservan entre celdas.
import sys, json, math, random
from browser import window


class Salida:
    def write(self, s):
        window.pyout(s)

    def flush(self):
        pass


sys.stdout = Salida()
sys.stderr = Salida()
G = {"__name__": "__main__"}


def grafica(valores, titulo=""):
    """Dibuja una línea con los valores de una lista."""
    window.pyplot(json.dumps({"type": "line", "title": titulo, "series": [{"name": "valores", "y": [float(v) for v in valores]}]}))


# ---- ayudas para las comprobaciones de los retos
def _t(r, nombre, cond, msg=""):
    r.append([nombre, bool(cond), "" if cond else msg])


def _try(f, *a):
    try:
        return f(*a)
    except BaseException as e:
        return "ERROR: " + type(e).__name__ + ": " + str(e)


def _cerca(a, b, tol=1e-6):
    try:
        return abs(a - b) <= tol
    except BaseException:
        return False


def ayuda():
    print("Escribe Python y pulsa Enter. Las variables se guardan entre lecciones.")
    print("Prueba: 2 + 2   ·   import math   ·   math.sqrt(16)   ·   [i * i for i in range(5)]")
    print("grafica(lista, 'título') dibuja una línea con los valores de una lista.")


for _n in ("grafica", "ayuda", "_t", "_try", "_cerca"):
    G[_n] = globals()[_n]


def _info_error(e):
    linea = getattr(e, "lineno", None)
    tb = e.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == "<string>":
            linea = tb.tb_lineno
        tb = tb.tb_next
    return [type(e).__name__, str(e), linea]


def correr(code):
    try:
        exec(code, G)
    except BaseException as e:
        window.pyerr(json.dumps(_info_error(e)))


def repl(code):
    try:
        try:
            exec("_ans = (" + code + "\n)", G)
        except SyntaxError:
            exec(code, G)
            return
        r = G.pop("_ans", None)
        if r is not None:
            print(repr(r))
    except BaseException as e:
        window.pyerr(json.dumps(_info_error(e)))


def comprobar(src):
    T = dict(G)
    T["U"] = G
    try:
        exec(src, T)
        res = T["probar"]()
    except BaseException as e:
        res = [["No se pudo comprobar el reto", False, type(e).__name__ + ": " + str(e)]]
    window.pychk(json.dumps(res))


def reiniciar():
    keep = {k: G[k] for k in ("__name__", "grafica", "ayuda", "_t", "_try", "_cerca")}
    G.clear()
    G.update(keep)


window.pycorrer = correr
window.pyrepl = repl
window.pycomprobar = comprobar
window.pyreiniciar = reiniciar
