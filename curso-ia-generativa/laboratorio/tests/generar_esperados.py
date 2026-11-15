"""Ejecuta cada programa de corpus.py en CPython y guarda su salida en esperados.json."""
import json, pathlib, subprocess, sys
aqui = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(aqui))
from corpus import C
res = []
for code in C:
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    exc = None
    if r.returncode != 0:
        lineas = [l for l in r.stderr.strip().split("\n") if l]
        exc = lineas[-1].split(":")[0].strip() if lineas else "Error"
    res.append({"code": code, "out": r.stdout, "exc": exc})
(aqui / "esperados.json").write_text(json.dumps(res))
print(len(res), "programas;", sum(1 for r in res if r["exc"]), "terminan con error")
