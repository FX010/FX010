"""Genera 4000 expresiones aritméticas aleatorias y su resultado en CPython (fuzz.json)."""
import json, pathlib, random
random.seed(2024)
def atom():
    r = random.random()
    if r < 0.45: return str(random.choice([0, 1, 2, 3, 5, 7, 10, 13, 100, 255, -4, -7]))
    if r < 0.8: return random.choice(["0.5", "1.5", "2.25", "0.1", "3.14159", "1e3", "2.5e-3", "100.0", "0.3", "7.0", "-2.5", "1e-5", "123456.789"])
    if r < 0.9: return random.choice(["True", "False"])
    return random.choice(["(1+2)", "(3-5)", "abs(-3)", "round(2.567, 1)", "int(7.9)", "max(1, 2)", 'len("abc")'])
def expr(d):
    if d == 0 or random.random() < 0.25: return atom()
    op = random.choice(["+", "-", "*", "/", "//", "%", "**", "+", "-", "*", "<", "==", ">=", "and", "or"])
    a, b = expr(d - 1), expr(d - 1)
    if op == "**": b = random.choice(["0", "1", "2", "3", "0.5", "-1", "2.0", "-2"])
    return f"({a} {op} {b})" if random.random() < 0.8 else f"{a} {op} {b}"
casos = []
for _ in range(4000):
    e = expr(random.randint(1, 4))
    try: out = repr(eval(e))
    except Exception as ex: out = "EXC:" + type(ex).__name__
    casos.append([e, out])
(pathlib.Path(__file__).resolve().parent / "fuzz.json").write_text(json.dumps(casos))
print(len(casos), "expresiones")
