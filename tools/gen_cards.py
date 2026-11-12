"""Genera las tarjetas de proyecto (SVG) y las imágenes de vista previa social."""
import html, math, os

def esc(t): return html.escape(t, quote=False)

SANS = "Segoe UI, Helvetica, Arial, sans-serif"
SERIF = "Fraunces, 'Iowan Old Style', Georgia, 'Times New Roman', serif"
ROUND = "Fredoka, Nunito, Segoe UI, Helvetica, Arial, sans-serif"


def card(w, h, bg, bg2, ink, mut, acc, chipbg, kicker, title, sub, desc, metrics, tags, font, deco, aria, titlesize):
    k = min(w / 560, h / 340)
    r = int(20 * k)
    s = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{esc(aria)}">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{bg}"/><stop offset="1" stop-color="{bg2}"/></linearGradient>
<clipPath id="c"><rect width="{w}" height="{h}" rx="{r}"/></clipPath></defs>
<rect width="{w}" height="{h}" rx="{r}" fill="url(#g)"/>
<g clip-path="url(#c)">{deco}</g>
'''
    x = int(36 * k); y = int(50 * k)
    s += f'<text x="{x}" y="{y}" font-family="{SANS}" font-size="{12*k:.1f}" font-weight="700" letter-spacing="{2.4*k:.1f}" fill="{acc}">{esc(kicker)}</text>\n'
    s += f'<text x="{x-2}" y="{y+int(66*k)}" font-family="{font}" font-size="{titlesize*k:.0f}" font-weight="700" fill="{ink}">{esc(title)}</text>\n'
    s += f'<text x="{x}" y="{y+int(98*k)}" font-family="{SANS}" font-size="{19*k:.1f}" font-weight="600" fill="{acc}">{esc(sub)}</text>\n'
    for i, l in enumerate(desc):
        s += f'<text x="{x}" y="{y+int((134+i*23)*k)}" font-family="{SANS}" font-size="{15.5*k:.1f}" fill="{mut}">{esc(l)}</text>\n'
    gap = int(12 * k)
    mw = (w - 2 * x - 2 * gap) / 3
    mh = int(62 * k)
    my = h - int(104 * k)
    for i, (v, l) in enumerate(metrics):
        mx = x + i * (mw + gap)
        s += f'<rect x="{mx:.0f}" y="{my}" width="{mw:.0f}" height="{mh}" rx="{int(10*k)}" fill="{chipbg}"/>\n'
        s += f'<text x="{mx+int(14*k):.0f}" y="{my+int(30*k)}" font-family="{SANS}" font-size="{24*k:.0f}" font-weight="800" fill="{ink}">{esc(v)}</text>\n'
        s += f'<text x="{mx+int(14*k):.0f}" y="{my+int(50*k)}" font-family="{SANS}" font-size="{11*k:.1f}" fill="{mut}">{esc(l)}</text>\n'
    s += f'<text x="{x}" y="{h-int(22*k)}" font-family="{SANS}" font-size="{12.5*k:.1f}" fill="{mut}">{esc(tags)}</text>\n</svg>'
    return s


def fami_deco(cx, cy, s):
    rings = "".join(
        f'<circle cx="{cx}" cy="{cy}" r="{R*s:.0f}" fill="none" stroke="#516472" stroke-opacity="{o}" stroke-width="{2.5*s:.1f}"/>'
        for R, o in ((40, .5), (80, .36), (120, .25), (160, .16)))
    nodes = [(40, 20), (80, 140), (80, 250), (120, 75), (120, 205), (160, 330), (160, 105)]
    dots = "".join(
        f'<circle cx="{cx+R*s*math.cos(math.radians(a)):.1f}" cy="{cy+R*s*math.sin(math.radians(a)):.1f}" r="{6*s:.1f}" fill="#89674b"/>'
        for R, a in nodes)
    return rings + dots + f'<circle cx="{cx}" cy="{cy}" r="{18*s:.0f}" fill="#516472"/><circle cx="{cx}" cy="{cy}" r="{7*s:.0f}" fill="#f2f0e3"/>'


def rappi_deco(w, h, s):
    g = "".join(f'<line x1="{x}" y1="0" x2="{x-40*s}" y2="{h}"/>' for x in range(int(w * 0.6), w + 80, int(60 * s)))
    g += "".join(f'<line x1="{int(w*0.55)}" y1="{y}" x2="{w}" y2="{y-20*s}"/>' for y in range(int(30 * s), h, int(60 * s)))
    return (f'<g stroke="#ffffff" stroke-opacity=".07" stroke-width="{4*s}" stroke-linecap="round">{g}</g>'
            f'<path d="M{w*0.68:.0f} {h*0.96:.0f} C {w*0.76:.0f} {h*0.8:.0f}, {w*0.84:.0f} {h*0.6:.0f}, {w*0.94:.0f} {h*0.2:.0f}" fill="none" stroke="#DE3410" stroke-width="{4*s}" stroke-dasharray="2 {10*s}" stroke-linecap="round"/>'
            f'<circle cx="{w*0.94:.0f}" cy="{h*0.2:.0f}" r="{11*s}" fill="#DE3410"/><circle cx="{w*0.94:.0f}" cy="{h*0.2:.0f}" r="{4.5*s}" fill="#1A1A22"/>')


FAMI = dict(bg="#f2f0e3", bg2="#e1d7c8", ink="#516472", mut="#5c6a74", acc="#89674b", chipbg="#fbfaf3",
            kicker="SENDA ZERO · DLAB ICUBO UDD 2026", title="FAMI", font=ROUND, titlesize=66,
            metrics=[("89%", "utilidad percibida"), ("17.144", "mayores en Quinta Normal"), ("915 MHz", "LoRaWAN, sin SIM")],
            tags="LoRaWAN · ESP32 · React Native · IoT")
RAPPI = dict(bg="#1A1A22", bg2="#2a1a18", ink="#ffffff", mut="#b9b9c6", acc="#FF6A47", chipbg="#262631",
             kicker="DLAB UDD 2026 · DESAFÍO 18 · RAPPI CHILE", title="Rappi Hunters", font=SANS, titlesize=48,
             metrics=[("3", "vistas: app · oficina · gemelo"), ("7", "escenarios simulados"), ("6", "hallazgos clave")],
             tags="JavaScript · HTML/CSS · Simulación de agentes")

os.makedirs("assets", exist_ok=True)
os.makedirs("social", exist_ok=True)

# Tarjetas del perfil (560x340)
W, H = 560, 340
open("assets/card-fami.svg", "w", encoding="utf8").write(card(
    W, H, FAMI["bg"], FAMI["bg2"], FAMI["ink"], FAMI["mut"], FAMI["acc"], FAMI["chipbg"], FAMI["kicker"], FAMI["title"],
    "Redes que protegen", ["Teleasistencia comunitaria para adultos mayores:", "wearable LoRaWAN sin SIM + app vecinal."],
    FAMI["metrics"], FAMI["tags"], FAMI["font"], fami_deco(490, 80, .8), "FAMI · Redes que protegen", FAMI["titlesize"]))
open("assets/card-rappi.svg", "w", encoding="utf8").write(card(
    W, H, RAPPI["bg"], RAPPI["bg2"], RAPPI["ink"], RAPPI["mut"], RAPPI["acc"], RAPPI["chipbg"], RAPPI["kicker"], RAPPI["title"],
    "Prototipo + gemelo digital", ["Repartidores reportan locales nuevos en sus", "tiempos de espera; la Oficina los gestiona."],
    RAPPI["metrics"], RAPPI["tags"], RAPPI["font"], rappi_deco(W, H, 1), "Rappi Hunters · Prototipo y gemelo digital", RAPPI["titlesize"]))

# Vista previa social (1280x640)
SW, SH = 1280, 640
open("social/fami.svg", "w", encoding="utf8").write(card(
    SW, SH, FAMI["bg"], FAMI["bg2"], FAMI["ink"], FAMI["mut"], FAMI["acc"], FAMI["chipbg"], FAMI["kicker"], FAMI["title"],
    "Redes que protegen · teleasistencia comunitaria",
    ["Wearable LoRaWAN sin SIM + app vecinal para activar", "a los vecinos más cercanos en los minutos que importan."],
    FAMI["metrics"], FAMI["tags"], FAMI["font"], fami_deco(1090, 190, 1.8), "FAMI", FAMI["titlesize"]))
open("social/rappi.svg", "w", encoding="utf8").write(card(
    SW, SH, RAPPI["bg"], RAPPI["bg2"], RAPPI["ink"], RAPPI["mut"], RAPPI["acc"], RAPPI["chipbg"], RAPPI["kicker"], RAPPI["title"],
    "Prototipo web + gemelo digital",
    ["Repartidores reportan locales nuevos mientras esperan un", "pedido; la Oficina los gestiona y el gemelo prueba las reglas."],
    RAPPI["metrics"], RAPPI["tags"], RAPPI["font"], rappi_deco(SW, SH, 2), "Rappi Hunters", RAPPI["titlesize"]))
