/* Laboratorio Neuronal: aplicación. Depende de kernel.js (kernelLib, solo para los widgets), pyinterp.js (el intérprete de Python) y lessons.js. */
(function () {
  "use strict";
  const L = {}; kernelLib(L);   // biblioteca para los widgets (hilo principal)

  // ============================================================ utilidades
  function h(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const k in attrs || {}) {
      const v = attrs[k];
      if (v === false || v == null) continue;
      if (k === "class") e.className = v; else if (k === "html") e.innerHTML = v; else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v === true ? "" : v);
    }
    for (const c of kids.flat()) if (c != null && c !== false) e.append(c.nodeType ? c : document.createTextNode(String(c)));
    return e;
  }
  const $ = (s, r = document) => r.querySelector(s);
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const store = {
    get(k, d) { try { const v = localStorage.getItem("ln:" + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem("ln:" + k, JSON.stringify(v)); } catch (e) { /* sin almacenamiento */ } },
    del(k) { try { localStorage.removeItem("ln:" + k); } catch (e) { /* nada */ } },
  };
  const fmtN = (x, p = 3) => (Number.isFinite(x) ? (Math.abs(x) >= 1e5 || (Math.abs(x) < 1e-3 && x !== 0) ? x.toExponential(2) : String(parseFloat(x.toPrecision(p)))) : String(x));
  const rgb = (c) => { c = c.trim(); if (c[0] === "#") { if (c.length === 4) c = "#" + [...c.slice(1)].map((x) => x + x).join(""); return [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16)); } const m = c.match(/[\d.]+/g); return m ? m.slice(0, 3).map(Number) : [0, 0, 0]; };
  const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
  const rgbs = (a, al = 1) => `rgba(${a[0]},${a[1]},${a[2]},${al})`;

  // ---- registro de dibujos que se repintan al cambiar tema o tamaño
  const drawables = new Set();
  function reg(el, draw) { const o = { el, draw }; drawables.add(o); draw(); return o; }
  function redrawAll() { for (const o of [...drawables]) { if (!o.el.isConnected) { drawables.delete(o); continue; } try { o.draw(); } catch (e) { /* ignorar */ } } }
  let rz = 0; window.addEventListener("resize", () => { cancelAnimationFrame(rz); rz = requestAnimationFrame(redrawAll); });
  new MutationObserver(() => redrawAll()).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  try { matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => setTimeout(redrawAll, 50)); } catch (e) { /* nada */ }

  function fit(cv, w, hgt) {
    const d = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = Math.round(w * d); cv.height = Math.round(hgt * d); cv.style.width = w + "px"; cv.style.height = hgt + "px";
    const g = cv.getContext("2d"); g.setTransform(d, 0, 0, d, 0, 0); return g;
  }
  function theme() { return { ink: css("--ink"), muted: css("--muted"), line: css("--line"), surface: css("--surface"), surface2: css("--surface2"), accent: css("--accent"), cls: [css("--c1"), css("--c2"), css("--c3"), css("--c4")] }; }
  function niceTicks(a, b, n = 5) {
    const span = b - a || 1, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0)), r = step0 / mag, step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag, out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + step * 1e-6; v += step) out.push(+v.toFixed(12));
    return out;
  }
  const FONT = '11px "Inter", system-ui, sans-serif';

  // ============================================================ gráficos
  function drawLine(g, W, H, spec, T) {
    const m = { l: 48, r: 14, t: spec.title ? 30 : 12, b: 28 }, pw = W - m.l - m.r, ph = H - m.t - m.b, log = !!spec.ylog;
    const all = spec.series.flatMap((s) => s.y).filter((v) => Number.isFinite(v));
    const tf = (v) => (log ? Math.log10(Math.max(v, 1e-30)) : v);
    let lo = Math.min(...all.map(tf)), hi = Math.max(...all.map(tf));
    if (log) { lo = Math.floor(lo); hi = Math.ceil(hi); if (hi === lo) hi = lo + 1; } else { const pd = (hi - lo || 1) * 0.08; lo -= pd; hi += pd; }
    const n = Math.max(...spec.series.map((s) => s.y.length));
    const X = (i) => m.l + (pw * i) / Math.max(1, n - 1), Y = (v) => m.t + ph - ((tf(v) - lo) / (hi - lo)) * ph;
    g.font = FONT; g.textBaseline = "middle"; g.fillStyle = T.muted; g.strokeStyle = T.line; g.lineWidth = 1;
    const ticks = log ? (() => { const out = [], st = Math.max(1, Math.ceil((hi - lo) / 6)); for (let p = lo; p <= hi; p += st) out.push(p); return out; })() : niceTicks(lo, hi, Math.max(2, Math.min(5, Math.floor(ph / 32))));
    g.textAlign = "right";
    for (const tk of ticks) { const y = m.t + ph - ((tk - lo) / (hi - lo)) * ph; g.beginPath(); g.moveTo(m.l, y); g.lineTo(m.l + pw, y); g.stroke(); g.fillText(log ? "1e" + tk : fmtN(tk, 3), m.l - 6, y); }
    g.textAlign = "center"; g.textBaseline = "top";
    for (const tk of niceTicks(0, n - 1, 6)) g.fillText(String(Math.round(tk) + (spec.x1 ? 1 : 0)), X(tk), m.t + ph + 6);
    spec.series.forEach((s, k) => {
      g.strokeStyle = T.cls[k % 4]; g.lineWidth = 2.2; g.beginPath();
      s.y.forEach((v, i) => { const x = X(i), y = Y(v); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
    });
    g.textAlign = "left"; g.textBaseline = "middle";
    if (spec.title) { g.fillStyle = T.ink; g.font = '600 12px "Inter", system-ui, sans-serif'; g.fillText(spec.title, m.l, 12); }
    let lx = W - m.r; g.font = FONT; g.textAlign = "right";
    for (let k = spec.series.length - 1; k >= 0; k--) { const nm = spec.series[k].name || "serie " + (k + 1), w = g.measureText(nm).width; g.fillStyle = T.muted; g.fillText(nm, lx, 12); g.fillStyle = T.cls[k % 4]; g.fillRect(lx - w - 16, 9, 10, 6); lx -= w + 28; }
  }
  function drawBar(g, W, H, spec, T) {
    const m = { l: 128, r: 54, t: spec.title ? 28 : 10, b: 22 }, pw = W - m.l - m.r, n = spec.categories.length, bh = (H - m.t - m.b) / n;
    const mx = Math.max(...spec.values, spec.ref ? spec.ref.value : 0) * 1.05;
    g.font = FONT; g.textBaseline = "middle";
    if (spec.title) { g.fillStyle = T.ink; g.font = '600 12px "Inter", system-ui, sans-serif'; g.textAlign = "left"; g.fillText(spec.title, 8, 12); g.font = FONT; }
    spec.categories.forEach((c, i) => {
      const y = m.t + i * bh, w = (spec.values[i] / mx) * pw, hot = spec.highlight === i;
      g.fillStyle = T.muted; g.textAlign = "right"; g.fillText(c, m.l - 8, y + bh / 2);
      g.fillStyle = hot ? T.cls[1] : T.cls[0]; g.globalAlpha = hot ? 1 : 0.85; g.beginPath(); g.roundRect(m.l, y + bh * 0.18, Math.max(w, 2), bh * 0.64, 4); g.fill(); g.globalAlpha = 1;
      g.fillStyle = T.ink; g.textAlign = "left"; g.fillText(fmtN(spec.values[i], 3), m.l + w + 6, y + bh / 2);
    });
    if (spec.ref) {
      const x = m.l + (spec.ref.value / mx) * pw; g.strokeStyle = T.cls[2]; g.lineWidth = 2; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(x, m.t - 2); g.lineTo(x, H - m.b + 2); g.stroke(); g.setLineDash([]);
      g.fillStyle = T.cls[2]; g.textAlign = "left"; g.fillText(spec.ref.label, Math.min(x + 6, W - 150), H - 8);
    }
  }
  function drawScatter(g, W, H, spec, T) {
    const m = { l: 36, r: 12, t: spec.title ? 28 : 10, b: 24 }, pw = W - m.l - m.r, ph = H - m.t - m.b, xs = spec.points.map((p) => p[0]), ys = spec.points.map((p) => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    g.font = FONT; g.fillStyle = T.muted; g.textAlign = "right"; g.textBaseline = "middle";
    for (const tk of niceTicks(y0, y1, 4)) g.fillText(fmtN(tk, 2), m.l - 6, m.t + ph - ((tk - y0) / (y1 - y0 || 1)) * ph);
    g.textAlign = "center"; for (const tk of niceTicks(x0, x1, 5)) g.fillText(fmtN(tk, 2), m.l + ((tk - x0) / (x1 - x0 || 1)) * pw, H - 10);
    if (spec.title) { g.fillStyle = T.ink; g.textAlign = "left"; g.font = '600 12px "Inter", system-ui, sans-serif'; g.fillText(spec.title, m.l, 12); }
    for (const p of spec.points) { g.fillStyle = T.cls[(p[2] || 0) % 4]; g.beginPath(); g.arc(m.l + ((p[0] - x0) / (x1 - x0 || 1)) * pw, m.t + ph - ((p[1] - y0) / (y1 - y0 || 1)) * ph, 3.2, 0, 7); g.fill(); }
  }
  function drawHeat(g, W, H, spec, T) {
    const M = spec.matrix, r = M.length, c = M[0].length, ml = spec.ylabels ? 64 : 24, mt = (spec.title ? 26 : 8) + (spec.xlabels ? 18 : 0), cw = (W - ml - 10) / c, ch = (H - mt - 10) / r;
    const flat = M.flat(), lo = spec.min ?? Math.min(...flat), hi = spec.max ?? Math.max(...flat), A = rgb(T.surface2), B = rgb(T.accent);
    g.font = FONT; g.textBaseline = "middle";
    if (spec.title) { g.fillStyle = T.ink; g.textAlign = "left"; g.font = '600 12px "Inter", system-ui, sans-serif'; g.fillText(spec.title, 8, 12); g.font = FONT; }
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) {
      const t = hi === lo ? 0 : (M[i][j] - lo) / (hi - lo); g.fillStyle = rgbs(mix(A, B, Math.max(0, Math.min(1, t)))); g.fillRect(ml + j * cw + 1, mt + i * ch + 1, cw - 2, ch - 2);
      if (r <= 8 && c <= 8 && cw > 30) { g.fillStyle = t > 0.55 ? T.surface : T.ink; g.textAlign = "center"; g.fillText(M[i][j].toFixed(2), ml + j * cw + cw / 2, mt + i * ch + ch / 2); }
    }
    g.fillStyle = T.muted;
    if (spec.ylabels) { g.textAlign = "right"; spec.ylabels.forEach((s, i) => g.fillText(s.length > 9 ? s.slice(0, 8) + "…" : s, ml - 6, mt + i * ch + ch / 2)); }
    if (spec.xlabels) { g.textAlign = "center"; spec.xlabels.forEach((s, j) => g.fillText(s.length > 9 ? s.slice(0, 8) + "…" : s, ml + j * cw + cw / 2, mt - 9)); }
  }
  function drawBoundary(g, W, H, spec, T) {
    const m = { l: 6, r: 6, t: spec.title ? 26 : 6, b: 6 }, pw = W - m.l - m.r, ph = H - m.t - m.b, res = spec.res, [x0, x1, y0, y1] = spec.bounds, cw = pw / res, ch = ph / res;
    for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) { g.fillStyle = rgbs(rgb(T.cls[spec.grid[j * res + i] % 4]), 0.26); g.fillRect(m.l + i * cw, m.t + j * ch, cw + 0.6, ch + 0.6); }
    for (const p of spec.points) { g.fillStyle = T.cls[p[2] % 4]; g.strokeStyle = T.surface; g.lineWidth = 1.2; g.beginPath(); g.arc(m.l + ((p[0] - x0) / (x1 - x0)) * pw, m.t + ((y1 - p[1]) / (y1 - y0)) * ph, 3, 0, 7); g.fill(); g.stroke(); }
    if (spec.title) { g.fillStyle = T.ink; g.font = '600 12px "Inter", system-ui, sans-serif'; g.textAlign = "left"; g.textBaseline = "middle"; g.fillText(spec.title, 6, 12); }
  }
  function renderPlot(spec, host, width) {
    const cv = h("canvas", { role: "img", "aria-label": spec.title || "gráfico" }); host.append(cv);
    const kind = spec.type || "line";
    reg(cv, () => {
      const w = Math.max(220, Math.min(width || 560, (host.clientWidth || 560) - 4)), hh = kind === "bar" ? Math.max(120, 38 * spec.categories.length + 44) : kind === "heat" ? Math.round(Math.min(w, 300) * (spec.matrix.length / Math.max(spec.matrix[0].length, 1)) * 0.8 + 60) : kind === "boundary" || kind === "scatter" ? Math.round(w * 0.78) : Math.round(w * 0.5);
      const g = fit(cv, w, Math.min(Math.max(hh, 160), 420)), T = theme(), Wp = w, Hp = Math.min(Math.max(hh, 160), 420);
      g.clearRect(0, 0, Wp, Hp);
      ({ line: drawLine, bar: drawBar, scatter: drawScatter, heat: drawHeat, boundary: drawBoundary }[kind] || drawLine)(g, Wp, Hp, spec, T);
    });
    return cv;
  }

  // ============================================================ motor de Python (intérprete propio, sin eval)
  const { PyInterp, PyError } = PyInterpLib;
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  class PyKernel {
    constructor() { this.q = Promise.resolve(); this.busyCb = () => {}; this.buf = ""; this.sink = null; this.interp = new PyInterp({ stdout: (s) => this.write(s), plot: (spec) => { if (this.sink) this.sink({ type: "plot", spec }); } }); }
    write(s) { this.buf += s; let i; while ((i = this.buf.indexOf("\n")) >= 0) { const line = this.buf.slice(0, i); this.buf = this.buf.slice(i + 1); if (this.sink) this.sink({ type: "log", text: line }); } }
    flush() { if (this.buf && this.sink) this.sink({ type: "log", text: this.buf }); this.buf = ""; }
    enqueue(job) { const p = this.q.then(job); this.q = p.catch(() => {}); return p; }
    run(code, sink, opts = {}) {
      return this.enqueue(async () => {
        this.busyCb(true); await new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
        this.sink = sink || null; this.buf = ""; let err = null;
        try { if (opts.repl) this.interp.repl(code, opts.ms || 6000); else this.interp.run(code, opts.ms || 6000); }
        catch (e) { err = e instanceof PyError ? [e.pyType, e.pyMsg, e.line == null ? this.interp.line : e.line] : ["JavascriptError", String((e && e.message) || e), null]; }
        this.flush(); this.sink = null; this.busyCb(false);
        return err ? { ok: false, err } : { ok: true };
      });
    }
    check(src) { return this.enqueue(async () => ({ results: this.interp.check(src, 8000) })); }
    reset() { return this.enqueue(async () => { this.interp.reset(); }); }
  }
  const kernel = new PyKernel();

  // ---- errores explicados en español sencillo
  function explain(err, code) {
    const [t, msg, line] = err, at = line ? " (línea " + line + ")" : "";
    if (t === "NotSupported") return msg + at + ". Este laboratorio usa un Python simplificado; para eso usa Python completo.";
    if (/__TIMEOUT__/.test(msg)) return "Tu código tardó más de 6 segundos y lo detuve. Revisa que cada bucle while llegue a terminar.";
    if (/(^|[^\w])_{3}([^\w]|$)/.test(code) && /NameError|SyntaxError/.test(t)) return "Todavía quedan huecos ___ por completar" + at + ". Sustitúyelos por tu código.";
    let m;
    if (t === "NameError" && (m = msg.match(/name '(.*)' is not defined/))) return "Python no conoce «" + m[1] + "»" + at + ". Revisa cómo lo escribiste (las mayúsculas cuentan) y que lo hayas creado antes de usarlo.";
    if (t === "IndentationError") return "La sangría no es correcta" + at + ". Lo que va dentro de un if, for o def empieza con 4 espacios.";
    if (t === "SyntaxError") return "Hay un error de escritura" + at + ". Suele ser un paréntesis o una comilla sin cerrar, o faltar los dos puntos «:» al final de un if, for o def.";
    if (t === "TypeError") return /unsupported operand|can only concatenate|must be str|must be real/.test(msg) ? "Estás mezclando tipos que no se pueden combinar" + at + ", por ejemplo texto con números. Convierte con str() o int()." : /positional argument|required/.test(msg) ? "Llamaste a una función con más o menos datos de los que espera" + at + "." : "Un valor no tiene el tipo que se esperaba" + at + ".";
    if (t === "ZeroDivisionError") return "Dividiste entre cero" + at + ".";
    if (t === "IndexError") return "Pediste una posición que no existe en la lista" + at + ". La primera posición es la 0.";
    if (t === "KeyError") return "Esa clave no existe en el diccionario" + at + ".";
    if (t === "ValueError") return "Un valor no es válido para esa operación" + at + ".";
    if (t === "AttributeError") return "Ese objeto no tiene lo que le pides" + at + ". Revisa el nombre.";
    if (t === "ModuleNotFoundError") return "No encuentro ese módulo" + at + ".";
    return "Python encontró un problema" + at + ".";
  }
  function addError(host, err, code) { const e = h("div", { class: "ln err" }, h("b", null, explain(err, code)), h("small", null, err[0] + ": " + err[1])); host.append(e); host.scrollTop = host.scrollHeight; }
  function addLine(host, text, cls) { const e = h("div", { class: "ln " + (cls || "") }, text); host.append(e); host.scrollTop = host.scrollHeight; return e; }
  function sinkFor(host) { return (m) => { if (m.type === "log") addLine(host, m.text); else if (m.type === "plot") { try { renderPlot(m.spec, host, 560); } catch (e) { addLine(host, "No se pudo dibujar el gráfico.", "err"); } } }; }
  async function execute(code, host, opts) {
    host.replaceChildren();
    const r = await kernel.run(code, sinkFor(host), opts);
    if (!r.ok) addError(host, r.err, code);
    return r;
  }

  // ============================================================ editor con resaltado (Python)
  const PYKW = /^(def|return|if|elif|else|for|while|in|not|and|or|is|import|from|as|pass|break|continue|class|lambda|try|except|finally|with|raise|True|False|None)$/;
  const PYBI = /^(print|len|range|sum|max|min|round|abs|zip|enumerate|list|dict|str|int|float|sorted|map|math|random|grafica)$/;
  function highlight(src) {
    const re = /(#[^\n]*)|("(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')|(\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b)|(_{3,})|([A-Za-z_]\w*)/g;
    let out = "", last = 0, m;
    while ((m = re.exec(src))) {
      out += esc(src.slice(last, m.index)); last = re.lastIndex;
      if (m[1]) out += '<span class="tk-c">' + esc(m[1]) + "</span>";
      else if (m[2]) out += '<span class="tk-s">' + esc(m[2]) + "</span>";
      else if (m[3]) out += '<span class="tk-n">' + m[3] + "</span>";
      else if (m[4]) out += '<span class="tk-b">' + m[4] + "</span>";
      else if (PYKW.test(m[5])) out += '<span class="tk-k">' + m[5] + "</span>";
      else if (PYBI.test(m[5])) out += '<span class="tk-f">' + m[5] + "</span>";
      else out += m[5];
    }
    return out + esc(src.slice(last));
  }
  function makeEditor(initial, { onChange, onRun, label } = {}) {
    const pre = h("pre", { "aria-hidden": "true" }), ta = h("textarea", { spellcheck: "false", autocapitalize: "off", autocomplete: "off", autocorrect: "off", "aria-label": label || "Editor de código", rows: 4 });
    const wrap = h("div", { class: "editor" }, pre, ta);
    const sync = () => { pre.innerHTML = highlight(ta.value) + "\n"; ta.style.height = "auto"; ta.style.height = Math.max(88, ta.scrollHeight + 2) + "px"; pre.scrollLeft = ta.scrollLeft; };
    ta.value = initial; sync();
    requestAnimationFrame(() => { sync(); setTimeout(sync, 120); });
    try { document.fonts.ready.then(sync); } catch (e) { /* sin API de fuentes */ }
    reg(wrap, sync);
    ta.addEventListener("input", () => { sync(); onChange && onChange(ta.value); });
    ta.addEventListener("scroll", () => { pre.scrollLeft = ta.scrollLeft; });
    ta.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); onRun && onRun(); return; }
      const s = ta.selectionStart, en = ta.selectionEnd, v = ta.value, ls = v.lastIndexOf("\n", s - 1) + 1;
      if (e.key === "Tab") {
        e.preventDefault();
        if (e.shiftKey) { const m = v.slice(ls).match(/^ {1,4}/); if (m) { ta.setRangeText("", ls, ls + m[0].length, "preserve"); ta.setSelectionRange(Math.max(ls, s - m[0].length), Math.max(ls, s - m[0].length)); ta.dispatchEvent(new Event("input")); } }
        else { ta.setRangeText("    ", s, en, "end"); ta.dispatchEvent(new Event("input")); }
      } else if (e.key === "Enter") {
        e.preventDefault(); const before = v.slice(ls, s), ind = (before.match(/^\s*/) || [""])[0], extra = /:\s*(#.*)?$/.test(before) ? "    " : "";
        ta.setRangeText("\n" + ind + extra, s, en, "end"); ta.dispatchEvent(new Event("input"));
      }
    });
    return { el: wrap, get value() { return ta.value; }, set value(x) { ta.value = x; sync(); }, focus() { ta.focus(); } };
  }

  // ============================================================ progreso
  const state = { done: store.get("done", {}), current: null };
  const saveDone = () => store.set("done", state.done);
  const isDone = (id) => !!state.done[id];
  const unlocked = (i) => i === 0 || isDone(LESSONS[i - 1].id);

  // ---- celda de código: sirve para ejemplos (sin reto) y para retos
  function makeCode(o) {
    const key = "code:" + o.id, saved = store.get(key, null);
    const out = h("div", { class: "out", "aria-live": "polite" });
    const ed = makeEditor(saved ?? o.code, { onChange: (v) => store.set(key, v), onRun: () => run(), label: o.label });
    const btnRun = h("button", { class: "btn primary", onclick: () => run(), title: "Ctrl/Cmd + Enter" }, "▶ Ejecutar");
    const btnReset = h("button", { class: "btn small", onclick: () => { ed.value = o.code; store.del(key); out.replaceChildren(); if (o.reset) o.reset(); } }, "Restablecer");
    async function run() { btnRun.disabled = true; try { return await execute(ed.value, out); } finally { btnRun.disabled = false; } }
    const parts = [h("div", { class: "cell-bar" }, h("span", { class: "tag" }, o.tag || "Python"), h("span", { class: "sp" }), btnReset, btnRun), ed.el, out];
    return { el: h("div", { class: "cell" }, parts), ed, out, run, bar: parts[0], btnRun };
  }
  function makeTask(lesson, onPass) {
    const c = makeCode({ id: lesson.id, code: lesson.code, tag: "Completa los huecos ___ y comprueba", label: "Editor del reto de la lección " + lesson.num, reset: () => { tests.replaceChildren(); win.hidden = true; } });
    const tests = h("div", { class: "tests" }), win = h("div", { class: "win", hidden: true }), hints = h("div", { class: "hints" });
    let fails = 0, shown = 0;
    const btnCheck = h("button", { class: "btn good", onclick: () => check() }, "✓ Comprobar reto");
    const skip = h("button", { class: "btn small", hidden: true, onclick: () => { state.done[lesson.id] = state.done[lesson.id] || "saltado"; saveDone(); onPass("saltado"); } }, "Saltar este reto");
    const bHint = h("button", { class: "btn small", onclick: () => { if (shown >= lesson.hints.length) return; hints.append(h("div", { class: "hint" }, h("b", null, "Pista " + (shown + 1) + " de " + lesson.hints.length + ": "), lesson.hints[shown])); shown++; if (shown >= lesson.hints.length) { bHint.disabled = true; bHint.textContent = "Sin más pistas"; } } }, "💡 Pista");
    c.bar.insertBefore(bHint, c.bar.children[2]); c.bar.append(btnCheck); c.bar.insertBefore(skip, c.bar.children[2]);
    async function check() {
      btnCheck.disabled = true; tests.replaceChildren(h("div", { class: "test" }, h("span"), "Comprobando…"));
      try {
        const r = await c.run();
        if (!r.ok) { tests.replaceChildren(h("div", { class: "test no" }, h("span", { class: "ic" }, "✕"), h("div", null, "Tu código tiene un error. Corrígelo y vuelve a comprobar."))); fails++; skip.hidden = fails < 1; return; }
        const res = (await kernel.check(lesson.check)).results;
        tests.replaceChildren(...res.map((t) => h("div", { class: "test " + (t[1] ? "ok" : "no") }, h("span", { class: "ic" }, t[1] ? "✓" : "✕"), h("div", null, t[0], t[1] ? null : h("small", null, t[2])))));
        if (res.length && res.every((t) => t[1])) { win.hidden = false; win.textContent = "¡Reto superado! " + res.length + " de " + res.length + " comprobaciones correctas."; if (state.done[lesson.id] !== "pasado") { state.done[lesson.id] = "pasado"; saveDone(); } onPass("pasado"); }
        else { win.hidden = true; fails++; skip.hidden = fails < 1; }
      } finally { btnCheck.disabled = false; }
    }
    const aid = h("details", { class: "aid" }, h("summary", null, "¿Te atascaste? Ver la solución"), h("pre", null, lesson.solution), h("button", { class: "btn small", style: "margin-top:8px", onclick: () => { c.ed.value = lesson.solution; store.set("code:" + lesson.id, lesson.solution); } }, "Copiar al editor"));
    return h("div", null, c.el, hints, tests, win, aid);
  }

  // ============================================================ widgets
  const widgets = {};
  function slider(label, min, max, step, value, onInput, fmtv) {
    const out = h("output", null, fmtv ? fmtv(value) : fmtN(value)), inp = h("input", { type: "range", min, max, step, value, "aria-label": label });
    inp.addEventListener("input", () => { out.textContent = fmtv ? fmtv(+inp.value) : fmtN(+inp.value); onInput(+inp.value); });
    return { el: h("div", { class: "ctl" }, h("label", null, label, out), inp), set(v) { inp.value = v; out.textContent = fmtv ? fmtv(v) : fmtN(v); } };
  }
  function select(label, opts, value, onChange) {
    const s = h("select", { "aria-label": label }, opts.map(([v, t]) => h("option", { value: v, selected: v === value }, t)));
    s.addEventListener("change", () => onChange(s.value)); return h("div", { class: "ctl" }, h("span", { class: "lbl" }, label), s);
  }

  // ---- 1. neurona
  widgets.neurona = (el) => {
    const st = { w1: 1.6, w2: 1.0, b: -0.4, act: "tanh", px: 0.9, py: -0.5 };
    const A = { tanh: [Math.tanh, (y) => (y + 1) / 2], sigmoide: [(z) => 1 / (1 + Math.exp(-z)), (y) => y], relu: [(z) => Math.max(0, z), (y) => Math.min(y / 2, 1)], escalón: [(z) => (z > 0 ? 1 : 0), (y) => y] };
    const cv = h("canvas", { role: "img", "aria-label": "Mapa de la salida de la neurona" }), ro = h("div", { class: "readout" });
    const ctl = h("div", { class: "controls" }, slider("peso w₁", -4, 4, 0.1, st.w1, (v) => { st.w1 = v; draw(); }).el, slider("peso w₂", -4, 4, 0.1, st.w2, (v) => { st.w2 = v; draw(); }).el, slider("sesgo b", -3, 3, 0.1, st.b, (v) => { st.b = v; draw(); }).el, select("activación φ", Object.keys(A).map((k) => [k, k]), st.act, (v) => { st.act = v; draw(); }));
    el.append(h("div", { class: "widget cols" }, ctl, h("div", { class: "canvas-wrap" }, cv, h("div", { style: "height:10px" }), ro)));
    const small = document.createElement("canvas"); small.width = small.height = 72;
    const toX = (px, S) => ((px + 2) / 4) * S, toY = (py, S) => ((2 - py) / 4) * S;
    function draw() {
      const S = Math.max(220, Math.min(380, (cv.parentElement.clientWidth || 380))), g = fit(cv, S, S), T = theme(), [f, nrm] = A[st.act], lo = rgb(T.cls[0]), hi = rgb(T.cls[1]);
      const sg = small.getContext("2d"), im = sg.createImageData(72, 72);
      for (let j = 0; j < 72; j++) for (let i = 0; i < 72; i++) { const x1 = -2 + (4 * (i + 0.5)) / 72, x2 = 2 - (4 * (j + 0.5)) / 72, c = mix(lo, hi, Math.max(0, Math.min(1, nrm(f(st.w1 * x1 + st.w2 * x2 + st.b))))); const k = (j * 72 + i) * 4; im.data[k] = c[0]; im.data[k + 1] = c[1]; im.data[k + 2] = c[2]; im.data[k + 3] = 235; }
      sg.putImageData(im, 0, 0); g.imageSmoothingEnabled = true; g.drawImage(small, 0, 0, S, S);
      g.strokeStyle = T.line; g.lineWidth = 1; g.beginPath(); g.moveTo(S / 2, 0); g.lineTo(S / 2, S); g.moveTo(0, S / 2); g.lineTo(S, S / 2); g.stroke();
      g.fillStyle = T.ink; g.font = FONT; g.textAlign = "left"; g.fillText("x₂", S / 2 + 6, 12); g.textAlign = "right"; g.fillText("x₁", S - 6, S / 2 - 6);
      g.strokeStyle = T.ink; g.lineWidth = 2; g.setLineDash([6, 5]); g.beginPath();
      if (Math.abs(st.w2) > 1e-6) { g.moveTo(0, toY(-(st.w1 * -2 + st.b) / st.w2, S)); g.lineTo(S, toY(-(st.w1 * 2 + st.b) / st.w2, S)); } else if (Math.abs(st.w1) > 1e-6) { const x = -st.b / st.w1; g.moveTo(toX(x, S), 0); g.lineTo(toX(x, S), S); }
      g.stroke(); g.setLineDash([]);
      g.fillStyle = T.surface; g.strokeStyle = T.ink; g.lineWidth = 2; g.beginPath(); g.arc(toX(st.px, S), toY(st.py, S), 6, 0, 7); g.fill(); g.stroke();
      const z = st.w1 * st.px + st.w2 * st.py + st.b, y = f(z);
      ro.innerHTML = `punto (x₁, x₂) = (<b>${st.px.toFixed(2)}</b>, <b>${st.py.toFixed(2)}</b>)<br>z = w₁x₁ + w₂x₂ + b = <b>${z.toFixed(3)}</b><br>y = ${st.act}(z) = <b>${y.toFixed(3)}</b><br><span style="color:var(--muted)">línea discontinua: z = 0 (donde cambia de lado)</span>`;
    }
    cv.addEventListener("click", (e) => { const r = cv.getBoundingClientRect(); st.px = -2 + (4 * (e.clientX - r.left)) / r.width; st.py = 2 - (4 * (e.clientY - r.top)) / r.height; draw(); });
    reg(cv, draw);
  };

  // ---- 2. gradiente
  widgets.gradiente = (el) => {
    const f = (x) => 0.05 * x ** 4 - 0.6 * x * x + 0.3 * x, df = (x) => 0.2 * x ** 3 - 1.2 * x + 0.3;
    const st = { x0: 3, lr: 0.1, x: 3, path: [3], run: null, msg: "" };
    const cv = h("canvas", { role: "img", "aria-label": "Curva de pérdida con la bola que desciende" }), stats = h("div", { class: "stats" });
    const sx = slider("punto de partida x₀", -3.8, 3.8, 0.1, st.x0, (v) => { st.x0 = v; reset(); }), sl = slider("tasa de aprendizaje η", 0.01, 1.2, 0.01, st.lr, (v) => { st.lr = v; }, (v) => v.toFixed(2));
    const step = () => { st.x = st.x - st.lr * df(st.x); st.path.push(st.x); if (!Number.isFinite(st.x) || Math.abs(st.x) > 6) { st.msg = "Se salió de la zona: la tasa es demasiado grande y el descenso diverge."; stop(); } draw(); };
    function stop() { clearInterval(st.run); st.run = null; bRun.textContent = "▶ Ejecutar"; }
    function reset() { stop(); st.x = st.x0; st.path = [st.x0]; st.msg = ""; draw(); }
    const bStep = h("button", { class: "btn", onclick: step }, "Paso"), bRun = h("button", { class: "btn primary", onclick: () => { if (st.run) return stop(); st.msg = ""; bRun.textContent = "❚❚ Pausa"; st.run = setInterval(() => { if (st.path.length > 80) return stop(); step(); }, 110); } }, "▶ Ejecutar"), bReset = h("button", { class: "btn", onclick: reset }, "Reiniciar");
    el.append(h("div", { class: "widget cols" }, h("div", { class: "controls" }, sx.el, sl.el, h("div", { class: "btn-row" }, bStep, bRun, bReset), stats), h("div", { class: "canvas-wrap" }, cv)));
    function draw() {
      const W = Math.max(240, Math.min(540, cv.parentElement.clientWidth || 540)), H = Math.round(W * 0.62), g = fit(cv, W, H), T = theme(), m = { l: 34, r: 12, t: 12, b: 24 }, pw = W - m.l - m.r, ph = H - m.t - m.b;
      const X = (x) => m.l + ((x + 4) / 8) * pw, Y = (y) => m.t + ph - ((y + 3.2) / 8.6) * ph;
      g.clearRect(0, 0, W, H); g.font = FONT; g.fillStyle = T.muted; g.strokeStyle = T.line; g.lineWidth = 1; g.textAlign = "center";
      for (const t of [-4, -2, 0, 2, 4]) { g.beginPath(); g.moveTo(X(t), m.t); g.lineTo(X(t), m.t + ph); g.stroke(); g.fillText(String(t), X(t), H - 8); }
      g.textAlign = "right"; for (const t of [-2, 0, 2, 4]) { g.beginPath(); g.moveTo(m.l, Y(t)); g.lineTo(m.l + pw, Y(t)); g.stroke(); g.fillText(String(t), m.l - 6, Y(t) + 4); }
      g.strokeStyle = T.cls[0]; g.lineWidth = 2.5; g.beginPath(); for (let i = 0; i <= 160; i++) { const x = -4 + (8 * i) / 160; i ? g.lineTo(X(x), Y(f(x))) : g.moveTo(X(x), Y(f(x))); } g.stroke();
      g.fillStyle = rgbs(rgb(T.cls[1]), 0.6); for (const p of st.path.slice(0, -1)) { if (Math.abs(p) <= 4) { g.beginPath(); g.arc(X(p), Y(f(p)), 3, 0, 7); g.fill(); } }
      const x = Math.max(-4, Math.min(4, st.x)), y = f(x), sl = df(x);
      g.strokeStyle = T.cls[1]; g.lineWidth = 1.6; g.beginPath(); g.moveTo(X(x - 1), Y(y - sl)); g.lineTo(X(x + 1), Y(y + sl)); g.stroke();
      g.fillStyle = T.cls[1]; g.strokeStyle = T.surface; g.lineWidth = 2; g.beginPath(); g.arc(X(x), Y(y), 7, 0, 7); g.fill(); g.stroke();
      stats.innerHTML = `<span>paso <b>${st.path.length - 1}</b></span><span>x <b>${fmtN(st.x)}</b></span><span>f(x) <b>${fmtN(f(st.x))}</b></span><span>f′(x) <b>${fmtN(df(st.x))}</b></span>` + (st.msg ? `<span style="color:var(--bad);flex-basis:100%">${st.msg}</span>` : "");
    }
    reg(cv, draw);
  };

  // ---- 4. playground
  widgets.playground = (el) => {
    const st = { data: "espirales", capas: 2, ancho: 16, act: "tanh", optim: "sgd", lr: 0.4, model: null, ds: null, loss: [], paso: 0, run: false, raf: 0, last: null };
    const mkData = () => { st.ds = st.data === "espirales" ? L.spirals(100, 3, 0.2, 0) : st.data === "lunas" ? L.moons(240, 0.12, 0) : L.xorData(240, 0.18, 0); };
    const mkModel = () => { const k = st.data === "espirales" ? 3 : 2, dims = [2, ...Array(st.capas).fill(st.ancho), k]; st.model = new L.MLP(dims, { act: st.act, optim: st.optim, seed: 1 }); st.loss = []; st.paso = 0; st.last = null; };
    const cv = h("canvas", { role: "img", "aria-label": "Frontera de decisión de la red" }), cl = h("canvas", { role: "img", "aria-label": "Curva de pérdida" }), stats = h("div", { class: "stats" });
    const reset = () => { st.run = false; bPlay.textContent = "▶ Entrenar"; mkData(); mkModel(); draw(true); };
    const sLr = slider("tasa de aprendizaje", 0.01, 1, 0.01, st.lr, (v) => { st.lr = v; }, (v) => v.toFixed(2));
    const ctl = h("div", { class: "controls" },
      select("datos", [["espirales", "Espirales (3 clases)"], ["lunas", "Dos lunas"], ["xor", "XOR"]], st.data, (v) => { st.data = v; reset(); }),
      select("capas ocultas", [["1", "1"], ["2", "2"], ["3", "3"]], "2", (v) => { st.capas = +v; reset(); }),
      select("neuronas por capa", [["2", "2"], ["4", "4"], ["8", "8"], ["16", "16"], ["32", "32"]], "16", (v) => { st.ancho = +v; reset(); }),
      select("activación", [["tanh", "tanh"], ["relu", "ReLU"], ["sigmoid", "sigmoide"]], st.act, (v) => { st.act = v; reset(); }),
      select("optimizador", [["sgd", "SGD"], ["momentum", "SGD + momentum"], ["adam", "Adam"]], st.optim, (v) => { st.optim = v; if (v === "adam") { st.lr = 0.02; sLr.set(0.02); } else if (st.lr < 0.1) { st.lr = 0.4; sLr.set(0.4); } reset(); }), sLr.el);
    const bPlay = h("button", { class: "btn primary", onclick: () => { st.run = !st.run; bPlay.textContent = st.run ? "❚❚ Pausa" : "▶ Entrenar"; if (st.run) loop(); } }, "▶ Entrenar"), bStep = h("button", { class: "btn", onclick: () => { for (let i = 0; i < 10; i++) one(); draw(true); } }, "+10 pasos"), bReset = h("button", { class: "btn", onclick: reset }, "Reiniciar");
    el.append(h("div", { class: "widget cols" }, h("div", null, ctl, h("div", { style: "height:12px" }), h("div", { class: "btn-row" }, bPlay, bStep, bReset)), h("div", { class: "canvas-wrap" }, cv, h("div", { style: "height:8px" }), cl, h("div", { style: "height:8px" }), stats)));
    function one() { const r = st.model.trainStep(st.ds.X, st.ds.y, st.lr); st.last = r; st.loss.push(r.loss); st.paso++; if (!Number.isFinite(r.loss)) { st.run = false; bPlay.textContent = "▶ Entrenar"; } }
    function loop() { if (!st.run || !el.isConnected) return; for (let i = 0; i < 6; i++) one(); draw(true); if (st.paso >= 6000) { st.run = false; bPlay.textContent = "▶ Entrenar"; return; } st.raf = requestAnimationFrame(loop); }
    function draw(full) {
      const W = Math.max(240, Math.min(420, cv.parentElement.clientWidth || 420)), T = theme(), g = fit(cv, W, W), gp = L.gridPredict(st.model, st.ds, 40);
      g.clearRect(0, 0, W, W); drawBoundary(g, W, W, { res: gp.res, grid: gp.grid, bounds: gp.bounds, points: st.ds.X.map((p, i) => [p[0], p[1], st.ds.y[i]]) }, T);
      const h2 = 90, g2 = fit(cl, W, h2); g2.clearRect(0, 0, W, h2);
      if (st.loss.length > 1) drawLine(g2, W, h2, { title: "pérdida", series: [{ name: "entrenamiento", y: st.loss.filter((_, i) => i % Math.max(1, Math.floor(st.loss.length / 200)) === 0) }] }, T); else { g2.fillStyle = T.muted; g2.font = FONT; g2.fillText("La curva de pérdida aparecerá al entrenar.", 8, 40); }
      const r = st.last || st.model.evaluate(st.ds.X, st.ds.y);
      stats.innerHTML = `<span>paso <b>${st.paso}</b></span><span>pérdida <b>${r.loss.toFixed(3)}</b></span><span>precisión <b>${(r.acc * 100).toFixed(1)} %</b></span><span>parámetros <b>${st.model.nParams()}</b></span>`;
    }
    mkData(); mkModel(); reg(cv, () => draw(true));
  };

  // ---- 7. modelo de lenguaje
  widgets.lenguaje = (el) => {
    const S = ["el gato", "la niña", "el ingeniero", "la máquina", "el robot", "mi vecino"], Vb = ["come", "mira", "busca", "construye", "analiza"], O = ["pescado", "el motor", "un puente", "los datos", "la red"], A = ["", " rápido", " con calma", " hoy"];
    const g = L.rng(1), pick = (a) => a[g.int(a.length)]; let txt = ""; for (let i = 0; i < 3000; i++) txt += pick(S) + " " + pick(Vb) + " " + pick(O) + pick(A) + ".\n";
    const cut = Math.floor(txt.length * 0.9), tr = txt.slice(0, cut), va = txt.slice(cut), chars = [...new Set(txt)].sort(), Vn = chars.length, floor = Math.log(600) / (txt.length / 3000), ORD = [0, 1, 2, 3, 4, 6, 8];
    const tabs = []; for (let k = 0; k <= 8; k++) tabs.push(new Map());
    for (let i = 0; i < tr.length; i++) for (let k = 0; k <= 8 && i - k >= 0; k++) { const c = tr.slice(i - k, i); let m = tabs[k].get(c); if (!m) { m = { tot: 0, c: {} }; tabs[k].set(c, m); } m.c[tr[i]] = (m.c[tr[i]] || 0) + 1; m.tot++; }
    const prob = (n, ctx, ch) => { let p = 1 / Vn; for (let k = 0; k <= n; k++) { if (ctx.length < k) break; const m = tabs[k].get(k ? ctx.slice(ctx.length - k) : ""); if (!m) break; p = ((m.c[ch] || 0) + 0.5 * p) / (m.tot + 0.5); } return p; };
    const nll = (n) => { let s = 0; for (let i = 1; i < va.length; i++) s -= Math.log(prob(n, va.slice(Math.max(0, i - n), i), va[i])); return s / (va.length - 1); };
    const losses = ORD.map(nll), st = { n: 8, T: 0.8 }, host = h("div", { class: "canvas-wrap" }), outp = h("div", { class: "readout", style: "min-height:6em;white-space:pre-wrap" });
    const spec = { type: "bar", title: "pérdida de validación (nats por carácter)", categories: ORD.map((n) => (n === 0 ? "sin contexto" : n === 1 ? "bigrama (1)" : "contexto de " + n)), values: losses, ref: { value: floor, label: "cota teórica " + floor.toFixed(3) } };
    renderPlot(spec, host, 560);
    const gen = () => {
      const lines = []; for (let r = 0; r < 5; r++) {
        let ctx = "\n", s = ""; for (let i = 0; i < 70; i++) {
          const pr = chars.map((c) => Math.pow(prob(st.n, ctx.slice(-st.n || ctx.length), c), 1 / st.T)), tot = pr.reduce((a, b) => a + b, 0); let u = Math.random() * tot, k = 0; while (k < pr.length - 1 && u >= pr[k]) { u -= pr[k]; k++; }
          const c = chars[k]; if (c === "\n") break; s += c; ctx += c;
        } lines.push(s);
      } outp.textContent = lines.join("\n");
    };
    const bGen = h("button", { class: "btn primary", onclick: gen }, "Generar 5 frases");
    el.append(h("div", { class: "widget" }, host, h("p", { style: "margin:0;font-size:14px;color:var(--muted)" }, "Corpus de 3 000 frases. Cota teórica: ", h("b", { style: "color:var(--ink)" }, floor.toFixed(3) + " nats/carácter"), ". Sin contexto (todas las letras con su frecuencia) la pérdida es ", losses[0].toFixed(2), "; con 8 caracteres llega a ", losses[6].toFixed(3), "."),
      h("div", { class: "widget cols", style: "grid-template-columns:minmax(0,260px) minmax(0,1fr)" }, h("div", { class: "controls" }, select("contexto al generar", ORD.filter((n) => n > 0).map((n) => [String(n), n + " caracteres"]), "8", (v) => { st.n = +v; }), slider("temperatura τ", 0.2, 2, 0.05, st.T, (v) => { st.T = v; }, (v) => v.toFixed(2)).el, bGen), outp))); gen();
  };

  // ---- 8. difusión
  widgets.difusion = (el) => {
    const T = 200, ab = []; let p = 1; for (const b of L.linspace(1e-4, 0.06, T)) { p *= 1 - b; ab.push(p); }
    const g0 = L.rng(3), N = 350, M = 700, mk = () => { const u = g0.next(), t = 1.5 * Math.PI * (1 + 2 * u); return [t * Math.cos(t), t * Math.sin(t)]; };
    const raw = Array.from({ length: M }, mk), mxa = Math.max(...raw.flat().map(Math.abs)), data = raw.map((q) => [(q[0] / mxa) * 1.5 + 0.03 * g0.normal(), (q[1] / mxa) * 1.5 + 0.03 * g0.normal()]);
    const eps = Array.from({ length: N }, () => [g0.normal(), g0.normal()]), st = { t: 0, mode: "fwd", pts: null, steps: 20, tm: null, info: "" };
    const cv = h("canvas", { role: "img", "aria-label": "Puntos de la espiral con ruido" }), stats = h("div", { class: "stats" });
    const fwdPts = (t) => eps.map((e, i) => [Math.sqrt(ab[t]) * data[i][0] + Math.sqrt(1 - ab[t]) * e[0], Math.sqrt(ab[t]) * data[i][1] + Math.sqrt(1 - ab[t]) * e[1]]);
    const sT = slider("paso de ruido t", 0, T - 1, 1, 0, (v) => { stop(); st.mode = "fwd"; st.t = v; st.pts = null; draw(); });
    function stop() { clearInterval(st.tm); st.tm = null; }
    const bD = h("button", { class: "btn", onclick: () => { stop(); st.mode = "fwd"; st.t = 0; st.tm = setInterval(() => { st.t = Math.min(T - 1, st.t + 4); sT.set(st.t); draw(); if (st.t >= T - 1) stop(); }, 40); } }, "Destruir"),
      bG = h("button", { class: "btn primary", onclick: () => {
        stop(); st.mode = "rev"; const ts = L.linspace(T - 1, 0, st.steps).map(Math.round); let x = Array.from({ length: N }, () => [g0.normal(), g0.normal()]), i = 0; st.pts = x; st.t = ts[0]; sT.set(st.t); draw();
        st.tm = setInterval(() => {
          const t = ts[i], a = ab[t], ap = i + 1 < ts.length ? ab[ts[i + 1]] : 1, sa = Math.sqrt(a), s1 = Math.sqrt(1 - a);
          x = x.map((xi) => { // denoiser ideal: media posterior de x0 dado xt, sobre el conjunto de datos
            let mxl = -Infinity; const lg = data.map((q) => { const dx = xi[0] - sa * q[0], dy = xi[1] - sa * q[1], v = -(dx * dx + dy * dy) / (2 * (1 - a) + 1e-9); if (v > mxl) mxl = v; return v; });
            let sw = 0, x0 = 0, y0 = 0; for (let j = 0; j < data.length; j++) { const w = Math.exp(lg[j] - mxl); sw += w; x0 += w * data[j][0]; y0 += w * data[j][1]; } x0 /= sw; y0 /= sw;
            const e0 = (xi[0] - sa * x0) / s1, e1 = (xi[1] - sa * y0) / s1; return [Math.sqrt(ap) * x0 + Math.sqrt(1 - ap) * e0, Math.sqrt(ap) * y0 + Math.sqrt(1 - ap) * e1];
          });
          st.pts = x; st.t = i + 1 < ts.length ? ts[i + 1] : 0; sT.set(st.t); draw(); i++; if (i >= ts.length) { stop(); st.info = "Generación terminada: las muestras vuelven a formar la espiral."; draw(); }
        }, 380); st.info = "";
      } }, "Generar desde ruido");
    const stp = select("pasos al generar (DDIM)", [["5", "5"], ["10", "10"], ["20", "20"], ["50", "50"]], "20", (v) => { st.steps = +v; });
    el.append(h("div", { class: "widget cols" }, h("div", { class: "controls" }, sT.el, h("div", { class: "btn-row" }, bD, bG), stp, stats), h("div", { class: "canvas-wrap" }, cv)));
    function draw() {
      const W = Math.max(240, Math.min(460, cv.parentElement.clientWidth || 460)), g = fit(cv, W, W), Tm = theme(), X = (v) => ((v + 3) / 6) * W, Y = (v) => W - ((v + 3) / 6) * W;
      g.clearRect(0, 0, W, W); g.fillStyle = rgbs(rgb(Tm.cls[0]), 0.28); for (const q of data) { g.beginPath(); g.arc(X(q[0]), Y(q[1]), 2.2, 0, 7); g.fill(); }
      const pts = st.mode === "rev" && st.pts ? st.pts : fwdPts(st.t); g.fillStyle = Tm.cls[1]; for (const q of pts) { g.beginPath(); g.arc(X(q[0]), Y(q[1]), 2.8, 0, 7); g.fill(); }
      g.font = FONT; g.fillStyle = Tm.muted; g.textAlign = "left"; g.fillText("azul: datos originales · ámbar: xₜ", 8, W - 8);
      const a = ab[Math.min(T - 1, Math.max(0, st.t))]; stats.innerHTML = `<span>t <b>${st.t}</b></span><span>ᾱₜ <b>${fmtN(a, 3)}</b></span><span>señal √ᾱ <b>${(Math.sqrt(a) * 100).toFixed(0)} %</b></span><span>ruido √(1−ᾱ) <b>${(Math.sqrt(1 - a) * 100).toFixed(0)} %</b></span>` + (st.info ? `<span style="color:var(--good);flex-basis:100%">${st.info}</span>` : "");
    }
    reg(cv, draw);
  };

  // ---- temperatura
  widgets.temperatura = (el) => {
    const cats = ["A", "B", "C", "D", "E"], p = [0.05, 0.5, 0.25, 0.15, 0.05], st = { T: 1 };
    const host = h("div", { class: "canvas-wrap" }), info = h("div", { class: "readout" });
    const sl = slider("temperatura T", 0.1, 3, 0.05, 1, (v) => { st.T = v; draw(); }, (v) => v.toFixed(2));
    el.append(h("div", { class: "widget cols" }, h("div", { class: "controls" }, sl.el, info), host));
    function draw() {
      const w = p.map((x) => Math.pow(x, 1 / st.T)), s = w.reduce((a, b) => a + b, 0), q = w.map((x) => x / s);
      host.replaceChildren(); renderPlot({ type: "bar", title: "probabilidad de cada opción tras la temperatura", categories: cats.map((c, i) => c + " · antes " + Math.round(p[i] * 100) + " %"), values: q }, host, 520);
      info.innerHTML = `La opción B sale el <b>${Math.round(q[1] * 100)} %</b> de las veces.<br><span style="color:var(--muted)">${st.T < 0.5 ? "Temperatura baja: casi siempre la más probable." : st.T > 1.8 ? "Temperatura alta: casi todo parejo, más variedad y más errores." : "Cerca de 1: respeta las probabilidades originales."}</span>`;
    }
    draw();
  };

  // ============================================================ interfaz principal
  const nav = $("#nav"), main = $("#main"), MOD = Object.fromEntries(MODULES.map((m) => [m.id, m]));
  const TOTAL_MIN = LESSONS.reduce((s, l) => s + l.minutes, 0);
  let dockOpen = false;
  function renderNav() {
    const parts = [h("button", { class: "btn small home-btn", onclick: () => go("inicio") }, "Inicio y guía")];
    for (const m of MODULES) {
      const ls = LESSONS.filter((l) => l.module === m.id);
      parts.push(h("h2", null, m.title + " · " + ls.reduce((s, l) => s + l.minutes, 0) + " min"));
      parts.push(h("ul", { class: "lesson-list" }, ls.map((l) => {
        const i = LESSONS.indexOf(l), ok = unlocked(i), s = state.done[l.id], badge = s === "pasado" ? "✓" : s === "saltado" ? "↷" : String(l.num);
        return h("li", null, h("button", { class: "lesson-item", disabled: !ok, "data-state": s || (ok ? "open" : "locked"), "aria-current": state.current === l.id ? "true" : "false", onclick: () => go(l.id) },
          h("span", { class: "badge", html: ok ? esc(badge) : '<svg viewBox="0 0 24 24"><rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/></svg>' }), h("span", null, h("b", null, l.title), h("small", null, l.minutes + " min"))));
      })));
    }
    parts.push(h("h2", null, "Próximamente"), ...PROXIMAS.map((p) => h("div", { class: "soon" }, h("b", null, p.title), h("small", null, p.note))));
    parts.push(h("div", { class: "side-foot" }, h("span", null, "Tu progreso se guarda en este navegador."), h("button", { class: "btn small", onclick: resetProgress }, "Reiniciar progreso")));
    nav.replaceChildren(...parts);
    const n = LESSONS.filter((l) => isDone(l.id)).length, left = LESSONS.filter((l) => !isDone(l.id)).reduce((s, l) => s + l.minutes, 0);
    $("#progressText").textContent = n + " de " + LESSONS.length + (left ? " · quedan ≈ " + left + " min" : " · ¡completo!"); $("#progressFill").style.width = (100 * n) / LESSONS.length + "%";
  }
  function resetProgress() { if (!resetProgress.armed) { resetProgress.armed = true; const b = nav.querySelector(".side-foot .btn"); b.textContent = "Pulsa otra vez para confirmar"; setTimeout(() => { resetProgress.armed = false; b.textContent = "Reiniciar progreso"; }, 3500); return; } resetProgress.armed = false; state.done = {}; saveDone(); LESSONS.forEach((l) => { store.del("code:" + l.id); (l.examples || []).forEach((e, k) => store.del("code:" + l.id + "-ex" + k)); }); go("inicio", true); }
  function go(id, force) { if (state.current === id && !force) { closeNav(); return; } location.hash = id === "inicio" ? "inicio" : "leccion-" + id; if (state.current !== id || force) show(id); closeNav(); }
  const firstOpen = () => LESSONS.find((l, i) => !isDone(l.id) && unlocked(i)) || LESSONS[LESSONS.length - 1];

  function renderHome() {
    state.current = "inicio"; renderNav(); drawables.clear();
    const pyDone = LESSONS.filter((l) => l.module === "python").every((l) => isDone(l.id)), started = LESSONS.some((l) => isDone(l.id));
    const startBtn = h("button", { class: "btn primary big", onclick: () => go(firstOpen().id) }, started ? "Continuar donde lo dejé →" : "Empezar →");
    const skipBtn = pyDone ? null : h("button", { class: "btn big", onclick: () => { LESSONS.filter((l) => l.module === "python").forEach((l) => { if (!state.done[l.id]) state.done[l.id] = "saltado"; }); saveDone(); go("neurona"); } }, "Ya sé Python: ir a redes neuronales");
    const mods = MODULES.map((m, k) => { const ls = LESSONS.filter((l) => l.module === m.id), n = ls.filter((l) => isDone(l.id)).length; return h("div", { class: "mod" }, h("span", { class: "chip" }, "Módulo " + (k + 1)), h("h3", null, m.title), h("p", null, m.blurb), h("ul", null, ls.map((l) => h("li", null, l.title, h("small", null, " · " + l.minutes + " min")))), h("small", { class: "modp" }, n + " de " + ls.length + " completadas")); });
    main.replaceChildren(h("article", { class: "lesson" },
      h("div", { class: "lesson-head" }, h("div", { class: "chips" }, h("span", { class: "chip" }, "≈ " + TOTAL_MIN + " minutos en total"), h("span", { class: "chip mute" }, "Sin instalar nada"), h("span", { class: "chip mute" }, "Python real")),
        h("h1", null, "Aprende Python e IA en una hora"), h("p", { class: "lead" }, "Lecciones de 3 a 8 minutos. Lees poco, ejecutas código de verdad y ves el resultado al instante. Los retos se comprueban solos.")),
      h("div", { class: "btn-row" }, startBtn, skipBtn),
      h("section", { class: "section" }, h("h2", null, "Cómo funciona cada lección"), h("div", { class: "steps" },
        h("div", null, h("b", null, "1 · Lee"), h("span", null, "Una idea corta, sin tecnicismos.")), h("div", null, h("b", null, "2 · Prueba"), h("span", null, "Ejecuta ejemplos y cámbialos.")), h("div", null, h("b", null, "3 · Completa"), h("span", null, "Rellena los huecos ___ y pulsa Comprobar.")))),
      h("section", { class: "section" }, h("h2", null, "Tres módulos"), h("div", { class: "mods" }, mods)),
      h("p", { class: "note" }, "Funciona sin instalar nada y sin conexión una vez cargada la página. Ejecuta un Python simplificado que cubre lo esencial: todo lo que aprendes aquí funciona igual en Python real. No incluye clases, try/except ni librerías externas como numpy. Si te atascas, cada reto tiene pistas y solución. La consola de abajo (botón «Consola») es un Python libre para probar lo que quieras.")));
    window.scrollTo({ top: 0 });
  }

  function show(id) {
    if (id === "inicio") return renderHome();
    const i = Math.max(0, LESSONS.findIndex((l) => l.id === id)), l = LESSONS[i]; if (!unlocked(i)) return show(firstOpen().id);
    state.current = l.id; renderNav(); drawables.clear();
    const nextBtn = h("div", { class: "callout next", hidden: !isDone(l.id) || i === LESSONS.length - 1 }, h("span", null, h("b", null, "Siguiente: "), (LESSONS[i + 1] || {}).title), h("button", { class: "btn primary", onclick: () => go(LESSONS[i + 1].id) }, "Ir a la lección " + (l.num + 1) + " →"));
    const done = h("div", { class: "callout", hidden: !(i === LESSONS.length - 1 && isDone(l.id)) }, h("b", null, "¡Terminaste el recorrido! "), "Ya sabes lo básico de Python y cómo piensan las redes y los modelos generativos. La consola es tuya para seguir probando.");
    const task = makeTask(l, () => { renderNav(); nextBtn.hidden = i === LESSONS.length - 1; if (i === LESSONS.length - 1) done.hidden = false; });
    const exs = (l.examples || []).map((e, k) => { const c = makeCode({ id: l.id + "-ex" + k, code: e.code, tag: "Ejemplo · ejecútalo y cámbialo", label: "Ejemplo " + (k + 1) }); return h("div", { class: "example" }, h("p", { class: "ex-note" }, e.note), c.el); });
    const wEl = h("div");
    const lesson = h("article", { class: "lesson" },
      h("div", { class: "lesson-head" }, h("div", { class: "chips" }, h("span", { class: "chip" }, MOD[l.module].title), h("span", { class: "chip mute" }, "Lección " + l.num + " de " + LESSONS.length), h("span", { class: "chip mute" }, "≈ " + l.minutes + " min")), h("h1", null, l.title), h("p", { class: "goal" }, h("b", null, "Al terminar: "), l.goal)),
      h("section", { class: "section" }, h("h2", null, "La idea"), h("div", { class: "prose", html: l.intro })),
      exs.length ? h("section", { class: "section" }, h("h2", null, "Pruébalo"), ...exs) : null,
      l.widget ? h("section", { class: "section" }, h("h2", null, "Míralo en vivo"), h("div", { class: "panel" }, h("header", null, h("h3", null, l.widgetTitle), h("p", null, l.widgetNote)), wEl)) : null,
      h("section", { class: "section" }, h("h2", null, "Tu reto"), h("div", { class: "prose", html: l.taskIntro }), task, nextBtn, done),
      h("section", { class: "section" }, h("h2", null, "Lo que te llevas"), h("div", { class: "callout prose", html: l.outro })),
      h("div", { class: "lesson-nav" }, i > 0 ? h("button", { class: "btn", onclick: () => go(LESSONS[i - 1].id) }, "← " + LESSONS[i - 1].title) : h("span"), unlocked(i + 1) && LESSONS[i + 1] ? h("button", { class: "btn", onclick: () => go(LESSONS[i + 1].id) }, LESSONS[i + 1].title + " →") : h("span")));
    main.replaceChildren(lesson); if (l.widget) widgets[l.widget](wEl); window.scrollTo({ top: 0 });
  }
  function closeNav() { document.body.classList.remove("nav-open"); const s = $(".scrim"); if (s) s.remove(); }
  function openNav() { document.body.classList.add("nav-open"); if (!$(".scrim")) document.body.append(h("div", { class: "scrim", onclick: closeNav })); }
  $("#menuBtn").addEventListener("click", () => (document.body.classList.contains("nav-open") ? closeNav() : openNav()));

  // ---- tema
  function applyTheme(t) { if (t) document.documentElement.setAttribute("data-theme", t); else document.documentElement.removeAttribute("data-theme"); }
  applyTheme(store.get("theme", null));
  $("#themeBtn").addEventListener("click", () => { const cur = document.documentElement.getAttribute("data-theme"), dark = cur ? cur === "dark" : matchMedia("(prefers-color-scheme: dark)").matches, t = dark ? "light" : "dark"; store.set("theme", t); applyTheme(t); });

  // ---- consola de Python (REPL)
  const dock = $("#dock"), dOut = $("#dockOut"), dIn = $("#dockIn"), dSt = $("#dockSt"), hist = [], hp = { i: 0 };
  function setDock(open) { dockOpen = open; dock.hidden = !open; document.body.style.setProperty("--dock-h", open ? dock.offsetHeight + "px" : "0px"); $("#consoleBtn").setAttribute("aria-pressed", String(open)); if (open) { dIn.focus(); redrawAll(); if (!dOut.children.length) welcome(); } }
  $("#consoleBtn").addEventListener("click", () => setDock(!dockOpen));
  $("#dockClose").addEventListener("click", () => setDock(false));
  $("#dockClear").addEventListener("click", () => dOut.replaceChildren());
  $("#dockReset").addEventListener("click", async () => { await kernel.reset(); addLine(dOut, "Python reiniciado: se borraron las variables.", "sys"); });
  $("#dockHelp").addEventListener("click", () => submit("ayuda()"));
  window.addEventListener("resize", () => { if (dockOpen) document.body.style.setProperty("--dock-h", dock.offsetHeight + "px"); });
  kernel.busyCb = (b) => { dSt.classList.toggle("busy", b); dSt.lastChild.textContent = b ? "ejecutando…" : "Python listo"; };
  function welcome() { addLine(dOut, "Consola de Python. Comparte variables con las lecciones, como un cuaderno.", "sys"); addLine(dOut, "Escribe algo y pulsa Enter. Prueba: 2 + 2  ·  import math  ·  math.sqrt(16)  ·  ayuda()", "sys"); }
  async function submit(code) {
    code = code.replace(/\s+$/, ""); if (!code.trim()) return; hist.push(code); hp.i = hist.length; addLine(dOut, "› " + code.split("\n").join("\n  "), "ech");
    const r = await kernel.run(code, sinkFor(dOut), { repl: true });
    if (!r.ok) addError(dOut, r.err, code);
  }
  dIn.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      const v = dIn.value, lines = v.split("\n"), last = lines[lines.length - 1];
      if (/:\s*$/.test(last) || (lines.length > 1 && last.trim() !== "")) { e.preventDefault(); const ind = (last.match(/^\s*/) || [""])[0] + (/:\s*$/.test(last) ? "    " : ""); dIn.setRangeText("\n" + ind, dIn.selectionStart, dIn.selectionEnd, "end"); dIn.dispatchEvent(new Event("input")); return; }
      e.preventDefault(); dIn.value = ""; dIn.style.height = "auto"; submit(v);
    } else if (e.key === "Tab") { e.preventDefault(); dIn.setRangeText("    ", dIn.selectionStart, dIn.selectionEnd, "end"); }
    else if (e.key === "ArrowUp" && !dIn.value.includes("\n") && hp.i > 0) { e.preventDefault(); dIn.value = hist[--hp.i]; }
    else if (e.key === "ArrowDown" && !dIn.value.includes("\n") && hp.i < hist.length) { e.preventDefault(); hp.i++; dIn.value = hp.i === hist.length ? "" : hist[hp.i]; }
  });
  dIn.addEventListener("input", () => { dIn.style.height = "auto"; dIn.style.height = Math.min(140, dIn.scrollHeight) + "px"; });
  window.__ln = { kernel, LESSONS, state, go, show, setDock };   // para depuración desde la consola del navegador

  // ---- arranque
  function route() { const hsh = location.hash, m = hsh.match(/^#leccion-([a-z]+)$/); if (m && LESSONS.some((l) => l.id === m[1])) show(m[1]); else if (hsh === "#inicio" || !LESSONS.some((l) => isDone(l.id))) renderHome(); else show(firstOpen().id); }
  window.addEventListener("hashchange", () => { const m = location.hash.match(/^#leccion-([a-z]+)$/); if (m && m[1] !== state.current && LESSONS.some((l) => l.id === m[1])) show(m[1]); else if (location.hash === "#inicio" && state.current !== "inicio") renderHome(); });
  route();
})();
