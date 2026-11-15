/* Laboratorio Neuronal: aplicación. Depende de kernel.js (kernelLib) y lessons.js (LESSONS, PROXIMAS). */
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
    const ticks = log ? (() => { const out = [], st = Math.max(1, Math.ceil((hi - lo) / 6)); for (let p = lo; p <= hi; p += st) out.push(p); return out; })() : niceTicks(lo, hi, 5);
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

  // ============================================================ kernel
  class Kernel {
    constructor() { this.seq = 0; this.q = Promise.resolve(); this.cur = null; this.busyCb = () => {}; this.mode = "worker"; this.ready = this.boot(); }
    boot() {
      return new Promise((resolve) => {
        const fallback = () => { if (this.mode === "main" && this._ok) return; this.mode = "main"; try { if (this.w) this.w.terminate(); } catch (e) { /* */ } L2(); this._ok = true; resolve(); };
        const L2 = () => { window.__emit = (m) => this.onMsg(m); kernelLib(window); };
        try {
          const src = "self.__emit=function(m){postMessage(m)};(" + kernelLib.toString() + ")(self);onmessage=function(e){self.__handle(e.data)};";
          this.url = URL.createObjectURL(new Blob([src], { type: "text/javascript" }));
          this.w = new Worker(this.url);
          let done = false;
          this.w.onerror = () => { if (!done) { done = true; fallback(); } };
          this.w.onmessage = (e) => { if (!done && e.data && e.data.type === "result" && e.data.id === 0) { done = true; this._ok = true; clearTimeout(tm); resolve(); } else this.onMsg(e.data); };
          const tm = setTimeout(() => { if (!done) { done = true; fallback(); } }, 2500);
          this.w.postMessage({ type: "run", id: 0, code: "1" });
        } catch (e) { fallback(); }
      });
    }
    restart() { try { this.w && this.w.terminate(); } catch (e) { /* */ } this._ok = false; this.cur = null; this.ready = this.boot(); }
    onMsg(m) {
      if (m.type === "log" || m.type === "plot") { if (this.cur && this.cur.sink) this.cur.sink(m); return; }
      if (this.cur && m.id === this.cur.id) { clearTimeout(this.cur.timer); const c = this.cur; this.cur = null; this.busyCb(false); c.resolve(m); }
    }
    enqueue(msg, sink) {
      const p = this.q.then(async () => {
        await this.ready;
        return new Promise((resolve) => {
          const id = ++this.seq; msg.id = id; this.busyCb(true);
          const c = { id, sink, resolve };
          if (this.mode === "worker") c.timer = setTimeout(() => {
            this.restart(); this.busyCb(false);
            resolve({ ok: false, error: "Tiempo agotado (8 s). ¿Hay un bucle infinito? Reinicié el kernel, así que las variables anteriores se perdieron.", results: [{ name: "Ejecución", ok: false, msg: "El código tardó más de 8 s y se detuvo." }], timeout: true });
          }, 8000);
          this.cur = c;
          if (this.mode === "worker") this.w.postMessage(msg); else window.__handle(msg);
        });
      });
      this.q = p.catch(() => {}); return p;
    }
    run(code, sink) { return this.enqueue({ type: "run", code }, sink); }
    check(fn) { return this.enqueue({ type: "check", code: fn.toString() }); }
    reset() { this.restart(); }
  }
  const kernel = new Kernel();

  // ============================================================ editor con resaltado
  const KW = /^(function|return|const|let|var|if|else|for|while|of|in|new|class|true|false|null|undefined|break|continue|typeof|instanceof|throw|try|catch|async|await)$/;
  const LIBN = /^(print|plot|plotBoundary|Value|V|MLP|Math|randn|matmul|transpose|zeros|softmax|dot|argmax|range|linspace|sum|mean|std|rng|moons|spirals|xorData|gradcheck|ayuda)$/;
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  function highlight(src) {
    const re = /(\/\/[^\n]*)|("(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`)|(\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b)|([A-Za-z_$][\w$]*)/g;
    let out = "", last = 0, m;
    while ((m = re.exec(src))) {
      out += esc(src.slice(last, m.index)); last = re.lastIndex;
      if (m[1]) out += '<span class="tk-c">' + esc(m[1]) + "</span>";
      else if (m[2]) out += '<span class="tk-s">' + esc(m[2]) + "</span>";
      else if (m[3]) out += '<span class="tk-n">' + m[3] + "</span>";
      else if (KW.test(m[4])) out += '<span class="tk-k">' + m[4] + "</span>";
      else if (LIBN.test(m[4])) out += '<span class="tk-f">' + m[4] + "</span>";
      else out += esc(m[4]);
    }
    return out + esc(src.slice(last));
  }
  function makeEditor(initial, { onChange, onRun, label } = {}) {
    const pre = h("pre", { "aria-hidden": "true" }), ta = h("textarea", { spellcheck: "false", autocapitalize: "off", autocomplete: "off", autocorrect: "off", "aria-label": label || "Editor de código", rows: 6 });
    const wrap = h("div", { class: "editor" }, pre, ta);
    const sync = () => { pre.innerHTML = highlight(ta.value) + "\n"; ta.style.height = "auto"; ta.style.height = Math.max(120, ta.scrollHeight + 2) + "px"; pre.scrollLeft = ta.scrollLeft; };
    ta.value = initial; sync();
    requestAnimationFrame(() => { sync(); setTimeout(sync, 120); });          // se mide de nuevo ya insertado en la página
    try { document.fonts.ready.then(sync); } catch (e) { /* sin API de fuentes */ }
    reg(wrap, sync);
    ta.addEventListener("input", () => { sync(); onChange && onChange(ta.value); });
    ta.addEventListener("scroll", () => { pre.scrollLeft = ta.scrollLeft; });
    ta.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); onRun && onRun(); return; }
      const s = ta.selectionStart, en = ta.selectionEnd, v = ta.value;
      if (e.key === "Tab") { e.preventDefault(); ta.setRangeText("  ", s, en, "end"); ta.dispatchEvent(new Event("input")); }
      else if (e.key === "Enter") {
        e.preventDefault(); const ls = v.lastIndexOf("\n", s - 1) + 1, ind = (v.slice(ls, s).match(/^\s*/) || [""])[0], extra = /[{(\[]\s*$/.test(v.slice(ls, s)) ? "  " : "";
        ta.setRangeText("\n" + ind + extra, s, en, "end"); ta.dispatchEvent(new Event("input"));
      }
    });
    return { el: wrap, get value() { return ta.value; }, set value(x) { ta.value = x; sync(); }, focus() { ta.focus(); } };
  }

  // ============================================================ salida de celdas y consola
  function addLine(host, text, cls) { const e = h("div", { class: "ln " + (cls || "") }, text); host.append(e); host.scrollTop = host.scrollHeight; return e; }
  function sinkFor(host) { return (m) => { if (m.type === "log") addLine(host, m.text); else if (m.type === "plot") { try { renderPlot(m.spec, host, 560); } catch (e) { addLine(host, "No se pudo dibujar el gráfico: " + e.message, "err"); } } }; }
  async function execute(code, host) {
    host.replaceChildren();
    const r = await kernel.run(code, sinkFor(host));
    if (!r.ok) addLine(host, r.error, "err"); else if (r.has) addLine(host, "← " + r.value, "res");
    return r;
  }

  // ============================================================ progreso
  const state = { done: store.get("done", {}), current: null };
  const saveDone = () => store.set("done", state.done);
  const isDone = (id) => !!state.done[id];
  const unlocked = (i) => i === 0 || isDone(LESSONS[i - 1].id);

  function makeCell(lesson, onPass) {
    const key = "code:" + lesson.id, saved = store.get(key, null);
    const out = h("div", { class: "out", "aria-live": "polite" }), tests = h("div", { class: "tests" }), win = h("div", { class: "win", hidden: true }), tag = h("span", { class: "tag" }, "Código · JavaScript");
    const ed = makeEditor(saved ?? lesson.code, { onChange: (v) => store.set(key, v), onRun: () => run(), label: "Editor de la lección " + lesson.num });
    let fails = 0;
    const btnRun = h("button", { class: "btn primary", onclick: () => run(), title: "Ctrl/Cmd + Enter" }, "▶ Ejecutar");
    const btnCheck = h("button", { class: "btn good", onclick: () => check() }, "✓ Comprobar reto");
    const btnReset = h("button", { class: "btn small", onclick: () => { ed.value = lesson.code; store.del(key); out.replaceChildren(); tests.replaceChildren(); win.hidden = true; } }, "Restablecer");
    const skip = h("button", { class: "btn small", hidden: true, onclick: () => { state.done[lesson.id] = state.done[lesson.id] || "saltado"; saveDone(); onPass("saltado"); } }, "Saltar este reto");
    async function run() { btnRun.disabled = true; try { return await execute(ed.value, out); } finally { btnRun.disabled = false; } }
    async function check() {
      btnCheck.disabled = true; tests.replaceChildren(h("div", { class: "test" }, h("span"), "Comprobando…"));
      try {
        const r = await run(); if (r.timeout) { tests.replaceChildren(); return; }
        if (!r.ok) { tests.replaceChildren(h("div", { class: "test no" }, h("span", { class: "ic" }, "✕"), h("div", null, "Tu código tiene un error. Corrígelo y vuelve a comprobar."))); fails++; skip.hidden = fails < 2; return; }
        const c = await kernel.check(lesson.check), res = c.results || [];
        tests.replaceChildren(...res.map((t) => h("div", { class: "test " + (t.ok ? "ok" : "no") }, h("span", { class: "ic" }, t.ok ? "✓" : "✕"), h("div", null, t.name, t.ok ? null : h("small", null, t.msg)))));
        if (res.length && res.every((t) => t.ok)) { win.hidden = false; win.textContent = "Reto superado. " + res.length + " de " + res.length + " comprobaciones correctas."; if (state.done[lesson.id] !== "pasado") { state.done[lesson.id] = "pasado"; saveDone(); } onPass("pasado"); }
        else { win.hidden = true; fails++; skip.hidden = fails < 2; }
      } finally { btnCheck.disabled = false; }
    }
    const aid = h("details", { class: "aid" }, h("summary", null, "Pistas y solución"),
      h("ol", null, lesson.hints.map((t) => h("li", null, t))),
      h("details", { style: "margin-top:10px" }, h("summary", { style: "font-weight:600;font-size:13px" }, "Ver solución completa"), h("pre", null, lesson.solution), h("button", { class: "btn small", style: "margin-top:8px", onclick: () => { ed.value = lesson.solution; store.set(key, lesson.solution); } }, "Copiar al editor")));
    const el = h("div", { class: "cell" }, h("div", { class: "cell-bar" }, tag, h("span", { class: "sp" }), btnReset, skip, btnRun, btnCheck), ed.el, out, tests, win, aid);
    return el;
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

  // ---- 3. grafo de retropropagación
  widgets.grafo = (el) => {
    const w = new L.Value(0.5), x = new L.Value(2), b = new L.Value(-0.3), m = w.mul(x), z = m.add(b), a = z.tanh(), d = a.sub(1), Lo = d.pow(2); Lo.backward();
    const N = [["w", "w", w, 60, 45], ["x", "x", x, 60, 120], ["b", "b", b, 60, 195], ["m", "m = w·x", m, 200, 82], ["z", "z = m + b", z, 330, 140], ["a", "a = tanh z", a, 460, 140], ["d", "d = a − 1", d, 590, 140], ["L", "L = d²", Lo, 700, 140]];
    const E = [["w", "m"], ["x", "m"], ["m", "z"], ["b", "z"], ["z", "a"], ["a", "d"], ["d", "L"]], pos = Object.fromEntries(N.map((n) => [n[0], n]));
    const NS = "http://www.w3.org/2000/svg", svg = document.createElementNS(NS, "svg"); svg.setAttribute("viewBox", "0 0 760 250"); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "Grafo computacional de L");
    const mk = (t, at, txt) => { const e = document.createElementNS(NS, t); for (const k in at) e.setAttribute(k, at[k]); if (txt != null) e.textContent = txt; return e; };
    for (const [p, q] of E) { const A = pos[p], B = pos[q]; svg.append(mk("line", { class: "g-edge", x1: A[3] + 48, y1: A[4], x2: B[3] - 48, y2: B[4] })); }
    const groups = {};
    for (const [id, name, v, cx, cy] of N) {
      const gr = mk("g", { class: "g-node" }); gr.append(mk("rect", { x: cx - 48, y: cy - 27, width: 96, height: 54 }), mk("text", { class: "nm", x: cx, y: cy - 8 }, name), mk("text", { class: "vl", x: cx, y: cy + 14 }, fmtN(v.data, 4)), mk("text", { class: "gr", x: cx, y: cy + 45 }, "∂L = " + fmtN(v.grad, 3)));
      svg.append(gr); groups[id] = gr;
    }
    const fwd = ["m", "z", "a", "d", "L"], bwd = ["L", "d", "a", "z", "m", "w", "x", "b"]; let timers = [];
    const clear = () => { timers.forEach(clearTimeout); timers = []; };
    const reset = () => { clear(); for (const k in groups) groups[k].setAttribute("class", "g-node"); ["w", "x", "b"].forEach((k) => groups[k].setAttribute("class", "g-node on has-v")); note.textContent = "Las entradas valen w = 0,5, x = 2, b = −0,3."; };
    const seq = (ids, cls, text, delay = 500, then) => { ids.forEach((id, i) => timers.push(setTimeout(() => { const g = groups[id]; g.setAttribute("class", "g-node on has-v" + (cls === "g" || g.classList.contains("has-g") ? " has-g" : "")); if (cls === "g") g.classList.add("has-g"); }, i * delay))); timers.push(setTimeout(() => { note.textContent = text; then && then(); }, ids.length * delay)); };
    const note = h("p", { class: "readout", style: "margin:0" }), allFwd = () => fwd.forEach((k) => groups[k].setAttribute("class", "g-node on has-v"));
    const bF = h("button", { class: "btn primary", onclick: () => { reset(); note.textContent = "Calculando hacia delante…"; seq(fwd, "v", "Hacia delante: L = 0,1565. Ahora propaga los gradientes hacia atrás."); } }, "Hacia delante"),
      bB = h("button", { class: "btn", onclick: () => { clear(); allFwd(); ["w", "x", "b"].forEach((k) => groups[k].setAttribute("class", "g-node on has-v")); for (const k of bwd) groups[k].classList.remove("has-g"); note.textContent = "Propagando hacia atrás: cada nodo multiplica el gradiente que recibe por su derivada local…"; seq(bwd, "g", "Listo. dL/dw = −1,004, dL/db = −0,502, dL/dx = −0,251. Son los números de la guía."); } }, "Hacia atrás"), bR = h("button", { class: "btn", onclick: reset }, "Reiniciar");
    el.append(h("div", { class: "widget" }, h("div", { class: "btn-row" }, bF, bB, bR), h("div", { class: "graph-wrap" }, svg), note)); reset();
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

  // ---- 5. inicialización
  widgets.init = (el) => {
    const INITS = [["N(0, 1)", () => 1], ["N(0, 0.01)", () => 0.01], ["Xavier", (f) => Math.sqrt(1 / f)], ["He", (f) => Math.sqrt(2 / f)]], cache = {}, st = { act: "relu" };
    function sim(scale, act, layers = 30, width = 128, batch = 64) {
      const g = L.rng(7); let hh = Float64Array.from({ length: batch * width }, () => g.normal()), res = []; const s = scale(width);
      for (let l = 0; l < layers; l++) {
        const W = Float64Array.from({ length: width * width }, () => g.normal() * s), O = new Float64Array(batch * width);
        for (let i = 0; i < batch; i++) for (let p = 0; p < width; p++) { const a = hh[i * width + p]; if (a !== 0) for (let j = 0; j < width; j++) O[i * width + j] += a * W[p * width + j]; }
        for (let q = 0; q < O.length; q++) O[q] = act === "relu" ? (O[q] > 0 ? O[q] : 0) : Math.tanh(O[q]);
        hh = O; let m = 0, v = 0; for (const x of hh) m += x; m /= hh.length; for (const x of hh) v += (x - m) ** 2; res.push(Math.sqrt(v / hh.length));
      } return res;
    }
    const cv = h("canvas", { role: "img", "aria-label": "Desviación estándar por capa" }), sum = h("div", { class: "readout" });
    const sel = select("activación", [["relu", "ReLU"], ["tanh", "tanh"]], "relu", (v) => { st.act = v; draw(); });
    el.append(h("div", { class: "widget" }, h("div", { style: "max-width:260px" }, sel), h("div", { class: "canvas-wrap" }, cv), sum));
    function draw() {
      if (!cache[st.act]) cache[st.act] = INITS.map(([n, s]) => ({ name: n, y: sim(s, st.act) }));
      const series = cache[st.act], W = Math.max(260, Math.min(640, cv.parentElement.clientWidth || 640)), g = fit(cv, W, Math.round(W * 0.5)); g.clearRect(0, 0, W, W);
      drawLine(g, W, Math.round(W * 0.5), { title: "std de las activaciones por capa (escala log)", ylog: true, x1: true, series }, theme());
      sum.innerHTML = "<b>En la capa 30:</b> " + series.map((s) => `${s.name} → ${fmtN(s.y[29], 2)}`).join(" · ") + "<br><span style='color:var(--muted)'>Una línea plana es una señal que sobrevive. Las que caen o se disparan hacen imposible entrenar.</span>";
    }
    reg(cv, draw);
  };

  // ---- 6. atención
  widgets.atencion = (el) => {
    const st = { text: "el robot busca los datos hoy", k: 4, scale: true, causal: true };
    const inp = h("input", { type: "text", value: st.text, "aria-label": "Frase", maxlength: 60 }), cvHost = h("div", { class: "canvas-wrap" }), info = h("div", { class: "readout" });
    inp.addEventListener("input", () => { st.text = inp.value; draw(); });
    const sd = slider("dimensión d", 1, 8, 1, st.k, (v) => { st.k = v; draw(); }, (v) => String(2 ** v));
    const cs = h("input", { type: "checkbox", id: "w6s", checked: true }), cc = h("input", { type: "checkbox", id: "w6c", checked: true });
    cs.addEventListener("change", () => { st.scale = cs.checked; draw(); }); cc.addEventListener("change", () => { st.causal = cc.checked; draw(); });
    const hash = (s) => { let x = 5381; for (const c of s) x = ((x * 33) ^ c.charCodeAt(0)) >>> 0; return x; };
    el.append(h("div", { class: "widget cols" }, h("div", { class: "controls" }, h("div", { class: "ctl" }, h("span", { class: "lbl" }, "frase (hasta 8 palabras)"), inp), sd.el, h("label", { class: "ctl", style: "display:flex;gap:8px;align-items:center" }, cs, "dividir por √d"), h("label", { class: "ctl", style: "display:flex;gap:8px;align-items:center" }, cc, "máscara causal")), h("div", null, cvHost, h("div", { style: "height:10px" }), info)));
    let cv = null;
    function compute() {
      const toks = st.text.trim().split(/\s+/).filter(Boolean).slice(0, 8), d = 2 ** st.k; if (!toks.length) return null;
      const E = toks.map((t) => { const g = L.rng(hash(t)); return Array.from({ length: d }, () => g.normal()); }), proj = (seed) => { const g = L.rng(seed * 1000 + d); return Array.from({ length: d }, () => Array.from({ length: d }, () => g.normal() / Math.sqrt(d))); };
      const Q = L.matmul(E, proj(1)), K = L.matmul(E, proj(2)), S = L.matmul(Q, L.transpose(K)).map((r) => r.map((s) => (st.scale ? s / Math.sqrt(d) : s)));
      const P = S.map((r, i) => L.softmax(r.map((s, j) => (st.causal && j > i ? -Infinity : s))));
      return { toks, P, d };
    }
    function draw() {
      const r = compute(); if (!r) { cvHost.replaceChildren(); info.textContent = "Escribe una frase."; return; }
      cvHost.replaceChildren(); const spec = { type: "heat", title: "pesos de atención (fila = quién mira, columna = a quién)", matrix: r.P, xlabels: r.toks, ylabels: r.toks, min: 0, max: 1 }; cv = renderPlot(spec, cvHost, 480);
      const mx = r.P.map((row) => Math.max(...row)), avg = mx.reduce((a, b) => a + b, 0) / mx.length;
      info.innerHTML = `d = <b>${r.d}</b> · peso máximo medio por fila: <b>${avg.toFixed(2)}</b> (1,00 = mira a una sola palabra)<br><span style="color:var(--muted)">${!st.scale && r.d >= 64 ? "Sin dividir por √d, con d grande el softmax se vuelve casi 0/1." : st.scale ? "Con el escalado, la mezcla se mantiene repartida aunque d crezca." : "Prueba subir d con el escalado desactivado."}</span>`;
    }
    draw();
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

  // ============================================================ interfaz principal
  const nav = $("#nav"), main = $("#main");
  let dockOpen = false;
  function renderNav() {
    const items = LESSONS.map((l, i) => {
      const ok = unlocked(i), s = state.done[l.id], badge = s === "pasado" ? "✓" : s === "saltado" ? "↷" : ok ? String(l.num) : "";
      const b = h("button", { class: "lesson-item", disabled: !ok, "data-state": s || (ok ? "open" : "locked"), "aria-current": state.current === l.id ? "true" : "false", onclick: () => go(l.id) },
        h("span", { class: "badge", html: ok ? esc(badge) : '<svg viewBox="0 0 24 24"><rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/></svg>' }), h("span", null, h("b", null, l.title), h("small", null, l.week + " · " + l.minutes + " min")));
      return h("li", null, b);
    });
    nav.replaceChildren(h("h2", null, "Lecciones"), h("ul", { class: "lesson-list" }, items), h("h2", null, "Próximamente"), ...PROXIMAS.map((p) => h("div", { class: "soon" }, h("b", null, p.title), h("small", null, p.week + " · " + p.note))),
      h("div", { class: "side-foot" }, h("span", null, "Tu progreso se guarda en este navegador. Cuando se añadan lecciones, aparecerán aquí sin perder lo hecho."), h("button", { class: "btn small", onclick: resetProgress }, "Reiniciar progreso")));
    const n = LESSONS.filter((l) => isDone(l.id)).length; $("#progressText").textContent = n + " de " + LESSONS.length + " completadas"; $("#progressFill").style.width = (100 * n) / LESSONS.length + "%";
  }
  function resetProgress() { if (!resetProgress.armed) { resetProgress.armed = true; const b = nav.querySelector(".side-foot .btn"); b.textContent = "Pulsa otra vez para confirmar"; setTimeout(() => { resetProgress.armed = false; b.textContent = "Reiniciar progreso"; }, 3500); return; } resetProgress.armed = false; state.done = {}; saveDone(); LESSONS.forEach((l) => store.del("code:" + l.id)); go(LESSONS[0].id, true); }
  function go(id, force) { if (state.current === id && !force) { closeNav(); return; } location.hash = "leccion-" + id; if (state.current !== id || force) show(id); closeNav(); }
  function show(id) {
    const i = Math.max(0, LESSONS.findIndex((l) => l.id === id)), l = LESSONS[i]; if (!unlocked(i)) { return show(LESSONS[0].id); }
    state.current = l.id; renderNav(); drawables.clear();
    const nextBtn = h("div", { class: "callout next", hidden: !isDone(l.id) || i === LESSONS.length - 1 }, h("span", null, h("b", null, "Siguiente: "), (LESSONS[i + 1] || {}).title), h("button", { class: "btn primary", onclick: () => go(LESSONS[i + 1].id) }, "Ir a la lección " + (l.num + 1) + " →"));
    const cell = makeCell(l, (s) => { renderNav(); nextBtn.hidden = i === LESSONS.length - 1; if (i === LESSONS.length - 1) { done.hidden = false; } });
    const wEl = h("div"), done = h("div", { class: "callout", hidden: !(i === LESSONS.length - 1 && isDone(l.id)) }, h("b", null, "Completaste todas las lecciones disponibles. "), "Pronto habrá más. Mientras tanto, la consola de abajo es tuya: pruébalo todo.");
    const lesson = h("article", { class: "lesson" },
      h("div", { class: "lesson-head" }, h("div", { class: "chips" }, h("span", { class: "chip" }, "Lección " + l.num + " de " + LESSONS.length), h("span", { class: "chip mute" }, "Guía del curso · " + l.week), h("span", { class: "chip mute" }, "≈ " + l.minutes + " min")), h("h1", null, l.title)),
      h("section", { class: "section" }, h("h2", null, "La idea"), h("div", { class: "prose", html: l.intro })),
      h("section", { class: "section" }, h("h2", null, "Pruébalo"), h("div", { class: "panel" }, h("header", null, h("h3", null, l.widgetTitle), h("p", null, l.widgetNote)), wEl)),
      h("section", { class: "section" }, h("h2", null, "Ahora programa tú"), h("div", { class: "prose", html: l.taskIntro }), cell, nextBtn, done),
      h("section", { class: "section" }, h("h2", null, "Lo que te llevas"), h("div", { class: "callout prose", html: l.outro })),
      h("div", { class: "lesson-nav" }, i > 0 ? h("button", { class: "btn", onclick: () => go(LESSONS[i - 1].id) }, "← " + LESSONS[i - 1].title) : h("span"), unlocked(i + 1) && LESSONS[i + 1] ? h("button", { class: "btn", onclick: () => go(LESSONS[i + 1].id) }, LESSONS[i + 1].title + " →") : h("span")));
    main.replaceChildren(lesson); widgets[l.widget](wEl); window.scrollTo({ top: 0 });
  }
  function closeNav() { document.body.classList.remove("nav-open"); const s = $(".scrim"); if (s) s.remove(); }
  function openNav() { document.body.classList.add("nav-open"); if (!$(".scrim")) document.body.append(h("div", { class: "scrim", onclick: closeNav })); }
  $("#menuBtn").addEventListener("click", () => (document.body.classList.contains("nav-open") ? closeNav() : openNav()));

  // ---- tema
  function applyTheme(t) { if (t) document.documentElement.setAttribute("data-theme", t); else document.documentElement.removeAttribute("data-theme"); }
  applyTheme(store.get("theme", null));
  $("#themeBtn").addEventListener("click", () => { const dark = document.documentElement.getAttribute("data-theme") ? document.documentElement.getAttribute("data-theme") === "dark" : matchMedia("(prefers-color-scheme: dark)").matches; const t = dark ? "light" : "dark"; store.set("theme", t); applyTheme(t); });

  // ---- consola (REPL)
  const dock = $("#dock"), dOut = $("#dockOut"), dIn = $("#dockIn"), dSt = $("#dockSt"), hist = [], hp = { i: 0 };
  function setDock(open) { dockOpen = open; dock.hidden = !open; document.body.style.setProperty("--dock-h", open ? dock.offsetHeight + "px" : "0px"); $("#consoleBtn").setAttribute("aria-pressed", String(open)); if (open) { dIn.focus(); redrawAll(); } }
  $("#consoleBtn").addEventListener("click", () => { setDock(!dockOpen); if (dockOpen && !dOut.children.length) welcome(); });
  $("#dockClose").addEventListener("click", () => setDock(false));
  $("#dockClear").addEventListener("click", () => dOut.replaceChildren());
  $("#dockReset").addEventListener("click", () => { kernel.reset(); addLine(dOut, "Kernel reiniciado: se borraron las variables.", "sys"); });
  $("#dockHelp").addEventListener("click", () => submit("ayuda()"));
  window.addEventListener("resize", () => { if (dockOpen) document.body.style.setProperty("--dock-h", dock.offsetHeight + "px"); });
  kernel.busyCb = (b) => { dSt.classList.toggle("busy", b); dSt.lastChild.textContent = b ? "ejecutando…" : "kernel listo"; };
  function welcome() { addLine(dOut, "Consola de JavaScript. Comparte variables con las celdas de las lecciones, como un cuaderno.", "sys"); addLine(dOut, "Escribe una expresión y pulsa Enter (Shift+Enter para otra línea). Prueba: ayuda()  ·  moons(5)  ·  new Value(3).mul(4)", "sys"); }
  async function submit(code) {
    code = code.trim(); if (!code) return; hist.push(code); hp.i = hist.length; addLine(dOut, "› " + code, "ech");
    const r = await kernel.run(code, sinkFor(dOut));
    if (!r.ok) addLine(dOut, r.error, "err"); else if (r.has) addLine(dOut, r.value, "res");
  }
  dIn.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); const v = dIn.value; dIn.value = ""; dIn.style.height = "auto"; submit(v); }
    else if (e.key === "ArrowUp" && (dIn.selectionStart === 0 || !dIn.value.includes("\n")) && hp.i > 0) { e.preventDefault(); dIn.value = hist[--hp.i]; }
    else if (e.key === "ArrowDown" && hp.i < hist.length) { e.preventDefault(); hp.i++; dIn.value = hp.i === hist.length ? "" : hist[hp.i]; }
  });
  dIn.addEventListener("input", () => { dIn.style.height = "auto"; dIn.style.height = Math.min(120, dIn.scrollHeight) + "px"; });
  window.__ln = { kernel, LESSONS, state, go, show, setDock };   // para depuración desde la consola del navegador

  // ---- arranque
  function route() { const m = location.hash.match(/^#leccion-([a-z]+)$/), id = m ? m[1] : null, first = LESSONS.find((l, i) => !isDone(l.id) && unlocked(i)) || LESSONS[LESSONS.length - 1]; show(id && LESSONS.some((l) => l.id === id) ? id : first.id); }
  window.addEventListener("hashchange", () => { const m = location.hash.match(/^#leccion-([a-z]+)$/); if (m && m[1] !== state.current && LESSONS.some((l) => l.id === m[1])) show(m[1]); });
  route();
})();
