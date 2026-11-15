/* Kernel del laboratorio: todo lo que el alumno puede usar en sus celdas y en la consola.
   Se ejecuta dentro de un Web Worker (si falla, en la página). `G` es el objeto global. */
function kernelLib(G) {
  // ------------------------------------------------------------ formato de salida
  function num(x) {
    if (Number.isInteger(x)) return String(x);
    if (!Number.isFinite(x)) return String(x);
    const a = Math.abs(x);
    if (a !== 0 && (a < 1e-4 || a >= 1e7)) return x.toExponential(4).replace(/\.?0+e/, "e");
    return String(parseFloat(x.toPrecision(6)));
  }
  function fmt(v, depth = 0) {
    if (v === undefined) return "undefined";
    if (v === null) return "null";
    if (typeof v === "number") return num(v);
    if (typeof v === "string") return depth ? JSON.stringify(v) : v;
    if (typeof v === "boolean") return String(v);
    if (typeof v === "function") return "ƒ " + (v.name || "anónima");
    if (v instanceof Error) return v.name + ": " + v.message;
    if (v && typeof v.toString === "function" && v.constructor && v.constructor.name === "Value") return v.toString();
    if (ArrayBuffer.isView(v)) v = Array.from(v);
    if (Array.isArray(v)) {
      if (depth >= 3) return "[…]";
      const lim = depth === 0 && Array.isArray(v[0]) ? 6 : 12;
      const parts = v.slice(0, lim).map((e) => fmt(e, depth + 1));
      return "[" + parts.join(", ") + (v.length > lim ? `, … (${v.length})` : "") + "]";
    }
    if (typeof v === "object") {
      if (depth >= 2) return "{…}";
      const ks = Object.keys(v).slice(0, 12);
      return "{ " + ks.map((k) => k + ": " + fmt(v[k], depth + 1)).join(", ") + (Object.keys(v).length > 12 ? ", …" : "") + " }";
    }
    return String(v);
  }
  const print = (...a) => G.__emit({ type: "log", text: a.map((x) => fmt(x)).join(" ") });
  const plot = (spec) => G.__emit({ type: "plot", spec });

  // ------------------------------------------------------------ azar reproducible
  function rng(seed = 1) {
    let a = seed >>> 0;
    const next = () => {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const normal = () => {
      let u = 0;
      while (u === 0) u = next();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next());
    };
    return { next, normal, int: (n) => Math.floor(next() * n) };
  }

  // ------------------------------------------------------------ álgebra con arreglos
  const range = (n) => Array.from({ length: n }, (_, i) => i);
  const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => (n === 1 ? a : a + ((b - a) * i) / (n - 1)));
  const sum = (v) => v.reduce((s, x) => s + x, 0);
  const mean = (v) => sum(v) / v.length;
  const std = (v) => { const m = mean(v); return Math.sqrt(mean(v.map((x) => (x - m) ** 2))); };
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  const argmax = (v) => v.reduce((bi, x, i) => (x > v[bi] ? i : bi), 0);
  const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
  const transpose = (A) => A[0].map((_, j) => A.map((row) => row[j]));
  function matmul(A, B) {
    const n = A.length, k = B.length, m = B[0].length;
    const C = zeros(n, m);
    for (let i = 0; i < n; i++) for (let p = 0; p < k; p++) { const a = A[i][p]; if (a !== 0) for (let j = 0; j < m; j++) C[i][j] += a * B[p][j]; }
    return C;
  }
  function randn(r, c, seed = 0) { const g = rng(seed); return Array.from({ length: r }, () => Array.from({ length: c }, () => g.normal())); }
  function softmax(v) {
    const m = Math.max(...v);
    const e = v.map((x) => Math.exp(x - m));
    const s = sum(e);
    return e.map((x) => x / s);
  }

  // ------------------------------------------------------------ autograd escalar (micrograd)
  class Value {
    constructor(data, prev = [], op = "") { this.data = data; this.grad = 0; this._prev = prev; this._op = op; this._back = () => {}; }
    static w(x) { return x instanceof Value ? x : new Value(x); }
    add(o) { o = Value.w(o); const out = new Value(this.data + o.data, [this, o], "+"); out._back = () => { this.grad += out.grad; o.grad += out.grad; }; return out; }
    mul(o) { o = Value.w(o); const out = new Value(this.data * o.data, [this, o], "*"); out._back = () => { this.grad += o.data * out.grad; o.grad += this.data * out.grad; }; return out; }
    pow(k) { const out = new Value(this.data ** k, [this], "**" + k); out._back = () => { this.grad += k * this.data ** (k - 1) * out.grad; }; return out; }
    neg() { return this.mul(-1); }
    sub(o) { return this.add(Value.w(o).neg()); }
    div(o) { return this.mul(Value.w(o).pow(-1)); }
    exp() { const out = new Value(Math.exp(this.data), [this], "exp"); out._back = () => { this.grad += out.data * out.grad; }; return out; }
    log() { const out = new Value(Math.log(this.data), [this], "log"); out._back = () => { this.grad += (1 / this.data) * out.grad; }; return out; }
    tanh() { const t = Math.tanh(this.data); const out = new Value(t, [this], "tanh"); out._back = () => { this.grad += (1 - t * t) * out.grad; }; return out; }
    relu() { const out = new Value(Math.max(0, this.data), [this], "relu"); out._back = () => { this.grad += (out.data > 0 ? 1 : 0) * out.grad; }; return out; }
    sigmoid() { const s = 1 / (1 + Math.exp(-this.data)); const out = new Value(s, [this], "sigmoid"); out._back = () => { this.grad += s * (1 - s) * out.grad; }; return out; }
    backward() {
      const topo = [], seen = new Set();
      const build = (v) => { if (!seen.has(v)) { seen.add(v); v._prev.forEach(build); topo.push(v); } };
      build(this);
      this.grad = 1;
      for (let i = topo.length - 1; i >= 0; i--) topo[i]._back();
    }
    toString() { return `Value(data=${num(this.data)}, grad=${num(this.grad)})`; }
  }
  const V = (x) => Value.w(x);
  /** Compara el gradiente de autodiff con la derivada numérica. f recibe Values y devuelve un Value. */
  function gradcheck(f, xs, h = 1e-6) {
    const vs = xs.map((x) => new Value(x));
    f(...vs).backward();
    let worst = 0;
    xs.forEach((x, i) => {
      const p = xs.slice(), q = xs.slice();
      p[i] += h; q[i] -= h;
      const num_ = (f(...p.map((z) => new Value(z))).data - f(...q.map((z) => new Value(z))).data) / (2 * h);
      worst = Math.max(worst, Math.abs(vs[i].grad - num_) / Math.max(1e-12, Math.abs(vs[i].grad) + Math.abs(num_)));
    });
    return worst;
  }

  // ------------------------------------------------------------ datos sintéticos
  function moons(n = 200, ruido = 0.1, seed = 0) {
    const g = rng(seed), X = [], y = [];
    for (let i = 0; i < n; i++) {
      const c = i % 2, t = g.next() * Math.PI;
      const p = c === 0 ? [Math.cos(t), Math.sin(t)] : [1 - Math.cos(t), 0.5 - Math.sin(t)];
      X.push([p[0] + ruido * g.normal(), p[1] + ruido * g.normal()]); y.push(c);
    }
    return { X, y };
  }
  function spirals(nPorClase = 100, clases = 3, ruido = 0.2, seed = 0) {
    const g = rng(seed), X = [], y = [];
    for (let k = 0; k < clases; k++) for (let i = 0; i < nPorClase; i++) {
      const r = i / (nPorClase - 1), t = k * 4 + (i / (nPorClase - 1)) * 4 + ruido * g.normal();
      X.push([r * Math.sin(t), r * Math.cos(t)]); y.push(k);
    }
    return { X, y };
  }
  function xorData(n = 200, ruido = 0.15, seed = 0) {
    const g = rng(seed), X = [], y = [];
    for (let i = 0; i < n; i++) { const a = g.next() < 0.5 ? -1 : 1, b = g.next() < 0.5 ? -1 : 1; X.push([a * 0.7 + ruido * g.normal(), b * 0.7 + ruido * g.normal()]); y.push(a * b > 0 ? 1 : 0); }
    return { X, y };
  }

  // ------------------------------------------------------------ MLP vectorizado con backprop manual
  class MLP {
    constructor(dims, opts = {}) {
      this.dims = dims; this.act = opts.act || "tanh"; this.optim = opts.optim || "sgd"; this.t = 0;
      const g = rng(opts.seed ?? 0);
      this.W = []; this.b = []; this.mW = []; this.vW = []; this.mb = []; this.vb = [];
      for (let l = 0; l < dims.length - 1; l++) {
        const fi = dims[l], fo = dims[l + 1];
        const sd = this.act === "relu" ? Math.sqrt(2 / fi) : Math.sqrt(1 / fi) * (this.act === "tanh" ? 5 / 3 : 1);
        this.W.push(Float64Array.from({ length: fi * fo }, () => g.normal() * sd));
        this.b.push(new Float64Array(fo));
        this.mW.push(new Float64Array(fi * fo)); this.vW.push(new Float64Array(fi * fo));
        this.mb.push(new Float64Array(fo)); this.vb.push(new Float64Array(fo));
      }
    }
    nParams() { return this.W.reduce((s, w, i) => s + w.length + this.b[i].length, 0); }
    _f(z) { return this.act === "relu" ? (z > 0 ? z : 0) : this.act === "sigmoid" ? 1 / (1 + Math.exp(-z)) : Math.tanh(z); }
    _df(z, a) { return this.act === "relu" ? (z > 0 ? 1 : 0) : this.act === "sigmoid" ? a * (1 - a) : 1 - a * a; }
    _flat(X) { const n = X.length, d = X[0].length, F = new Float64Array(n * d); for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) F[i * d + j] = X[i][j]; return F; }
    _forward(F, n) {
      const L = this.W.length, as = [F], zs = [];
      for (let l = 0; l < L; l++) {
        const fi = this.dims[l], fo = this.dims[l + 1], A = as[l], W = this.W[l], b = this.b[l];
        const Z = new Float64Array(n * fo);
        for (let i = 0; i < n; i++) for (let j = 0; j < fo; j++) { let s = b[j]; for (let p = 0; p < fi; p++) s += A[i * fi + p] * W[p * fo + j]; Z[i * fo + j] = s; }
        zs.push(Z);
        if (l < L - 1) { const O = new Float64Array(n * fo); for (let q = 0; q < O.length; q++) O[q] = this._f(Z[q]); as.push(O); } else as.push(Z);
      }
      return { as, zs };
    }
    _softmax(Z, n, k) {
      const P = new Float64Array(n * k);
      for (let i = 0; i < n; i++) { let m = -Infinity; for (let j = 0; j < k; j++) m = Math.max(m, Z[i * k + j]); let s = 0; for (let j = 0; j < k; j++) { const e = Math.exp(Z[i * k + j] - m); P[i * k + j] = e; s += e; } for (let j = 0; j < k; j++) P[i * k + j] /= s; }
      return P;
    }
    predict(X) {
      const n = X.length, k = this.dims[this.dims.length - 1], { as } = this._forward(this._flat(X), n), Z = as[as.length - 1], out = [];
      for (let i = 0; i < n; i++) { let bj = 0; for (let j = 1; j < k; j++) if (Z[i * k + j] > Z[i * k + bj]) bj = j; out.push(bj); }
      return out;
    }
    evaluate(X, y) {
      const n = X.length, k = this.dims[this.dims.length - 1], { as } = this._forward(this._flat(X), n), P = this._softmax(as[as.length - 1], n, k);
      let loss = 0, ok = 0;
      for (let i = 0; i < n; i++) { loss -= Math.log(P[i * k + y[i]] + 1e-12); let bj = 0; for (let j = 1; j < k; j++) if (P[i * k + j] > P[i * k + bj]) bj = j; if (bj === y[i]) ok++; }
      return { loss: loss / n, acc: ok / n };
    }
    /** Un paso de descenso con todo el lote. Devuelve { loss, acc } ANTES de actualizar. */
    trainStep(X, y, lr = 0.1, opts = {}) {
      const n = X.length, k = this.dims[this.dims.length - 1], L = this.W.length, wd = opts.weightDecay || 0;
      const { as, zs } = this._forward(this._flat(X), n), P = this._softmax(as[L], n, k);
      let loss = 0, ok = 0;
      let dZ = new Float64Array(n * k);
      for (let i = 0; i < n; i++) {
        loss -= Math.log(P[i * k + y[i]] + 1e-12);
        let bj = 0; for (let j = 1; j < k; j++) if (P[i * k + j] > P[i * k + bj]) bj = j; if (bj === y[i]) ok++;
        for (let j = 0; j < k; j++) dZ[i * k + j] = (P[i * k + j] - (j === y[i] ? 1 : 0)) / n;
      }
      this.t++;
      for (let l = L - 1; l >= 0; l--) {
        const fi = this.dims[l], fo = this.dims[l + 1], A = as[l], W = this.W[l];
        const dW = new Float64Array(fi * fo), db = new Float64Array(fo);
        for (let i = 0; i < n; i++) for (let j = 0; j < fo; j++) { const d = dZ[i * fo + j]; db[j] += d; for (let p = 0; p < fi; p++) dW[p * fo + j] += A[i * fi + p] * d; }
        let dA = null;
        if (l > 0) {
          dA = new Float64Array(n * fi);
          for (let i = 0; i < n; i++) for (let p = 0; p < fi; p++) { let s = 0; for (let j = 0; j < fo; j++) s += dZ[i * fo + j] * W[p * fo + j]; dA[i * fi + p] = s * this._df(zs[l - 1][i * fi + p], A[i * fi + p]); }
        }
        this._update(l, dW, db, lr, wd);
        dZ = dA;
      }
      return { loss: loss / n, acc: ok / n };
    }
    _update(l, dW, db, lr, wd) {
      const W = this.W[l], b = this.b[l];
      if (this.optim === "adam") {
        const b1 = 0.9, b2 = 0.999, c1 = 1 - b1 ** this.t, c2 = 1 - b2 ** this.t;
        for (let q = 0; q < W.length; q++) { this.mW[l][q] = b1 * this.mW[l][q] + (1 - b1) * dW[q]; this.vW[l][q] = b2 * this.vW[l][q] + (1 - b2) * dW[q] * dW[q]; W[q] -= lr * (this.mW[l][q] / c1 / (Math.sqrt(this.vW[l][q] / c2) + 1e-8) + wd * W[q]); }
        for (let q = 0; q < b.length; q++) { this.mb[l][q] = b1 * this.mb[l][q] + (1 - b1) * db[q]; this.vb[l][q] = b2 * this.vb[l][q] + (1 - b2) * db[q] * db[q]; b[q] -= lr * (this.mb[l][q] / c1 / (Math.sqrt(this.vb[l][q] / c2) + 1e-8)); }
      } else if (this.optim === "momentum") {
        for (let q = 0; q < W.length; q++) { this.mW[l][q] = 0.9 * this.mW[l][q] + dW[q]; W[q] -= lr * (this.mW[l][q] + wd * W[q]); }
        for (let q = 0; q < b.length; q++) { this.mb[l][q] = 0.9 * this.mb[l][q] + db[q]; b[q] -= lr * this.mb[l][q]; }
      } else {
        for (let q = 0; q < W.length; q++) W[q] -= lr * (dW[q] + wd * W[q]);
        for (let q = 0; q < b.length; q++) b[q] -= lr * db[q];
      }
    }
  }
  /** Rejilla de predicciones del modelo para dibujar la frontera de decisión. */
  function gridPredict(model, data, res = 48) {
    const xs = data.X.map((p) => p[0]), ys = data.X.map((p) => p[1]);
    const pad = 0.25, x0 = Math.min(...xs) - pad, x1 = Math.max(...xs) + pad, y0 = Math.min(...ys) - pad, y1 = Math.max(...ys) + pad;
    const pts = [];
    for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) pts.push([x0 + ((x1 - x0) * (i + 0.5)) / res, y1 - ((y1 - y0) * (j + 0.5)) / res]);
    return { grid: model.predict(pts), res, bounds: [x0, x1, y0, y1] };
  }
  function plotBoundary(model, data, titulo = "Frontera de decisión") {
    const g = gridPredict(model, data);
    plot({ type: "boundary", title: titulo, grid: g.grid, res: g.res, bounds: g.bounds, points: data.X.map((p, i) => [p[0], p[1], data.y[i]]) });
  }

  // ------------------------------------------------------------ ayuda
  function ayuda() {
    print([
      "Funciones disponibles en la consola y en las celdas:",
      "  print(...)  plot({type:'line'|'scatter'|'heat'|'bar', ...})  plotBoundary(modelo, datos)",
      "  Value(x)    autograd escalar: a.add(b) .sub .mul .div .pow(k) .exp() .log() .tanh() .relu() .sigmoid() .backward()",
      "  V(x)        envuelve un número en Value      gradcheck(f, [x1, x2, ...])",
      "  matmul(A,B) transpose(A) zeros(r,c) randn(r,c,seed) softmax(v) dot(a,b) argmax(v)",
      "  range(n) linspace(a,b,n) sum(v) mean(v) std(v)       rng(seed) → { next(), normal(), int(n) }",
      "  moons(n,ruido,seed) spirals(nPorClase,clases,ruido,seed) xorData(n,ruido,seed)",
      "  new MLP([2,16,3], { act:'tanh'|'relu'|'sigmoid', optim:'sgd'|'momentum'|'adam', seed })",
      "      .trainStep(X, y, lr) → {loss, acc}   .evaluate(X, y)   .predict(X)   .nParams()",
      "Las variables que declares con const o let se conservan entre celdas (como en un cuaderno).",
    ].join("\n"));
  }

  // ------------------------------------------------------------ ejecución
  function prep(code) {
    return code
      .replace(/^(const|let)\s/gm, "var ")
      .replace(/^class\s+([A-Za-z_$][\w$]*)/gm, "var $1 = class $1");
  }
  G.__handle = function (msg) {
    if (msg.type === "run") {
      try {
        const r = (0, eval)(prep(msg.code));
        G.__emit({ type: "result", id: msg.id, ok: true, has: r !== undefined, value: r === undefined ? "" : fmt(r) });
      } catch (e) {
        G.__emit({ type: "result", id: msg.id, ok: false, error: e && e.name ? e.name + ": " + e.message : String(e) });
      }
    } else if (msg.type === "check") {
      let results;
      try { const fn = (0, eval)("(" + msg.code + ")"); results = fn(); }
      catch (e) { results = [{ name: "Preparación del reto", ok: false, msg: String(e && e.message ? e.message : e) }]; }
      G.__emit({ type: "check", id: msg.id, results });
    }
  };

  Object.assign(G, { fmt, num, print, plot, rng, range, linspace, sum, mean, std, dot, argmax, zeros, transpose, matmul, randn, softmax,
    Value, V, gradcheck, moons, spirals, xorData, MLP, gridPredict, plotBoundary, ayuda });
  if (G.console) { G.console.log = print; G.console.info = print; G.console.warn = print; G.console.error = (...a) => print("⚠", ...a); }
}
if (typeof module !== "undefined") module.exports = kernelLib;
