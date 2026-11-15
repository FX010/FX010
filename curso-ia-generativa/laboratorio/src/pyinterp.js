/* pyinterp.js — intérprete de un subconjunto de Python 3, escrito desde cero, sin eval ni dependencias.
   Cubre lo que usa un curso para empezar: variables, números, texto, listas, tuplas, diccionarios,
   if/elif/else, for, while, def, lambda, comprensiones, f-strings, import math/random y las funciones
   integradas más comunes. Lo que no soporta (clases, try/except, with, etc.) avisa con un error claro.
   Fidelidad: int exactos (BigInt), repr de floats como CPython, random con el mismo generador (MT19937). */
(function (root) {
  "use strict";

  // ============================================================ errores y valores
  class PyError extends Error {
    constructor(type, msg, line) { super(msg); this.pyType = type; this.pyMsg = msg; this.line = line == null ? null : line; }
  }
  class PyTuple { constructor(items) { this.items = items; } }
  class PyRange { constructor(start, stop, step) { this.start = start; this.stop = stop; this.step = step; } len() { const s = this.step; const n = s > 0n ? (this.stop - this.start + s - 1n) / s : (this.start - this.stop - s - 1n) / -s; return n > 0n ? n : 0n; } }
  class PyIter { constructor(items, kind) { this.items = items; this.kind = kind; } }
  class PyDict {
    constructor() { this.m = new Map(); }
    static key(k) {
      if (typeof k === "bigint") return "n" + k;
      if (typeof k === "boolean") return "n" + (k ? 1 : 0);
      if (typeof k === "number") return Number.isInteger(k) ? "n" + k : "f" + k;
      if (typeof k === "string") return "s" + k;
      if (k === null) return "N";
      if (k instanceof PyTuple) return "t(" + k.items.map(PyDict.key).join(",") + ")";
      throw new PyError("TypeError", "unhashable type: '" + tname(k) + "'");
    }
    get(k) { const e = this.m.get(PyDict.key(k)); return e ? e[1] : undefined; }
    has(k) { return this.m.has(PyDict.key(k)); }
    set(k, v) { const kk = PyDict.key(k), e = this.m.get(kk); if (e) e[1] = v; else this.m.set(kk, [k, v]); }
    del(k) { return this.m.delete(PyDict.key(k)); }
    get size() { return this.m.size; }
    keys() { return [...this.m.values()].map((e) => e[0]); }
    values() { return [...this.m.values()].map((e) => e[1]); }
    entries() { return [...this.m.values()]; }
  }
  class PyFunc { constructor(name, params, body, env, defaults, isLambda) { this.name = name; this.params = params; this.body = body; this.env = env; this.defaults = defaults; this.isLambda = isLambda; } }
  class Builtin { constructor(name, fn, typeName) { this.name = name; this.fn = fn; this.typeName = typeName || null; } }
  class PyModule { constructor(name, attrs) { this.name = name; this.attrs = attrs; } }
  class Env {
    constructor(parent) { this.vars = new Map(); this.parent = parent || null; this.globals = null; }
    lookup(name) { for (let e = this; e; e = e.parent) { if (e.vars.has(name)) return e.vars.get(name); } return undefined; }
    root() { let e = this; while (e.parent) e = e.parent; return e; }
    set(name, v) { if (this.globals && this.globals.has(name)) this.root().vars.set(name, v); else this.vars.set(name, v); }
  }
  class Signal { constructor(kind, value) { this.kind = kind; this.value = value; } }
  const BREAK = new Signal("break"), CONTINUE = new Signal("continue");

  const isInt = (v) => typeof v === "bigint", isFloat = (v) => typeof v === "number";
  function tname(v) {
    if (v === null) return "NoneType"; if (typeof v === "bigint") return "int"; if (typeof v === "number") return "float"; if (typeof v === "boolean") return "bool";
    if (typeof v === "string") return "str"; if (Array.isArray(v)) return "list"; if (v instanceof PyTuple) return "tuple"; if (v instanceof PyDict) return "dict";
    if (v instanceof PyRange) return "range"; if (v instanceof PyFunc || v instanceof Builtin) return "function"; if (v instanceof PyModule) return "module"; if (v instanceof PyIter) return v.kind;
    return "object";
  }

  // ============================================================ representación de texto
  function floatRepr(x) {
    if (Number.isNaN(x)) return "nan"; if (x === Infinity) return "inf"; if (x === -Infinity) return "-inf"; if (x === 0) return Object.is(x, -0) ? "-0.0" : "0.0";
    const ax = Math.abs(x);
    if (ax >= 1e16 || ax < 1e-4) {
      const [m, e] = x.toExponential().split("e"), n = parseInt(e, 10);
      return m + "e" + (n < 0 ? "-" : "+") + String(Math.abs(n)).padStart(2, "0");
    }
    const s = String(x);
    return /[.e]/.test(s) ? s : s + ".0";
  }
  function strRepr(s) {
    const q = s.includes("'") && !s.includes('"') ? '"' : "'";
    let o = q;
    for (const ch of s) {
      if (ch === "\\") o += "\\\\"; else if (ch === "\n") o += "\\n"; else if (ch === "\t") o += "\\t"; else if (ch === "\r") o += "\\r"; else if (ch === q) o += "\\" + q;
      else if (ch.charCodeAt(0) < 32 || ch.charCodeAt(0) === 127) o += "\\x" + ch.charCodeAt(0).toString(16).padStart(2, "0"); else o += ch;
    }
    return o + q;
  }
  function pyrepr(v) {
    if (v === null) return "None"; if (v === true) return "True"; if (v === false) return "False";
    if (typeof v === "bigint") return v.toString(); if (typeof v === "number") return floatRepr(v); if (typeof v === "string") return strRepr(v);
    if (Array.isArray(v)) return "[" + v.map(pyrepr).join(", ") + "]";
    if (v instanceof PyTuple) return v.items.length === 1 ? "(" + pyrepr(v.items[0]) + ",)" : "(" + v.items.map(pyrepr).join(", ") + ")";
    if (v instanceof PyDict) return "{" + v.entries().map(([k, x]) => pyrepr(k) + ": " + pyrepr(x)).join(", ") + "}";
    if (v instanceof PyRange) return "range(" + v.start + ", " + v.stop + (v.step !== 1n ? ", " + v.step : "") + ")";
    if (v instanceof PyFunc) return "<function " + v.name + ">"; if (v instanceof Builtin) return v.typeName ? "<class '" + v.typeName + "'>" : "<built-in function " + v.name + ">";
    if (v instanceof PyModule) return "<module '" + v.name + "'>"; if (v instanceof PyIter) return /^dict_/.test(v.kind) ? v.kind + "([" + v.items.map(pyrepr).join(", ") + "])" : "<" + v.kind + " object>";
    return String(v);
  }
  const pystr = (v) => (typeof v === "string" ? v : pyrepr(v));

  // ============================================================ aritmética y comparaciones
  const num = (v) => (typeof v === "boolean" ? (v ? 1n : 0n) : v);
  const isNumeric = (v) => typeof v === "bigint" || typeof v === "number" || typeof v === "boolean";
  const toNumber = (v) => (typeof v === "bigint" ? Number(v) : typeof v === "boolean" ? (v ? 1 : 0) : v);
  function floorDivBig(a, b) { let q = a / b; if ((a % b !== 0n) && ((a < 0n) !== (b < 0n))) q -= 1n; return q; }
  function modBig(a, b) { const r = a % b; return r !== 0n && ((r < 0n) !== (b < 0n)) ? r + b : r; }
  function modFloat(a, b) { const r = a % b; return r !== 0 && ((r < 0) !== (b < 0)) ? r + b : r; }
  // ---- potencias y divisiones de floats con la misma semántica que CPython
  function ldexp(x, k) { while (k > 1000) { x *= Math.pow(2, 1000); k -= 1000; } while (k < -1000) { x *= Math.pow(2, -1000); k += 1000; } return x * Math.pow(2, k); }
  const bitlen = (b) => b.toString(2).length;
  function bigScale(numr, den, exp2) {          // numr/den · 2^exp2, redondeado correctamente a double
    const sh = Math.max(0, 66 + bitlen(den) - bitlen(numr)), top = numr << BigInt(sh);
    let q = top / den; const rem = top % den; q = q * 2n + (rem !== 0n ? 1n : 0n); let ex = exp2 - sh - 1;
    const bl = bitlen(q); if (bl > 70) { const s2 = bl - 70, lost = q & ((1n << BigInt(s2)) - 1n); q = (q >> BigInt(s2)) | (lost !== 0n ? 1n : 0n); ex += s2; }
    return ldexp(Number(q), ex);
  }
  function exactPowInt(x, n) {                  // x finito y distinto de 0, n entero
    const dv = new DataView(new ArrayBuffer(8)); dv.setFloat64(0, x); const bits = dv.getBigUint64(0), neg = bits >> 63n === 1n;
    let e = Number((bits >> 52n) & 0x7ffn), m = bits & 0xfffffffffffffn; if (e === 0) e = 1; else m |= 1n << 52n; e -= 1075;
    const k = Math.abs(n), N = m ** BigInt(k), E = e * k, v = n >= 0 ? bigScale(N, 1n, E) : bigScale(1n, N, -E);
    return neg && k % 2 === 1 ? -v : v;
  }
  function fpow(x, y) {
    if (y === 0) return 1; if (x === 1) return 1;
    if (x < 0 && !Number.isInteger(y) && Number.isFinite(y)) throw new PyError("NotSupported", "elevar un número negativo a una potencia con decimales da un número complejo, que no está disponible aquí");
    let r;
    if (y === 0.5 && x >= 0) r = Math.sqrt(x);
    else if (Number.isInteger(y) && Math.abs(y) <= 1024 && Number.isFinite(x) && x !== 0) r = exactPowInt(x, y);
    else r = Math.pow(x, y);
    if (!Number.isFinite(r) && Number.isFinite(x) && Number.isFinite(y) && x !== 0) throw new PyError("OverflowError", "(34, 'Numerical result out of range')");
    return r;
  }
  function floatDivmod(vx, wx) {                 // igual que float_divmod de CPython
    let mod = vx % wx, div = (vx - mod) / wx;
    if (mod) { if ((wx < 0) !== (mod < 0)) { mod += wx; div -= 1; } } else mod = wx < 0 ? -0 : 0;
    let fl;
    if (div) { fl = Math.floor(div); if (div - fl > 0.5) fl += 1; } else fl = (vx / wx) < 0 || Object.is(vx / wx, -0) ? -0 : 0;
    return [fl, mod];
  }
  function opSym(op) { return op; }
  function arith(op, a, b) {
    if (isNumeric(a) && isNumeric(b)) {
      a = num(a); b = num(b);
      if (typeof a === "bigint" && typeof b === "bigint") {
        switch (op) {
          case "+": return a + b; case "-": return a - b; case "*": return a * b;
          case "/": if (b === 0n) throw new PyError("ZeroDivisionError", "division by zero"); return Number(a) / Number(b);
          case "//": if (b === 0n) throw new PyError("ZeroDivisionError", "integer division or modulo by zero"); return floorDivBig(a, b);
          case "%": if (b === 0n) throw new PyError("ZeroDivisionError", "integer modulo by zero"); return modBig(a, b);
          case "**": if (b < 0n) { if (a === 0n) throw new PyError("ZeroDivisionError", "0.0 cannot be raised to a negative power"); return fpow(Number(a), Number(b)); } if (b > 100000n) throw new PyError("OverflowError", "exponente demasiado grande"); return a ** b;
        }
      }
      const x = toNumber(a), y = toNumber(b);
      switch (op) {
        case "+": return x + y; case "-": return x - y; case "*": return x * y;
        case "/": if (y === 0) throw new PyError("ZeroDivisionError", "float division by zero"); return x / y;
        case "//": if (y === 0) throw new PyError("ZeroDivisionError", "float floor division by zero"); return floatDivmod(x, y)[0];
        case "%": if (y === 0) throw new PyError("ZeroDivisionError", "float modulo"); return floatDivmod(x, y)[1];
        case "**": if (x === 0 && y < 0) throw new PyError("ZeroDivisionError", "0.0 cannot be raised to a negative power"); return fpow(x, y);
      }
    }
    if (op === "+") {
      if (typeof a === "string" && typeof b === "string") return a + b;
      if (Array.isArray(a) && Array.isArray(b)) return a.concat(b);
      if (a instanceof PyTuple && b instanceof PyTuple) return new PyTuple(a.items.concat(b.items));
      if (typeof a === "string") throw new PyError("TypeError", 'can only concatenate str (not "' + tname(b) + '") to str');
      if (Array.isArray(a)) throw new PyError("TypeError", 'can only concatenate list (not "' + tname(b) + '") to list');
    }
    if (op === "*") {
      const rep = (s, n) => { n = Number(num(n)); return n <= 0 ? null : n; };
      if ((typeof a === "string" || Array.isArray(a) || a instanceof PyTuple) && (typeof b === "bigint" || typeof b === "boolean")) { const n = Math.max(0, Number(num(b))); return typeof a === "string" ? a.repeat(n) : Array.isArray(a) ? [].concat(...Array(n).fill(a)) : new PyTuple([].concat(...Array(n).fill(a.items))); }
      if ((typeof b === "string" || Array.isArray(b) || b instanceof PyTuple) && (typeof a === "bigint" || typeof a === "boolean")) return arith("*", b, a);
    }
    if (op === "%" && typeof a === "string") return percentFormat(a, b);
    throw new PyError("TypeError", "unsupported operand type(s) for " + opSym(op) + ": '" + tname(a) + "' and '" + tname(b) + "'");
  }
  function pyEq(a, b) {
    if (isNumeric(a) && isNumeric(b)) { a = num(a); b = num(b); return typeof a === "bigint" && typeof b === "bigint" ? a === b : toNumber(a) === toNumber(b); }
    if (typeof a === "string" || typeof b === "string") return a === b;
    if (a === null || b === null) return a === b;
    if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((x, i) => pyEq(x, b[i]));
    if (a instanceof PyTuple && b instanceof PyTuple) return a.items.length === b.items.length && a.items.every((x, i) => pyEq(x, b.items[i]));
    if (a instanceof PyDict && b instanceof PyDict) return a.size === b.size && a.entries().every(([k, v]) => b.has(k) && pyEq(v, b.get(k)));
    return a === b;
  }
  function cmp(a, b, op) {
    if (isNumeric(a) && isNumeric(b)) { a = num(a); b = num(b); const x = typeof a === "bigint" && typeof b === "bigint" ? a : toNumber(a), y = typeof a === "bigint" && typeof b === "bigint" ? b : toNumber(b); return op === "<" ? x < y : op === "<=" ? x <= y : op === ">" ? x > y : x >= y; }
    if (typeof a === "string" && typeof b === "string") return op === "<" ? a < b : op === "<=" ? a <= b : op === ">" ? a > b : a >= b;
    const sa = Array.isArray(a) ? a : a instanceof PyTuple ? a.items : null, sb = Array.isArray(b) ? b : b instanceof PyTuple ? b.items : null;
    if (sa && sb && Array.isArray(a) === Array.isArray(b)) {
      for (let i = 0; i < Math.min(sa.length, sb.length); i++) { if (!pyEq(sa[i], sb[i])) return cmp(sa[i], sb[i], op); }
      return op === "<" ? sa.length < sb.length : op === "<=" ? sa.length <= sb.length : op === ">" ? sa.length > sb.length : sa.length >= sb.length;
    }
    throw new PyError("TypeError", "'" + op + "' not supported between instances of '" + tname(a) + "' and '" + tname(b) + "'");
  }
  function truthy(v) {
    if (v === null || v === false) return false; if (v === true) return true;
    if (typeof v === "bigint") return v !== 0n; if (typeof v === "number") return v !== 0; if (typeof v === "string") return v.length > 0;
    if (Array.isArray(v)) return v.length > 0; if (v instanceof PyTuple) return v.items.length > 0; if (v instanceof PyDict) return v.size > 0; if (v instanceof PyRange) return v.len() > 0n;
    return true;
  }

  // ============================================================ formato (f-strings, format, %)
  function fmtSpec(v, spec) {
    if (!spec) return pystr(v);
    const m = spec.match(/^(?:(.)?([<>^=]))?([+\- ])?(0)?(\d+)?(,)?(?:\.(\d+))?([a-zA-Z%])?$/);
    if (!m) throw new PyError("ValueError", "Invalid format specifier '" + spec + "'");
    let [, fill, align, sign, zero, width, comma, prec, type] = m;
    let s;
    if (!type && isNumeric(v)) { const nv = num(v); if (typeof nv === "bigint") type = "d"; else if (prec != null) type = "g"; else type = "r"; }
    if (type === "f" || type === "F" || type === "e" || type === "E" || type === "%" || type === "g") {
      if (!isNumeric(v)) throw new PyError("ValueError", "Unknown format code '" + type + "' for object of type '" + tname(v) + "'");
      const x = toNumber(num(v)), p = prec == null ? 6 : +prec;
      s = type === "%" ? (Math.abs(x) * 100).toFixed(p) + "%" : type === "f" || type === "F" ? Math.abs(x).toFixed(p) : type === "g" ? String(+Math.abs(x).toPrecision(p || 1)) : Math.abs(x).toExponential(p).replace(/e([+-])(\d)$/, "e$10$2");
      if (comma) { const [i, d] = s.split("."); s = i.replace(/\B(?=(\d{3})+(?!\d))/g, ",") + (d != null ? "." + d : ""); }
      s = (x < 0 || Object.is(x, -0) ? "-" : sign === "+" ? "+" : sign === " " ? " " : "") + s;
    } else if (type === "r") {
      const x = toNumber(num(v)); s = floatRepr(Math.abs(x)); if (comma) { const [i, d] = s.split("."); s = i.replace(/\B(?=(\d{3})+(?!\d))/g, ",") + (d != null ? "." + d : ""); }
      s = (x < 0 || Object.is(x, -0) ? "-" : sign === "+" ? "+" : sign === " " ? " " : "") + s;
    } else if (type === "d") {
      if (!(typeof v === "bigint" || typeof v === "boolean")) throw new PyError("ValueError", "Unknown format code 'd' for object of type '" + tname(v) + "'");
      const x = num(v); s = (x < 0n ? "-" : sign === "+" ? "+" : "") + (x < 0n ? -x : x).toString(); if (comma) s = s.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    } else { s = pystr(v); if (prec != null && typeof v === "string") s = s.slice(0, +prec); }
    if (width) {
      const w = +width, numeric = isNumeric(v);
      if (zero && !align) { const neg = /^[+\- ]/.test(s) ? s[0] : ""; s = neg + s.slice(neg.length).padStart(w - neg.length, "0"); }
      else { const a = align || (numeric ? ">" : "<"), f = fill || " "; s = a === "<" ? s.padEnd(w, f) : a === ">" ? s.padStart(w, f) : a === "^" ? (() => { const t = Math.max(0, w - s.length), l = Math.floor(t / 2); return f.repeat(l) + s + f.repeat(t - l); })() : s.padStart(w, f); }
    }
    return s;
  }
  function percentFormat(fmt, args) {
    const vals = args instanceof PyTuple ? args.items : [args]; let i = 0;
    const out = fmt.replace(/%([-+ 0]*)(\d+)?(?:\.(\d+))?([sdifrxXeEgc%])/g, (_, flags, w, p, t) => {
      if (t === "%") return "%";
      if (i >= vals.length) throw new PyError("TypeError", "not enough arguments for format string");
      const v = vals[i++]; let s;
      if (t === "s") s = pystr(v); else if (t === "r") s = pyrepr(v);
      else if (t === "d" || t === "i") s = (typeof v === "number" ? BigInt(Math.trunc(v)) : num(v)).toString();
      else if (t === "f") s = toNumber(num(v)).toFixed(p == null ? 6 : +p); else if (t === "e" || t === "E") s = toNumber(num(v)).toExponential(p == null ? 6 : +p).replace(/e([+-])(\d)$/, "e$10$2"); else if (t === "g") s = String(+toNumber(num(v)).toPrecision(p == null ? 6 : +p || 1));
      else if (t === "x" || t === "X") { s = num(v).toString(16); if (t === "X") s = s.toUpperCase(); } else s = String(v);
      if (t === "s" && p != null) s = s.slice(0, +p);
      if (flags.includes("+") && /[dfie]/.test(t) && !s.startsWith("-")) s = "+" + s;
      if (w) { const n = +w; s = flags.includes("-") ? s.padEnd(n) : flags.includes("0") && /[dfie]/.test(t) ? s.padStart(n, "0") : s.padStart(n); }
      return s;
    });
    if (i < vals.length && !(args instanceof PyDict)) throw new PyError("TypeError", "not all arguments converted during string formatting");
    return out;
  }
  function strFormat(fmt, args, kw) {
    let auto = 0;
    return fmt.replace(/\{\{|\}\}|\{([^{}!:]*)(?:!([rs]))?(?::([^{}]*))?\}/g, (all, name, conv, spec) => {
      if (all === "{{") return "{"; if (all === "}}") return "}";
      let v;
      if (name === "" || name == null) v = args[auto++]; else if (/^\d+$/.test(name)) v = args[+name]; else v = kw ? kw[name] : undefined;
      if (v === undefined) throw new PyError(name && !/^\d+$/.test(name) ? "KeyError" : "IndexError", name && !/^\d+$/.test(name) ? "'" + name + "'" : "Replacement index " + (name === "" ? auto - 1 : name) + " out of range for positional args tuple");
      if (conv === "r") v = pyrepr(v); else if (conv === "s") v = pystr(v);
      return fmtSpec(v, spec || "");
    });
  }

  // ============================================================ generador aleatorio (MT19937, igual que CPython)
  class MT {
    constructor() { this.mt = new Uint32Array(624); this.i = 625; this.gaussNext = null; this.seedWith(Date.now() ^ ((Math.random() * 4294967296) >>> 0)); }
    initGenrand(s) { const mt = this.mt; mt[0] = s >>> 0; for (let i = 1; i < 624; i++) { const p = mt[i - 1] ^ (mt[i - 1] >>> 30); mt[i] = (Math.imul(1812433253, p) + i) >>> 0; } this.i = 624; }
    initByArray(key) {
      this.initGenrand(19650218); const mt = this.mt; let i = 1, j = 0, k = Math.max(624, key.length);
      for (; k; k--) { const p = mt[i - 1] ^ (mt[i - 1] >>> 30); mt[i] = ((mt[i] ^ Math.imul(p, 1664525)) + key[j] + j) >>> 0; i++; j++; if (i >= 624) { mt[0] = mt[623]; i = 1; } if (j >= key.length) j = 0; }
      for (k = 623; k; k--) { const p = mt[i - 1] ^ (mt[i - 1] >>> 30); mt[i] = ((mt[i] ^ Math.imul(p, 1566083941)) - i) >>> 0; i++; if (i >= 624) { mt[0] = mt[623]; i = 1; } }
      mt[0] = 0x80000000; this.i = 624;
    }
    seedWith(n) { let x = typeof n === "bigint" ? (n < 0n ? -n : n) : BigInt(Math.abs(Math.trunc(n))); const key = []; if (x === 0n) key.push(0); while (x > 0n) { key.push(Number(x & 0xffffffffn)); x >>= 32n; } this.initByArray(key); this.gaussNext = null; }
    genrand() {
      const mt = this.mt;
      if (this.i >= 624) {
        let kk = 0, y;
        for (; kk < 624 - 397; kk++) { y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7fffffff); mt[kk] = mt[kk + 397] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0); }
        for (; kk < 623; kk++) { y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7fffffff); mt[kk] = mt[kk + (397 - 624)] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0); }
        y = (mt[623] & 0x80000000) | (mt[0] & 0x7fffffff); mt[623] = mt[396] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0); this.i = 0;
      }
      let y = mt[this.i++]; y ^= y >>> 11; y ^= (y << 7) & 0x9d2c5680; y ^= (y << 15) & 0xefc60000; y ^= y >>> 18; return y >>> 0;
    }
    random() { const a = this.genrand() >>> 5, b = this.genrand() >>> 6; return (a * 67108864 + b) / 9007199254740992; }
    randbelow(n) { if (n <= 0) throw new PyError("ValueError", "empty range for randrange()"); const k = n.toString(2).length; let r = this.genrand() >>> (32 - k); while (r >= n) r = this.genrand() >>> (32 - k); return r; }
    gauss(mu, sigma) {
      let z = this.gaussNext; this.gaussNext = null;
      if (z === null) { const x2pi = this.random() * 2 * Math.PI, g2rad = Math.sqrt(-2 * Math.log(1 - this.random())); z = Math.cos(x2pi) * g2rad; this.gaussNext = Math.sin(x2pi) * g2rad; }
      return mu + z * sigma;
    }
  }

  // ============================================================ tokenizador
  const KEYWORDS = new Set(["False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield"]);
  const UNSUPPORTED = { class: "las clases", try: "try/except", except: "try/except", finally: "try/except", with: "with", raise: "raise", assert: "assert", del: "del", yield: "yield", async: "async", await: "await", nonlocal: "nonlocal" };
  const OPS = ["**=", "//=", "==", "!=", "<=", ">=", "+=", "-=", "*=", "/=", "%=", "**", "//", "->", "+", "-", "*", "/", "%", "<", ">", "=", "(", ")", "[", "]", "{", "}", ",", ":", ".", ";", "@", "&", "|", "^", "~"];
  const ESC = { n: "\n", t: "\t", r: "\r", "\\": "\\", "'": "'", '"': '"', "0": "\0", a: "\x07", b: "\b", f: "\f", v: "\v" };

  function tokenize(src) {
    const toks = [], lines = src.replace(/\r\n?/g, "\n").split("\n"), indents = [0], brackets = [];
    let li = 0, i = 0, line = 0, text = lines, atLineStart = true;
    const push = (t, v, l, extra) => toks.push(Object.assign({ t, v, line: l }, extra || {}));
    const full = lines.join("\n");
    let pos = 0, curLine = 1, bol = true;
    const n = full.length;
    function err(type, msg, l) { throw new PyError(type, msg, l == null ? curLine : l); }
    while (pos < n) {
      if (bol && brackets.length === 0) {
        let col = 0, p = pos;
        while (p < n && (full[p] === " " || full[p] === "\t" || full[p] === "\f")) { col += full[p] === "\t" ? 4 : 1; p++; }
        if (p >= n || full[p] === "\n" || full[p] === "#") { while (p < n && full[p] !== "\n") p++; pos = p + 1; curLine++; continue; }
        if (col > indents[indents.length - 1]) { indents.push(col); push("INDENT", null, curLine); }
        else { while (col < indents[indents.length - 1]) { indents.pop(); push("DEDENT", null, curLine); } if (col !== indents[indents.length - 1]) err("IndentationError", "unindent does not match any outer indentation level"); }
        pos = p; bol = false;
      }
      const ch = full[pos];
      if (ch === "\n") { if (brackets.length === 0) { if (toks.length && toks[toks.length - 1].t !== "NEWLINE") push("NEWLINE", null, curLine); bol = true; } pos++; curLine++; continue; }
      if (ch === " " || ch === "\t" || ch === "\f") { pos++; continue; }
      if (ch === "#") { while (pos < n && full[pos] !== "\n") pos++; continue; }
      if (ch === "\\" && full[pos + 1] === "\n") { pos += 2; curLine++; continue; }
      // cadenas
      const sm = full.slice(pos, pos + 4).match(/^([rRfFbBuU]{0,2})('''|"""|'|")/);
      if (sm && (sm[1] === "" || /^[rRfFbBuU]+$/.test(sm[1]))) {
        const prefix = sm[1].toLowerCase(), q = sm[2], raw = prefix.includes("r"), isF = prefix.includes("f"), startLine = curLine;
        let p = pos + sm[0].length, body = "";
        for (;;) {
          if (p >= n) err("SyntaxError", "unterminated string literal (detected at line " + curLine + ")", startLine);
          if (full.startsWith(q, p)) { p += q.length; break; }
          const c = full[p];
          if (c === "\n") { if (q.length === 1) err("SyntaxError", "unterminated string literal (detected at line " + curLine + ")", startLine); curLine++; }
          if (c === "\\" && !raw) {
            const d = full[p + 1];
            if (d === "\n") { p += 2; curLine++; continue; }
            if (d === "x") { body += String.fromCharCode(parseInt(full.substr(p + 2, 2), 16)); p += 4; continue; }
            if (d === "u") { body += String.fromCharCode(parseInt(full.substr(p + 2, 4), 16)); p += 6; continue; }
            body += d in ESC ? ESC[d] : "\\" + d; p += 2; continue;
          }
          if (c === "\\" && raw) { body += c + (full[p + 1] || ""); p += 2; continue; }
          body += c; p++;
        }
        if (isF) push("FSTR", parseFString(body, startLine, raw), startLine); else push("STR", body, startLine);
        pos = p; continue;
      }
      // números
      const hm = full.slice(pos, pos + 70).match(/^0([xX][0-9a-fA-F_]+|[bB][01_]+|[oO][0-7_]+)/);
      if (hm) { push("NUM", BigInt(hm[0].replace(/_/g, "")), curLine); pos += hm[0].length; continue; }
      const nm = full.slice(pos, pos + 64).match(/^(?:\d[\d_]*(?:\.[\d_]*)?(?:[eE][+-]?\d+)?|\.\d[\d_]*(?:[eE][+-]?\d+)?)/);
      if (nm) {
        const s = nm[0].replace(/_/g, ""), after = full[pos + nm[0].length];
        if (after && /[A-Za-z_]/.test(after)) err("SyntaxError", "invalid decimal literal");
        push("NUM", /[.eE]/.test(s) ? parseFloat(s) : BigInt(s), curLine); pos += nm[0].length; continue;
      }
      const wm = full.slice(pos, pos + 128).match(/^[A-Za-z_À-￿][\wÀ-￿]*/);
      if (wm) { push(KEYWORDS.has(wm[0]) ? "KW" : "NAME", wm[0], curLine); pos += wm[0].length; continue; }
      const op = OPS.find((o) => full.startsWith(o, pos));
      if (op) {
        if ("([{".includes(op)) brackets.push([op, curLine]);
        else if (")]}".includes(op)) { const top = brackets.pop(); if (!top || "([{".indexOf(top[0]) !== ")]}".indexOf(op)) err("SyntaxError", top ? "closing parenthesis '" + op + "' does not match opening parenthesis '" + top[0] + "'" : "unmatched '" + op + "'"); }
        push("OP", op, curLine); pos += op.length; continue;
      }
      err("SyntaxError", "invalid syntax");
    }
    if (brackets.length) { const b = brackets[brackets.length - 1]; throw new PyError("SyntaxError", "'" + b[0] + "' was never closed", b[1]); }
    if (toks.length && toks[toks.length - 1].t !== "NEWLINE") push("NEWLINE", null, curLine);
    while (indents.length > 1) { indents.pop(); push("DEDENT", null, curLine); }
    push("EOF", null, curLine);
    return toks;
  }
  function parseFString(body, line, raw) {
    const parts = []; let lit = "", i = 0;
    while (i < body.length) {
      const c = body[i];
      if (c === "{" && body[i + 1] === "{") { lit += "{"; i += 2; continue; }
      if (c === "}" && body[i + 1] === "}") { lit += "}"; i += 2; continue; }
      if (c === "}") throw new PyError("SyntaxError", "f-string: single '}' is not allowed", line);
      if (c === "{") {
        if (lit) { parts.push(lit); lit = ""; }
        let depth = 0, j = i + 1, q = null;
        for (; j < body.length; j++) {
          const d = body[j];
          if (q) { if (d === q) q = null; continue; }
          if (d === "'" || d === '"') { q = d; continue; }
          if ("([{".includes(d)) depth++; else if (")]}".includes(d)) { if (depth === 0 && d === "}") break; depth--; }
        }
        if (j >= body.length) throw new PyError("SyntaxError", "f-string: expecting '}'", line);
        let inner = body.slice(i + 1, j), spec = null, conv = null, d2 = 0, q2 = null, cut = -1;
        for (let k = 0; k < inner.length; k++) {
          const d = inner[k];
          if (q2) { if (d === q2) q2 = null; continue; }
          if (d === "'" || d === '"') { q2 = d; continue; }
          if ("([{".includes(d)) d2++; else if (")]}".includes(d)) d2--;
          else if (d2 === 0 && d === ":") { cut = k; break; }
          else if (d2 === 0 && d === "!" && inner[k + 1] !== "=") { cut = k; break; }
        }
        let exprSrc = inner, debug = null;
        if (cut >= 0) { exprSrc = inner.slice(0, cut); const rest = inner.slice(cut); if (rest[0] === "!") { conv = rest[1]; spec = rest.slice(2).startsWith(":") ? rest.slice(3) : null; } else spec = rest.slice(1); }
        const dm = exprSrc.match(/^([\s\S]*[^=!<>\s])\s*=\s*$/);
        if (dm) { debug = exprSrc; exprSrc = dm[1]; if (conv === null && spec === null) conv = "r"; }
        if (debug !== null) parts.push(debug);
        parts.push({ expr: exprSrc.trim(), spec, conv, line }); i = j + 1; continue;
      }
      lit += c; i++;
    }
    if (lit) parts.push(lit);
    return parts;
  }

  // ============================================================ analizador sintáctico
  class Parser {
    constructor(src) { this.toks = tokenize(src); this.p = 0; }
    get tok() { return this.toks[this.p]; }
    peekIs(t, v) { const k = this.tok; return k.t === t && (v === undefined || k.v === v); }
    isOp(v) { return this.tok.t === "OP" && this.tok.v === v; }
    isKw(v) { return this.tok.t === "KW" && this.tok.v === v; }
    err(msg, tok) { const k = tok || this.tok; throw new PyError("SyntaxError", msg || "invalid syntax", k.line); }
    expectOp(v) { if (!this.isOp(v)) this.err(v === ":" ? "expected ':'" : "invalid syntax"); return this.toks[this.p++]; }
    expectKw(v) { if (!this.isKw(v)) this.err(); return this.toks[this.p++]; }
    acceptOp(v) { if (this.isOp(v)) { this.p++; return true; } return false; }
    acceptKw(v) { if (this.isKw(v)) { this.p++; return true; } return false; }
    program() { const body = []; while (this.tok.t !== "EOF") { if (this.tok.t === "NEWLINE") { this.p++; continue; } if (this.tok.t === "INDENT") throw new PyError("IndentationError", "unexpected indent", this.tok.line); body.push(...this.statement()); } return body; }
    block(kind, headLine) {
      this.expectOp(":");
      if (this.tok.t === "NEWLINE") {
        this.p++;
        if (this.tok.t !== "INDENT") throw new PyError("IndentationError", "expected an indented block after " + kind + " on line " + headLine, this.tok.line);
        this.p++; const body = [];
        while (this.tok.t !== "DEDENT" && this.tok.t !== "EOF") { if (this.tok.t === "NEWLINE") { this.p++; continue; } body.push(...this.statement()); }
        if (this.tok.t === "DEDENT") this.p++;
        return body;
      }
      return this.simpleLine();
    }
    statement() {
      const k = this.tok;
      if (k.t === "KW") {
        if (UNSUPPORTED[k.v]) throw new PyError("NotSupported", "«" + UNSUPPORTED[k.v] + "» todavía no está disponible en este laboratorio", k.line);
        if (k.v === "if") return [this.ifStmt()];
        if (k.v === "while") { this.p++; const test = this.expr(); const body = this.block("'while' statement", k.line); return [{ type: "While", test, body, line: k.line }]; }
        if (k.v === "for") { this.p++; const target = this.targetList(); this.expectKw("in"); const iter = this.testList(); const body = this.block("'for' statement", k.line); return [{ type: "For", target, iter, body, line: k.line }]; }
        if (k.v === "def") return [this.funcDef()];
      }
      return this.simpleLine();
    }
    simpleLine() {
      const out = [this.small()];
      while (this.acceptOp(";")) { if (this.tok.t === "NEWLINE") break; out.push(this.small()); }
      if (this.tok.t !== "NEWLINE" && this.tok.t !== "EOF") this.err();
      if (this.tok.t === "NEWLINE") this.p++;
      return out;
    }
    ifStmt() {
      const k = this.tok; this.p++; const test = this.expr(); const body = this.block("'if' statement", k.line); let orelse = [];
      if (this.isKw("elif")) { const e = this.tok; orelse = [this.elifAsIf()]; }
      else if (this.isKw("else")) { const e = this.tok; this.p++; orelse = this.block("'else' statement", e.line); }
      return { type: "If", test, body, orelse, line: k.line };
    }
    elifAsIf() { const k = this.tok; this.p++; const test = this.expr(); const body = this.block("'elif' statement", k.line); let orelse = []; if (this.isKw("elif")) orelse = [this.elifAsIf()]; else if (this.isKw("else")) { const e = this.tok; this.p++; orelse = this.block("'else' statement", e.line); } return { type: "If", test, body, orelse, line: k.line }; }
    funcDef() {
      const k = this.tok; this.p++; const nameTok = this.tok; if (nameTok.t !== "NAME") this.err(); this.p++;
      this.expectOp("("); const params = [], defaults = []; let seenDefault = false;
      while (!this.isOp(")")) {
        if (this.isOp("*") || this.isOp("**")) throw new PyError("NotSupported", "los parámetros *args y **kwargs todavía no están disponibles", this.tok.line);
        const pt = this.tok; if (pt.t !== "NAME") this.err(); this.p++; params.push(pt.v);
        if (this.acceptOp("=")) { seenDefault = true; defaults.push(this.expr()); } else { if (seenDefault) this.err("non-default argument follows default argument", pt); defaults.push(undefined); }
        if (!this.acceptOp(",")) break;
      }
      this.expectOp(")"); const body = this.block("function definition", k.line);
      return { type: "Def", name: nameTok.v, params, defaults, body, line: k.line };
    }
    small() {
      const k = this.tok;
      if (k.t === "KW") {
        if (UNSUPPORTED[k.v]) throw new PyError("NotSupported", "«" + UNSUPPORTED[k.v] + "» todavía no está disponible en este laboratorio", k.line);
        switch (k.v) {
          case "pass": this.p++; return { type: "Pass", line: k.line };
          case "break": this.p++; return { type: "Break", line: k.line };
          case "continue": this.p++; return { type: "Continue", line: k.line };
          case "return": { this.p++; let v = null; if (this.tok.t !== "NEWLINE" && !this.isOp(";") && this.tok.t !== "EOF") v = this.testList(); return { type: "Return", value: v, line: k.line }; }
          case "global": { this.p++; const names = []; do { names.push(this.tok.v); this.p++; } while (this.acceptOp(",")); return { type: "Global", names, line: k.line }; }
          case "import": { this.p++; const mods = []; do { let name = this.tok.v; this.p++; while (this.acceptOp(".")) { name += "." + this.tok.v; this.p++; } let as = name; if (this.acceptKw("as")) { as = this.tok.v; this.p++; } mods.push([name, as]); } while (this.acceptOp(",")); return { type: "Import", mods, line: k.line }; }
          case "from": { this.p++; let name = this.tok.v; this.p++; while (this.acceptOp(".")) { name += "." + this.tok.v; this.p++; } this.expectKw("import"); const names = []; if (this.isOp("*")) { this.p++; names.push(["*", "*"]); } else { const par = this.acceptOp("("); do { const nm = this.tok.v; this.p++; let as = nm; if (this.acceptKw("as")) { as = this.tok.v; this.p++; } names.push([nm, as]); } while (this.acceptOp(",") && !this.isOp(")")); if (par) this.expectOp(")"); } return { type: "FromImport", mod: name, names, line: k.line }; }
        }
      }
      const first = this.testList();
      if (this.tok.t === "OP" && ["+=", "-=", "*=", "/=", "//=", "%=", "**="].includes(this.tok.v)) {
        const op = this.tok.v.slice(0, -1); this.p++; const value = this.testList(); this.checkTarget(first, k); return { type: "AugAssign", op, target: first, value, line: k.line };
      }
      if (this.isOp("=")) {
        const targets = [first];
        while (this.acceptOp("=")) targets.push(this.testList());
        const value = targets.pop(); targets.forEach((t) => this.checkTarget(t, k)); return { type: "Assign", targets, value, line: k.line };
      }
      return { type: "Expr", value: first, line: k.line };
    }
    checkTarget(t, k) {
      if (t.type === "Name" || t.type === "Subscript") return;
      if (t.type === "Tuple" || t.type === "List") { t.items.forEach((x) => this.checkTarget(x, k)); return; }
      if (t.type === "Attr") throw new PyError("NotSupported", "asignar atributos (objeto.campo = …) todavía no está disponible", k.line);
      throw new PyError("SyntaxError", "cannot assign to expression here. Maybe you meant '==' instead of '='?", k.line);
    }
    targetList() { const first = this.targetAtom(); if (!this.isOp(",")) return first; const items = [first]; while (this.acceptOp(",")) { if (this.isKw("in")) break; items.push(this.targetAtom()); } return { type: "Tuple", items, line: first.line }; }
    targetAtom() { if (this.isOp("(")) { this.p++; const t = this.targetList(); this.expectOp(")"); return t; } const k = this.tok; if (k.t !== "NAME") this.err(); this.p++; return { type: "Name", name: k.v, line: k.line }; }
    testList() { const k = this.tok, first = this.expr(); if (!this.isOp(",")) return first; const items = [first]; while (this.acceptOp(",")) { if (this.tok.t === "NEWLINE" || this.isOp("=") || this.isOp(")") || this.isOp("]") || this.tok.t === "EOF" || this.isOp(":") || this.isOp(";")) break; items.push(this.expr()); } return { type: "Tuple", items, line: k.line }; }
    expr() {
      const k = this.tok;
      if (this.isKw("lambda")) return this.lambda();
      const body = this.orTest();
      if (this.isKw("if")) { this.p++; const test = this.orTest(); this.expectKw("else"); const orelse = this.expr(); return { type: "IfExp", test, body, orelse, line: k.line }; }
      return body;
    }
    lambda() {
      const k = this.tok; this.p++; const params = [], defaults = [];
      while (!this.isOp(":")) { const pt = this.tok; if (pt.t !== "NAME") this.err(); this.p++; params.push(pt.v); if (this.acceptOp("=")) defaults.push(this.expr()); else defaults.push(undefined); if (!this.acceptOp(",")) break; }
      this.expectOp(":"); const body = this.expr(); return { type: "Lambda", params, defaults, body, line: k.line };
    }
    orTest() { let l = this.andTest(); while (this.isKw("or")) { const k = this.tok; this.p++; l = { type: "BoolOp", op: "or", left: l, right: this.andTest(), line: k.line }; } return l; }
    andTest() { let l = this.notTest(); while (this.isKw("and")) { const k = this.tok; this.p++; l = { type: "BoolOp", op: "and", left: l, right: this.notTest(), line: k.line }; } return l; }
    notTest() { if (this.isKw("not")) { const k = this.tok; this.p++; return { type: "Not", operand: this.notTest(), line: k.line }; } return this.comparison(); }
    comparison() {
      const k = this.tok, left = this.arithExpr(), ops = [], rights = [];
      for (;;) {
        const t = this.tok; let op = null;
        if (t.t === "OP" && ["<", ">", "==", "!=", "<=", ">="].includes(t.v)) { op = t.v; this.p++; }
        else if (t.t === "KW" && t.v === "in") { op = "in"; this.p++; }
        else if (t.t === "KW" && t.v === "not" && this.toks[this.p + 1].t === "KW" && this.toks[this.p + 1].v === "in") { op = "not in"; this.p += 2; }
        else if (t.t === "KW" && t.v === "is") { this.p++; if (this.acceptKw("not")) op = "is not"; else op = "is"; }
        if (!op) break; ops.push(op); rights.push(this.arithExpr());
      }
      return ops.length ? { type: "Compare", left, ops, rights, line: k.line } : left;
    }
    arithExpr() { let l = this.term(); while (this.tok.t === "OP" && (this.tok.v === "+" || this.tok.v === "-")) { const k = this.tok; this.p++; l = { type: "BinOp", op: k.v, left: l, right: this.term(), line: k.line }; } return l; }
    term() { let l = this.factor(); while (this.tok.t === "OP" && ["*", "/", "//", "%"].includes(this.tok.v)) { const k = this.tok; this.p++; l = { type: "BinOp", op: k.v, left: l, right: this.factor(), line: k.line }; } return l; }
    factor() { if (this.tok.t === "OP" && (this.tok.v === "-" || this.tok.v === "+")) { const k = this.tok; this.p++; return { type: "Unary", op: k.v, operand: this.factor(), line: k.line }; } return this.power(); }
    power() { const base = this.postfix(); if (this.isOp("**")) { const k = this.tok; this.p++; return { type: "BinOp", op: "**", left: base, right: this.factor(), line: k.line }; } return base; }
    postfix() {
      let e = this.atom();
      for (;;) {
        const k = this.tok;
        if (this.isOp("(")) { this.p++; const { args, kwargs } = this.callArgs(); e = { type: "Call", func: e, args, kwargs, line: k.line }; }
        else if (this.isOp("[")) { this.p++; const idx = this.subscript(); this.expectOp("]"); e = { type: "Subscript", value: e, index: idx, line: k.line }; }
        else if (this.isOp(".")) { this.p++; const nt = this.tok; if (nt.t !== "NAME" && nt.t !== "KW") this.err(); this.p++; e = { type: "Attr", value: e, name: nt.v, line: k.line }; }
        else break;
      }
      return e;
    }
    callArgs() {
      const args = [], kwargs = [];
      while (!this.isOp(")")) {
        if (this.isOp("*") || this.isOp("**")) throw new PyError("NotSupported", "pasar *lista o **dic a una función todavía no está disponible", this.tok.line);
        if (this.tok.t === "NAME" && this.toks[this.p + 1].t === "OP" && this.toks[this.p + 1].v === "=") { const name = this.tok.v; this.p += 2; kwargs.push([name, this.expr()]); }
        else {
          const k = this.tok, e = this.expr();
          if (this.isKw("for")) args.push({ type: "Comp", kind: "gen", elt: e, gens: this.compFor(), line: k.line }); else args.push(e);
        }
        if (!this.acceptOp(",")) break;
      }
      this.expectOp(")"); return { args, kwargs };
    }
    subscript() {
      const k = this.tok;
      const part = () => { if (this.isOp(":") || this.isOp("]")) return null; return this.expr(); };
      let lo = part();
      if (this.isOp(":")) { this.p++; const hi = part(); let step = null; if (this.acceptOp(":")) step = part(); return { type: "Slice", lo, hi, step, line: k.line }; }
      if (this.isOp(",")) { const items = [lo]; while (this.acceptOp(",")) { if (this.isOp("]")) break; items.push(this.expr()); } return { type: "Tuple", items, line: k.line }; }
      return lo;
    }
    compFor() {
      const gens = [];
      while (this.isKw("for")) {
        this.p++; const target = this.targetList(); this.expectKw("in"); const iter = this.orTest(); const ifs = [];
        while (this.isKw("if")) { this.p++; ifs.push(this.orTestNoCond()); }
        gens.push({ target, iter, ifs });
      }
      return gens;
    }
    orTestNoCond() { return this.orTest(); }
    atom() {
      const k = this.tok;
      switch (k.t) {
        case "NUM": this.p++; return { type: "Const", value: k.v, line: k.line };
        case "STR": case "FSTR": return this.strings();
        case "NAME": this.p++; return { type: "Name", name: k.v, line: k.line };
        case "KW":
          if (k.v === "True") { this.p++; return { type: "Const", value: true, line: k.line }; }
          if (k.v === "False") { this.p++; return { type: "Const", value: false, line: k.line }; }
          if (k.v === "None") { this.p++; return { type: "Const", value: null, line: k.line }; }
          if (k.v === "lambda") return this.lambda();
          if (UNSUPPORTED[k.v]) throw new PyError("NotSupported", "«" + UNSUPPORTED[k.v] + "» todavía no está disponible en este laboratorio", k.line);
          this.err(); break;
        case "OP":
          if (k.v === "(") {
            this.p++;
            if (this.acceptOp(")")) return { type: "Tuple", items: [], line: k.line };
            const first = this.expr();
            if (this.isKw("for")) { const gens = this.compFor(); this.expectOp(")"); return { type: "Comp", kind: "gen", elt: first, gens, line: k.line }; }
            if (this.isOp(",")) { const items = [first]; while (this.acceptOp(",")) { if (this.isOp(")")) break; items.push(this.expr()); } this.expectOp(")"); return { type: "Tuple", items, line: k.line }; }
            this.expectOp(")"); return first;
          }
          if (k.v === "[") {
            this.p++;
            if (this.acceptOp("]")) return { type: "List", items: [], line: k.line };
            const first = this.expr();
            if (this.isKw("for")) { const gens = this.compFor(); this.expectOp("]"); return { type: "Comp", kind: "list", elt: first, gens, line: k.line }; }
            const items = [first]; while (this.acceptOp(",")) { if (this.isOp("]")) break; items.push(this.expr()); } this.expectOp("]"); return { type: "List", items, line: k.line };
          }
          if (k.v === "{") {
            this.p++;
            if (this.acceptOp("}")) return { type: "Dict", pairs: [], line: k.line };
            const first = this.expr();
            if (this.isOp(":")) {
              this.p++; const v = this.expr();
              if (this.isKw("for")) { const gens = this.compFor(); this.expectOp("}"); return { type: "Comp", kind: "dict", elt: first, val: v, gens, line: k.line }; }
              const pairs = [[first, v]]; while (this.acceptOp(",")) { if (this.isOp("}")) break; const kk = this.expr(); this.expectOp(":"); pairs.push([kk, this.expr()]); } this.expectOp("}"); return { type: "Dict", pairs, line: k.line };
            }
            throw new PyError("NotSupported", "los conjuntos {a, b} todavía no están disponibles; usa una lista", k.line);
          }
      }
      this.err(); return null;
    }
    strings() {
      const k = this.tok, parts = [];
      while (this.tok.t === "STR" || this.tok.t === "FSTR") {
        const t = this.tok; this.p++;
        if (t.t === "STR") parts.push(t.v);
        else for (const q of t.v) { if (typeof q === "string") parts.push(q); else { const sub = new Parser(q.expr); const e = sub.expr(); if (sub.tok.t !== "NEWLINE" && sub.tok.t !== "EOF") throw new PyError("SyntaxError", "f-string: invalid syntax", t.line); e.line = t.line; parts.push({ expr: e, spec: q.spec, conv: q.conv }); } }
      }
      if (parts.every((x) => typeof x === "string")) return { type: "Const", value: parts.join(""), line: k.line };
      return { type: "FString", parts, line: k.line };
    }
  }
  function parseProgram(src) { return new Parser(src).program(); }
  function parseExpression(src) { const p = new Parser(src); while (p.tok.t === "NEWLINE") p.p++; const e = p.testList(); while (p.tok.t === "NEWLINE") p.p++; if (p.tok.t !== "EOF") p.err(); return e; }

  function exactTie(ax, nd) {   // ¿ax·10^nd es exactamente k + 0.5? (comprobación exacta con BigInt)
    const dv = new DataView(new ArrayBuffer(8)); dv.setFloat64(0, ax); const bits = dv.getBigUint64(0);
    let e = Number((bits >> 52n) & 0x7ffn), m = bits & 0xfffffffffffffn; if (e === 0) e = 1; else m |= 1n << 52n; e -= 1075;
    if (e >= 0) return false; const numr = m * 10n ** BigInt(nd) * 2n, d = 1n << BigInt(-e);
    return numr % d === 0n && (numr / d) % 2n === 1n;
  }

  // ============================================================ intérprete
  class PyInterp {
    constructor(opts) {
      this.out = (opts && opts.stdout) || (() => {}); this.plot = (opts && opts.plot) || (() => {});
      this.rng = new MT(); this.steps = 0; this.deadline = Infinity; this.depth = 0; this.line = 0;
      this.builtins = this.makeBuiltins(); this.modules = this.makeModules(); this.reset();
    }
    reset() { this.globals = new Env(null); }
    tick() { if ((++this.steps & 4095) === 0 && Date.now() > this.deadline) throw new PyError("TimeoutError", "__TIMEOUT__", this.line); }

    // ---- API pública
    run(code, ms) { this.deadline = Date.now() + (ms || 6000); const prog = parseProgram(code); try { this.execBlock(prog, this.globals); } finally { this.deadline = Infinity; } }
    repl(code, ms) {
      this.deadline = Date.now() + (ms || 6000);
      try {
        let ex = null; try { ex = parseExpression(code); } catch (e) { ex = null; }
        if (ex) { const v = this.ev(ex, this.globals); if (v !== null) this.out(pyrepr(v) + "\n"); return; }
        this.execBlock(parseProgram(code), this.globals);
      } finally { this.deadline = Infinity; }
    }
    check(src, ms) {
      this.deadline = Date.now() + (ms || 8000);
      try {
        const U = new PyDict(); for (const [k, v] of this.globals.vars) U.set(k, v);
        const env = new Env(this.globals); env.vars.set("U", U);
        this.execBlock(parseProgram(src), env);
        const probar = env.vars.get("probar"); if (!probar) throw new PyError("NameError", "name 'probar' is not defined");
        const res = this.call(probar, [], {}, 0);
        return res.map((t) => [t.items ? t.items[0] : t[0], truthy(t[1]), t[2]]);
      } catch (e) {
        if (e instanceof PyError) return [["No se pudo comprobar el reto", false, e.pyType + ": " + e.pyMsg]];
        throw e;
      } finally { this.deadline = Infinity; }
    }

    // ---- ejecución de sentencias
    execBlock(stmts, env) { for (const s of stmts) { const r = this.exec(s, env); if (r) return r; } return null; }
    exec(s, env) {
      this.line = s.line; this.tick();
      switch (s.type) {
        case "Expr": this.ev(s.value, env); return null;
        case "Assign": { const v = this.ev(s.value, env); for (const t of s.targets) this.assign(t, v, env); return null; }
        case "AugAssign": { const cur = this.ev(s.target, env), r = this.binop(s.op, cur, this.ev(s.value, env)); this.assign(s.target, Array.isArray(cur) && s.op === "+" ? (cur.push(...r.slice(cur.length)), cur) : r, env); return null; }
        case "If": { for (let st = s; ;) { if (truthy(this.ev(st.test, env))) return this.execBlock(st.body, env); if (st.orelse.length === 1 && st.orelse[0].type === "If" && st.orelse[0] !== st) { st = st.orelse[0]; this.line = st.line; continue; } return this.execBlock(st.orelse, env); } }
        case "While": { while (truthy(this.ev(s.test, env))) { this.tick(); const r = this.execBlock(s.body, env); if (r) { if (r === BREAK) break; if (r === CONTINUE) continue; return r; } } return null; }
        case "For": {
          const it = this.iterate(this.ev(s.iter, env));
          for (const v of it) { this.tick(); this.assign(s.target, v, env); const r = this.execBlock(s.body, env); if (r) { if (r === BREAK) break; if (r === CONTINUE) continue; return r; } }
          return null;
        }
        case "Def": { const defaults = s.defaults.map((d) => (d ? this.ev(d, env) : undefined)); env.set(s.name, new PyFunc(s.name, s.params, s.body, env, defaults, false)); return null; }
        case "Return": return new Signal("return", s.value ? this.ev(s.value, env) : null);
        case "Break": return BREAK; case "Continue": return CONTINUE; case "Pass": return null;
        case "Global": { if (!env.globals) env.globals = new Set(); s.names.forEach((n) => env.globals.add(n)); return null; }
        case "Import": for (const [name, as] of s.mods) { const m = this.modules[name]; if (!m) throw new PyError("ModuleNotFoundError", "No module named '" + name + "'", s.line); env.set(as, m); } return null;
        case "FromImport": { const m = this.modules[s.mod]; if (!m) throw new PyError("ModuleNotFoundError", "No module named '" + s.mod + "'", s.line); for (const [n, as] of s.names) { if (n === "*") { for (const k in m.attrs) env.set(k, m.attrs[k]); continue; } if (!(n in m.attrs)) throw new PyError("ImportError", "cannot import name '" + n + "' from '" + s.mod + "'", s.line); env.set(as, m.attrs[n]); } return null; }
      }
      throw new PyError("NotSupported", "instrucción no soportada: " + s.type, s.line);
    }
    assign(t, v, env) {
      switch (t.type) {
        case "Name": env.set(t.name, v); return;
        case "Tuple": case "List": { const items = this.iterate(v), arr = [...items]; if (arr.length !== t.items.length) throw new PyError(arr.length < t.items.length ? "ValueError" : "ValueError", arr.length < t.items.length ? "not enough values to unpack (expected " + t.items.length + ", got " + arr.length + ")" : "too many values to unpack (expected " + t.items.length + ")", t.line); t.items.forEach((x, i) => this.assign(x, arr[i], env)); return; }
        case "Subscript": {
          const obj = this.ev(t.value, env), idx = this.ev(t.index, env);
          if (Array.isArray(obj)) { if (t.index.type === "Slice") throw new PyError("NotSupported", "asignar a un trozo de lista todavía no está disponible", t.line); const i = this.listIndex(obj, idx); obj[i] = v; return; }
          if (obj instanceof PyDict) { obj.set(idx, v); return; }
          throw new PyError("TypeError", "'" + tname(obj) + "' object does not support item assignment", t.line);
        }
      }
      throw new PyError("SyntaxError", "cannot assign to this expression", t.line);
    }
    listIndex(arr, idx) { if (typeof idx === "boolean") idx = idx ? 1n : 0n; if (typeof idx !== "bigint") throw new PyError("TypeError", "list indices must be integers or slices, not " + tname(idx)); let i = Number(idx); if (i < 0) i += arr.length; if (i < 0 || i >= arr.length) throw new PyError("IndexError", "list index out of range"); return i; }
    iterate(v) {
      if (Array.isArray(v)) return v.slice(); if (typeof v === "string") return [...v]; if (v instanceof PyTuple) return v.items; if (v instanceof PyIter) return v.items;
      if (v instanceof PyDict) return v.keys();
      if (v instanceof PyRange) { const self = this; return (function* () { const s = v.step; for (let x = v.start; s > 0n ? x < v.stop : x > v.stop; x += s) yield x; })(); }
      throw new PyError("TypeError", "'" + tname(v) + "' object is not iterable");
    }

    // ---- expresiones
    binop(op, a, b) { return arith(op, a, b); }
    ev(n, env) {
      switch (n.type) {
        case "Const": return n.value;
        case "Name": { const v = env.lookup(n.name); if (v !== undefined) return v; if (n.name in this.builtins) return this.builtins[n.name]; this.line = n.line; throw new PyError("NameError", "name '" + n.name + "' is not defined", n.line); }
        case "BinOp": { const a = this.ev(n.left, env), b = this.ev(n.right, env); this.line = n.line; return arith(n.op, a, b); }
        case "Unary": { const v = this.ev(n.operand, env); this.line = n.line; if (!isNumeric(v)) throw new PyError("TypeError", "bad operand type for unary " + n.op + ": '" + tname(v) + "'", n.line); const x = num(v); return n.op === "-" ? (typeof x === "bigint" ? -x : -x) : x; }
        case "BoolOp": { const l = this.ev(n.left, env); if (n.op === "or") return truthy(l) ? l : this.ev(n.right, env); return truthy(l) ? this.ev(n.right, env) : l; }
        case "Not": return !truthy(this.ev(n.operand, env));
        case "IfExp": return truthy(this.ev(n.test, env)) ? this.ev(n.body, env) : this.ev(n.orelse, env);
        case "Compare": {
          let l = this.ev(n.left, env);
          for (let i = 0; i < n.ops.length; i++) {
            const r = this.ev(n.rights[i], env), op = n.ops[i]; this.line = n.line; let ok;
            if (op === "==") ok = pyEq(l, r); else if (op === "!=") ok = !pyEq(l, r);
            else if (op === "in" || op === "not in") { ok = this.contains(r, l); if (op === "not in") ok = !ok; }
            else if (op === "is") ok = l === r || (isNumeric(l) && isNumeric(r) && false); else if (op === "is not") ok = l !== r;
            else ok = cmp(l, r, op);
            if (!ok) return false; l = r;
          }
          return true;
        }
        case "List": return n.items.map((x) => this.ev(x, env));
        case "Tuple": return new PyTuple(n.items.map((x) => this.ev(x, env)));
        case "Dict": { const d = new PyDict(); for (const [k, v] of n.pairs) d.set(this.ev(k, env), this.ev(v, env)); return d; }
        case "FString": { let o = ""; for (const p of n.parts) { if (typeof p === "string") o += p; else { let v = this.ev(p.expr, env); if (p.conv === "r") v = pyrepr(v); else if (p.conv === "s") v = pystr(v); o += fmtSpec(v, p.spec || ""); } } return o; }
        case "Lambda": return new PyFunc("<lambda>", n.params, [{ type: "Return", value: n.body, line: n.line }], env, n.defaults.map((d) => (d ? this.ev(d, env) : undefined)), true);
        case "Call": return this.evalCall(n, env);
        case "Attr": { const o = this.ev(n.value, env); this.line = n.line; return this.getAttr(o, n.name, n.line); }
        case "Subscript": { const o = this.ev(n.value, env); this.line = n.line; return this.subscript(o, n.index, env, n.line); }
        case "Comp": return this.comprehension(n, env);
        case "Slice": throw new PyError("SyntaxError", "invalid syntax", n.line);
      }
      throw new PyError("NotSupported", "expresión no soportada: " + n.type, n.line);
    }
    contains(container, item) {
      if (typeof container === "string") { if (typeof item !== "string") throw new PyError("TypeError", "'in <string>' requires string as left operand, not " + tname(item)); return container.includes(item); }
      if (Array.isArray(container)) return container.some((x) => pyEq(x, item));
      if (container instanceof PyTuple) return container.items.some((x) => pyEq(x, item));
      if (container instanceof PyDict) return container.has(item);
      if (container instanceof PyRange) { for (const x of this.iterate(container)) if (pyEq(x, item)) return true; return false; }
      throw new PyError("TypeError", "argument of type '" + tname(container) + "' is not iterable");
    }
    subscript(o, indexNode, env, line) {
      if (indexNode.type === "Slice") {
        const lo = indexNode.lo ? this.ev(indexNode.lo, env) : null, hi = indexNode.hi ? this.ev(indexNode.hi, env) : null, st = indexNode.step ? this.ev(indexNode.step, env) : null;
        const seq = typeof o === "string" ? [...o] : Array.isArray(o) ? o : o instanceof PyTuple ? o.items : null;
        if (!seq) throw new PyError("TypeError", "'" + tname(o) + "' object is not subscriptable", line);
        const step = st === null ? 1 : Number(num(st)); if (step === 0) throw new PyError("ValueError", "slice step cannot be zero", line);
        const len = seq.length, clamp = (v, d, lowDefault) => { if (v === null) return d; let x = Number(num(v)); if (x < 0) x += len; return step > 0 ? Math.max(0, Math.min(len, x)) : Math.max(-1, Math.min(len - 1, x)); };
        const a = clamp(lo, step > 0 ? 0 : len - 1), b = clamp(hi, step > 0 ? len : -1), out = [];
        if (step > 0) for (let i = a; i < b; i += step) out.push(seq[i]); else for (let i = a; i > b; i += step) out.push(seq[i]);
        return typeof o === "string" ? out.join("") : Array.isArray(o) ? out : new PyTuple(out);
      }
      const idx = this.ev(indexNode, env);
      if (o instanceof PyDict) { const v = o.get(idx); if (v === undefined) throw new PyError("KeyError", pyrepr(idx), line); return v; }
      if (typeof o === "string" || Array.isArray(o) || o instanceof PyTuple) {
        const seq = typeof o === "string" ? o : Array.isArray(o) ? o : o.items;
        if (typeof idx !== "bigint" && typeof idx !== "boolean") throw new PyError("TypeError", (typeof o === "string" ? "string" : tname(o)) + " indices must be integers or slices, not " + tname(idx), line);
        let i = Number(num(idx)); const L = seq.length; if (i < 0) i += L;
        if (i < 0 || i >= L) throw new PyError("IndexError", (typeof o === "string" ? "string" : tname(o)) + " index out of range", line);
        return typeof o === "string" ? seq[i] : seq[i];
      }
      if (o instanceof PyRange) { const arr = [...this.iterate(o)]; let i = Number(num(idx)); if (i < 0) i += arr.length; if (i < 0 || i >= arr.length) throw new PyError("IndexError", "range object index out of range", line); return arr[i]; }
      throw new PyError("TypeError", "'" + tname(o) + "' object is not subscriptable", line);
    }
    comprehension(n, env) {
      const out = n.kind === "dict" ? new PyDict() : [];
      const loop = (gi, e) => {
        if (gi === n.gens.length) { if (n.kind === "dict") out.set(this.ev(n.elt, e), this.ev(n.val, e)); else out.push(this.ev(n.elt, e)); return; }
        const g = n.gens[gi];
        for (const v of this.iterate(this.ev(g.iter, e))) { this.tick(); const e2 = new Env(e); this.assign(g.target, v, e2); if (g.ifs.every((c) => truthy(this.ev(c, e2)))) loop(gi + 1, e2); }
      };
      loop(0, new Env(env));
      return n.kind === "gen" ? new PyIter(out, "generator") : out;
    }
    evalCall(n, env) {
      const f = this.ev(n.func, env), args = n.args.map((a) => this.ev(a, env));
      let kw = null; if (n.kwargs.length) { kw = {}; for (const [k, v] of n.kwargs) kw[k] = this.ev(v, env); }
      this.line = n.line; return this.call(f, args, kw, n.line);
    }
    call(f, args, kw, line) {
      this.tick();
      if (f instanceof Builtin) return f.fn(args, kw || {}, this);
      if (f instanceof PyFunc) {
        if (++this.depth > 250) { this.depth--; throw new PyError("RecursionError", "maximum recursion depth exceeded", line); }
        try {
          const env = new Env(f.env), P = f.params;
          if (args.length > P.length) throw new PyError("TypeError", f.name + "() takes " + P.length + " positional argument" + (P.length === 1 ? "" : "s") + " but " + args.length + " " + (args.length === 1 ? "was" : "were") + " given", line);
          for (let i = 0; i < P.length; i++) {
            if (i < args.length) env.vars.set(P[i], args[i]);
            else if (kw && P[i] in kw) env.vars.set(P[i], kw[P[i]]);
            else if (f.defaults[i] !== undefined) env.vars.set(P[i], f.defaults[i]);
            else throw new PyError("TypeError", f.name + "() missing 1 required positional argument: '" + P[i] + "'", line);
          }
          if (kw) for (const k in kw) if (!P.includes(k)) throw new PyError("TypeError", f.name + "() got an unexpected keyword argument '" + k + "'", line);
          const r = this.execBlock(f.body, env); return r && r.kind === "return" ? r.value : null;
        } finally { this.depth--; }
      }
      throw new PyError("TypeError", "'" + tname(f) + "' object is not callable", line);
    }

    // ---- atributos y métodos
    getAttr(o, name, line) {
      if (o instanceof PyModule) { if (name in o.attrs) return o.attrs[name]; throw new PyError("AttributeError", "module '" + o.name + "' has no attribute '" + name + "'", line); }
      const m = this.methodsOf(o); if (m && name in m) { const fn = m[name]; return new Builtin(name, (a, k) => fn(o, a, k, this)); }
      throw new PyError("AttributeError", "'" + tname(o) + "' object has no attribute '" + name + "'", line);
    }
    methodsOf(o) { return typeof o === "string" ? STR_M : Array.isArray(o) ? LIST_M : o instanceof PyDict ? DICT_M : o instanceof PyTuple ? TUPLE_M : isFloat(o) ? FLOAT_M : null; }

    // ---- funciones integradas
    makeBuiltins() {
      const B = {}, def = (name, fn, tn) => { B[name] = new Builtin(name, fn, tn); }, self = this;
      const argc = (name, a, min, max) => { if (a.length < min || a.length > max) throw new PyError("TypeError", name + "() takes " + (min === max ? "exactly " + min : "at most " + max) + " argument" + (max === 1 ? "" : "s") + " (" + a.length + " given)"); };
      def("print", (a, kw, I) => { const sep = kw.sep == null ? " " : kw.sep, end = kw.end == null ? "\n" : kw.end; I.out(a.map(pystr).join(sep) + end); return null; });
      def("len", (a) => { argc("len", a, 1, 1); const v = a[0]; if (typeof v === "string" || Array.isArray(v)) return BigInt(v.length); if (v instanceof PyTuple) return BigInt(v.items.length); if (v instanceof PyDict) return BigInt(v.size); if (v instanceof PyRange) return v.len(); if (v instanceof PyIter) return BigInt(v.items.length); throw new PyError("TypeError", "object of type '" + tname(v) + "' has no len()"); });
      def("range", (a) => { argc("range", a, 1, 3); const g = (x) => { if (typeof x === "boolean") return x ? 1n : 0n; if (typeof x !== "bigint") throw new PyError("TypeError", "'" + tname(x) + "' object cannot be interpreted as an integer"); return x; }; if (a.length === 1) return new PyRange(0n, g(a[0]), 1n); const step = a.length === 3 ? g(a[2]) : 1n; if (step === 0n) throw new PyError("ValueError", "range() arg 3 must not be zero"); return new PyRange(g(a[0]), g(a[1]), step); }, "range");
      def("sum", (a, kw, I) => {
        const arr = [...I.iterate(a[0])]; let r = a.length > 1 ? a[1] : kw.start !== undefined ? kw.start : 0n, i = 0;
        if (typeof r === "bigint") { while (i < arr.length && (typeof arr[i] === "bigint" || typeof arr[i] === "boolean")) { r += num(arr[i]); i++; } if (i < arr.length) { r = arith("+", r, arr[i]); i++; } }
        if (typeof r === "number") {   // suma con compensación de Neumaier, igual que CPython 3.12+
          let f = r, c = 0;
          while (i < arr.length) { const it = arr[i]; let x; if (typeof it === "number") x = it; else if (typeof it === "bigint" || typeof it === "boolean") x = Number(num(it)); else break; const t = f + x; if (Math.abs(f) >= Math.abs(x)) c += (f - t) + x; else c += (x - t) + f; f = t; i++; }
          if (c !== 0 && Number.isFinite(c)) f += c; r = f;
        }
        for (; i < arr.length; i++) r = arith("+", r, arr[i]);
        return r;
      });
      const minmax = (name, better) => def(name, (a, kw, I) => {
        const items = a.length === 1 ? [...I.iterate(a[0])] : a; if (!items.length) { if (kw.default !== undefined) return kw.default; throw new PyError("ValueError", name + "() arg is an empty sequence"); }
        const key = kw.key ? (x) => I.call(kw.key, [x], null) : (x) => x; let best = items[0], bk = key(best);
        for (let i = 1; i < items.length; i++) { const k = key(items[i]); if (better(k, bk)) { best = items[i]; bk = k; } } return best;
      });
      minmax("max", (k, b) => cmp(k, b, ">")); minmax("min", (k, b) => cmp(k, b, "<"));
      def("abs", (a) => { argc("abs", a, 1, 1); const v = num(a[0]); if (typeof v === "bigint") return v < 0n ? -v : v; if (typeof v === "number") return Math.abs(v); throw new PyError("TypeError", "bad operand type for abs(): '" + tname(a[0]) + "'"); });
      def("round", (a) => {
        argc("round", a, 1, 2); const v = num(a[0]); if (!isNumeric(v)) throw new PyError("TypeError", "type " + tname(a[0]) + " doesn't define __round__ method");
        const nd = a.length === 2 && a[1] !== null ? Number(num(a[1])) : null, x = toNumber(v);
        const half = (z) => { const f = Math.floor(z), d = z - f; return d < 0.5 ? f : d > 0.5 ? f + 1 : f % 2 === 0 ? f : f + 1; };
        if (typeof v === "bigint") { if (nd === null || nd >= 0) return v; const sc = 10n ** BigInt(-nd); const q = floorDivBig(v, sc), r = v - q * sc, twice = r * 2n; return (twice < sc || (twice === sc && q % 2n === 0n) ? q : q + 1n) * sc; }
        if (nd === null) { if (!Number.isFinite(x)) throw new PyError("OverflowError", "cannot convert float " + floatRepr(x) + " to integer"); return BigInt(half(x)); }
        if (!Number.isFinite(x)) return x;
        if (nd < 0) { const sc = Math.pow(10, -nd); return half(x / sc) * sc; }
        const ax = Math.abs(x); let r;
        if (nd <= 100 && exactTie(ax, nd)) { const f = Math.floor(ax * Math.pow(10, nd)); r = (f % 2 === 0 ? f : f + 1) / Math.pow(10, nd); } else r = Number(ax.toFixed(Math.min(nd, 100)));
        return x < 0 || Object.is(x, -0) ? -r : r;
      });
      def("int", (a) => { if (!a.length) return 0n; const v = a[0]; if (typeof v === "bigint") return v; if (typeof v === "boolean") return v ? 1n : 0n; if (typeof v === "number") { if (!Number.isFinite(v)) throw new PyError("OverflowError", "cannot convert float infinity to integer"); return BigInt(Math.trunc(v)); } if (typeof v === "string") { const s = v.trim().replace(/_/g, ""); if (!/^[+-]?\d+$/.test(s)) throw new PyError("ValueError", "invalid literal for int() with base 10: " + strRepr(v)); return BigInt(s); } throw new PyError("TypeError", "int() argument must be a string or a real number, not '" + tname(v) + "'"); }, "int");
      def("float", (a) => { if (!a.length) return 0; const v = a[0]; if (isNumeric(v)) return toNumber(num(v)); if (typeof v === "string") { const s = v.trim().toLowerCase(); if (s === "inf" || s === "+inf" || s === "infinity") return Infinity; if (s === "-inf") return -Infinity; if (s === "nan") return NaN; if (!/^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/.test(s.replace(/_/g, ""))) throw new PyError("ValueError", "could not convert string to float: " + strRepr(v)); return parseFloat(s.replace(/_/g, "")); } throw new PyError("TypeError", "float() argument must be a string or a real number, not '" + tname(v) + "'"); }, "float");
      def("str", (a) => (a.length ? pystr(a[0]) : ""), "str");
      def("repr", (a) => pyrepr(a[0]));
      def("bool", (a) => (a.length ? truthy(a[0]) : false), "bool");
      def("list", (a, kw, I) => (a.length ? [...I.iterate(a[0])] : []), "list");
      def("tuple", (a, kw, I) => new PyTuple(a.length ? [...I.iterate(a[0])] : []), "tuple");
      def("dict", (a, kw, I) => { const d = new PyDict(); if (a.length) { const src = a[0]; if (src instanceof PyDict) src.entries().forEach(([k, v]) => d.set(k, v)); else for (const p of I.iterate(src)) { const kv = [...I.iterate(p)]; d.set(kv[0], kv[1]); } } for (const k in kw) d.set(k, kw[k]); return d; }, "dict");
      def("sorted", (a, kw, I) => { const items = [...I.iterate(a[0])], key = kw.key ? (x) => I.call(kw.key, [x], null) : (x) => x, ks = items.map((x, i) => [key(x), x, i]); ks.sort((p, q) => (cmp(p[0], q[0], "<") ? -1 : cmp(q[0], p[0], "<") ? 1 : p[2] - q[2])); let out = ks.map((p) => p[1]); if (kw.reverse !== undefined && truthy(kw.reverse)) { ks.sort((p, q) => (cmp(p[0], q[0], "<") ? 1 : cmp(q[0], p[0], "<") ? -1 : p[2] - q[2])); out = ks.map((p) => p[1]); } return out; });
      def("reversed", (a, kw, I) => new PyIter([...I.iterate(a[0])].reverse(), "reversed"));
      def("enumerate", (a, kw, I) => { const s = a.length > 1 ? a[1] : 0n; return new PyIter([...I.iterate(a[0])].map((v, i) => new PyTuple([s + BigInt(i), v])), "enumerate"); });
      def("zip", (a, kw, I) => { const ls = a.map((x) => [...I.iterate(x)]), n = ls.length ? Math.min(...ls.map((l) => l.length)) : 0; return new PyIter(Array.from({ length: n }, (_, i) => new PyTuple(ls.map((l) => l[i]))), "zip"); });
      def("map", (a, kw, I) => { const ls = a.slice(1).map((x) => [...I.iterate(x)]), n = Math.min(...ls.map((l) => l.length)); return new PyIter(Array.from({ length: n }, (_, i) => I.call(a[0], ls.map((l) => l[i]), null)), "map"); });
      def("filter", (a, kw, I) => new PyIter([...I.iterate(a[1])].filter((x) => truthy(a[0] === null ? x : I.call(a[0], [x], null))), "filter"));
      def("any", (a, kw, I) => { for (const v of I.iterate(a[0])) if (truthy(v)) return true; return false; });
      def("all", (a, kw, I) => { for (const v of I.iterate(a[0])) if (!truthy(v)) return false; return true; });
      def("isinstance", (a) => { const t = a[1], names = t instanceof PyTuple ? t.items.map((x) => x.typeName) : [t.typeName], tn = tname(a[0]); return names.some((n) => n === tn || (n === "int" && tn === "bool")); });
      def("callable", (a) => a[0] instanceof PyFunc || a[0] instanceof Builtin);
      def("type", (a) => { const n = tname(a[0]); return B[n] || new Builtin(n, () => null, n); });
      def("input", () => { throw new PyError("NotSupported", "input() no está disponible en este laboratorio; escribe los valores directamente en el código"); });
      def("divmod", (a) => new PyTuple([arith("//", a[0], a[1]), arith("%", a[0], a[1])]));
      def("pow", (a) => arith("**", a[0], a[1]));
      def("chr", (a) => String.fromCodePoint(Number(a[0]))); def("ord", (a) => BigInt(a[0].codePointAt(0)));
      def("grafica", (a, kw, I) => { const vals = [...I.iterate(a[0])].map((v) => toNumber(num(v))); I.plot({ type: "line", title: a.length > 1 ? pystr(a[1]) : kw.titulo != null ? pystr(kw.titulo) : "", series: [{ name: "valores", y: vals }] }); return null; });
      def("ayuda", (a, kw, I) => { I.out("Escribe Python y pulsa Enter. Las variables se guardan entre lecciones.\nPrueba: 2 + 2   ·   import math   ·   math.sqrt(16)   ·   [i * i for i in range(5)]\ngrafica(lista, 'título') dibuja una línea con los valores de una lista.\n"); return null; });
      // ayudas de las comprobaciones de los retos
      def("_t", (a) => { a[0].push([a[1], truthy(a[2]), truthy(a[2]) ? "" : a.length > 3 ? a[3] : ""]); return null; });
      def("_try", (a, kw, I) => { try { return I.call(a[0], a.slice(1), null); } catch (e) { if (e instanceof PyError && e.pyType !== "TimeoutError") return "ERROR: " + e.pyType + ": " + e.pyMsg; throw e; } });
      def("_cerca", (a) => { try { const d = Math.abs(toNumber(num(a[0])) - toNumber(num(a[1]))); return d <= (a.length > 2 ? toNumber(num(a[2])) : 1e-6); } catch (e) { return false; } });
      B.True = true; B.False = false; B.None = null;
      return B;
    }
    makeModules() {
      const f1 = (name, fn) => new Builtin(name, (a) => { if (a.length !== 1) throw new PyError("TypeError", name + "() takes exactly one argument (" + a.length + " given)"); if (!isNumeric(a[0])) throw new PyError("TypeError", "must be real number, not " + tname(a[0])); return fn(toNumber(num(a[0]))); });
      const domain = (x) => { if (Number.isNaN(x)) throw new PyError("ValueError", "math domain error"); return x; };
      const math = {
        pi: Math.PI, e: Math.E, tau: 2 * Math.PI, inf: Infinity, nan: NaN,
        sqrt: f1("sqrt", (x) => { if (x < 0) throw new PyError("ValueError", "math domain error"); return Math.sqrt(x); }), exp: f1("exp", Math.exp), tanh: f1("tanh", Math.tanh), sin: f1("sin", Math.sin), cos: f1("cos", Math.cos), tan: f1("tan", Math.tan),
        atan: f1("atan", Math.atan), asin: f1("asin", (x) => domain(Math.asin(x))), acos: f1("acos", (x) => domain(Math.acos(x))), fabs: f1("fabs", Math.abs), sinh: f1("sinh", Math.sinh), cosh: f1("cosh", Math.cosh),
        floor: f1("floor", (x) => BigInt(Math.floor(x))), ceil: f1("ceil", (x) => BigInt(Math.ceil(x))), trunc: f1("trunc", (x) => BigInt(Math.trunc(x))), degrees: f1("degrees", (x) => (x * 180) / Math.PI), radians: f1("radians", (x) => (x * Math.PI) / 180),
        log: new Builtin("log", (a) => { const x = toNumber(num(a[0])); if (x <= 0) throw new PyError("ValueError", "math domain error"); return a.length > 1 ? Math.log(x) / Math.log(toNumber(num(a[1]))) : Math.log(x); }),
        log10: f1("log10", (x) => { if (x <= 0) throw new PyError("ValueError", "math domain error"); return Math.log10(x); }), log2: f1("log2", (x) => { if (x <= 0) throw new PyError("ValueError", "math domain error"); return Math.log2(x); }),
        pow: new Builtin("pow", (a) => { const x = toNumber(num(a[0])), y = toNumber(num(a[1])); if (x < 0 && !Number.isInteger(y)) throw new PyError("ValueError", "math domain error"); return fpow(x, y); }), atan2: new Builtin("atan2", (a) => Math.atan2(toNumber(num(a[0])), toNumber(num(a[1])))),
        isclose: new Builtin("isclose", (a, kw) => { const x = toNumber(num(a[0])), y = toNumber(num(a[1])), rt = kw.rel_tol != null ? kw.rel_tol : 1e-9, at = kw.abs_tol != null ? kw.abs_tol : 0; return Math.abs(x - y) <= Math.max(rt * Math.max(Math.abs(x), Math.abs(y)), at); }),
        isnan: f1("isnan", (x) => Number.isNaN(x)), isinf: f1("isinf", (x) => Math.abs(x) === Infinity), hypot: new Builtin("hypot", (a) => Math.hypot(...a.map((v) => toNumber(num(v))))),
        factorial: new Builtin("factorial", (a) => { let r = 1n; for (let i = 2n; i <= num(a[0]); i++) r *= i; return r; }),
      };
      const R = this.rng, rg = (name, fn) => new Builtin(name, fn);
      const random = {
        random: rg("random", () => R.random()), seed: rg("seed", (a) => { if (!a.length || a[0] === null) R.seedWith(Date.now()); else if (typeof a[0] === "string") { let h = 0; for (const c of a[0]) h = (h * 31 + c.charCodeAt(0)) >>> 0; R.seedWith(h); } else { const sv = num(a[0]); R.seedWith(typeof sv === "bigint" ? sv : Math.trunc(sv)); } return null; }),
        randint: rg("randint", (a) => { const lo = num(a[0]), hi = num(a[1]); if (hi < lo) throw new PyError("ValueError", "empty range for randrange() (" + lo + ", " + (hi + 1n) + ", " + (hi + 1n - lo) + ")"); return lo + BigInt(R.randbelow(Number(hi - lo + 1n))); }),
        randrange: rg("randrange", (a) => { const start = a.length > 1 ? num(a[0]) : 0n, stop = a.length > 1 ? num(a[1]) : num(a[0]); return start + BigInt(R.randbelow(Number(stop - start))); }),
        uniform: rg("uniform", (a) => { const x = toNumber(num(a[0])), y = toNumber(num(a[1])); return x + (y - x) * R.random(); }),
        gauss: rg("gauss", (a) => R.gauss(a.length ? toNumber(num(a[0])) : 0, a.length > 1 ? toNumber(num(a[1])) : 1)),
        normalvariate: rg("normalvariate", (a) => R.gauss(a.length ? toNumber(num(a[0])) : 0, a.length > 1 ? toNumber(num(a[1])) : 1)),
        choice: rg("choice", (a, kw, I) => { const s = [...I.iterate(a[0])]; if (!s.length) throw new PyError("IndexError", "Cannot choose from an empty sequence"); return s[R.randbelow(s.length)]; }),
        shuffle: rg("shuffle", (a) => { const x = a[0]; for (let i = x.length - 1; i >= 1; i--) { const j = R.randbelow(i + 1); const t = x[i]; x[i] = x[j]; x[j] = t; } return null; }),
        sample: rg("sample", (a, kw, I) => { const pool = [...I.iterate(a[0])], k = Number(num(a[1])), out = []; for (let i = 0; i < k; i++) { const j = R.randbelow(pool.length - i); out.push(pool[j]); pool[j] = pool[pool.length - i - 1]; } return out; }),
      };
      return { math: new PyModule("math", math), random: new PyModule("random", random), time: new PyModule("time", { time: new Builtin("time", () => Date.now() / 1000) }) };
    }
  }

  // ============================================================ métodos de tipos integrados
  const needStr = (v, m) => { if (typeof v !== "string") throw new PyError("TypeError", m + " arg must be str, not " + tname(v)); return v; };
  const STR_M = {
    upper: (s) => s.toUpperCase(), lower: (s) => s.toLowerCase(), capitalize: (s) => s.charAt(0).toUpperCase() + s.slice(1).toLowerCase(), title: (s) => s.replace(/\b\w/g, (c) => c.toUpperCase()),
    strip: (s, a) => (a.length ? s.replace(new RegExp("^[" + a[0].replace(/[\]\\^-]/g, "\\$&") + "]+|[" + a[0].replace(/[\]\\^-]/g, "\\$&") + "]+$", "g"), "") : s.trim()), lstrip: (s) => s.trimStart(), rstrip: (s) => s.trimEnd(),
    split: (s, a) => { if (!a.length || a[0] === null) return s.split(/\s+/).filter((x) => x.length); const sep = needStr(a[0], "must be"); let parts = s.split(sep); if (a.length > 1) { const mx = Number(num(a[1])); if (parts.length > mx + 1) parts = parts.slice(0, mx).concat(parts.slice(mx).join(sep)); } return parts; },
    join: (s, a, k, I) => [...I.iterate(a[0])].map((x) => { if (typeof x !== "string") throw new PyError("TypeError", "sequence item: expected str instance, " + tname(x) + " found"); return x; }).join(s),
    replace: (s, a) => s.split(a[0]).join(a[1]), startswith: (s, a) => (a[0] instanceof PyTuple ? a[0].items.some((p) => s.startsWith(p)) : s.startsWith(a[0])), endswith: (s, a) => (a[0] instanceof PyTuple ? a[0].items.some((p) => s.endsWith(p)) : s.endsWith(a[0])),
    find: (s, a) => BigInt(s.indexOf(a[0])), index: (s, a) => { const i = s.indexOf(a[0]); if (i < 0) throw new PyError("ValueError", "substring not found"); return BigInt(i); }, count: (s, a) => BigInt(a[0] === "" ? s.length + 1 : s.split(a[0]).length - 1),
    isdigit: (s) => /^\d+$/.test(s), isalpha: (s) => /^[A-Za-zÀ-￿]+$/.test(s), isspace: (s) => /^\s+$/.test(s), isupper: (s) => s !== s.toLowerCase() && s === s.toUpperCase(), islower: (s) => s !== s.toUpperCase() && s === s.toLowerCase(),
    format: (s, a, kw) => strFormat(s, a, kw), zfill: (s, a) => s.padStart(Number(num(a[0])), "0"), center: (s, a) => { const w = Number(num(a[0])), t = Math.max(0, w - s.length), l = Math.floor(t / 2); return " ".repeat(l) + s + " ".repeat(t - l); }, ljust: (s, a) => s.padEnd(Number(num(a[0]))), rjust: (s, a) => s.padStart(Number(num(a[0]))),
  };
  const LIST_M = {
    append: (l, a) => { l.push(a[0]); return null; }, extend: (l, a, k, I) => { l.push(...I.iterate(a[0])); return null; }, insert: (l, a) => { let i = Number(num(a[0])); if (i < 0) i = Math.max(0, l.length + i); l.splice(Math.min(i, l.length), 0, a[1]); return null; },
    pop: (l, a) => { if (!l.length) throw new PyError("IndexError", "pop from empty list"); const i = a.length ? Number(num(a[0])) : -1, j = i < 0 ? l.length + i : i; if (j < 0 || j >= l.length) throw new PyError("IndexError", "pop index out of range"); return l.splice(j, 1)[0]; },
    remove: (l, a) => { const i = l.findIndex((x) => pyEq(x, a[0])); if (i < 0) throw new PyError("ValueError", "list.remove(x): x not in list"); l.splice(i, 1); return null; },
    index: (l, a) => { const i = l.findIndex((x) => pyEq(x, a[0])); if (i < 0) throw new PyError("ValueError", pyrepr(a[0]) + " is not in list"); return BigInt(i); }, count: (l, a) => BigInt(l.filter((x) => pyEq(x, a[0])).length),
    sort: (l, a, kw, I) => { const key = kw.key ? (x) => I.call(kw.key, [x], null) : (x) => x, rev = kw.reverse !== undefined && truthy(kw.reverse), ks = l.map((x, i) => [key(x), x, i]); ks.sort((p, q) => { const d = cmp(p[0], q[0], "<") ? -1 : cmp(q[0], p[0], "<") ? 1 : 0; return rev ? -d || p[2] - q[2] : d || p[2] - q[2]; }); ks.forEach((p, i) => { l[i] = p[1]; }); return null; },
    reverse: (l) => { l.reverse(); return null; }, copy: (l) => l.slice(), clear: (l) => { l.length = 0; return null; },
  };
  const DICT_M = {
    get: (d, a) => { const v = d.get(a[0]); return v === undefined ? (a.length > 1 ? a[1] : null) : v; }, keys: (d) => new PyIter(d.keys(), "dict_keys"), values: (d) => new PyIter(d.values(), "dict_values"), items: (d) => new PyIter(d.entries().map(([k, v]) => new PyTuple([k, v])), "dict_items"),
    pop: (d, a) => { const v = d.get(a[0]); if (v === undefined) { if (a.length > 1) return a[1]; throw new PyError("KeyError", pyrepr(a[0])); } d.del(a[0]); return v; }, update: (d, a, kw) => { if (a.length) a[0].entries().forEach(([k, v]) => d.set(k, v)); for (const k in kw) d.set(k, kw[k]); return null; },
    setdefault: (d, a) => { if (!d.has(a[0])) d.set(a[0], a.length > 1 ? a[1] : null); return d.get(a[0]); }, copy: (d) => { const n = new PyDict(); d.entries().forEach(([k, v]) => n.set(k, v)); return n; }, clear: (d) => { d.m.clear(); return null; },
  };
  const TUPLE_M = { count: (t, a) => BigInt(t.items.filter((x) => pyEq(x, a[0])).length), index: (t, a) => { const i = t.items.findIndex((x) => pyEq(x, a[0])); if (i < 0) throw new PyError("ValueError", "tuple.index(x): x not in tuple"); return BigInt(i); } };
  const FLOAT_M = { is_integer: (x) => Number.isInteger(x) };

  const api = { PyInterp, PyError, PyTuple, PyDict, pyrepr, pystr, parseProgram, parseExpression, tokenize };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.PyInterpLib = api;
})(typeof window !== "undefined" ? window : globalThis);
