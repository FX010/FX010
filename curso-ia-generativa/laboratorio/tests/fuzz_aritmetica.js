const fs = require("fs"), path = require("path");
const { PyInterp, PyError } = require("../src/pyinterp.js");
const casos = JSON.parse(fs.readFileSync(path.join(__dirname, "fuzz.json"), "utf8"));
let bad = 0, complejos = 0;
for (const [e, esperado] of casos) {
  let out = "", got;
  const I = new PyInterp({ stdout: (s) => { out += s; } });
  try { I.run("print(repr(" + e + "))"); got = out.replace(/\n$/, ""); } catch (err) { got = err instanceof PyError ? "EXC:" + err.pyType : "JS:" + err.message; }
  if (got !== esperado) { if (got === "EXC:NotSupported") complejos++; else { bad++; console.log("DIF:", e, "\n  CPython:", esperado, "\n  nuestro:", got); } }
}
console.log("expresiones:", casos.length, "| diferencias:", bad, "| números complejos (no soportados a propósito):", complejos);
