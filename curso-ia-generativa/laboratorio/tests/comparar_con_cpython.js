// Compara la salida del intérprete con la de CPython para cada programa del corpus.
const fs = require("fs"), path = require("path");
const { PyInterp, PyError } = require("../src/pyinterp.js");
const exp = JSON.parse(fs.readFileSync(path.join(__dirname, "esperados.json"), "utf8"));
let bad = 0, nosup = 0;
exp.forEach((e, i) => {
  let out = "", exc = null, msg = "";
  const I = new PyInterp({ stdout: (s) => { out += s; } });
  try { I.run(e.code); } catch (err) { if (err instanceof PyError) { exc = err.pyType; msg = err.pyMsg; } else { exc = "JSERR"; msg = err.message; } }
  if (exc === "NotSupported") nosup++;
  const igual = out === e.out && ((exc || null) === (e.exc || null) || exc === "NotSupported");
  if (!igual) { bad++; console.log("--- #" + i + "\n" + e.code + "\n  CPython:", JSON.stringify(e.out), e.exc, "\n  nuestro:", JSON.stringify(out), exc, msg); }
});
console.log("programas:", exp.length, "| diferencias:", bad, "| no soportados a propósito:", nosup);
