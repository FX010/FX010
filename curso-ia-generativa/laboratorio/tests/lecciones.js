// Cada lección: con los huecos sin completar el reto debe FALLAR, y con la solución debe PASAR.
const { PyInterp, PyError } = require("../src/pyinterp.js");
const { LESSONS } = require("../src/lessons.js");
let mal = 0;
for (const L of LESSONS) {
  const I = new PyInterp({ stdout: () => {} });
  const run = (c) => { try { I.run(c); return null; } catch (e) { if (e instanceof PyError) return e.pyType + ": " + e.pyMsg; throw e; } };
  const e0 = run(L.code), c0 = I.check(L.check); I.reset();
  const e1 = run(L.solution), c1 = I.check(L.check);
  const ok = (c0.some((t) => !t[1]) || e0 !== null) && e1 === null && c1.length > 0 && c1.every((t) => t[1]);
  if (!ok) mal++;
  console.log(ok ? "OK " : "MAL", L.num, L.id);
}
console.log(mal ? "HAY LECCIONES CON PROBLEMAS: " + mal : "Todas las lecciones correctas");
process.exit(mal ? 1 : 0);
