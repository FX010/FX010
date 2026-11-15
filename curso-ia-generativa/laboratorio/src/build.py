#!/usr/bin/env python3
"""Une kernel.js + lessons.js + app.js + styles.css en dos archivos:
   dist/artifact.html            fragmento para publicar como Artifact (sin doctype/head/body)
   ../laboratorio-neuronal.html  documento completo que se abre con doble clic en cualquier navegador
"""
import pathlib
src = pathlib.Path(__file__).resolve().parent
dist = src.parent
(dist / "dist").mkdir(exist_ok=True)
t = (src / "template.html").read_text()
for tag, f in [("CSS", "styles.css"), ("KERNEL", "kernel.js"), ("LESSONS", "lessons.js"), ("APP", "app.js")]:
    t = t.replace(f"/*__{tag}__*/", (src / f).read_text().replace("</script", "<\\/script"))
(dist / "dist" / "artifact.html").write_text(t)
head, _, rest = t.partition("<header")
full = '<!doctype html>\n<html lang="es">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n' + head + '</head>\n<body>\n<header' + rest + "\n</body>\n</html>\n"
(dist / "laboratorio-neuronal.html").write_text(full)
print("OK", len(t) // 1024, "KB")
