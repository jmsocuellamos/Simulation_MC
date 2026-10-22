# -*- coding: utf-8 -*-
"""Genera index.html: la aplicación Streamlit empaquetada con stlite (Streamlit + Pyodide) para
ejecutarse íntegramente en el navegador, sin servidor (apta para GitHub Pages).

Uso:  python build_html.py          (lee el código de ./app y escribe ./index.html)

Cada vez que modifiques algún fichero de ./app, vuelve a ejecutar este script y sube el nuevo index.html.
"""
import json
import pathlib

STLITE = "0.85.1"     # versión de @stlite/browser
RAIZ = pathlib.Path(__file__).parent
CARPETA_APP = RAIZ / "app"
FICHEROS = ["app.py", "motor.py", "kpis.py", "validacion.py", "graficos.py", "informe.py"]
# Paquetes que se instalan en el navegador (numpy, pandas, scipy, matplotlib y lxml vienen con Pyodide)
REQUISITOS = ["simpy", "numpy", "pandas", "scipy", "matplotlib", "seaborn", "python-docx", "openpyxl"]

PLANTILLA = """<!doctype html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="X-UA-Compatible" content="IE=edge" />
  <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
  <title>Simulador Urgencias</title>
  <meta name="description" content="Simulación Monte Carlo de la red de colas del Servicio de Urgencias (SimPy + Streamlit en el navegador)." />
  <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🏥</text></svg>" />
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/browser@__STLITE__/build/stlite.css" />
  <style>
    #carga { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; max-width: 640px; margin: 12vh auto; padding: 0 16px;
             color: #1f2937; line-height: 1.5; }
    #carga h1 { font-size: 1.4rem; margin-bottom: .3rem; }
    #carga p { color: #4b5563; }
    .barra { height: 6px; background: #e5e7eb; border-radius: 3px; overflow: hidden; margin-top: 1.2rem; }
    .barra div { height: 100%; width: 40%; background: #2f5496; border-radius: 3px; animation: mover 1.4s ease-in-out infinite; }
    @keyframes mover { 0% { margin-left: -40%; } 100% { margin-left: 100%; } }
    @media (prefers-color-scheme: dark) { body { background: #0e1117; } #carga { color: #f3f4f6; } #carga p { color: #9ca3af; } }
  </style>
</head>
<body>
  <div id="carga">
    <h1>🏥 Simulador de la red de colas de Urgencias</h1>
    <p>Cargando Python en el navegador e instalando los paquetes (SimPy, pandas, SciPy, matplotlib…).
       La primera vez puede tardar uno o dos minutos; después el navegador lo guarda en caché.</p>
    <div class="barra"><div></div></div>
  </div>
  <div id="root"></div>
  <script type="module">
    import { mount } from "https://cdn.jsdelivr.net/npm/@stlite/browser@__STLITE__/build/stlite.js";
    const ficheros = __FICHEROS__;
    const carga = document.getElementById("carga");
    const raiz = document.getElementById("root");
    new MutationObserver((_, obs) => { if (raiz.childElementCount) { carga.remove(); obs.disconnect(); } })
      .observe(raiz, { childList: true });
    mount(
      {
        requirements: __REQUISITOS__,
        entrypoint: "app.py",
        files: ficheros,
      },
      raiz,
    );
  </script>
</body>
</html>
"""


def main():
    ficheros = {}
    for nombre in FICHEROS:
        ficheros[nombre] = (CARPETA_APP / nombre).read_text(encoding="utf-8")
    # json.dumps produce literales JS válidos; se escapa "</" para no cerrar el <script> por accidente
    js_ficheros = json.dumps(ficheros, ensure_ascii=False, indent=1).replace("</", "<\\/")
    html = (PLANTILLA.replace("__STLITE__", STLITE)
            .replace("__REQUISITOS__", json.dumps(REQUISITOS))
            .replace("__FICHEROS__", js_ficheros))
    (RAIZ / "index.html").write_text(html, encoding="utf-8")
    (RAIZ / ".nojekyll").write_text("", encoding="utf-8")
    print(f"index.html generado ({len(html) / 1024:.0f} KB) con {len(ficheros)} ficheros Python.")


if __name__ == "__main__":
    main()
