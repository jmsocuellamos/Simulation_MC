# Simulador del taller de mantenimiento y reacondicionamiento eólico · versión web (GitHub Pages)

`index.html` contiene la aplicación Streamlit completa, que se ejecuta **dentro del navegador** gracias a
[stlite](https://github.com/whitphx/stlite) (Streamlit + Pyodide). No hace falta servidor: basta con
alojar el fichero en cualquier hosting estático, como GitHub Pages.

## Contenido

| Fichero | Para qué sirve |
| --- | --- |
| `index.html` | La aplicación lista para publicar, con todo el código Python incluido |
| `.nojekyll` | Indica a GitHub Pages que sirva los ficheros tal cual |
| `app/` | Código fuente de la aplicación (el mismo que la versión local) |
| `build_html.py` | Regenera `index.html` a partir del código de `app/` |

## Publicar en GitHub Pages

1. Crea un repositorio en GitHub (por ejemplo, `simulador-taller-eolico`).
2. Sube el contenido de esta carpeta a la raíz de la rama `main`: `index.html`, `.nojekyll`, `app/`,
   `build_html.py` y este README. Puedes arrastrarlo todo con *Add file → Upload files* o usar git:
   ```bash
   git init && git add . && git commit -m "Simulador del taller eólico"
   git branch -M main
   git remote add origin https://github.com/<usuario>/simulador-taller-eolico.git
   git push -u origin main
   ```
3. En el repositorio, ve a **Settings → Pages**. En *Build and deployment* elige *Source: Deploy from a branch*,
   la rama **main** y la carpeta **/ (root)**, y pulsa *Save*.
4. Al cabo de uno o dos minutos la aplicación estará en `https://<usuario>.github.io/simulador-taller-eolico/`.

## Modificar la aplicación

Edita los ficheros de `app/` y regenera la página antes de subirla:

```bash
python build_html.py
```

Para probarla en tu ordenador antes de publicarla, sírvela con un servidor local (abrir `index.html`
con doble clic también suele funcionar):

```bash
python -m http.server 8000      # y abre http://localhost:8000
```

La misma carpeta `app/` funciona también como aplicación local con Streamlit, que es mucho más rápida:

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

## Diferencias con la versión local

- **Primera carga**: el navegador descarga Python (Pyodide) y los paquetes SimPy, pandas, SciPy, matplotlib,
  seaborn, python-docx y openpyxl. La primera vez tarda uno o dos minutos; después quedan en la caché.
- **Velocidad**: todo se calcula en el navegador, en un único hilo y unas 2–3 veces más despacio que en Python
  nativo, y sin ejecución en paralelo. Con unas 275 órdenes al año, cada réplica de 5 años es ligera: el estudio por defecto (30 réplicas × 4 escenarios) tarda solo unos minutos.
- **Privacidad**: los datos y resultados no salen de tu ordenador. La configuración se puede descargar en JSON
  y volver a cargar más tarde desde la barra lateral.
- El informe Word y el Excel de réplicas se descargan igual que en la versión local.
