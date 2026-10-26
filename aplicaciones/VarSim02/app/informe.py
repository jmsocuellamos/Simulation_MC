# -*- coding: utf-8 -*-
"""Informe Word (.docx) de riesgo y retorno de planes de inversión: teoría, simulación, planes y sensibilidad."""
import datetime
import io
import math

import numpy as np
import pandas as pd
import docx
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

import inversion as iv
from inversion import num, pct, eur, fmt_kpi, fmt_mc, fmt_obj
import graficos as gr


def _sombrear(celda, color):
    tcPr = celda._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), color)
    tcPr.append(shd)


ANCHO_UTIL_CM = 17.0          # A4 con márgenes de 2 cm


def _cm_por_caracter(pt):
    return 0.0205 * pt          # anchura media aproximada de un carácter de Calibri


def _anchos(cab, filas, pt):
    """Anchura de cada columna según su contenido; se reparte el ancho útil sin superar la página."""
    nat = []
    for j, c in enumerate(cab):
        palabras = max((len(w) for w in str(c).split()), default=1)
        # los textos largos (con espacios) pueden partirse en varias líneas; los números, no
        celdas = max((min(len(f[j]), 30) if " " in f[j].strip() and len(f[j]) > 30 else len(f[j]) for f in filas),
                     default=1)
        nat.append(max(palabras, celdas, 3) * _cm_por_caracter(pt) + 0.3)
    total = sum(nat)
    return nat, total


class Informe:
    """Envoltorio sencillo de python-docx con helpers para títulos, párrafos, tablas y figuras."""

    def __init__(self, titulo, subtitulo=""):
        self.doc = docx.Document()
        sec = self.doc.sections[0]
        sec.page_width, sec.page_height = Cm(21), Cm(29.7)
        sec.left_margin = sec.right_margin = Cm(2)
        sec.top_margin = sec.bottom_margin = Cm(2)
        estilo = self.doc.styles["Normal"]
        estilo.font.name = "Calibri"; estilo.font.size = Pt(10.5)
        self.n_tabla, self.n_fig = 0, 0
        self.doc.add_heading(titulo, level=0)
        if subtitulo:
            p = self.doc.add_paragraph(subtitulo); p.runs[0].italic = True

    def h(self, texto, nivel=1):
        self.doc.add_heading(texto, level=nivel)

    def p(self, texto, negrita_inicio=None):
        par = self.doc.add_paragraph()
        if negrita_inicio and texto.startswith(negrita_inicio):
            par.add_run(negrita_inicio).bold = True
            par.add_run(texto[len(negrita_inicio):])
        else:
            par.add_run(texto)
        par.paragraph_format.space_after = Pt(6)
        return par

    def vinetas(self, items):
        for it in items:
            self.doc.add_paragraph(it, style="List Bullet")

    def tabla(self, df, titulo, formato=None, indice=True, ancho_fuente=9, nombre_indice=""):
        """Tabla con anchos de columna calculados para que quepa siempre en el ancho de la página: si el contenido
        no cabe se reduce la letra (hasta 7 pt) y se reparten las columnas proporcionalmente."""
        self.n_tabla += 1
        cap = self.doc.add_paragraph()
        cap.add_run(f"Tabla {self.n_tabla}. {titulo}").bold = True
        cap.paragraph_format.keep_with_next = True
        cols = list(df.columns)
        cab = ([nombre_indice or (df.index.name or "")] if indice else []) + [str(c) for c in cols]
        filas = []
        for idx, fila in df.iterrows():
            filas.append(([num(idx, 0) if isinstance(idx, (int, np.integer)) else str(idx)] if indice else []) +
                         [formato(idx, c, fila[c]) if formato else _fmt(fila[c]) for c in cols])
        pt = ancho_fuente
        nat, total = _anchos(cab, filas, pt)
        while total > ANCHO_UTIL_CM and pt > 7:
            pt -= 0.5
            nat, total = _anchos(cab, filas, pt)
        anchos = [w * ANCHO_UTIL_CM / total for w in nat] if total > ANCHO_UTIL_CM else \
                 [w * min(ANCHO_UTIL_CM / total, 1.6) for w in nat]
        t = self.doc.add_table(rows=1, cols=len(cab))
        t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        tblPr = t._tbl.tblPr
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
        # la fila de cabecera se repite si la tabla continúa en otra página
        trPr = t.rows[0]._tr.get_or_add_trPr()
        rep = OxmlElement("w:tblHeader"); rep.set(qn("w:val"), "true"); trPr.append(rep)
        for j, col in enumerate(t.columns):          # rejilla de la tabla (la que usan Word y LibreOffice)
            col.width = Cm(anchos[j])
        for j, c in enumerate(cab):
            cel = t.rows[0].cells[j]; cel.text = ""; cel.width = Cm(anchos[j])
            r = cel.paragraphs[0].add_run(str(c)); r.bold = True; r.font.size = Pt(pt)
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            if j > 0 or not indice:
                cel.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
            _sombrear(cel, "2F5496")
        for i, vals in enumerate(filas):
            celdas = t.add_row().cells
            for j, v in enumerate(vals):
                celdas[j].text = ""; celdas[j].width = Cm(anchos[j])
                r = celdas[j].paragraphs[0].add_run(v); r.font.size = Pt(pt)
                if j > 0 or not indice:
                    celdas[j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
                if i % 2 == 1:
                    _sombrear(celdas[j], "EAF0F8")
        self.doc.add_paragraph()

    def figura(self, fig, titulo, ancho_cm=16.5):
        self.n_fig += 1
        self.doc.add_picture(gr.fig_a_png(fig), width=Cm(ancho_cm))
        self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.doc.paragraphs[-1].paragraph_format.keep_with_next = True
        cap = self.doc.add_paragraph(); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cap.add_run(f"Figura {self.n_fig}. {titulo}"); r.italic = True; r.font.size = Pt(9)

    def salto(self):
        self.doc.add_page_break()

    def guardar(self):
        buf = io.BytesIO()
        self.doc.save(buf)
        return buf.getvalue()


def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "Sí" if v else "No"
    if v is None:
        return "—"
    if isinstance(v, (int, np.integer)):
        return num(v, 0)
    if isinstance(v, (float, np.floating)):
        return "—" if math.isnan(v) else num(v, 3)
    return str(v)


def fmt_tramos(fila, col, v):
    if col.startswith(("Teórica", "Simulada")):
        return pct(v, 2)
    if "€" in col:
        return eur(v, 0)
    return str(v)


def fmt_euros(fila, col, v):
    return eur(v, 0)


def fmt_sens(fila, col, v):
    if col == "Simulaciones totales":
        return num(v, 0)
    if col == "Error relativo":
        return pct(v, 3)
    return eur(v, 1) if "σ" in col else eur(v, 0)


def tabla_sens(df):
    return df.drop(columns=["Medias por réplica"])


def generar_informe(res, comunes, ajustes, titulo="Riesgo y retorno de planes de inversión", sens=None):
    inf = Informe(titulo, f"Informe generado automáticamente el {datetime.datetime.now():%d/%m/%Y %H:%M} · "
                          "transformada inversa, simulación Monte Carlo y comparación de planes de inversión")
    I0 = comunes["I0"]
    # 1. Planteamiento
    inf.h("1. Planteamiento y parámetros", 1)
    inf.p(f"Una empresa dispone de una inversión inicial I₀ = {eur(I0)} y puede elegir entre varios planes. El retorno R "
          "de cada plan es una variable aleatoria discreta: toma los valores v₁, …, v_k con probabilidades p₁, …, p_k. "
          "R se genera por el método de la transformada inversa: se lanza U ~ Unif(0, 1) y se toma el menor i con "
          "U ≤ F(i) = p₁ + … + pᵢ. Para cada plan se estudian el retorno esperado E(R) = Σ pᵢ·vᵢ, el riesgo "
          "σ(R) = √(Σ pᵢ·(vᵢ − E(R))²) y la probabilidad de pérdida patrimonial P(R < I₀).")
    esc = pd.DataFrame({n: {"Tramos": num(r["teoria"]["tramos"], 0),
                            "Retornos posibles (€)": " / ".join(num(x, 0) for x in r["teoria"]["v"]),
                            "Probabilidades": " / ".join(pct(x, 0) for x in r["teoria"]["p"])}
                        for n, r in res.items()})
    inf.tabla(esc, "Planes de inversión (escenarios) analizados", nombre_indice="Parámetro", formato=lambda f, c, v: v)
    desc = [f"{n}: {r['esc']['descripcion']}" for n, r in res.items() if r["esc"].get("descripcion")]
    if desc:
        inf.vinetas(desc)
    ob = comunes["objetivos"]
    def o(k, f):
        return f(ob[k]) if ob.get(k) is not None else "no se evalúa"
    com = pd.DataFrame({"Valor": {
        "Inversión inicial I₀": eur(I0),
        "Objetivo · rentabilidad esperada mínima": o("RENT_MIN", lambda v: pct(v, 1)),
        "Objetivo · probabilidad de pérdida máxima": o("PLOSS_MAX", lambda v: pct(v, 1)),
        "Objetivo · riesgo σ(R) máximo": o("SIGMA_MAX", lambda v: eur(v)),
        "Objetivo · resultado neto mínimo en el peor 5 %": o("P5NETO_MIN", lambda v: eur(v)),
        "Simulación · simulaciones por réplica (nsim)": num(ajustes["nsim"], 0),
        "Simulación · réplicas (semillas)": f"{ajustes['replicas']} (semillas {ajustes['semilla']} … "
                                            f"{int(ajustes['semilla']) + int(ajustes['replicas']) - 1})"}})
    inf.tabla(com, "Parámetros comunes, objetivos y ajustes de simulación", nombre_indice="Parámetro",
              formato=lambda f, c, v: v)

    # 2. Síntesis
    inf.h("2. Síntesis y recomendación", 1)
    for n, r in res.items():
        inf.p(iv.texto_escenario(n, r), negrita_inicio=n + ":")
    inf.p("Recomendación: " + iv.texto_recomendacion(res), negrita_inicio="Recomendación:")
    if len(res) > 1:
        inf.tabla(iv.tabla_comparativa(res), "Comparación de planes (filas «Teórico»: fórmulas exactas; filas "
                  "«Simulado»: media de las réplicas)", formato=fmt_kpi, nombre_indice="KPI")
        inf.tabla(iv.tabla_objetivos(res), "Cumplimiento de los objetivos (valores teóricos)", nombre_indice="Objetivo",
                  formato=lambda f, c, v: str(v))
        inf.figura(gr.fig_riesgo_retorno(res), "Riesgo frente a retorno: valores teóricos y simulados con su IC95 %")
        inf.figura(gr.fig_comparativa(res), "Retorno esperado, riesgo, probabilidad de pérdida y rentabilidad por plan (teóricos)")
        inf.figura(gr.fig_validacion(res), "Error relativo de la simulación frente a los valores teóricos")

    # 3. Detalle por plan
    for k, (n, r) in enumerate(res.items(), start=3):
        inf.salto()
        t, mc = r["teoria"], r["mc"]
        inf.h(f"{k}. Plan «{n}»", 1)
        if r["esc"].get("descripcion"):
            inf.p(r["esc"]["descripcion"])
        inf.h(f"{k}.1 Resultados teóricos", 2)
        teo = pd.DataFrame({"Valor": {
            "Retorno esperado E(R) = Σ pᵢ·vᵢ": eur(t["E"]),
            "Riesgo σ(R)": eur(t["sd"], 1),
            "Coeficiente de variación σ(R)/E(R)": pct(t["CV"], 2),
            "Resultado neto esperado E(R) − I₀": eur(t["neto"]),
            "Rentabilidad esperada (E(R) − I₀)/I₀": pct(t["rent"], 2),
            "Probabilidad de pérdida P(R < I₀)": pct(t["Ploss"], 2),
            "Pérdida esperada E[max(I₀ − R, 0)]": eur(t["perdida_media"]),
            "Percentil 5 de R": eur(t["P5"]),
            "Resultado neto en el peor 5 % (P5 − I₀)": eur(t["P5_neto"]),
            "Retorno mínimo / máximo posible": f"{eur(t['min'])} / {eur(t['max'])}"}})
        inf.tabla(teo, f"KPIs teóricos · {n}", nombre_indice="Medida", formato=lambda f, c, v: v)
        inf.tabla(r["objetivos"], f"Objetivos (valores teóricos) · {n}", formato=fmt_obj, nombre_indice="Objetivo")

        inf.h(f"{k}.2 Resultados simulados", 2)
        inf.p(f"{ajustes['replicas']} réplicas independientes de {num(ajustes['nsim'], 0)} simulaciones por "
              "transformada inversa. Cada KPI se calcula en cada réplica; la tabla muestra la media de las réplicas y su "
              "IC95 % (t de Student). Todos los planes usan las mismas semillas (números aleatorios comunes).")
        inf.tabla(r["resumen_mc"], f"KPIs simulados frente a teóricos · {n}", formato=fmt_mc, nombre_indice="KPI")
        inf.figura(gr.fig_replicas(mc, t, f"Retorno medio en cada réplica · {n}"),
                   f"Media de cada réplica con su IC95 % y valor teórico · {n}")

        inf.h(f"{k}.3 Distribución del retorno (teórica y simulada)", 2)
        inf.tabla(iv.tabla_tramos(t, mc), f"Tramos del plan e intervalo de U de la transformada inversa · {n}",
                  formato=fmt_tramos, nombre_indice="Tramo")
        inf.figura(gr.fig_pmf(t, mc, f"Distribución del retorno · {n}"),
                   f"Probabilidad de cada retorno: teórica (barras) y simulada (rombos) · {n}")
        inf.figura(gr.fig_cdf(t, mc["R"], f"Función de distribución del retorno · {n}"),
                   f"Función de distribución teórica y empírica del retorno, con la inversión inicial I₀ · {n}")
        inf.tabla(iv.tabla_cuantiles(t, mc), f"Percentiles del retorno · {n}", formato=fmt_euros, nombre_indice="Percentil")

        inf.h(f"{k}.4 Funcionamiento simulado", 2)
        u0 = float(mc["traza"]["U"].iloc[0])
        inf.figura(gr.fig_escalera(t, u0, f"Transformada inversa · {n}"),
                   f"Probabilidades acumuladas F(i) y primera simulación (U = {num(u0, 2)}) · {n}")
        inf.figura(gr.fig_traza(mc["traza"], t, f"Primeras {len(mc['traza'])} simulaciones · {n}"),
                   f"U generado y tramo obtenido en las primeras {len(mc['traza'])} simulaciones · {n}")

    if sens is not None:
        inf.salto()
        df, cfg = sens["df"], sens["cfg"]
        inf.h(f"{len(res) + 3}. Sensibilidad del estimador Monte Carlo", 1)
        inf.p(f"Plan «{cfg['plan']}». Se varía nsim entre {num(df.index.min(), 0)} y {num(df.index.max(), 0)} "
              f"({len(df)} puntos en escala logarítmica) con {cfg['replicas']} réplicas por punto (semillas "
              f"{cfg['semilla']} … {cfg['semilla'] + cfg['replicas'] - 1}). " + sens["texto"])
        inf.figura(gr.fig_sens_convergencia(df), "Convergencia de la estimación hacia el retorno teórico")
        inf.figura(gr.fig_sens_variabilidad(df), "Variabilidad de la media estimada entre réplicas y predicción σ(R)/√nsim")
        inf.tabla(tabla_sens(df), "Barrido del número de simulaciones", formato=fmt_sens, nombre_indice="nsim")

    inf.salto()
    inf.h("Anexo. Metodología", 1)
    inf.vinetas([
        "Transformada inversa: F(i) = p₁ + … + pᵢ; con U ~ Unif(0, 1), R = vᵢ para el menor i tal que U ≤ F(i). Así "
        "P(R = vᵢ) = P(F(i−1) < U ≤ F(i)) = pᵢ.",
        "Retorno esperado E(R) = Σ pᵢ·vᵢ ; riesgo σ(R) = √(Σ pᵢ·(vᵢ − E(R))²) ; CV = σ(R)/E(R).",
        "Probabilidad de pérdida P(R < I₀) = Σ_{vᵢ < I₀} pᵢ ; pérdida esperada = Σ pᵢ·max(I₀ − vᵢ, 0).",
        "Rentabilidad esperada = (E(R) − I₀)/I₀ ; percentiles: menor valor x con P(R ≤ x) ≥ q.",
        "IC95 % de los KPIs simulados: media de las réplicas ± t₀.₉₇₅·s/√R. En la sensibilidad, IC95 % de la media "
        "combinada: media ± 1,96·s/√(simulaciones); la σ de la media de nsim simulaciones es σ(R)/√nsim.",
        "Basado en el Ejemplo 2 de «B-70. Distribuciones derivadas», Javier Morales y Asun Martínez Mayoral, equipo "
        "IA4LEGOS (CC BY-SA 4.0).",
    ])
    return inf.guardar()
