# -*- coding: utf-8 -*-
"""Informe Word (.docx) del estudio de caso Bio-Distri: beneficio de un proyecto de calidad, programa plurianual con y
sin mantenimiento, escenarios, mantenimiento, sensibilidad y ampliación."""
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

import biodistri as bd
from biodistri import num, pct, keur, fmt_kpi, fmt_res, fmt_col, fmt_obj, fmt_val
import graficos as gr


def _sombrear(celda, color):
    tcPr = celda._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), color)
    tcPr.append(shd)


ANCHO_UTIL_CM = 17.0          # A4 con márgenes de 2 cm
MAX_COLS_ESC = 5              # escenarios por tabla cuando las columnas son escenarios


def _cm_por_caracter(pt):
    return 0.0205 * pt          # anchura media aproximada de un carácter de Calibri


def _anchos(cab, filas, pt):
    """Anchura de cada columna según su contenido; se reparte el ancho útil sin superar la página."""
    nat = []
    for j, c in enumerate(cab):
        palabras = max((len(w) for w in str(c).split()), default=1) + 1      # cabecera en negrita
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

    def vinetas(self, items, numeradas=False):
        for it in items:
            self.doc.add_paragraph(it, style="List Number" if numeradas else "List Bullet")

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
        if total > ANCHO_UTIL_CM:
            # la primera columna (nombre de la fila) conserva su anchura natural para no partir palabras
            w0 = min(nat[0], 0.4 * ANCHO_UTIL_CM) if indice else 0.0
            resto = sum(nat[1:]) if indice else total
            f_ = (ANCHO_UTIL_CM - w0) / resto
            anchos = ([w0] + [w * f_ for w in nat[1:]]) if indice else [w * f_ for w in nat]
        else:
            anchos = [w * min(ANCHO_UTIL_CM / total, 1.6) for w in nat]
        t = self.doc.add_table(rows=1, cols=len(cab))
        t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        tblPr = t._tbl.tblPr
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
        trPr = t.rows[0]._tr.get_or_add_trPr()
        rep = OxmlElement("w:tblHeader"); rep.set(qn("w:val"), "true"); trPr.append(rep)
        for j, col in enumerate(t.columns):
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

    def tabla_escenarios(self, df, titulo, formato=None, nombre_indice=""):
        """Tabla cuyas columnas son escenarios: se parte en bloques de MAX_COLS_ESC columnas para que quepa."""
        cols = list(df.columns)
        if len(cols) <= MAX_COLS_ESC:
            self.tabla(df, titulo, formato, nombre_indice=nombre_indice)
            return
        n = math.ceil(len(cols) / MAX_COLS_ESC)
        tam = math.ceil(len(cols) / n)
        for k in range(n):
            self.tabla(df[cols[k * tam:(k + 1) * tam]], f"{titulo} ({k + 1}/{n})", formato, nombre_indice=nombre_indice)

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


def _sub(df, filas):
    return df.loc[[f for f in filas if f in df.index]]


def _txt(f, c, v):
    return v if isinstance(v, str) else _fmt(v)




# ---------------------------------------------------------------------------------------------- tablas
FILAS_A = ["Beneficio esperado E(Y) (k€)", "Desviación típica σ(Y) (k€)", "Probabilidad de pérdida P(Y < 0)",
           "VaR 95 % · percentil 5 de Y (k€)", "CVaR 5 % · media del 5 % peor (k€)",
           "Probabilidad de penalización P(R < umbral)"]
FILAS_B = ["VAN esperado con mantenimiento (k€)", "VAN esperado sin mantenimiento (k€)",
           "Ganancia media pareada E(Δ) (k€)", "P(el mantenimiento compensa) P(Δ > 0)",
           "VaR 95 % del VAN con mantenimiento (k€)", "VaR 95 % del VAN sin mantenimiento (k€)",
           "CVaR 5 % del VAN con mantenimiento (k€)", "CVaR 5 % del VAN sin mantenimiento (k€)",
           "σ(Δ) por centro · diseño pareado (k€)", "σ(Δ) por centro · diseño independiente (k€)"]


def tabla_tipos(t, resA):
    """O1.3 por tipo de proyecto: teórico y simulado (media de las réplicas)."""
    pf = t["A"]["por_tipo"]
    filas = {}
    for f in bd.PROYECTOS:
        nf = bd.NOMBRE[f]
        g = lambda k: resA.loc[k, "Simulado · media"]
        filas[nf] = {"Teórico · E(Y)": pf[f]["EY"], "Simulado · E(Y)": g(f"E(Y | {nf}) (k€)"),
                     "Teórico · CV": pf[f]["CV"], "Simulado · CV": g(f"CV(Y | {nf})"),
                     "Teórico · P(Y < 0)": pf[f]["Pperd"], "Simulado · P(Y < 0)": g(f"P(Y < 0 | {nf})"),
                     "Teórico · VaR 95 %": pf[f]["VaR"], "Teórico · peso en las pérdidas": pf[f]["peso_perd"]}
    return pd.DataFrame(filas).T


def fmt_tipos(fila, col, v):
    if col == "p" or "P(" in col or "peso" in col:
        return pct(v, 2)
    if "CV" in col:
        return num(v, 3)
    return keur(v)


def tabla_tipos_B(t, mcB):
    pt = t["B"]["por_tipo"]; rep = mcB["replicas"]
    filas = {}
    for f in bd.PROYECTOS:
        nf = bd.NOMBRE[f]
        filas[nf] = {"p": pt[f]["p"], "Teórico · E(VAN) con": pt[f]["E_con"],
                     "Teórico · E(Δ)": pt[f]["E_gan"], "Simulado · E(Δ)": rep[f"E(Δ | {nf})"].mean(),
                     "Teórico · P(Δ > 0)": pt[f]["P_gan"], "Ahorro medio S̄": pt[f]["ES"],
                     "C_mant de indiferencia": pt[f]["C_ind"]}
    return pd.DataFrame(filas).T


def fmt_tipos_B(fila, col, v):
    if col == "p" or "P(" in col or "Cuota" in col:
        return pct(v, 1)
    return keur(v, 2)


def tabla_extras_B(t, mcB, esc):
    B = t["B"]
    filas = {
        "Factores de descuento d_t": " / ".join(num(x, 3) for x in bd.descuentos(esc)),
        "Peso del ahorro perdido Σ d_t·(1 − δ^(t−1))": num(float((bd.descuentos(esc) * (1 - float(esc['delta']) ** np.arange(int(esc['H'])))).sum()), 3),
        "Coste de mantenimiento descontado C·Σ d_t (k€)": keur(bd.c_efectivo(esc) * bd.descuentos(esc).sum(), 2),
        "C_mant de indiferencia (O6.1)": f"{keur(B['C_ind'], 3)}/año (actual {keur(float(esc['C_mant']), 2)}/año)",
        "Subvención que iguala las políticas (O8.5)": pct(B["X_ind"], 1) if B["X_ind"] > 0 else "no hace falta (ya compensa)",
        "Reducción de varianza del diseño pareado (O4.1)": f"× {num(B['var_independiente'] / B['var_pareada'], 0)}" if B["var_pareada"] > 0 else "—",
        "P(VAN < 0) con mantenimiento": pct(B["Pperd_con"], 4),
    }
    return pd.DataFrame({"Valor": filas})


def tabla_parametros(res):
    filas = {}
    for n, r in res.items():
        e = r["esc"]
        filas[n] = {"p (IoT / IA / Emp. / Aud.)": " / ".join(num(x, 2) for x in bd.normalizar(e["p"])),
                    "V_stock (k€)": num(e["V"], 0),
                    "Penalización K / c / umbral": f"{num(e['K'], 0)} / {num(e['c_pen'], 0)} / {pct(e['umbral_R'], 0)}",
                    "Garantía (% de I)": pct(e["garantia"], 0),
                    "H / λ / δ": f"{e['H']} / {num(e['lam'], 2)} / {num(e['delta'], 2)}",
                    "C_mant (k€) / subvención": f"{num(e['C_mant'], 1)} / {pct(e.get('subvencion') or 0, 0)}",
                    "r / t₀": f"{pct(e['r'], 0)} / {e.get('t0', 1)}",
                    "Política": bd.POLITICAS.get(e.get("politica", "con"), "—")}
    return pd.DataFrame(filas)


def tabla_dif(dif, cols):
    return dif[[c for c in cols if c in dif.columns]]


COLS_DIF_A = ["ΔE(Y) por proyecto · teórico (k€)", "ΔE(Y) por proyecto · simulado (k€)", "IC95 ΔE(Y) (k€)",
              "ΔP(Y < 0) · teórico", "ΔVaR 95 % · teórico (k€)", "ΔCVaR 5 % · teórico (k€)"]
COLS_DIF_B = ["ΔE(VAN) política · teórico (k€)", "ΔE(VAN) política · simulado (k€)", "IC95 ΔE(VAN) (k€)",
              "ΔCVaR 5 % del VAN · teórico (k€)", "Ganancia del mantenimiento · teórico (k€)"]


def fmt_sens(fila, col, v):
    if col == "Error relativo":
        return pct(v, 3)
    return keur(v, 2)


def tabla_sens(df):
    return df.drop(columns=["Medias por réplica"])


def tabla_sens_param(df):
    d = df.copy()
    d["Variación"] = d["Variación"].map(lambda v: f"{'+' if v > 0 else ''}{num(100 * v, 0)} %")
    return d.set_index(["Parámetro", "Variación"])


def fmt_sens_param(fila, col, v):
    if "Var(Y)" in col:
        return num(v, 0)
    return keur(v)


def fmt_tabla(fila, col, v):
    if isinstance(v, (bool, np.bool_)):
        return "Sí" if v else "No"
    return fmt_val(col, v)


def indice_fmt(df, fn):
    out = df.copy(); out.index = [fn(i) for i in df.index]; return out


def tabla_racha(df):
    """Tabla compacta de la racha: R medio, S y VAN acumulado por política (con racha)."""
    cols = ["Sin racha · R medio", "Con racha · R medio", "Con racha · P(penalización)", "Con racha · S con mant. (k€)",
            "Con racha · S sin mant. (k€)", "Sin racha · VAN acum. con (k€)", "Con racha · VAN acum. con (k€)",
            "Con racha · VAN acum. sin (k€)"]
    d = df[cols].copy()
    d.columns = ["R sin racha", "R con racha", "P(penalización) con racha", "S con mant. (k€)", "S sin mant. (k€)",
                 "VAN acum. con · sin racha (k€)", "VAN acum. con · con racha (k€)", "VAN acum. sin · con racha (k€)"]
    return d


def fmt_racha(fila, col, v):
    if col.startswith("R ") or "P(" in col:
        return pct(v, 1)
    return keur(v, 1)


FMT_IDX = {"delta": lambda v: num(v, 2), "C_mant": lambda v: keur(v, 2), "H": lambda v: f"{int(v)} años",
           "subvencion": lambda v: pct(v, 0)}

TEXTO_O86 = ("Para completar la comparación haría falta información que el simulador no contiene: el valor residual o "
             "de desmantelamiento del proyecto antiguo, si el proyecto nuevo incorpora mejoras tecnológicas (otra "
             "distribución de R o S), los costes de transición e interrupción del servicio, la posible transferencia de "
             "la experiencia acumulada al proyecto nuevo y la evolución del coste de mantenimiento con la antigüedad.")


# ---------------------------------------------------------------------------------------------- informe
def generar_informe(res, com, aj, base, mant=None, sens=None, ampliacion=None, sens_est=None,
                    titulo="Rentabilidad de los proyectos de calidad en Bio-Distri"):
    inf = Informe(titulo, f"Informe generado automáticamente el {datetime.datetime.now():%d/%m/%Y %H:%M} · beneficio de un "
                          "proyecto por composición, programa plurianual con y sin mantenimiento, escenarios, "
                          "sensibilidad y ampliación · cantidades en miles de € (k€)")
    inf.h("1. Planteamiento y parámetros", 1)
    inf.p("Bloque A · un proyecto. Cada centro elige un tipo de proyecto T (IoT, IA, Empaque o Auditoría) y, condicionadas "
          "a T, se simulan de forma independiente la inversión I ~ LN(μ, σ), la reducción de merma R ~ Beta(α, β) y el "
          "ahorro operativo S ~ Gamma(k, θ) (método de composición, por transformada inversa). El beneficio neto es "
          "Y = V_stock·R + S − I − Pen(R), con Pen(R) = K·I{R < 0,10} (K = 20 k€ en el modelo actual).")
    dist = pd.DataFrame({bd.NOMBRE[f]: {
        "Inversión I (k€)": f"LN(μ = {num(com['dist'][f]['mu_I'], 2)}; σ = {num(com['dist'][f]['sigma_I'], 2)})",
        "Reducción R": f"Beta({num(com['dist'][f]['a_R'], 1)}; {num(com['dist'][f]['b_R'], 1)})",
        "Ahorro S (k€)": f"Gamma(k = {num(com['dist'][f]['k_S'], 1)}; θ = {num(com['dist'][f]['theta_S'], 2)})"}
        for f in bd.PROYECTOS})
    inf.tabla(dist, "Distribuciones condicionadas al tipo de proyecto", nombre_indice="Variable", formato=_txt)
    inf.p("Bloque B · programa plurianual. Cada centro mantiene su tipo de proyecto H años; cada año se generan de nuevo "
          "I_t, R⁰_t y S⁰_t. La experiencia mejora la reducción de merma, R_t = 1 − (1 − R⁰_t)·λ^(t−1), y, si no se "
          "reinvierte, el ahorro se degrada, S_t = S⁰_t·δ^(t−1); con mantenimiento S_t = S⁰_t a cambio de C_mant al año. "
          "VAN = Σ Y_t/(1 + r)^(t−1). Las dos políticas se aplican sobre la misma base de números aleatorios (pareada).")
    inf.tabla_escenarios(tabla_parametros(res), "Escenarios analizados", formato=_txt, nombre_indice="Parámetro")
    desc = [f"{n}: {r['esc']['descripcion']}" for n, r in res.items() if r["esc"].get("descripcion")]
    if desc:
        inf.vinetas(desc)
    ob = com["objetivos"]
    o = lambda k, f: f(ob[k]) if ob.get(k) is not None else "no se evalúa"
    tc = pd.DataFrame({"Valor": {
        "Objetivo · E(Y) mínimo por proyecto": o("EY_MIN", keur),
        "Objetivo · P(Y < 0) máxima": o("PPERD_MAX", lambda v: pct(v, 1)),
        "Objetivo · E(VAN) mínimo por centro (política elegida)": o("VAN_MIN", keur),
        "Objetivo · CVaR 5 % mínimo del VAN (política elegida)": o("CVAR_MIN", keur),
        "Simulación A · proyectos por réplica × réplicas": f"{num(aj['nsim'], 0)} × {aj['replicas']} (semillas {aj['semilla']} …)",
        "Simulación B · centros por réplica × réplicas": f"{num(aj['centros'], 0)} × {aj['replicas_B']} (semillas {aj['semilla'] + 1000} …)"}})
    inf.tabla(tc, "Objetivos y ajustes de simulación", nombre_indice="Parámetro", formato=_txt)

    # 2. Síntesis
    inf.h("2. Síntesis y recomendación (O7.1)", 1)
    for n, r in res.items():
        inf.p(bd.texto_escenario(n, r), negrita_inicio=n + ":")
    inf.p("Recomendación sobre la política de mantenimiento:", negrita_inicio="Recomendación sobre la política de mantenimiento:")
    inf.vinetas(bd.recomendaciones(res, base, com), numeradas=True)
    inf.tabla_escenarios(bd.tabla_comparativa(res), "Comparación de escenarios (filas «Teórico»: fórmulas exactas y "
                         "convolución; filas «Simulado»: media de las macro-réplicas)", formato=fmt_kpi, nombre_indice="KPI")
    inf.tabla_escenarios(bd.tabla_objetivos(res), "Cumplimiento de los objetivos (valores teóricos)",
                         nombre_indice="Objetivo", formato=_txt)
    inf.figura(gr.fig_comparativa(res), "Beneficio por proyecto, probabilidad de pérdida, VAN por política y ganancia del "
               "mantenimiento por escenario")
    dif = bd.diferencias(res, base)
    if dif is not None:
        inf.p(f"Comparación frente a «{base}» (O3.1–O3.4 y escenarios del programa). Las diferencias simuladas son "
              "pareadas (mismas semillas en todos los escenarios), por lo que su IC95 % es mucho más estrecho que el de "
              "cada valor por separado. Positivo = mejora frente a la base.")
        inf.tabla(tabla_dif(dif, COLS_DIF_A), f"Beneficio y riesgo de un proyecto frente a «{base}»", formato=fmt_col,
                  nombre_indice="Escenario")
        inf.tabla(tabla_dif(dif, COLS_DIF_B), f"VAN del programa (política elegida) frente a «{base}»", formato=fmt_col,
                  nombre_indice="Escenario")
        inf.figura(gr.fig_diferencias(dif), f"Diferencia de E(VAN) frente a «{base}»: teórica y simulada (IC95 % pareado)")
    inf.figura(gr.fig_validacion(res), "Error estandarizado de la simulación frente a los valores teóricos (en unidades "
               "de error estándar; fuera de ±2 equivale a quedar fuera del IC95 %)")
    inf.h("Cuadro de mando propuesto", 2)
    inf.tabla(pd.DataFrame({"Para qué sirve": dict(bd.KPIS_CUADRO)}), "Cuadro de mando para el comité de dirección",
              nombre_indice="KPI", formato=_txt)

    sec = 3
    for n, r in res.items():
        inf.salto()
        t = r["teoria"]
        inf.h(f"{sec}. Escenario «{n}»", 1)
        if r["esc"].get("descripcion"):
            inf.p(r["esc"]["descripcion"])
        inf.h(f"{sec}.1 Bloque A · beneficio de un proyecto (O1 y O2.1)", 2)
        inf.tabla(_sub(r["resA"], FILAS_A), f"Beneficio de un proyecto: teórico y simulado con IC95 % · {n}",
                  formato=fmt_res, nombre_indice="KPI")
        inf.p(f"El VaR 95 % ({keur(t['A']['VaR'])}) es el umbral que solo el 5 % de los proyectos no alcanza; el CVaR 5 % "
              f"({keur(t['A']['CVaR'])}) es el beneficio medio de ese 5 % peor y, además del umbral, mide la gravedad de la "
              "cola: con un VaR positivo, un CVaR cercano a 0 o negativo indica que en los peores casos se pierde dinero.")
        inf.tabla(tabla_tipos(t, r["resA"]), f"Beneficio por tipo de proyecto (O1.3) · {n}", formato=fmt_tipos,
                  nombre_indice="Tipo")
        inf.figura(gr.fig_hist_Y(t["A"], r["mcA"]["Y"], f"Distribución del beneficio de un proyecto · {n}"),
                   f"Beneficio simulado frente a la distribución teórica, VaR y CVaR · {n}")
        inf.figura(gr.fig_por_tipo(t["A"], r["resA"], f"Beneficio por tipo de proyecto · {n}"),
                   f"Media, probabilidad de pérdida y variabilidad relativa por tipo · {n}")
        filas_d = [f"Contribución a Var(Y) · {x}" for x in bd.TERMINOS if abs(t["A"]["contrib"][bd.TERMINOS.index(x)]) > 1e-12]
        inf.tabla(_sub(r["resA"], filas_d), f"Descomposición de la varianza de Y (O2.1) · {n}", formato=fmt_res,
                  nombre_indice="Término")
        inf.figura(gr.fig_descomposicion(t["A"], r["resA"], f"Descomposición de Var(Y) · {n}"),
                   f"Parte de la varianza del beneficio atribuible a cada término · {n}")
        inf.h(f"{sec}.2 Bloque B · programa de {r['esc']['H']} años con y sin mantenimiento (O4 y O5.3)", 2)
        inf.tabla(_sub(r["resB"], FILAS_B), f"VAN por centro y ganancia del mantenimiento: teórico y simulado · {n}",
                  formato=fmt_res, nombre_indice="KPI")
        inf.tabla(tabla_tipos_B(t, r["mcB"]), f"Ganancia del mantenimiento por tipo de proyecto (O5.3) · {n}",
                  formato=fmt_tipos_B, nombre_indice="Tipo")
        inf.figura(gr.fig_hist_van(t["B"], r["mcB"], f"VAN por centro · {n}"),
                   f"VAN simulado de los centros con y sin mantenimiento frente a la distribución teórica · {n}")
        inf.figura(gr.fig_ganancia(t["B"], r["mcB"], f"Ganancia del mantenimiento · {n}"),
                   f"Distribución de Δ = VAN_con − VAN_sin y ganancia media por tipo · {n}")
        inf.tabla(tabla_extras_B(t, r["mcB"], r["esc"]), f"Descuento, coste de indiferencia y diseño pareado · {n}",
                  nombre_indice="Indicador", formato=_txt)
        inf.h(f"{sec}.3 Funcionamiento simulado", 2)
        pr = bd.un_proyecto(com, r["esc"], 0.5, 0.5, 0.5, 0.5)
        inf.figura(gr.fig_mecanismo(t["A"]["p"], 0.5, pr, f"Composición · un proyecto · {n}"),
                   f"Tipo elegido con U_T = 0,5 y beneficio por términos con U_I = U_R = U_S = 0,5 · {n}")
        inf.figura(gr.fig_traza_A(r["mcA"]["traza"], f"Primeros {len(r['mcA']['traza'])} proyectos · {n}"),
                   f"Beneficio y tipo de los primeros proyectos simulados (▼ = penalización) · {n}")
        inf.figura(gr.fig_ejemplo_B(r["mcB"]["ejemplo"], r["esc"], f"Programa simulado · {n}"),
                   f"Ahorro y beneficio por año en los primeros centros simulados · {n}")
        sec += 1

    if mant is not None:
        inf.salto()
        inf.h(f"{sec}. ¿De qué depende el valor del mantenimiento? (O5.1, O5.2, O6.1–O6.3, O8.5) · «{mant['escenario']}»", 1)
        inf.p("Δ = VAN_con − VAN_sin = Σ d_t·S⁰_t·(1 − δ^(t−1)) − C_mant·Σ d_t no depende de R ni de I: el mantenimiento "
              "solo protege el ahorro operativo. Por eso su valor crece con el ahorro medio del tipo de proyecto, con "
              "la severidad de la degradación (δ menor) y con el horizonte, y decrece con C_mant.")
        textos = {"delta": "O5.1 · tasa de degradación δ", "C_mant": "O5.2 y O6.1 · coste de mantenimiento",
                  "H": "O6.3 · horizonte del programa", "subvencion": "O8.5 · subvención del mantenimiento"}
        for par, df in mant["barridos"].items():
            inf.h(textos.get(par, par), 2)
            if par == "C_mant":
                inf.p(f"El coste de indiferencia es C* = {keur(mant['C_ind'], 3)}/año para la red (por tipo: "
                      + ", ".join(f"{bd.NOMBRE[f]} {keur(v, 2)}" for f, v in mant["C_ind_tipo"].items())
                      + "). Por encima de C* no compensa mantener en promedio; por debajo, sí.")
            if par == "H":
                inf.p("Con horizontes más largos los últimos años, en los que el ahorro sin mantenimiento está más degradado, "
                      "pesan más: la ganancia del mantenimiento aumenta con H aunque el descuento atenúa los años lejanos.")
            if par == "subvencion":
                inf.p(f"La subvención que iguala ambas políticas es X* = {pct(mant['X_ind'], 1)} del coste de mantenimiento.")
            dfi = indice_fmt(df[[c for c in df.columns if "E(Δ |" not in c]], FMT_IDX[par])
            inf.tabla(dfi, f"Ganancia del mantenimiento según {bd.PARAM_BARRIDO[par].split(' (')[0].lower()}",
                      formato=fmt_tabla, nombre_indice=bd.PARAM_BARRIDO[par])
            inf.figura(gr.fig_barrido(df), f"Ganancia del mantenimiento y P(Δ > 0) según {bd.PARAM_BARRIDO[par].split(' (')[0].lower()}")
        if mant.get("racha") is not None:
            df, cfg = mant["racha"]
            inf.h("O6.2 · racha de mala suerte", 2)
            inf.p(f"Centros de tipo {bd.NOMBRE[cfg['tipo']]} con R⁰ en el 10 % inferior de su distribución durante "
                  f"{cfg['anyos']} años desde el año {cfg['inicio']} (mismos números aleatorios en el resto). La racha reduce R "
                  "y el VAN acumulado; las penalizaciones solo pueden aparecer el primer año, porque el aprendizaje garantiza "
                  "R_t ≥ 1 − λ^(t−1) (con λ = 0,80, R_t ≥ 20 % desde el segundo año). El ahorro S no cambia (es "
                  "independiente de R), así que la ganancia del mantenimiento es la misma con o sin racha: el mantenimiento "
                  "no protege frente a este riesgo. Tras la racha, la curva de aprendizaje devuelve R a su senda.")
            inf.tabla(tabla_racha(df), "Evolución año a año con y sin racha (simulado)", formato=fmt_racha, nombre_indice="Año")
            inf.figura(gr.fig_racha(df), "R medio, ahorro S y VAN acumulado con y sin racha")
        sec += 1

    if sens is not None:
        inf.salto()
        inf.h(f"{sec}. Sensibilidad (O2.2 y O3.3) · escenario «{sens['escenario']}»", 1)
        df, elas = sens["param"]
        mayor = max(elas, key=lambda k: abs(elas[k]))
        inf.p("O2.2 · Se varía cada parámetro manteniendo fijos los demás. La elasticidad es la variación porcentual de "
              f"Var(Y) por cada 1 % de variación del parámetro; la mayor es la de «{mayor}» ({num(elas[mayor], 2)}). "
              "Los índices de Sobol confirman que la reducción de merma R y el tipo de proyecto T explican casi toda la "
              "varianza: los esfuerzos deben centrarse en la eficacia de la reducción de merma.")
        inf.figura(gr.fig_sens_param(df, elas), "Var(Y) al variar cada parámetro y elasticidad en el punto base")
        inf.tabla(pd.DataFrame({"Elasticidad de Var(Y)": elas}), "Elasticidad de Var(Y) por parámetro (teórico)",
                  formato=lambda f, c, v: num(v, 3), nombre_indice="Parámetro")
        inf.tabla(sens["sobol"], "Índices de Sobol de las entradas aleatorias (simulados; S₁ del tipo también teórico)",
                  formato=lambda f, c, v: pct(v, 1), nombre_indice="Entrada")
        inf.figura(gr.fig_sobol(sens["sobol"]), "Índices de Sobol de primer orden y totales")
        dv, ey3, er3 = sens["veq"]
        inf.p(f"O3.3 · E(Y | Empaque) = {keur(ey3)} es lineal en V_stock con pendiente E(R | Empaque) = {num(er3, 4)}. Con "
              f"V_stock = {keur(sens['V'], 0)} el Empaque ya tiene E(Y) > 0: el umbral de rentabilidad es "
              f"V_stock = {keur(dv.iloc[0]['V_stock necesario (k€)'])}. Para competir con los demás proyectos haría falta "
              "un V_stock mucho mayor (tabla).")
        inf.tabla(dv, "V_stock necesario para el Empaque según la meta (teórico)", formato=fmt_tabla, nombre_indice="Meta")
        inf.figura(gr.fig_vequilibrio(dv, sens["V"]), "V_stock que necesita el Empaque para cada meta")
        sec += 1

    if ampliacion is not None:
        A = ampliacion
        inf.salto()
        inf.h(f"{sec}. Ampliación optativa (O8) · escenario «{A['escenario']}»", 1)
        inf.p("O8.1 · Contraste H₀: E(Δ) = 0 frente a H₁: E(Δ) ≠ 0 con la t de Student sobre las diferencias pareadas de "
              "todos los centros simulados (en la red y por tipo de proyecto).")
        inf.tabla(A["contraste"], "Contraste de la ganancia media del mantenimiento", formato=fmt_tabla, nombre_indice="Grupo")
        inf.p("O8.2 · Empezar con experiencia previa (t₀ = 3) adelanta la curva de aprendizaje: R es mayor desde el primer "
              "año y el VAN esperado aumenta de forma apreciable; la ganancia del mantenimiento no cambia porque no depende de R.")
        inf.tabla(A["experiencia"], "VAN esperado según el punto de partida de la curva de aprendizaje", formato=fmt_tabla,
                  nombre_indice="Partida")
        da, ec, es = A["adaptativa"]
        mejor = da["Simulado · E(VAN) (k€)"].idxmax()
        inf.p(f"O8.3 · Política adaptativa: el centro mantiene el año t solo si el ahorro que observaría sin mantener, "
              f"S⁰_t·δ^n (n = años desde el último mantenimiento), cae por debajo de un umbral. El mejor umbral del barrido "
              f"es {keur(mejor, 1)}, con E(VAN) = {keur(da.loc[mejor, 'Simulado · E(VAN) (k€)'])} frente a "
              f"{keur(ec)} (siempre) y {keur(es)} (nunca). Como el mantenimiento apenas cambia el VAN frente al valor de la "
              "merma evitada, todas las políticas quedan muy próximas; un umbral óptimo bajo indica que conviene mantener "
              "casi nunca, y uno alto, casi siempre.")
        inf.tabla(indice_fmt(da, lambda v: keur(v, 1)), "Política adaptativa según el umbral de ahorro (simulado)", formato=fmt_tabla, nombre_indice="Umbral")
        inf.figura(gr.fig_adaptativa(da, ec, es), "E(VAN) de la política adaptativa frente a las políticas fijas")
        db, dv = A["bayes"]
        inf.p("O8.4 · Actualización bayesiana: cada centro tiene su propia λ (a priori uniforme). R₁ no informa sobre λ, pero "
              "R₂ = 1 − (1 − R⁰₂)·λ sí: con la verosimilitud exacta de R₂ se obtiene la distribución a posteriori de λ en "
              "una rejilla. Para decidir si continuar se compara el VAN esperado de los años restantes con la λ a posteriori.")
        inf.tabla(db, "Información que aportan los dos primeros años sobre λ (simulado)", formato=fmt_tabla, nombre_indice="Tipo")
        inf.tabla(indice_fmt(dv, lambda v: num(v, 3)), "VAN esperado de los años restantes según la λ estimada (teórico)", formato=fmt_tabla,
                  nombre_indice="λ supuesta")
        inf.p("O8.6 · Sustituir el proyecto a mitad del horizonte obliga a pagar otra inversión y reinicia la curva de "
              "aprendizaje, por lo que R vuelve a valores bajos justo cuando el proyecto original ya evitaba casi toda la "
              "merma: en este modelo la sustitución destruye valor. " + TEXTO_O86)
        inf.tabla(A["sustitucion"], "Mantener frente a sustituir a mitad del horizonte (simulado)", formato=fmt_tabla,
                  nombre_indice="Alternativa")
        sec += 1

    if sens_est is not None:
        df, cfg = sens_est["df"], sens_est["cfg"]
        inf.salto()
        inf.h(f"{sec}. Sensibilidad del estimador Monte Carlo", 1)
        inf.p(sens_est["texto"])
        inf.figura(gr.fig_sens_convergencia(df, cfg["metrica"]), "Convergencia de la estimación hacia el valor teórico")
        inf.figura(gr.fig_sens_variabilidad(df, cfg["metrica"]), "Variabilidad entre réplicas y σ teórica del estimador")
        inf.tabla(tabla_sens(df), "Barrido del tamaño de la simulación", formato=fmt_sens, nombre_indice="n")
        sec += 1

    inf.salto()
    inf.h("Anexo. Metodología y supuestos", 1)
    inf.vinetas([
        "Composición: T por transformada inversa sobre (p₁, …, p₄); I = exp(μ + σ·Φ⁻¹(U)), R = F⁻¹_Beta(U) y "
        "S = θ·F⁻¹_Gamma(U). Cada proyecto usa 4 uniformes, de modo que todos los escenarios comparten los mismos números "
        "aleatorios (diferencias pareadas).",
        "Momentos exactos: E y Var de cada término con la Beta truncada (E[R·I{R < u}] = μ_R·F_{α+1,β}(u), …), la "
        "lognormal y la Gamma; la contribución de cada término es Cov(término, Y)/Var(Y). En el programa, R_t < u ⇔ "
        "R⁰ < 1 − (1 − u)/λ^(t−1), de modo que las mismas fórmulas valen cada año.",
        "Distribución exacta en rejilla: cada término se discretiza con su función de distribución (paso 0,05 k€ para Y "
        "y 0,5 k€ para el VAN) y se convoluciona por FFT; de ahí salen P(Y < 0), el VaR (percentil 5) y el CVaR 5 %. La "
        "ganancia Δ es una suma de Gammas escaladas, también convolucionadas, para P(Δ > 0).",
        "Macro-réplicas: cada KPI se calcula en cada réplica y se da la media de las réplicas con IC95 % de la t de "
        "Student. El VaR y el CVaR de cada réplica son el percentil 5 y la media del 5 % peor de su muestra.",
        "Supuestos propios: penalización continua c·(0,10 − R)⁺ con c = 200 k€ (la multa máxima coincide con los 20 k€ "
        "del escalón); concentrar la cartera en IoT e IA respetando su proporción actual (58 % / 42 %); en la política "
        "adaptativa, el mantenimiento restaura el ahorro y reinicia la degradación; en la sustitución, el proyecto nuevo "
        "se mantiene y la experiencia no se transfiere. Todos se pueden cambiar en la aplicación.",
        "Material de Javier Morales, equipo IA4LEGOS (CC BY-SA 4.0).",
    ])
    return inf.guardar()
