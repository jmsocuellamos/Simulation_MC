# -*- coding: utf-8 -*-
"""Informe Word (.docx) del estudio de caso Eco-Flow: beneficio diario de la micro-red, batería de almacenamiento,
escenarios, dimensionamiento, sensibilidad y ampliación."""
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

import ecoflow as ef
from ecoflow import num, pct, eur, fmt_kpi, fmt_res, fmt_col, fmt_obj, fmt_val
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
FILAS_A = ["Beneficio diario esperado E(Y) (€)", "Desviación típica σ(Y) (€)", "Probabilidad de déficit P(B < 0)",
           "Probabilidad de comprar en el mercado", "VaR 95 % · percentil 5 de Y (€)", "CVaR 5 % · media del 5 % peor (€)",
           "Déficit medio comprado (MWh)", "Generación media E(G) (MWh)"]
FILAS_B = ["Beneficio diario esperado sin batería (€)", "Beneficio diario esperado con batería (€)",
           "Ganancia media diaria pareada E(Δ) (€)", "P(comprar en el mercado) sin batería",
           "P(comprar en el mercado) con batería", "Déficits cubiertos por completo por la batería",
           "VaR 95 % diario sin batería (€)", "VaR 95 % diario con batería (€)", "CVaR 5 % diario sin batería (€)",
           "CVaR 5 % diario con batería (€)", "Estado de carga medio (MWh)", "LOLE sin batería (días/año)",
           "LOLE con batería (días/año)", "σ(Δ) diario (€)"]


def tabla_tipos(t, resA):
    pf = t["A"]["por_tipo"]
    filas = {}
    for f in ef.ESCENARIOS:
        nf = ef.NOMBRE[f]
        g = lambda k: resA.loc[k, "Simulado · media"]
        filas[nf] = {"Teórico · E(Y)": pf[f]["EY"], "Simulado · E(Y)": g(f"E(Y | {nf}) (€)"),
                     "Teórico · CV": pf[f]["CV"], "Simulado · CV": g(f"CV(Y | {nf})"),
                     "Teórico · P(B < 0)": pf[f]["Pdef"], "Simulado · P(B < 0)": g(f"P(B < 0 | {nf})"),
                     "Teórico · VaR 95 %": pf[f]["VaR"], "Teórico · peso en la cola 5 %": pf[f]["peso_cola"]}
    return pd.DataFrame(filas).T


def fmt_tipos(fila, col, v):
    if "P(" in col or "peso" in col:
        return pct(v, 1)
    if "CV" in col:
        return num(v, 3)
    return eur(v)


def tabla_fuentes(com, esc):
    filas = {}
    for f in ef.ESCENARIOS:
        m = ef.momentos_fuentes(com, esc, f)
        filas[ef.NOMBRE[f]] = {"Solar E / σ": f"{num(m['Gs'][0], 1)} / {num(m['Gs'][1], 1)}",
                               "Eólica E / σ": f"{num(m['Gw'][0], 1)} / {num(m['Gw'][1], 1)}",
                               "Hidráulica E / σ": f"{num(m['Gh'][0], 1)} / {num(m['Gh'][1], 1)}",
                               "Demanda E / σ": f"{num(m['D'][0], 1)} / {num(m['D'][1], 1)}",
                               "Balance medio E(B)": num(m["Gs"][0] + m["Gw"][0] + m["Gh"][0] - m["D"][0], 1)}
    return pd.DataFrame(filas).T


def tabla_tipos_B(t, mcB):
    pt = t["B"]["por_tipo"]; rep = mcB["replicas"]
    filas = {}
    for f in ef.ESCENARIOS:
        nf = ef.NOMBRE[f]
        filas[nf] = {"p": pt[f]["p"], "Teórico · E(Δ) (€/día)": pt[f]["E_gan"], "Simulado · E(Δ) (€/día)": rep[f"E(Δ | {nf})"].mean(),
                     "Cuota de la ganancia": pt[f]["cuota_gan"], "P(B < 0)": pt[f]["Pdef"],
                     "Teórico · P(compra) con batería": pt[f]["Pspot_con"], "Déficit medio (MWh)": pt[f]["Edef"],
                     "Superávit medio (MWh)": pt[f]["Esup"]}
    return pd.DataFrame(filas).T


def fmt_tipos_B(fila, col, v):
    if col == "p" or "P(" in col or "Cuota" in col:
        return pct(v, 1)
    if "MWh" in col:
        return num(v, 2)
    return eur(v, 1)


def tabla_extras_B(t, mcB, esc):
    B = t["B"]
    filas = {
        "Ganancia anual de la batería (365 × E(Δ))": eur(B["gan_anual"]),
        "Amortización anual de la batería": f"{eur(B['amortizacion'])} ({eur(float(esc['amort']))} por MWh y año)",
        "Ingreso por servicios de red (O8.5)": eur(B["ingreso_red"]),
        "Ganancia neta anual": eur(B["neto_anual"]),
        "Tarifa de servicios de red para cubrir la amortización (O8.5)": (f"{eur(max(B['tarifa_equilibrio'], 0), 2)} por MWh y día"
                                                                          + (" (ya es rentable sin tarifa)" if B["tarifa_equilibrio"] <= 0 else "")),
        "Tarifa que cubre la amortización por sí sola": f"{eur(B['tarifa_sola'], 2)} por MWh y día",
        "Días al año que la batería evita comprar en el mercado (O6.3)": num(B["LOLE_sin"] - B["LOLE_con"], 1),
        "σ(Δ) diario · diseño pareado (O4.1)": f"{eur(mcB['sd_par'])} simulado · {eur(B['sd_gan'])} teórico",
        "σ(Y_con − Y_sin) · diseño independiente": eur(mcB["sd_ind"]) + " simulado",
        "Reducción de varianza del diseño pareado": f"× {num((mcB['sd_ind'] / mcB['sd_par']) ** 2, 0)}" if mcB["sd_par"] > 0 else "—",
    }
    return pd.DataFrame({"Valor": filas})


def tabla_parametros(res):
    filas = {}
    for n, r in res.items():
        e = r["esc"]
        filas[n] = {"p (Est. / Fro. / Mon. / Ant.)": " / ".join(num(x, 2) for x in ef.normalizar(e["p"])),
                    "P_exp / P_pen (€/MWh)": f"{num(e['P_exp'], 0)} / {num(e['P_pen'], 0)}" + (f" (CV {pct(e['cv_pen'], 0)})" if float(e.get('cv_pen') or 0) > 0 else ""),
                    "C_fijo (€) / c (€/MWh)": f"{num(e['C_fijo'], 0)} / {num(e['c_var'], 1)}",
                    "Solar ×factor / +MWh": f"{num(e['solar_factor'], 2)} / {num(e['solar_extra'], 1)}",
                    "Contrato (€/MWh; prima €/día)": f"{num(e['P_contrato'], 0)}; {num(e['prima'], 0)}" if e.get("contrato") else "no",
                    "Respuesta de demanda": f"−{pct(e['dr_red'], 0)} si B < −{num(e['dr_umbral'], 0)} ({num(e['dr_desc'], 0)} €/MWh)" if float(e["dr_red"]) > 0 else "no",
                    "Batería Cap / η / E₁": f"{num(e['cap'], 0)} / {num(e['eta'], 2)} / {pct(e['E1'], 0)}",
                    "Amortización (€/MWh·año) / tarifa (€/MWh·día)": f"{num(e['amort'], 0)} / {num(e.get('tarifa') or 0, 2)}"}
    return pd.DataFrame(filas)


def tabla_dif(dif, cols):
    return dif[[c for c in cols if c in dif.columns]]


COLS_DIF_A = ["ΔE(Y) diario sin batería · teórico (€)", "ΔE(Y) diario sin batería · simulado (€)", "IC95 ΔE(Y) (€)",
              "ΔP(B < 0) · teórico", "ΔVaR 95 % · teórico (€)", "ΔCVaR 5 % · teórico (€)"]
COLS_DIF_B = ["ΔE(Y) diario con batería · teórico (€)", "ΔE(Y) diario con batería · simulado (€)",
              "IC95 ΔE(Y) con batería (€)", "Ganancia de la batería · teórico (€/día)",
              "Δ ganancia neta anual de la batería · teórico (€)"]


def fmt_sens(fila, col, v):
    if col == "Error relativo":
        return pct(v, 3)
    return eur(v, 1)


def tabla_sens(df):
    return df.drop(columns=["Medias por réplica"])


def tabla_sens_param(df):
    d = df.copy()
    d["Variación"] = d["Variación"].map(lambda v: f"{'+' if v > 0 else ''}{num(100 * v, 0)} %")
    return d.set_index(["Parámetro", "Variación"])


def fmt_sens_param(fila, col, v):
    if "Var(Y)" in col:
        return num(v / 1e6, 2) + " M"
    return eur(v)


def fmt_tabla(fila, col, v):
    return fmt_val(col, v)


def indice_fmt(df, fn):
    out = df.copy(); out.index = [fn(i) for i in df.index]; return out


FMT_IDX = {"cap": lambda v: f"{num(v, 0)} MWh", "eta": lambda v: num(v, 2)}

TEXTO_O83 = ("O8.3 · Gestión activa: con un precio de exportación variable, la regla reactiva se amplía con una decisión "
             "más en cada día: si el precio del día supera un umbral, además del excedente se vende la energía almacenada "
             "(E·η). En simular_bateria basta con pasar la serie de precios y el umbral: la venta extra se suma al ingreso y "
             "el estado de carga se pone a 0. Con déficits tan frecuentes, la energía guardada vale casi siempre P_pen, de "
             "modo que solo compensa venderla a precios cercanos o superiores a P_pen.")
TEXTO_O84 = ("O8.4 · Valor de la información meteorológica: se simularía, con la misma base climática, una política que "
             "conoce el escenario de mañana (por ejemplo, reservar energía o venderla según si mañana será deficitario) y "
             "otra que no; la diferencia media de beneficio es el valor de la información perfecta, una cota superior de lo "
             "que merece la pena pagar por una predicción. Con una predicción imperfecta se simularía además el error de "
             "predicción (matriz de confusión del escenario). Con la regla actual (cargar primero, vender después) la "
             "información solo cambia las decisiones de vender lo almacenado, por lo que su valor es pequeño.")
TEXTO_O86 = ("Para completar la comparación haría falta información que el simulador no contiene: el coste de capital y la "
             "vida útil reales de la batería y de los paneles, la degradación de la batería con los ciclos, el perfil horario "
             "de generación y demanda (la batería también arbitra dentro del día), la correlación entre la nueva solar y el "
             "escenario climático (la solar extra rinde poco en los días Anticiclónicos, justo cuando más falta), las "
             "restricciones de terreno y de conexión a la red y la evolución futura de los precios.")


# ---------------------------------------------------------------------------------------------- informe
def generar_informe(res, com, aj, base, bat=None, sens=None, ampliacion=None, sens_est=None,
                    titulo="Gestión de riesgos en la micro-red de Eco-Flow Renewables"):
    inf = Informe(titulo, f"Informe generado automáticamente el {datetime.datetime.now():%d/%m/%Y %H:%M} · beneficio diario por "
                          "composición, batería de almacenamiento, escenarios, dimensionamiento, sensibilidad y ampliación")
    inf.h("1. Planteamiento y parámetros", 1)
    inf.p("Bloque A · un día. El escenario climático T (Estival, Frontal, Monzónico o Anticiclónico) se elige por "
          "transformada inversa y, condicionadas a T, se simulan de forma independiente la generación solar "
          "G_s ~ Tri(a, m, b), eólica G_w ~ Weibull(k, λ), hidráulica G_h ~ LN(μ, σ) y la demanda D ~ N(μ, σ). Con "
          "B = G − D, el beneficio del día es Y = P_exp·B⁺ − P_pen·B⁻ − C_fijo − c·G.")
    dist = pd.DataFrame({ef.NOMBRE[f]: {
        "Solar G_s (MWh)": "Tri(" + "; ".join(num(x, 0) for x in com["dist"][f]["tri"]) + ")",
        "Eólica G_w (MWh)": f"Weibull(k = {num(com['dist'][f]['weib'][0], 1)}; λ = {num(com['dist'][f]['weib'][1], 0)})",
        "Hidráulica G_h (MWh)": f"LN(μ = {num(com['dist'][f]['ln'][0], 1)}; σ = {num(com['dist'][f]['ln'][1], 2)})",
        "Demanda D (MWh)": f"N({num(com['dist'][f]['norm'][0], 0)}; {num(com['dist'][f]['norm'][1], 0)})"} for f in ef.ESCENARIOS})
    inf.tabla(dist, "Distribuciones condicionadas al escenario climático", nombre_indice="Variable", formato=_txt)
    inf.tabla(tabla_fuentes(com, res[base]["esc"] if base in res else ef.ESC_BASE),
              "Medias y desviaciones típicas teóricas por escenario (MWh)", nombre_indice="Escenario", formato=_txt)
    inf.p("Bloque B · batería. Cada día la batería se carga con el superávit (hasta su capacidad) y cubre el déficit con "
          "E·η; lo que no cubre se compra en el mercado. El estado de carga se arrastra de un día al siguiente durante "
          "365 días (E₁ dado). Ambas políticas, con y sin batería, se aplican sobre la misma base climática (pareada). "
          "Nota: el enunciado da tres valores de η (0,90 en el texto, 0,98 en la tabla y 0,95 en el código); se toma "
          "η = 0,98 como referencia, la del objetivo O5.2.")
    inf.tabla_escenarios(tabla_parametros(res), "Escenarios analizados", formato=_txt, nombre_indice="Parámetro")
    desc = [f"{n}: {r['esc']['descripcion']}" for n, r in res.items() if r["esc"].get("descripcion")]
    if desc:
        inf.vinetas(desc)
    ob = com["objetivos"]
    o = lambda k, f: f(ob[k]) if ob.get(k) is not None else "no se evalúa"
    tc = pd.DataFrame({"Valor": {
        "Objetivo · E(Y) diario mínimo con batería": o("EY_MIN", eur),
        "Objetivo · P(compra en el mercado) máxima con batería": o("PDEF_MAX", lambda v: pct(v, 1)),
        "Objetivo · CVaR 5 % diario mínimo (sin batería)": o("CVAR_MIN", eur),
        "Objetivo · ganancia neta anual mínima de la batería": o("NETO_MIN", eur),
        "Simulación A · días por réplica × réplicas": f"{num(aj['nsim'], 0)} × {aj['replicas']} (semillas {aj['semilla']} …)",
        "Simulación B · años de 365 días": f"{num(aj['anyos'], 0)} (semillas {aj['semilla'] + 1000} …; {num(aj['anyos'] * 365, 0)} días)"}})
    inf.tabla(tc, "Objetivos y ajustes de simulación", nombre_indice="Parámetro", formato=_txt)

    inf.h("2. Síntesis y recomendación (O7.1)", 1)
    for n, r in res.items():
        inf.p(ef.texto_escenario(n, r), negrita_inicio=n + ":")
    inf.p("Recomendación de inversión:", negrita_inicio="Recomendación de inversión:")
    inf.vinetas(ef.recomendaciones(res, base, com, bat["cap"] if bat else None), numeradas=True)
    inf.tabla_escenarios(ef.tabla_comparativa(res), "Comparación de escenarios (filas «Teórico»: fórmulas exactas, convolución "
                         "y cadena de Markov; filas «Simulado»: Monte Carlo)", formato=fmt_kpi, nombre_indice="KPI")
    inf.tabla_escenarios(ef.tabla_objetivos(res), "Cumplimiento de los objetivos (valores teóricos)", nombre_indice="Objetivo",
                         formato=_txt)
    inf.figura(gr.fig_comparativa(res), "Beneficio diario, días de déficit y de compra, CVaR y ganancia neta de la batería por escenario")
    dif = ef.diferencias(res, base)
    if dif is not None:
        inf.p(f"Comparación frente a «{base}» (O3.1–O3.4 y variantes de la batería). Las diferencias simuladas son pareadas "
              "(mismas semillas en todos los escenarios). Positivo = mejora frente a la base.")
        inf.tabla(tabla_dif(dif, COLS_DIF_A), f"Beneficio y riesgo diarios sin batería frente a «{base}»", formato=fmt_col,
                  nombre_indice="Escenario")
        inf.tabla(tabla_dif(dif, COLS_DIF_B), f"Beneficio con batería y rentabilidad de la batería frente a «{base}»",
                  formato=fmt_col, nombre_indice="Escenario")
        inf.figura(gr.fig_diferencias(dif), f"Diferencia de E(Y) diario con batería frente a «{base}»")
    inf.figura(gr.fig_validacion(res), "Error estandarizado de la simulación frente a los valores teóricos (fuera de ±2 "
               "equivale a quedar fuera del IC95 %)")
    inf.h("Cuadro de mando propuesto", 2)
    inf.tabla(pd.DataFrame({"Para qué sirve": dict(ef.KPIS_CUADRO)}), "Cuadro de mando para el comité de dirección",
              nombre_indice="KPI", formato=_txt)

    sec = 3
    for n, r in res.items():
        inf.salto()
        t = r["teoria"]
        inf.h(f"{sec}. Escenario «{n}»", 1)
        if r["esc"].get("descripcion"):
            inf.p(r["esc"]["descripcion"])
        inf.h(f"{sec}.1 Bloque A · beneficio de un día (O1 y O2.1)", 2)
        inf.tabla(_sub(r["resA"], FILAS_A), f"Beneficio diario: teórico y simulado con IC95 % · {n}", formato=fmt_res,
                  nombre_indice="KPI")
        inf.p(f"El VaR 95 % ({eur(t['A']['VaR'])}) es el beneficio que solo el 5 % de los días no alcanza; el CVaR 5 % "
              f"({eur(t['A']['CVaR'])}) es la media de esos días y mide además cuánto se pierde cuando se cruza el umbral. "
              "La cola está formada casi por completo por días Anticiclónicos.")
        inf.tabla(tabla_tipos(t, r["resA"]), f"Beneficio por escenario climático (O1.3) · {n}", formato=fmt_tipos,
                  nombre_indice="Escenario")
        inf.figura(gr.fig_hist_Y(t["A"], r["mcA"]["Y"], f"Distribución del beneficio diario · {n}"),
                   f"Beneficio simulado frente a la distribución teórica, VaR y CVaR · {n}")
        inf.figura(gr.fig_por_tipo(t["A"], r["resA"], f"Beneficio por escenario climático · {n}"),
                   f"Media, probabilidad de déficit y variabilidad relativa por escenario · {n}")
        filas_d = [f"Contribución a Var(Y) · {x}" for i, x in enumerate(ef.TERMINOS) if abs(t["A"]["contrib"][i]) > 1e-12]
        inf.tabla(_sub(r["resA"], filas_d), f"Descomposición de la varianza de Y (O2.1) · {n}", formato=fmt_res,
                  nombre_indice="Término")
        inf.figura(gr.fig_descomposicion(t["A"], r["resA"], f"Descomposición de Var(Y) · {n}"),
                   f"Parte de la varianza del beneficio atribuible a cada término · {n}")
        inf.h(f"{sec}.2 Bloque B · batería de {num(float(r['esc']['cap']), 0)} MWh (O4, O5.3 y O6.3)", 2)
        inf.tabla(_sub(r["resB"], FILAS_B), f"Beneficio diario con y sin batería: teórico y simulado · {n}", formato=fmt_res,
                  nombre_indice="KPI")
        inf.tabla(tabla_tipos_B(t, r["mcB"]), f"Ganancia de la batería por escenario climático (O5.3) · {n}",
                  formato=fmt_tipos_B, nombre_indice="Escenario")
        inf.figura(gr.fig_soc(t["B"], r["mcB"], r["esc"], f"Estado de carga · {n}"),
                   f"Estado de carga día a día (teórico y simulado) y su distribución media · {n}")
        inf.figura(gr.fig_hist_bat(t["A"], r["mcB"], f"Beneficio diario con y sin batería · {n}"),
                   f"Distribución del beneficio diario con y sin batería y CVaR 5 % · {n}")
        inf.figura(gr.fig_ganancia_tipo(t["B"], r["mcB"], f"Ganancia por escenario · {n}"),
                   f"Ganancia de la batería por escenario climático y déficit y superávit medios · {n}")
        inf.tabla(tabla_extras_B(t, r["mcB"], r["esc"]), f"Rentabilidad anual, LOLE y diseño pareado · {n}",
                  nombre_indice="Indicador", formato=_txt)
        inf.h(f"{sec}.3 Funcionamiento simulado", 2)
        dia = ef.un_dia(com, r["esc"], 0.5, 0.5, 0.5, 0.5, 0.5)
        inf.figura(gr.fig_mecanismo(t["A"]["p"], 0.5, dia, f"Composición · un día · {n}"),
                   f"Escenario elegido con U_T = 0,5 y fuentes, demanda y beneficio con U = 0,5 · {n}")
        inf.figura(gr.fig_traza_A(r["mcA"]["traza"], f"Primeros {len(r['mcA']['traza'])} días · {n}"),
                   f"Generación por fuente, demanda y beneficio de los primeros días simulados · {n}")
        sec += 1

    if bat is not None:
        inf.salto()
        inf.h(f"{sec}. Dimensionamiento y robustez de la batería (O5.1, O5.2, O6.1, O6.2) · «{bat['escenario']}»", 1)
        df = bat["cap"]; opt = bat["cap_opt"]
        inf.p(f"O5.1 y O6.1 · La ganancia diaria crece con la capacidad, pero con rendimientos decrecientes: los superávits "
              f"son escasos y una batería grande casi nunca se llena. Con una amortización de {eur(bat['amort'])} por MWh y año, "
              f"la capacidad que maximiza la ganancia neta anual es Cap* = {num(opt, 0)} MWh "
              f"({eur(df.loc[opt, 'Teórico · ganancia neta anual (€)'])} al año).")
        cols_c = [c for c in df.columns if c not in ("Teórico · LOLE con batería (días/año)", "Ganancia por MWh de capacidad (€/año)")]
        inf.tabla(indice_fmt(df[cols_c], FMT_IDX["cap"]), "Ganancia de la batería según su capacidad", formato=fmt_tabla,
                  nombre_indice="Capacidad")
        inf.figura(gr.fig_barrido(df, "cap", opt), "Ganancia diaria, ganancia anual, amortización y ganancia neta según la capacidad")
        de = bat["eta"]
        inf.p("O5.2 · La eficiencia de ida y vuelta escala la energía útil que devuelve la batería: la ganancia cae de forma "
              "aproximadamente proporcional a η.")
        inf.tabla(indice_fmt(de, FMT_IDX["eta"]), "Ganancia de la batería según su eficiencia", formato=fmt_tabla,
                  nombre_indice="η")
        inf.figura(gr.fig_barrido(de, "eta"), "Ganancia diaria y LOLE según la eficiencia de la batería")
        dr, rs, cfg = bat["racha"]
        inf.p(f"O6.2 · Racha de {cfg['dias']} días Anticiclónicos desde el día {cfg['inicio']} (teórico con la cadena de Markov "
              "y simulado con la misma base). Un día Anticiclónico tiene un déficit medio de unos 115 MWh, así que la batería "
              "se vacía el primer día de la racha y no aporta nada durante el resto: la racha cuesta casi lo mismo con y sin "
              "batería. Después se recupera en pocos días.")
        inf.tabla(pd.DataFrame({"Valor": rs}), "Resumen de la racha", nombre_indice="Indicador",
                  formato=lambda f, c, v: fmt_val(f, v))
        inf.figura(gr.fig_racha(dr, cfg["inicio"], cfg["dias"]), "Estado de carga y beneficio diario alrededor de la racha")
        sec += 1

    if sens is not None:
        inf.salto()
        inf.h(f"{sec}. Sensibilidad (O2.2 y O3.3) · escenario «{sens['escenario']}»", 1)
        df, elas = sens["param"]
        mayor = max(elas, key=lambda k: abs(elas[k]))
        inf.p("O2.2 · Se varía cada parámetro manteniendo fijos los demás; la elasticidad es la variación porcentual de Var(Y) "
              f"por cada 1 % de variación del parámetro. La mayor es la de «{mayor}» ({num(elas[mayor], 2)}). Los índices de "
              "Sobol muestran que el escenario climático T explica la mayor parte de la varianza: el riesgo es, ante todo, "
              "climático (días Anticiclónicos), y el precio de penalización lo amplifica.")
        inf.figura(gr.fig_sens_param(df, elas), "Var(Y) al variar cada parámetro y elasticidad en el punto base")
        inf.tabla(pd.DataFrame({"Elasticidad de Var(Y)": elas}), "Elasticidad de Var(Y) por parámetro (teórico)",
                  formato=lambda f, c, v: num(v, 3), nombre_indice="Parámetro")
        inf.tabla(sens["sobol"], "Índices de Sobol de las entradas aleatorias (simulados; S₁ del escenario también teórico)",
                  formato=lambda f, c, v: pct(v, 1), nombre_indice="Entrada")
        inf.figura(gr.fig_sobol(sens["sobol"]), "Índices de Sobol de primer orden y totales")
        ds, ms = sens["solar"]
        inf.p(f"O3.3 · La solar media actual es {num(ms, 1)} MWh/día. Para conseguir E(Y) ≥ 0 hace falta añadir "
              f"{num(ds.iloc[0]['MWh/día de media adicionales'], 1)} MWh/día si la ampliación aporta lo mismo todos los días, o "
              f"{num(ds.iloc[1]['MWh/día de media adicionales'], 1)} MWh/día de media si se amplía la capacidad (la nueva solar "
              "rinde poco en los días nublados, justo cuando más falta).")
        inf.tabla(ds, "Ampliación solar necesaria para E(Y) ≥ 0 (teórico)", formato=fmt_tabla, nombre_indice="Forma")
        sec += 1

    if ampliacion is not None:
        A = ampliacion
        inf.salto()
        inf.h(f"{sec}. Ampliación optativa (O8) · escenario «{A['escenario']}»", 1)
        dc, rho, teo = A["contraste"]
        inf.p(f"O8.1 · Contraste H₀: E(Δ) = 0. Los días de un mismo año no son independientes (la batería tiene memoria: la "
              f"correlación de Δ entre días consecutivos es {num(rho, 2)}), así que el contraste correcto usa las medias "
              "anuales como macro-réplicas independientes; tratar los días como independientes da un error estándar "
              "equivocado. La estimación sigue siendo la media por día.")
        inf.tabla(dc, "Contraste de la ganancia media de la batería", formato=fmt_tabla, nombre_indice="Procedimiento")
        inf.p("O8.2 · El estado de carga inicial solo afecta a los primeros días del año: el beneficio anual esperado apenas "
              "cambia entre empezar vacía, a mitad o llena (la diferencia es como mucho el valor de una carga completa).")
        inf.tabla(indice_fmt(A["inicial"], lambda v: pct(v, 0)), "Beneficio anual con batería según la carga inicial",
                  formato=fmt_tabla, nombre_indice="Carga inicial")
        inf.p(TEXTO_O83)
        inf.tabla(A["activa"], f"Gestión activa con precio de exportación aleatorio (CV {pct(A['cv_exp'], 0)}, simulado)",
                  formato=fmt_tabla, nombre_indice="Regla")
        inf.p(TEXTO_O84)
        tb = A["tarifa"]
        inf.p(f"O8.5 · Con una tarifa τ por MWh de capacidad y día, la batería se paga sola si 365·τ·Cap ≥ amortización: "
              f"τ ≥ {eur(tb['sola'], 2)} por MWh y día. Contando también la ganancia operativa, basta τ ≥ "
              f"{eur(max(tb['equilibrio'], 0), 2)} por MWh y día" + (" (ya es rentable sin tarifa)." if tb["equilibrio"] <= 0 else "."))
        dv, extra = A["solar_vs"]
        inf.p(f"O8.6 · Con el mismo coste anual que la batería se podrían añadir unos {num(extra, 1)} MWh/día de solar media "
              f"(supuesto: {eur(A['coste_solar'])} por MWh/día y año). " + TEXTO_O86)
        inf.tabla(dv, "Batería frente a ampliación solar del mismo coste (teórico)", formato=fmt_tabla, nombre_indice="Alternativa")
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
        "Composición: T por transformada inversa; G_s con la inversa de la triangular, G_w = λ·(−ln(1 − U))^(1/k), "
        "G_h = exp(μ + σ·Φ⁻¹(U)) y D = μ + σ·Φ⁻¹(U). Cada día usa 6 uniformes (la sexta, para el precio spot aleatorio), "
        "de modo que todos los escenarios comparten los mismos números aleatorios.",
        "Teoría del día: G = G_s + G_w + G_h se discretiza en una rejilla de 0,25 MWh convolucionando por FFT sus "
        "distribuciones exactas y D se integra con 100 cuantiles; sobre esa rejilla se calculan exactamente E(Y), Var(Y), "
        "la descomposición Cov(término, Y)/Var(Y), P(B < 0), la distribución de Y (paso de 2 €), el VaR y el CVaR.",
        "Teoría de la batería: el estado de carga es una cadena de Markov (los días son independientes dado el estado); se "
        "discretiza en 200 estados y la ley de B da la matriz de transición. Su ley día a día da la ganancia esperada, la "
        "carga media, P(compra), los déficits cubiertos y el LOLE = 365·P(compra). El VaR y el CVaR con batería son solo "
        "simulados.",
        "Unidad de análisis: las estimaciones de la batería son medias por día sobre todos los días simulados; los años solo "
        "se usan como macro-réplicas para el IC95 %, porque los días de un mismo año están correlacionados.",
        "Supuestos propios: precio spot lognormal de media 80 €/MWh y CV 50 % (O3.1); contrato a plazo a 60 €/MWh con una "
        "prima de 150 €/día (O3.2); ampliación solar del 30 % (O3.2); respuesta de demanda del 10 % si el déficit supera "
        "20 MWh con un descuento de 20 €/MWh no consumido (O3.4); amortización de la batería de 1.800 € por MWh y año "
        "(O6.1); coste de la solar de 9.000 € por MWh/día y año (O8.6). Todos se pueden cambiar en la aplicación.",
        "Material de Javier Morales, equipo IA4LEGOS (CC BY-SA 4.0).",
    ])
    return inf.guardar()
