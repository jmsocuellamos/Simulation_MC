# -*- coding: utf-8 -*-
"""Informe Word (.docx) del estudio de caso Staph-X: coste de un paciente, coste del brote con capacidad UCI limitada,
escenarios, capacidad, sensibilidad y ampliación."""
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

import staphx as sx
from staphx import num, pct, eur, fmt_kpi, fmt_res, fmt_col, fmt_obj, fmt_val
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
FILAS_A = ["Coste esperado E(Y) (€)", "Desviación típica σ(Y) (€)", "Probabilidad de complicación P(D > umbral)",
           "Días medios de ingreso E(D)", "Percentil 95 de Y (€)"]
FILAS_B = ["Coste total esperado del brote con capacidad (€)", "Coste total esperado sin restricción (€)",
           "Sobrecoste por falta de capacidad (€)", "Sobrecoste en % del coste total",
           "Percentil 95 del coste total (€) · aprox. normal", "Días con desbordamiento",
           "Pacientes derivados a la clínica externa", "Camas de alquiler usadas", "P(UCI llena dos días seguidos)",
           "Pacientes del brote", "Pacientes complicados del brote", "Coste total con reserva de camas e implantación (€)"]


def tabla_fenotipos(t, resA):
    """O1.3 por fenotipo: teórico y simulado (media de las réplicas)."""
    pf = t["A"]["por_fen"]
    filas = {}
    for f in sx.FENOTIPOS:
        nf = sx.NOMBRE_FEN[f]
        g = lambda k: resA.loc[k, "Simulado · media"]
        filas[nf] = {"Teórico · E(Y)": pf[f]["EY"], "Simulado · E(Y)": g(f"E(Y | {nf}) (€)"),
                     "Teórico · CV": pf[f]["CV"], "Simulado · CV": g(f"CV(Y | {nf})"),
                     "Teórico · P(comp.)": pf[f]["Pcomp"], "Simulado · P(comp.)": g(f"P(complicación | {nf})"),
                     "Teórico · peso en las complicaciones": pf[f]["peso_comp"],
                     "Simulado · peso en las complicaciones": g(f"Peso de {nf} en las complicaciones")}
    return pd.DataFrame(filas).T


def fmt_fen(fila, col, v):
    if col == "p" or "P(" in col or "peso" in col:
        return pct(v, 2)
    if "CV" in col:
        return num(v, 3)
    return eur(v)


def tabla_fases(t, resB):
    pf = t["B"]["por_fase"]
    filas = {}
    for f, d in pf.items():
        g = lambda k: resB.loc[k, "Simulado · media"]
        filas[f] = {"λ ingresos al día": d["lambda"], "% Persistente": d["p_persistente"],
                    "Complicados al día": d["lambda_comp"], "Teórico · sobrecoste": d["delta"],
                    "Simulado · sobrecoste": g(f"Sobrecoste · {f} (€)"), "Cuota del sobrecoste": d["cuota_delta"],
                    "Teórico · derivados": d["der"], "Simulado · derivados": g(f"Derivados · {f}")}
    return pd.DataFrame(filas).T


def fmt_fases(fila, col, v):
    if col in ("Días", "λ ingresos al día"):
        return num(v, 0)
    if "%" in col or "Cuota" in col:
        return pct(v, 1)
    if "derivados" in col or "Complicados" in col:
        return num(v, 2)
    if "por paciente" in col:
        return eur(v, 2)
    return eur(v)


def tabla_extras_B(t, mcB, com, esc):
    B = t["B"]
    rep = mcB["replicas"]
    filas = {
        "Ingresos por día: Poisson por tramos (O4)": " / ".join(f"{f}: Poisson({num(d['lambda'], 0)})" for f, d in B["por_fase"].items()),
        "Complicados por día (adelgazamiento de Poisson)": " / ".join(f"{f}: Poisson({num(d['lambda_comp'], 2)})" for f, d in B["por_fase"].items()),
        "Días esperados con la UCI llena (k ≥ camas)": f"{num(B['dias_llena'], 1)} de {B['ndias']}",
        "Pares esperados de días seguidos con la UCI llena (O6.2)": f"{num(B['pares_consecutivos'], 1)} teórico · {num(rep['Pares de días seguidos con UCI llena'].mean(), 1)} simulado",
        "P(UCI llena al menos dos días seguidos) (O6.2)": f"{pct(B['P_consecutivos'], 2)} teórico · {pct(rep['UCI llena dos días seguidos'].mean(), 2)} simulado",
        "σ del sobrecoste por brote · diseño pareado (O4.1, O8.1)": f"{eur(B['sd_delta'])} teórico · {eur(rep['Sobrecoste'].std(ddof=1))} simulado",
        "σ del sobrecoste · diseño independiente": eur(math.sqrt(B["var_independiente"])) + " teórico",
        "Reducción de varianza del diseño pareado": f"× {num(B['var_independiente'] / B['var_pareada'], 0)}" if B["var_pareada"] > 0 else "—",
        "Reserva de camas por brote (coste de oportunidad)": f"{eur(B['reserva'])} ({B['camas']} camas × {eur(com['coste_cama_dia'])} × {B['ndias']} días)",
    }
    if int(esc.get("tramo_Q") or 0) > 0:
        filas["Ahorro esperado del contrato por volumen (O5.2)"] = (f"{eur(B['ahorro_tramo'])} (desde el derivado {int(esc['tramo_Q']) + 1}: "
                                                                    f"{eur(esc['coste_tramo'])} en lugar de {eur(esc['C_ext'])})")
    return pd.DataFrame({"Valor": filas})


def tabla_parametros(res):
    filas = {}
    for n, r in res.items():
        e = r["esc"]; B = r["teoria"]["B"]
        filas[n] = {"C_fijo (+ extra) (€)": f"{num(e['C_fijo'], 0)} (+{num(e['coste_extra'], 0)})",
                    "C_día / C_UCI / C_ext (€)": f"{num(e['C_dia'], 0)} / {num(e['C_UCI'], 0)} / {num(e['C_ext'], 0)}",
                    "Devolución si complica": pct(e["devolucion"], 0),
                    "×L todos / ×L Persist. / ×R Persist.": f"{num(e['factor_L'], 2)} / {num(e['factor_L3'], 2)} / {num(e['factor_R3'], 2)}",
                    "Umbral (días) / λ Pico": f"{num(e['umbral'], 0)} / {num(B['por_fase'].get('Pico', {}).get('lambda', 0), 0)}",
                    "Camas propias / alquiler (€/uso)": f"{e['camas']} / {e['camas_alquiler']}" + (f" ({num(e['coste_alquiler'], 0)})" if e["camas_alquiler"] else ""),
                    "Contrato por volumen": f"desde {int(e['tramo_Q']) + 1}: {num(e['coste_tramo'], 0)} €" if int(e.get("tramo_Q") or 0) > 0 else "no",
                    "Implantación (€/brote)": num(e.get("coste_impl") or 0, 0)}
    return pd.DataFrame(filas)


def indice_pct(df, d=0):
    out = df.copy(); out.index = [pct(i, d) for i in df.index]; return out


def indice_cv(df):
    out = df.copy(); out.index = [f"CV = {num(i, 1)}" for i in df.index]; return out


def tabla_umbral(du):
    out = du.T.copy(); out.columns = [f"Umbral {num(u, 0)} días" for u in du.index]; return out


def tabla_dif(dif, cols):
    return dif[[c for c in cols if c in dif.columns]]


COLS_DIF_A = ["ΔE(Y) por paciente · teórico (€)", "ΔE(Y) por paciente · simulado (€)", "IC95 ΔE(Y) (€)",
              "ΔP95 de Y · teórico (€)", "Días de ingreso evitados por paciente", "Complicaciones evitadas por paciente"]
COLS_DIF_ICER = ["Δ coste por paciente con implantación · teórico (€)", "ICER (€ por día evitado)",
                 "ICER (€ por complicación evitada)"]
COLS_DIF_B = ["Ahorro por brote · teórico (€)", "Ahorro por brote · simulado (€)", "IC95 ahorro por brote (€)",
              "Ahorro en sobrecoste de capacidad · teórico (€)", "Coste de implantación (€/brote)"]


def fmt_sens(fila, col, v):
    if col == "Error relativo":
        return pct(v, 3)
    return eur(v)


def tabla_sens(df):
    return df.drop(columns=["Medias por réplica"])


def tabla_sens_param(df):
    d = df.copy()
    d["Variación"] = d["Variación"].map(lambda v: f"{'+' if v > 0 else ''}{num(100 * v, 0)} %")
    return d.set_index(["Parámetro", "Variación"])


def fmt_sens_param(fila, col, v):
    if "Var(Y)" in col:
        return num(v / 1e6, 3) + " M"
    if "P(" in col:
        return pct(v, 2)
    return eur(v)


def fmt_tabla(fila, col, v):
    """Formato genérico por nombre de columna (con la fila como respaldo)."""
    if isinstance(v, (bool, np.bool_)):
        return "Sí" if v else "No"
    if isinstance(v, str):
        return v
    if isinstance(v, tuple):
        return fmt_val(col, v)
    if "Reducción" in col or col.startswith("CV"):
        return pct(v, 0) if "Reducción" in col else num(v, 2)
    return fmt_val(col, v)


# ---------------------------------------------------------------------------------------------- informe
def generar_informe(res, com, aj, base, capacidad=None, sens=None, ampliacion=None, sens_est=None,
                    titulo="Farmacoeconomía de la terapia para Staph-X"):
    inf = Informe(titulo, f"Informe generado automáticamente el {datetime.datetime.now():%d/%m/%Y %H:%M} · coste de un "
                          "paciente por composición, coste del brote con capacidad UCI limitada, escenarios, capacidad, "
                          "sensibilidad y ampliación")
    # 1. Planteamiento
    inf.h("1. Planteamiento y parámetros", 1)
    inf.p("Bloque A · un paciente. Cada paciente tiene un fenotipo latente T (Rápido, Estándar o Persistente); "
          "condicionadas a T se simulan, de forma independiente, la carga bacteriana inicial L ~ Gamma(α_T; 1) y la tasa "
          "de depuración R ~ U(r₀, r₁) (método de composición, por transformada inversa). Los días de ingreso son "
          "D = máx(0, ln(L + 1)/R + ε), con ε ~ N(1; 0,2), y el coste Y = C_fijo + C_día·D + C_UCI·I{D > umbral}.")
    fen = pd.DataFrame({sx.NOMBRE_FEN[f]: {"Probabilidad p": pct(sx.normalizar(com["p"])[f - 1], 0),
                                           "Carga L (UFC/ml)": f"Gamma(α = {num(com['alpha'][f], 1)}; escala {num(com['escala_L'], 1)})",
                                           "Depuración R (log/día)": f"U({num(com['R'][f][0], 2)}; {num(com['R'][f][1], 2)})"}
                        for f in sx.FENOTIPOS})
    inf.tabla(fen, "Fenotipos y distribuciones condicionadas", nombre_indice="Variable", formato=_txt)
    inf.p("Bloque B · un brote de 90 días. Cada día ingresan N_d ~ Poisson(λ_fase) pacientes con la mezcla de "
          "fenotipos de la fase. Solo B camas UCI propias atienden a los complicados del día (orden de llegada "
          "aleatorio); el resto va a camas de alquiler (si las hay) o a la clínica externa (C_ext). Y_con e Y_sin se "
          "calculan con los mismos pacientes (simulación pareada) y Δ = Y_con − Y_sin es el sobrecoste por capacidad.")
    fas = pd.DataFrame({f["nombre"]: {"Días": num(f["dias"], 0), "λ (ingresos/día)": num(f["lambda"], 1),
                                      "p (Rápido / Estándar / Persistente)": " / ".join(num(x, 2) for x in sx.normalizar(f["p"]))}
                        for f in com["fases"]})
    inf.tabla(fas, "Fases del brote, ingresos y mezcla de fenotipos", nombre_indice="Parámetro", formato=_txt)
    inf.tabla_escenarios(tabla_parametros(res), "Escenarios analizados", formato=_txt, nombre_indice="Parámetro")
    desc = [f"{n}: {r['esc']['descripcion']}" for n, r in res.items() if r["esc"].get("descripcion")]
    if desc:
        inf.vinetas(desc)
    ob = com["objetivos"]
    o = lambda k, f: f(ob[k]) if ob.get(k) is not None else "no se evalúa"
    tc = pd.DataFrame({"Valor": {
        "ε ~ N(μ; σ)": f"N({num(com['eps_mu'], 2)}; {num(com['eps_sd'], 2)})",
        "Coste de oportunidad de una cama reservada": f"{eur(com['coste_cama_dia'])} por cama y día",
        "Objetivo · E(Y) máximo por paciente": o("EY_MAX", eur),
        "Objetivo · P(complicación) máxima": o("PCOMP_MAX", lambda v: pct(v, 1)),
        "Objetivo · coste total del brote (con reserva) máximo": o("TOTAL_MAX", eur),
        "Objetivo · pacientes derivados por brote máximo": o("EXT_MAX", lambda v: num(v, 1)),
        "Simulación A · pacientes por réplica × réplicas": f"{num(aj['nsim'], 0)} × {aj['replicas']} (semillas {aj['semilla']} …)",
        "Simulación B · brotes simulados": f"{num(aj['brotes'], 0)} (semillas {aj['semilla'] + 1000} …)"}})
    inf.tabla(tc, "Parámetros comunes, objetivos y ajustes de simulación", nombre_indice="Parámetro", formato=_txt)

    # 2. Síntesis
    inf.h("2. Síntesis y recomendaciones (O7.1)", 1)
    for n, r in res.items():
        inf.p(sx.texto_escenario(n, r), negrita_inicio=n + ":")
    cam_df, cam_opt = (capacidad["camas"], capacidad["camas_opt"]) if capacidad else (None, None)
    inf.p("Recomendaciones para el próximo brote:", negrita_inicio="Recomendaciones para el próximo brote:")
    inf.vinetas(sx.recomendaciones(res, base, com, cam_df, cam_opt), numeradas=True)
    inf.tabla_escenarios(sx.tabla_comparativa(res).iloc[5:], "Comparación de escenarios (filas «Teórico»: fórmulas "
                         "exactas; filas «Simulado»: media de las macro-réplicas)", formato=fmt_kpi, nombre_indice="KPI")
    inf.tabla_escenarios(sx.tabla_objetivos(res), "Cumplimiento de los objetivos (valores teóricos)",
                         nombre_indice="Objetivo", formato=_txt)
    inf.figura(gr.fig_comparativa(res), "Coste por paciente, complicaciones, coste total del brote y derivados por escenario")
    dif = sx.diferencias(res, base)
    if dif is not None:
        inf.p(f"Comparación frente a «{base}» (O3.1–O3.3, O5.1, O5.2, O6.3 y O8.6). Las diferencias simuladas son "
              "pareadas (mismas semillas en todos los escenarios), por lo que su IC95 % es mucho más estrecho que el de "
              "cada coste por separado. El ICER = Δcoste / Δefecto mide cuánto cuesta cada día de ingreso o cada "
              "complicación evitada; «dominante» significa que la alternativa ahorra dinero y mejora el resultado.")
        inf.tabla(tabla_dif(dif, COLS_DIF_A), f"Coste por paciente y efectividad frente a «{base}»", formato=fmt_col,
                  nombre_indice="Escenario")
        inf.tabla(tabla_dif(dif, COLS_DIF_ICER), f"Razón coste-efectividad incremental (ICER) frente a «{base}»",
                  formato=fmt_col, nombre_indice="Escenario")
        inf.tabla(tabla_dif(dif, COLS_DIF_B), f"Ahorro por brote frente a «{base}» (incluida la reserva de camas y la "
                  "implantación)", formato=fmt_col, nombre_indice="Escenario")
        inf.figura(gr.fig_ahorros(dif), f"Ahorro por brote frente a «{base}»: teórico y simulado (IC95 % pareado)")
    inf.figura(gr.fig_validacion(res), "Error relativo de la simulación frente a los valores teóricos")
    inf.h("Cuadro de mando propuesto", 2)
    inf.tabla(pd.DataFrame({"Para qué sirve": dict(sx.KPIS_CUADRO)}), "Cuadro de mando para la Dirección",
              nombre_indice="KPI", formato=_txt)

    # 3. Detalle por escenario
    sec = 3
    for n, r in res.items():
        inf.salto()
        t = r["teoria"]
        inf.h(f"{sec}. Escenario «{n}»", 1)
        if r["esc"].get("descripcion"):
            inf.p(r["esc"]["descripcion"])
        inf.h(f"{sec}.1 Bloque A · coste de un paciente (O1 y O2.1)", 2)
        inf.tabla(_sub(r["resA"], FILAS_A), f"Coste de un paciente: teórico y simulado con IC95 % · {n}",
                  formato=fmt_res, nombre_indice="KPI")
        inf.tabla(tabla_fenotipos(t, r["resA"]), f"Coste por fenotipo (O1.3) · {n}", formato=fmt_fen, nombre_indice="Fenotipo")
        inf.figura(gr.fig_hist_Y(t["A"], r["mcA"]["Y"], f"Distribución del coste de un paciente · {n}"),
                   f"Coste simulado frente a la distribución teórica (escala logarítmica) · {n}")
        inf.figura(gr.fig_por_fen(t["A"], r["resA"], f"Coste por fenotipo · {n}"),
                   f"Media, probabilidad de complicación y variabilidad relativa por fenotipo · {n}")
        filas_d = [f"Contribución a Var(Y) · {x}" for x in sx.TERMINOS]
        inf.tabla(_sub(r["resA"], filas_d), f"Descomposición de la varianza de Y (O2.1) · {n}", formato=fmt_res,
                  nombre_indice="Término")
        inf.figura(gr.fig_descomposicion(t["A"], r["resA"], f"Descomposición de Var(Y) · {n}"),
                   f"Parte de la varianza del coste atribuible a la estancia y al salto por complicación · {n}")
        inf.h(f"{sec}.2 Bloque B · coste del brote con capacidad UCI limitada (O4, O5.3 y O6.2)", 2)
        inf.tabla(_sub(r["resB"], FILAS_B), f"Coste del brote: teórico y simulado con IC95 % · {n}", formato=fmt_res,
                  nombre_indice="KPI")
        inf.tabla(tabla_fases(t, r["resB"]), f"Sobrecoste por fase del brote (O5.3) · {n}", formato=fmt_fases,
                  nombre_indice="Fase")
        inf.figura(gr.fig_hist_brote(t["B"], r["mcB"]["replicas"]["Coste total con capacidad"], f"Coste del brote · {n}"),
                   f"Coste total de los brotes simulados y percentil 95 · {n}")
        inf.figura(gr.fig_fases(t["B"], r["resB"], f"Fases del brote · {n}"),
                   f"Sobrecoste por fase y complicados esperados por día frente a las camas · {n}")
        inf.figura(gr.fig_dias(r["mcB"]["dias"], t["B"]["camas"], t["B"]["extra"], f"Primer brote simulado · {n}"),
                   f"Complicados por día y pacientes derivados en el primer brote simulado · {n}")
        inf.tabla(tabla_extras_B(t, r["mcB"], com, r["esc"]), f"Saturación, diseño pareado y reserva de camas · {n}",
                  nombre_indice="Indicador", formato=_txt)
        inf.h(f"{sec}.3 Funcionamiento simulado", 2)
        tr = r["mcA"]["traza"]
        uT = float(tr["U_T"].iloc[0])
        pac = sx.un_paciente(com, r["esc"], 0.95, 0.5, 0.5, 0.5)
        inf.figura(gr.fig_mecanismo(t["A"]["p"], 0.95, pac, r["esc"]["umbral"], f"Composición · un paciente · {n}"),
                   f"Fenotipo elegido con U_T = 0,95 y días y coste con U_L = U_R = U_ε = 0,5 · {n}")
        inf.figura(gr.fig_traza_A(tr, r["esc"], f"Primeros {len(tr)} pacientes · {n}"),
                   f"Coste por términos y fenotipo de los primeros {len(tr)} pacientes simulados · {n}")
        sec += 1

    if capacidad is not None:
        inf.salto()
        inf.h(f"{sec}. Capacidad UCI (O6.1, O6.3, O8.1, O8.2 y O8.5) · escenario «{capacidad['escenario']}»", 1)
        df, b = capacidad["camas"], capacidad["camas_opt"]
        inf.p(f"O6.1 · El sobrecoste esperado del brote baja al añadir camas, pero cada cama reservada tiene un coste de "
              f"oportunidad de {eur(com['coste_cama_dia'])} por día ({eur(df['Coste de reservarla (€)'].iloc[0])} por brote). "
              f"El mínimo de sobrecoste + reserva se alcanza con {b} camas: a partir de ahí el ahorro marginal de otra "
              "cama no compensa su reserva. La tabla incluye además el riesgo de UCI llena dos días seguidos (O6.2).")
        inf.tabla(df, "Barrido del número de camas UCI reservadas (teórico)", formato=fmt_tabla, nombre_indice="Camas")
        inf.figura(gr.fig_camas(df, b), "Sobrecoste, reserva y ahorro marginal por número de camas")
        if capacidad.get("cuota") is not None and len(capacidad["cuota"]):
            cq = capacidad["cuota"]
            inf.p("O8.5 · Si se alquilan camas adicionales solo durante el Pico, la cuota diaria máxima que compensa cada "
                  "cama es su ahorro marginal en sobrecoste del Pico dividido entre los días del Pico: "
                  f"{eur(cq.iloc[0, -1])} por día para la primera cama, {eur(cq.iloc[1, -1])} para la segunda, etc.")
            inf.tabla(cq, "Cuota diaria máxima por cama alquilada en el Pico (teórico)", formato=fmt_tabla,
                      nombre_indice="Camas extra")
            inf.figura(gr.fig_cuota(cq), "Cuota diaria máxima que el hospital debería pagar por cada cama extra en el Pico")
        if capacidad.get("pareada") is not None:
            dp, fac = capacidad["pareada"]
            inf.p(f"O4.1 y O8.1 · Con el diseño pareado, Y_con e Y_sin comparten pacientes: Var(Δ) = Var(Y_con) + Var(Y_sin) "
                  f"− 2·Cov(Y_con, Y_sin), y la covarianza es casi igual a las varianzas. Emparejar brotes distintos "
                  f"(independientes) multiplica la varianza del sobrecoste por {num(fac, 0)}: para la misma precisión harían "
                  f"falta unas {num(fac, 0)} veces más simulaciones.")
            inf.tabla(dp, "Diseño pareado frente a independiente (sobrecoste por brote)", formato=fmt_tabla,
                      nombre_indice="Diseño")
        if capacidad.get("ocupacion") is not None:
            do, ocu = capacidad["ocupacion"]
            inf.p("O8.2 · El modelo del caso decide cada día por separado («un día, una decisión»): un complicado solo "
                  "ocupa la cama el día de su ingreso. Si cada complicado ocupara la cama ⌈D − umbral⌉ días (supuesto), "
                  "las camas se liberarían más tarde y el modelo del caso subestima los derivados y el sobrecoste "
                  f"(ocupación media de las camas: {pct(ocu, 0)}). Un modelo M/M/c/K recoge esta dependencia entre días.")
            inf.tabla(do, "Modelo diario frente a ocupación de varios días (simulado, mismos pacientes)", formato=fmt_tabla,
                      nombre_indice="Modelo")
        sec += 1

    if sens is not None:
        inf.salto()
        inf.h(f"{sec}. Sensibilidad (O2.2 y O3.4) · escenario «{sens['escenario']}»", 1)
        df, elas = sens["param"]
        mayor = max(elas, key=lambda k: abs(elas[k]))
        inf.p("O2.2 · Se varía cada parámetro del modelo manteniendo fijos los demás. La elasticidad es la variación "
              f"porcentual de Var(Y) por cada 1 % de variación del parámetro. El de mayor impacto es «{mayor}» "
              f"(elasticidad {num(elas[mayor], 2)}). Los índices de Sobol de las entradas aleatorias confirman que el "
              "fenotipo T explica la mayor parte de Var(Y): conviene concentrar los esfuerzos en identificar y tratar "
              "a los Persistentes.")
        inf.figura(gr.fig_sens_param(df, elas), "Var(Y) al variar cada parámetro y elasticidad en el punto base")
        inf.tabla(pd.DataFrame({"Elasticidad de Var(Y)": elas}), "Elasticidad de Var(Y) por parámetro (teórico)",
                  formato=lambda f, c, v: num(v, 3), nombre_indice="Parámetro")
        inf.tabla(sens["sobol"], "Índices de Sobol de las entradas aleatorias (simulados; S₁ del fenotipo también teórico)",
                  formato=lambda f, c, v: pct(v, 1), nombre_indice="Entrada")
        inf.figura(gr.fig_sobol(sens["sobol"]), "Índices de Sobol de primer orden y totales")
        du = sens["umbral"]
        inf.p("O3.4 · Recalibrar el umbral de complicación cambia la distribución de Y con los mismos pacientes: con 5 días "
              "aumentan las complicaciones (más pacientes con el coste UCI y más presión sobre las camas); con 10 días se "
              "reducen, pero el criterio clínico podría retrasar la atención de pacientes graves.")
        inf.tabla(tabla_umbral(du), "Efecto del umbral de complicación (teórico y simulado con los mismos pacientes)",
                  formato=lambda f, c, v: fmt_val(f, v), nombre_indice="KPI")
        inf.figura(gr.fig_umbral(du), "Coste por paciente, complicaciones y sobrecoste según el umbral")
        sec += 1

    if ampliacion is not None:
        inf.salto()
        inf.h(f"{sec}. Ampliación optativa (O8.3, O8.4 y O8.6) · escenario «{ampliacion['escenario']}»", 1)
        df, opt = ampliacion["optim"]
        cfg = ampliacion["cfg"]
        inf.p(f"O8.3 · Decisión conjunta: camas reservadas b y reducción r de la carga del Persistente con un programa de "
              f"diagnóstico precoz de coste {eur(cfg['coste_10'])}·(10·r)² por brote (rendimientos decrecientes). "
              "Se minimiza el coste total esperado del brote (pacientes + sobrecoste + reserva + programa) sujeto a "
              f"reserva + programa ≤ {eur(cfg['presupuesto'])}. "
              + (f"Óptimo: {int(opt['Camas'])} camas y reducción del {pct(opt['Reducción de L₃'], 0)} "
                 f"(coste {eur(opt['Coste total esperado (€)'])})." if opt is not None else "No hay ninguna combinación factible."))
        inf.figura(gr.fig_optim(df, opt), "Coste total esperado por combinación de camas y diagnóstico precoz")
        dd = ampliacion["diagnostico"]
        inf.p("O8.6 · Análisis coste-beneficio del diagnóstico precoz: ahorro esperado por brote frente al coste del "
              "programa. Si solo se contara el ahorro en sobrecoste de capacidad, el programa solo compensaría para "
              "reducciones pequeñas; contando también el menor coste de los Persistentes, compensa en todo el rango.")
        inf.tabla(indice_pct(dd), "Ahorro esperado por brote al reducir la carga del Persistente (teórico)", formato=fmt_tabla,
                  nombre_indice="Reducción")
        dc = ampliacion["cox"]
        inf.p("O8.4 · En un proceso de Cox la intensidad del brote es aleatoria: λ_d·G_d con G_d ~ Gamma de media 1. "
              "Con la misma media de ingresos, la variabilidad extra concentra días con muchos complicados y aumenta el "
              "sobrecoste y su dispersión: el modelo de Poisson con λ fijo subestima el riesgo.")
        inf.tabla(indice_cv(dc), "Sobrecoste del brote según la variabilidad de la intensidad (simulado)", formato=fmt_tabla,
                  nombre_indice="CV")
        inf.figura(gr.fig_cox(dc), "Sobrecoste esperado con intensidad diaria aleatoria")
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
        "Composición: T por transformada inversa sobre (p₁, p₂, p₃); L | T por la inversa de la Gamma; R | T = r₀ + "
        "(r₁ − r₀)·U; ε = μ + σ·Φ⁻¹(U). Cada paciente usa 4 uniformes, de modo que todos los escenarios comparten los "
        "mismos números aleatorios (diferencias pareadas).",
        "Valores teóricos de un paciente: Z = ln(L + 1)/R se discretiza finamente (cuantiles de la Gamma × puntos de la "
        "uniforme) y, dado Z, D es una normal truncada en 0, con fórmulas cerradas para P(D > u), E(D), E(D²), E(D·I) y "
        "la distribución de Y (de la que salen los percentiles). La contribución de cada término es Cov(término, Y)/Var(Y).",
        "Brote: los complicados de cada día son Poisson(λ·q) por adelgazamiento, con q = Σ p_T·P(D > u | T); el "
        "sobrecoste diario, los días con desbordamiento, los derivados, la probabilidad de UCI llena dos días seguidos "
        "(cadena de días independientes) y la distribución del total de derivados (convolución) son exactos. El coste "
        "total del brote es una suma compuesta de Poisson; solo su percentil 95 usa la aproximación normal.",
        "Macro-réplicas: cada KPI se calcula en cada réplica (incluidos percentiles y probabilidades) y se da la media de "
        "las réplicas con IC95 % de la t de Student; el percentil 95 del coste del brote usa un IC por estadísticos de "
        "orden. Si un suceso es tan raro que ninguna réplica lo presenta, no se emite veredicto sobre el IC.",
        "Supuestos propios (no fijados en el enunciado): genérico con un 30 % más de carga bacteriana en todos los "
        "fenotipos; nuevo fármaco que duplica R del Persistente por 800 € adicionales; coste de oportunidad de "
        f"{eur(com['coste_cama_dia'])} por cama reservada y día; alquiler a 6.500 € por uso; contrato por volumen a "
        "7.000 € a partir del 16.º derivado; aislamiento con 50.000 € de implantación por brote; diagnóstico precoz con "
        "coste convexo; ocupación de ⌈D − umbral⌉ días en O8.2. Todos se pueden cambiar en la aplicación.",
        "Material de Javier Morales, equipo IA4LEGOS (CC BY-SA 4.0).",
    ])
    return inf.guardar()
