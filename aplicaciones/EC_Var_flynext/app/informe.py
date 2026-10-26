# -*- coding: utf-8 -*-
"""Informe Word (.docx) del estudio de caso Fly-Next: coste de una entrega, coste diario con tráfico, escenarios, optimización y sensibilidad."""
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

import flynext as fx
from flynext import num, pct, eur, fmt_kpi, fmt_res, fmt_dif, fmt_obj
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


def _sub(df, filas):
    return df.loc[[f for f in filas if f in df.index]]


def filas_A_global(umbral):
    u = num(umbral, 0)
    return ["Coste esperado E(Y) (€)", "Desviación típica σ(Y) (€)", f"P(Y > {u} €)", "Percentil 90 de Y (€)"]


def tabla_A_rutas(t, resA, umbral):
    """O1.3 por ruta: teórico y simulado (media de las réplicas) en una tabla compacta."""
    u = num(umbral, 0)
    pr = t["A"]["por_ruta"]
    filas = {}
    for r in fx.RUTAS:
        nr = fx.NOMBRE_RUTA[r]
        g = lambda k: resA.loc[k, "Simulado · media"]
        filas[nr] = {"p": pr[r]["p"], "Teórico · E(Y)": pr[r]["EY"], "Simulado · E(Y)": g(f"E(Y | {nr}) (€)"),
                     f"Teórico · P(Y > {u})": pr[r]["Pcola"], f"Simulado · P(Y > {u})": g(f"P(Y > {u} € | {nr})"),
                     "Teórico · CV": pr[r]["CV"], "Simulado · CV": g(f"CV(Y | {nr})"),
                     "Teórico · peso en la cola": pr[r]["peso_cola"],
                     "Simulado · peso en la cola": g(f"Peso de {nr} en la cola Y > {u} €")}
    return pd.DataFrame(filas).T


def fmt_rutas(fila, col, v):
    if col == "p" or "P(" in col or "peso" in col:
        return pct(v, 2)
    if "CV" in col:
        return num(v, 3)
    return eur(v, 2)


def tabla_franjas(t, resB):
    fr = t["B"]["franjas"]
    filas = {}
    for f in fx.FRANJAS:
        g = lambda k: resB.loc[k, "Simulado · media"]
        filas[f] = {"Pedidos/día": fr[f]["n"], "Teórico · E(Y_sin)": fr[f]["Ysin"], "Teórico · E(Y_con)": fr[f]["Ycon"],
                    "Simulado · E(Y_con)": g(f"E(Y_con) por pedido · {f} (€)"), "Teórico · E(Δ)": fr[f]["delta"],
                    "Simulado · E(Δ)": g(f"E(Δ) sobrecoste por pedido · {f} (€)"),
                    "Teórico · cuota del sobrecoste": fr[f]["cuota_sobre"],
                    "Simulado · cuota del sobrecoste": g(f"Cuota del sobrecoste diario · {f}")}
    return pd.DataFrame(filas).T


def fmt_franjas(fila, col, v):
    if "cuota" in col:
        return pct(v, 1)
    if "Pedidos" in col:
        return num(v, 1)
    return eur(v, 2)


def tabla_extras_B(t, mcB, com):
    B = t["B"]
    req = t["r_equilibrio"]
    filas = {
        "Pedidos por día: Poisson(Σλ) (O8.1)": f"Poisson({num(B['pedidos_poisson'], 0)}): media y varianza {num(B['pedidos_poisson'], 0)}",
        "Capacidad para una hora Punta con prob. ≥ 95 % (O6.1)": f"{num(B.get('cap95_hora', 0), 0)} pedidos/hora",
        "Capacidad para todas las horas Punta del día con prob. ≥ 95 %": f"{num(B.get('cap95_dia', 0), 0)} pedidos/hora",
        "Pedidos rechazados por día con la capacidad fijada (O6.3)": (f"{num(B['rechazados'], 2)} (capacidad {num(com['capacidad'], 0)} pedidos/hora)"
                                                                     if com.get("capacidad") else "sin límite de capacidad"),
        "Var(Δ) por pedido · diseño pareado (O8.4)": f"{num(B['var_pareada'], 1)} teórico · {num(mcB['var_pareada'], 1)} simulado",
        "Var(Y_con) + Var(Y_sin) · diseño independiente": f"{num(B['var_independiente'], 1)} teórico · {num(mcB['var_independiente'], 1)} simulado",
        "Reducción de varianza del diseño pareado": f"× {num(B['var_independiente'] / B['var_pareada'], 1)}" if B["var_pareada"] > 0 else "—",
        "Reducción del factor de retraso en Punta que iguala Punta y Llano (O5.3)": pct(req, 1) if req is not None and 0 <= req <= 1 else "no alcanzable con 0–100 %",
    }
    return pd.DataFrame({"Valor": filas})


def fmt_sens(fila, col, v):
    if col == "Error relativo":
        return pct(v, 3)
    return eur(v, 2)


def tabla_sens(df):
    return df.drop(columns=["Medias por réplica"])


def tabla_sens_econ(df):
    d = df.copy()
    d["Variación"] = d["Variación"].map(lambda v: f"{'+' if v > 0 else ''}{num(100 * v, 0)} %")
    d = d.set_index(["Parámetro", "Variación"])
    return d


def fmt_sens_econ(fila, col, v):
    if col == "Valor":
        return num(v, 2)
    if "Var" in col:
        return num(v, 1)
    return eur(v, 2)


def generar_informe(res, com, aj, base, opt=None, sens_econ=None, sens_est=None, titulo="Optimización de costes en Fly-Next"):
    inf = Informe(titulo, f"Informe generado automáticamente el {datetime.datetime.now():%d/%m/%Y %H:%M} · coste de una "
                          "entrega por composición, coste diario con franjas de tráfico, escenarios, optimización y sensibilidad")
    u = com["umbral_Y"]
    # 1. Planteamiento
    inf.h("1. Planteamiento y parámetros", 1)
    inf.p("Bloque A · coste de una entrega. Cada paquete se asigna a una ruta T (Urbana, Regional o Premium) y, "
          "condicionadas a la ruta, se simulan de forma independiente la distancia D, el consumo G y el retraso M "
          "(método de composición, por transformada inversa). El coste es Y = D·G·P_fuel + M·C_labor + peaje(D), con "
          "peaje(D) = K·I{D > umbral} + t·(D − umbral)⁺ (en el modelo actual t = 0).")
    dist = pd.DataFrame({fx.NOMBRE_RUTA[r]: {
        "Distancia D (km)": (f"Poisson(λ = {num(com['D'][r]['mu'], 1)})" if com["D"][r]["tipo"] == "poisson"
                             else f"N(μ = {num(com['D'][r]['mu'], 1)}; σ = {num(com['D'][r]['sd'], 1)})"),
        "Consumo G (L/km)": f"U({num(com['G'][r][0], 3)}; {num(com['G'][r][1], 3)})",
        "Retraso M (h)": f"Exp(media = {num(com['M'][r], 2)})"} for r in fx.RUTAS})
    inf.tabla(dist, "Distribuciones condicionadas a la ruta", nombre_indice="Variable", formato=lambda f, c, v: v)
    inf.p("Bloque B · franjas horarias. El día tiene 24 horas asignadas a tres franjas; en cada hora llegan "
          "N_h ~ Poisson(λ_franja) pedidos. El tráfico multiplica el retraso y el consumo de cada pedido según la franja "
          "y la ruta. Y_con e Y_sin se calculan con las mismas D, G, M (simulación pareada, números aleatorios comunes) y "
          "Δ = Y_con − Y_sin es el sobrecoste por tráfico.")
    fran = pd.DataFrame({f: {"Horas/día": num(sum(1 for h in range(24) if fx.HORAS_FRANJA[h] == f), 0),
                             "λ (pedidos/hora)": num(com["lambda"][f], 1),
                             "Factor de retraso (U / R / P)": " / ".join(num(com["f_retraso"][f][r], 2) for r in fx.RUTAS),
                             "Factor de consumo (U / R / P)": " / ".join(num(com["f_consumo"][f][r], 2) for r in fx.RUTAS)}
                         for f in fx.FRANJAS})
    inf.tabla(fran, "Franjas horarias, llegadas y factores de tráfico", nombre_indice="Parámetro", formato=lambda f, c, v: v)
    esc = pd.DataFrame({n: {"p (U / R / P)": " / ".join(num(x, 2) for x in r["teoria"]["A"]["p"]),
                            "P_fuel / C_labor": f"{num(r['esc']['P_fuel'], 2)} / {num(r['esc']['C_labor'], 1)}",
                            "K / t (€/km)": f"{num(r['esc']['K'], 0)} / {num(r['esc']['tarifa_km'], 2)}",
                            "×G proveedor": num(r["esc"]["factor_G"], 2),
                            "Urbanos Punta → Valle": pct(r["esc"]["desplazar"], 0),
                            "Premium externalizado": eur(r["esc"]["coste_ext"]) if r["esc"]["externalizar"] else "no",
                            "Coste de implantación (€/día)": eur(r["esc"].get("coste_impl") or 0)}
                        for n, r in res.items()})
    inf.tabla(esc, "Escenarios analizados", nombre_indice="Parámetro", formato=lambda f, c, v: v)
    desc = [f"{n}: {r['esc']['descripcion']}" for n, r in res.items() if r["esc"].get("descripcion")]
    if desc:
        inf.vinetas(desc)
    ob = com["objetivos"]
    o = lambda k, f: f(ob[k]) if ob.get(k) is not None else "no se evalúa"
    tc = pd.DataFrame({"Valor": {
        "Umbral de entrega cara": eur(u), "Entregas al año (ahorros anuales)": num(com["entregas_ano"], 0),
        "Presupuesto diario": eur(com["presupuesto"]),
        "Capacidad máxima por hora": num(com["capacidad"], 0) if com.get("capacidad") else "sin límite",
        "Objetivo · E(Y) máximo por entrega": o("EY_MAX", lambda v: eur(v, 2)),
        "Objetivo · P(Y > umbral) máxima": o("PCOLA_MAX", lambda v: pct(v, 1)),
        "Objetivo · coste diario máximo": o("DIA_MAX", lambda v: eur(v)),
        "Objetivo · P(coste diario > presupuesto) máxima": o("PPRES_MAX", lambda v: pct(v, 1)),
        "Simulación A · entregas por réplica × réplicas": f"{num(aj['nsim'], 0)} × {aj['replicas']} (semillas {aj['semilla']} …)",
        "Simulación B · días por réplica × réplicas": f"{num(aj['ndias'], 0)} × {aj['replicas_B']} (semillas {aj['semilla'] + 1000} …)"}})
    inf.tabla(tc, "Parámetros comunes, objetivos y ajustes de simulación", nombre_indice="Parámetro", formato=lambda f, c, v: v)

    # 2. Síntesis
    inf.h("2. Síntesis y recomendación (O7.1)", 1)
    for n, r in res.items():
        inf.p(fx.texto_escenario(n, r, com), negrita_inicio=n + ":")
    inf.p("Recomendación: " + fx.texto_recomendacion(res, base, com), negrita_inicio="Recomendación:")
    inf.tabla(fx.tabla_comparativa(res, com), "Comparación de escenarios (filas «Teórico»: fórmulas e integración "
              "numérica; filas «Simulado»: media de las macro-réplicas)", formato=fmt_kpi, nombre_indice="KPI")
    inf.tabla(fx.tabla_objetivos(res), "Cumplimiento de los objetivos (valores teóricos)", nombre_indice="Objetivo",
              formato=lambda f, c, v: str(v))
    inf.figura(gr.fig_comparativa(res, com), "Coste por entrega, entregas caras, coste diario y riesgo presupuestario por escenario")
    dif = fx.diferencias(res, base, com)
    if dif is not None:
        inf.p(f"Ahorros frente a «{base}» (O3.3, O3.4, O5.1 y O5.2). Las diferencias simuladas son pareadas: cada réplica "
              "usa las mismas semillas en todos los escenarios, por lo que el IC95 % del ahorro es mucho más estrecho que "
              "el de cada coste por separado.")
        cols_a = ["Ahorro por entrega · teórico (€)", "Ahorro por entrega · simulado (€)", "IC95 ahorro por entrega (€)",
                  "Ahorro anual · teórico (€)", "Cambio en σ(Y) · teórico (€)", "Cambio en P(Y > umbral) · teórico"]
        cols_b = ["Ahorro diario · teórico (€)", "Ahorro diario · simulado (€)", "IC95 ahorro diario (€)",
                  "Ahorro en sobrecoste diario · teórico (€)", "Coste de implantación (€/día)", "Ahorro diario neto · teórico (€)"]
        inf.tabla(dif[cols_a], f"Ahorro por entrega y anual ({num(com['entregas_ano'], 0)} entregas) frente a «{base}»",
                  formato=fmt_dif, nombre_indice="Escenario")
        inf.tabla(dif[cols_b], f"Ahorro diario frente a «{base}» (modelo con franjas)", formato=fmt_dif, nombre_indice="Escenario")
        inf.figura(gr.fig_ahorros(dif), f"Ahorro diario frente a «{base}»: teórico, simulado (IC95 % pareado) y neto de implantación")
    inf.figura(gr.fig_validacion(res, com), "Error relativo de la simulación frente a los valores teóricos")
    inf.h("Cuadro de mando propuesto (KPIs mensuales para la Dirección)", 2)
    inf.tabla(pd.DataFrame({"Por qué es relevante": dict(fx.KPIS_CUADRO)}), "Cuadro de mando conceptual",
              nombre_indice="KPI", formato=lambda f, c, v: v)

    # 3. Detalle por escenario
    sec = 3
    for n, r in res.items():
        inf.salto()
        t = r["teoria"]
        inf.h(f"{sec}. Escenario «{n}»", 1)
        if r["esc"].get("descripcion"):
            inf.p(r["esc"]["descripcion"])
        inf.h(f"{sec}.1 Bloque A · coste de una entrega (O1 y O2.1)", 2)
        inf.tabla(_sub(r["resA"], filas_A_global(u)), f"Coste de una entrega: teórico y simulado con IC95 % · {n}",
                  formato=fmt_res, nombre_indice="KPI")
        inf.tabla(tabla_A_rutas(t, r["resA"], u), f"Coste por ruta (O1.3) · {n}", formato=fmt_rutas, nombre_indice="Ruta")
        inf.figura(gr.fig_hist_Y(t["A"], r["mcA"]["Y"], u, f"Distribución del coste de una entrega · {n}"),
                   f"Coste simulado frente a la distribución teórica (escala logarítmica) · {n}")
        inf.figura(gr.fig_por_ruta(t["A"], r["resA"], u, f"Coste por ruta · {n}"), f"Media, cola y variabilidad relativa por ruta · {n}")
        filas_d = [f"Contribución a Var(Y) · {x}" for x in fx.TERMINOS]
        inf.tabla(_sub(r["resA"], filas_d), f"Descomposición de la varianza de Y (O2.1) · {n}", formato=fmt_res, nombre_indice="Término")
        inf.figura(gr.fig_descomposicion(t["A"], r["resA"], f"Descomposición de Var(Y) · {n}"),
                   f"Parte de la varianza del coste atribuible a cada término · {n}")
        inf.h(f"{sec}.2 Bloque B · coste diario con franjas de tráfico (O4, O5.3, O6 y O8)", 2)
        filas_b = [k for k in r["resB"].index if "por pedido" not in k and "Cuota" not in k]
        inf.tabla(_sub(r["resB"], filas_b), f"Coste diario: teórico y simulado con IC95 % · {n}", formato=fmt_res,
                  nombre_indice="KPI")
        inf.tabla(tabla_franjas(t, r["resB"]), f"Coste por pedido y sobrecoste por franja (O4.1 y O5.3) · {n}",
                  formato=fmt_franjas, nombre_indice="Franja")
        inf.figura(gr.fig_hist_dias(t["B"], r["mcB"]["dias"]["Coste con tráfico"], com["presupuesto"],
                                    f"Coste diario · {n}"),
                   f"Coste diario de los días simulados, percentil 95, CVaR 5 % y presupuesto · {n}")
        inf.figura(gr.fig_franjas(t["B"], r["resB"], f"Franjas horarias · {n}"),
                   f"Coste medio por pedido y cuota del sobrecoste por franja · {n}")
        inf.figura(gr.fig_horas(com, t["B"], f"Perfil horario del coste · {n}"), f"Coste esperado en cada hora del día · {n}")
        inf.tabla(tabla_extras_B(t, r["mcB"], com), f"Dimensionamiento, diseño pareado y equilibrio de franjas · {n}",
                  nombre_indice="Indicador", formato=lambda f, c, v: v)
        inf.h(f"{sec}.3 Funcionamiento simulado", 2)
        tr = r["mcA"]["traza"]
        e1 = fx.una_entrega(com, r["esc"], float(tr["U_T"].iloc[0]), 0.5, 0.5, 0.5, "Punta")
        inf.figura(gr.fig_mecanismo(t["A"]["p"], float(tr["U_T"].iloc[0]), e1["sin"], e1["con"],
                                    f"Composición · una entrega en hora Punta · {n}"),
                   f"Ruta elegida con U_T = {num(float(tr['U_T'].iloc[0]), 2)} y coste por términos con U_D = U_G = U_M = 0,5 · {n}")
        inf.figura(gr.fig_traza_A(tr, r["esc"], f"Primeras {len(tr)} entregas · {n}"),
                   f"Coste por términos y ruta de las primeras {len(tr)} entregas simuladas · {n}")
        sec += 1

    if opt is not None:
        inf.salto()
        inf.h(f"{sec}. Optimización de las probabilidades de ruta (O3.1)", 1)
        inf.p(f"Problema: minimizar ‖p − p₀‖² sujeto a Σ pⱼ·μⱼ ≤ (1 − {num(opt['reduccion'], 2)})·Σ p₀ⱼ·μⱼ, Σ pⱼ = 1 y pⱼ ≥ 0, "
              "con μⱼ = E(Y | T = j) y sin cambiar las distribuciones condicionadas. Como E(Y) es lineal en p, la región "
              "factible es un semiplano del símplex; se resuelve sobre una rejilla de paso 0,001 con las medias teóricas y "
              "con las estimadas por simulación, y la solución simulada se verifica re-simulando con las mismas semillas.")
        filas = {}
        for k_, et in (("teo", "Con medias teóricas"), ("sim", "Con medias simuladas")):
            s = opt[k_]
            if s is None:
                filas[et] = {"p (U / R / P)": "no factible", "E(Y) objetivo": "—", "E(Y) con p*": "—", "Distancia a p₀": "—"}
            else:
                filas[et] = {"p (U / R / P)": " / ".join(num(x, 3) for x in s["p"]), "E(Y) objetivo": eur(s["objetivo"], 2),
                             "E(Y) con p*": eur(s["E"], 2), "Distancia a p₀": num(s["distancia"], 3)}
        if opt.get("verif"):
            v = opt["verif"]
            filas["Verificación (re-simulación)"] = {"p (U / R / P)": "p* simulado", "E(Y) objetivo": "—",
                                                     "E(Y) con p*": f"{eur(v['E'], 2)} [{eur(v['lo'], 2)}; {eur(v['hi'], 2)}]",
                                                     "Distancia a p₀": f"teórico {eur(v['E_teo'], 2)}"}
        inf.tabla(pd.DataFrame(filas).T, "Solución del problema de optimización", nombre_indice="Cálculo", formato=lambda f, c, v: v)
        inf.figura(gr.fig_optimizacion(opt), "Coste esperado en el símplex de probabilidades, región factible y solución")
        sec += 1
    if sens_econ is not None:
        df, elas = sens_econ
        inf.salto()
        inf.h(f"{sec}. Sensibilidad de la varianza a los parámetros económicos (O2.2)", 1)
        mayor = max(elas, key=elas.get)
        inf.p("Se varía cada parámetro económico manteniendo fijos los demás. La elasticidad es la variación porcentual de "
              f"Var(Y) por cada 1 % de variación del parámetro. El parámetro con mayor impacto es {mayor} "
              f"(elasticidad {num(elas[mayor], 2)}): es donde conviene concentrar los esfuerzos.")
        inf.figura(gr.fig_sens_econ(df, elas), "Var(Y) relativa al variar cada parámetro y elasticidad en el punto base")
        inf.tabla(tabla_sens_econ(df), "Barrido de los parámetros económicos (teórico y simulado con números aleatorios comunes)",
                  formato=fmt_sens_econ, nombre_indice="Parámetro · variación")
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
    inf.h("Anexo. Metodología", 1)
    inf.vinetas([
        "Composición: T por transformada inversa sobre (p₁, p₂, p₃); D | T por la inversa de la Poisson o de la normal; "
        "G | T = g₀ + (g₁ − g₀)·U; M | T = −media·ln(1 − U). Cada entrega usa 5 uniformes, de modo que todos los "
        "escenarios comparten los mismos números aleatorios (diferencias pareadas).",
        "Momentos exactos: E(Y | T) = P_fuel·E(D)·E(G) + C_labor·E(M) + E(peaje); Var(Y) y la descomposición por términos "
        "usan E(D²), E[(D − c)⁺], E[D·I(D > c)]… (fórmulas cerradas para la normal y sumas exactas para la Poisson). "
        "La contribución de cada término es Cov(término, Y)/Var(Y) y las contribuciones suman el 100 %.",
        "Distribución de Y: P(Y ≤ y) se integra de forma exacta en G (uniforme) y M (exponencial) y numéricamente en D; de "
        "ella salen P(Y > umbral) y los percentiles teóricos.",
        "Coste diario: suma compuesta de Poisson. E = Σ_h E(N_h)·E(Y_h) y Var = Σ_h [E(N_h)·Var(Y_h) + Var(N_h)·E(Y_h)²]; "
        "su percentil 95, el CVaR 5 % (μ + σ·φ(z₀.₉₅)/0,05) y P(coste > presupuesto) usan la aproximación normal.",
        "Macro-réplicas: cada KPI se calcula en cada réplica (incluidos percentiles, CVaR y probabilidades) y se da la media "
        "de las réplicas con IC95 % de la t de Student.",
        "Material de Javier Morales, equipo IA4LEGOS (CC BY-SA 4.0).",
    ])
    return inf.guardar()
