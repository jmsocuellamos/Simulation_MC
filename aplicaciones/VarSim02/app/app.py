# -*- coding: utf-8 -*-
"""Riesgo y retorno de planes de inversión · aplicación Streamlit.
Transformada inversa, simulación Monte Carlo del retorno, comparación de planes (riesgo frente a retorno),
sensibilidad del estimador e informe Word.

Ejecutar con:   streamlit run app.py
"""
import copy
import datetime
import io
import json
import sys

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import inversion as iv
import graficos as gr
import informe
from inversion import num, pct, eur

st.set_page_config(page_title="Simulador · Riesgo y retorno", page_icon="📈", layout="wide")
EN_NAVEGADOR = sys.platform == "emscripten"

ESCENARIOS_DEF = {
    "A · fijo": {"valores": [27500.0], "probs": [100.0],
                 "descripcion": "Retorno fijo y seguro: 27.500 € con probabilidad 1."},
    "B · moderado": {"valores": [27000.0, 28000.0], "probs": [40.0, 60.0],
                     "descripcion": "Dos resultados posibles: 27.000 € (40 %) o 28.000 € (60 %)."},
    "C · agresivo": {"valores": [24000.0, 27000.0, 33000.0], "probs": [20.0, 50.0, 30.0],
                     "descripcion": "Tres escenarios: bajo 24.000 € (20 %), medio 27.000 € (50 %) y alto 33.000 € (30 %)."},
}
COMUNES_DEF = {"I0": 25000.0, "objetivos": dict(iv.OBJETIVOS_DEF)}
AJUSTES_DEF = {"nsim": 10000, "replicas": 5, "semilla": 1000}
SENS_DEF = {"nsim_min": 200, "nsim_max": 20000, "puntos": 6, "replicas": 5}
COL_V, COL_P = "Retorno (€)", "Probabilidad (%)"

# ==============================================================================================
# Estado de la sesión
# ==============================================================================================
ss = st.session_state
if "escenarios" not in ss:
    ss.escenarios = copy.deepcopy(ESCENARIOS_DEF)
    ss.comunes = copy.deepcopy(COMUNES_DEF)
    ss.ajustes = copy.deepcopy(AJUSTES_DEF)
    ss.sens_cfg = copy.deepcopy(SENS_DEF)
    ss.resultados = None
    ss.sens = None
    ss.informe_bytes = None
    ss.version = 0
    ss.esc_activo = list(ss.escenarios)[-1]
    ss.u_ejemplo = 0.55


def ver_fig(contenedor, fig):
    """Muestra una figura de matplotlib y la cierra para no acumular memoria entre ejecuciones."""
    contenedor.pyplot(fig, use_container_width=True)
    plt.close(fig)


def tabla_fmt(df, fn):
    out = df.copy().astype(object)
    for f in df.index:
        for c in df.columns:
            out.loc[f, c] = fn(f, c, df.loc[f, c])
    return out


def mostrar_tabla(df, fn=None, primera="", ancho_primera="medium", iconos=False):
    """Tabla a todo el ancho, con todos los valores ya formateados, la primera columna con el nombre de la fila y
    una altura que muestra todas las filas sin barra de desplazamiento."""
    d = (tabla_fmt(df, fn) if fn else df.astype(object)).copy()
    nombre = primera or df.index.name or " "
    d.index = [num(i, 0) if isinstance(i, (int, np.integer)) else str(i) for i in d.index]
    d.index.name = nombre
    d = d.reset_index()
    d.columns = [str(c) for c in d.columns]
    cfg = {nombre: st.column_config.TextColumn(nombre, width=ancho_primera)}
    for c in d.columns[1:]:
        et = c
        if iconos and c.startswith(("Teóric", "Simulad")):
            et = ("📐 " if c.startswith("Teóric") else "🎲 ") + c
        cfg[c] = st.column_config.TextColumn(et)
    st.dataframe(d.astype(str), use_container_width=True, hide_index=True, column_config=cfg,
                 height=38 + 35 * len(d))


def clave(*partes):
    """Clave de widget ligada a la versión de la configuración (al cargar o restablecer se regeneran)."""
    return "__".join(str(p) for p in partes) + f"__v{ss.version}"


def configuracion():
    return {"comunes": ss.comunes, "ajustes": ss.ajustes, "escenarios": ss.escenarios, "sensibilidad": ss.sens_cfg}


def cargar_configuracion(d):
    ss.comunes = {**copy.deepcopy(COMUNES_DEF), **d.get("comunes", {})}
    ss.comunes["objetivos"] = {**iv.OBJETIVOS_DEF, **d.get("comunes", {}).get("objetivos", {})}
    ss.ajustes = {**copy.deepcopy(AJUSTES_DEF), **{k: v for k, v in d.get("ajustes", {}).items() if k in AJUSTES_DEF}}
    ss.sens_cfg = {**copy.deepcopy(SENS_DEF), **d.get("sensibilidad", {})}
    ss.escenarios = d.get("escenarios") or copy.deepcopy(ESCENARIOS_DEF)
    ss.esc_activo = list(ss.escenarios)[-1]
    ss.resultados, ss.sens, ss.informe_bytes = None, None, None
    ss.version += 1


def teoria_plan(e):
    return iv.teoria(e["valores"], e["probs"], ss.comunes["I0"])


com, aj = ss.comunes, ss.ajustes

# ==============================================================================================
# Barra lateral
# ==============================================================================================
with st.sidebar:
    st.header("⚙️ Simulación")
    aj["nsim"] = int(st.number_input("Simulaciones por réplica (nsim)", 10, 10_000_000, int(aj["nsim"]), 1000,
                                     key=clave("nsim"), help="Número de retornos simulados en cada réplica."))
    aj["replicas"] = int(st.number_input("Réplicas (semillas)", 2, 100, int(aj["replicas"]), key=clave("rep"),
                                         help="Cada réplica usa una semilla distinta (semilla + i); el IC95 % se calcula "
                                              "con la variabilidad entre réplicas. Todos los planes usan las mismas "
                                              "semillas (números aleatorios comunes)."))
    aj["semilla"] = int(st.number_input("Semilla de inicialización", 0, 10**9, int(aj["semilla"]), key=clave("sem")))
    st.caption(f"Por plan: {num(aj['nsim'] * aj['replicas'], 0)} retornos generados por transformada inversa.")
    nombres = list(ss.escenarios)
    sel = st.multiselect("Planes a simular", nombres, default=nombres, key=clave("sel"))
    if EN_NAVEGADOR:
        st.caption("En el navegador la simulación es más lenta: con muchas simulaciones o réplicas puede tardar.")
    if st.button("▶️ Ejecutar análisis", type="primary", use_container_width=True, disabled=not sel):
        barra = st.progress(0.0, text="Simulando…")
        res = {}
        total = len(sel) * aj["replicas"]
        for i, n in enumerate(sel):
            def prog(k, m, i=i, n=n):
                barra.progress((i * aj["replicas"] + k) / total, text=f"{n}: réplica {k} de {m}")
            res[n] = iv.analizar_escenario(ss.escenarios[n], com, aj, prog)
        ss.resultados = {"res": res, "comunes": copy.deepcopy(com), "ajustes": copy.deepcopy(aj),
                         "fecha": f"{datetime.datetime.now():%d/%m/%Y %H:%M}"}
        ss.informe_bytes = None
        barra.progress(1.0, text="Análisis completado")

    st.divider()
    st.subheader("💾 Configuración")
    st.download_button("Descargar configuración (JSON)", json.dumps(configuracion(), ensure_ascii=False, indent=2),
                       "configuracion_inversion.json", "application/json", use_container_width=True)
    fich = st.file_uploader("Cargar configuración (JSON)", type="json", key=clave("json"))
    if fich is not None and st.button("Aplicar configuración cargada", use_container_width=True):
        cargar_configuracion(json.loads(fich.getvalue().decode("utf-8")))
        st.rerun()
    if st.button("↺ Restablecer valores por defecto", use_container_width=True):
        cargar_configuracion({})
        st.rerun()

# ==============================================================================================
# Cabecera
# ==============================================================================================
st.title("📈 Riesgo y retorno de planes de inversión")
st.caption("Transformada inversa para el retorno de cada plan, simulación Monte Carlo, comparación de riesgo frente a "
           "retorno, sensibilidad del estimador e informe Word. Basado en el Ejemplo 2 de «B-70. Distribuciones "
           "derivadas», Javier Morales y Asun Martínez Mayoral, equipo IA4LEGOS (CC BY-SA 4.0).")
tab_com, tab_esc, tab_res, tab_sens, tab_inf = st.tabs(["🌐 Parámetros comunes", "🏕️ Planes", "📊 Resultados",
                                                        "🔬 Sensibilidad", "📄 Informe"])

# ==============================================================================================
# 1) Parámetros comunes
# ==============================================================================================
with tab_com:
    st.subheader("Parámetros comunes a todos los planes")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("#### 💶 Inversión")
        com["I0"] = float(st.number_input("Inversión inicial I₀ (€)", 0.0, 1e12, float(com["I0"]), 500.0, key=clave("I0"),
                                          help="Capital invertido. Hay pérdida patrimonial si el retorno R es menor que I₀."))
    with c2:
        st.markdown("#### 🏆 Objetivos")
        st.caption("Criterios que deben cumplir los planes (se evalúan sobre los valores teóricos). "
                   "Desmarca un objetivo para no evaluarlo.")
        ob = com["objetivos"]
        def objetivo(clave_obj, etiqueta, defecto, factor, paso, ayuda):
            cc1, cc2 = st.columns([1, 2])
            activo = cc1.checkbox("Evaluar", value=ob.get(clave_obj) is not None, key=clave("obj_on", clave_obj))
            v = ob.get(clave_obj) if ob.get(clave_obj) is not None else defecto
            x = cc2.number_input(etiqueta, -1e12, 1e12, float(v) * factor, paso, key=clave("obj", clave_obj), help=ayuda,
                                 disabled=not activo)
            ob[clave_obj] = (x / factor) if activo else None
        objetivo("RENT_MIN", "Rentabilidad esperada mín. (%)", 0.102, 100, 0.1,
                 "Rentabilidad esperada mínima, (E(R) − I₀)/I₀.")
        objetivo("PLOSS_MAX", "P(pérdida) máxima (%)", 0.05, 100, 1.0,
                 "Probabilidad máxima admisible de pérdida patrimonial, P(R < I₀).")
        objetivo("SIGMA_MAX", "Riesgo σ(R) máximo (€)", 1000.0, 1, 100.0,
                 "Desviación típica máxima admisible del retorno.")
        objetivo("P5NETO_MIN", "Neto mínimo en el peor 5 % (€)", 0.0, 1, 100.0,
                 "Resultado neto mínimo en el percentil 5, P5(R) − I₀: el peor resultado salvo en 1 de cada 20 casos.")

# ==============================================================================================
# 2) Planes (escenarios)
# ==============================================================================================
with tab_esc:
    st.subheader("Planes de inversión")
    nombres = list(ss.escenarios)
    if ss.esc_activo not in nombres:
        ss.esc_activo = nombres[0]
    c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
    activo = c1.selectbox("Plan a editar", nombres, index=nombres.index(ss.esc_activo), key=clave("activo"))
    ss.esc_activo = activo
    nuevo = c2.text_input("Nombre para copiar / renombrar", key=clave("nuevo_nombre"), placeholder="nombre del plan")
    if c3.button("Copiar", use_container_width=True, help="Crea un plan nuevo copiando el actual"):
        if nuevo and nuevo not in ss.escenarios:
            ss.escenarios[nuevo] = copy.deepcopy(ss.escenarios[activo]); ss.esc_activo = nuevo; ss.version += 1; st.rerun()
        else:
            st.warning("Escribe un nombre nuevo que no exista.")
    if c4.button("Renombrar", use_container_width=True):
        if nuevo and nuevo not in ss.escenarios:
            ss.escenarios = {(nuevo if k == activo else k): v for k, v in ss.escenarios.items()}
            ss.esc_activo = nuevo; ss.version += 1; st.rerun()
        else:
            st.warning("Escribe un nombre nuevo que no exista.")

    e = ss.escenarios[activo]
    st.markdown(f"#### Tramos de «{activo}»")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.caption(f"Cada fila es un retorno posible y su probabilidad (hasta {iv.MAX_TRAMOS} tramos). Añade o borra filas "
                   "con los controles de la tabla; las probabilidades se normalizan si no suman 100 %.")
        k_df = clave("tramos", activo)
        k_base = "base_" + k_df
        if k_base not in ss:          # datos de partida fijos para el editor (evita aplicar dos veces los cambios)
            ss[k_base] = pd.DataFrame({COL_V: [float(x) for x in e["valores"]], COL_P: [float(x) for x in e["probs"]]})
        ed = st.data_editor(ss[k_base], key=k_df, num_rows="dynamic", use_container_width=True, hide_index=True,
                            column_config={COL_V: st.column_config.NumberColumn(COL_V, step=100.0, format="%.0f"),
                                           COL_P: st.column_config.NumberColumn(COL_P, min_value=0.0, max_value=100.0,
                                                                                step=1.0, format="%.1f")})
        ed = ed.dropna(how="any").head(iv.MAX_TRAMOS)
        if len(ed):
            e["valores"] = [float(x) for x in ed[COL_V]]
            e["probs"] = [float(x) for x in ed[COL_P]]
        else:
            st.error("El plan necesita al menos un tramo con retorno y probabilidad.")
        suma = sum(e["probs"])
        if abs(suma - 100) > 1e-9:
            st.warning(f"Las probabilidades suman {num(suma, 1)} %: se normalizan para sumar 100 %.")
        e["descripcion"] = st.text_area("Descripción", e.get("descripcion", ""), key=clave("desc", activo), height=90)

    t = teoria_plan(e)
    with c2:
        st.markdown("##### 📐 Vista previa teórica (fórmulas exactas, sin simular)")
        m = st.columns(3)
        m[0].metric("Retorno esperado E(R)", eur(t["E"]), help="Σ pᵢ·vᵢ")
        m[1].metric("Riesgo σ(R)", eur(t["sd"]), help="√(Σ pᵢ·(vᵢ − E(R))²)")
        m[2].metric("P(pérdida)", pct(t["Ploss"], 1), help=f"P(R < I₀ = {eur(com['I0'])})")
        m = st.columns(3)
        m[0].metric("Neto esperado E(R) − I₀", eur(t["neto"]))
        m[1].metric("Rentabilidad esperada", pct(t["rent"], 2))
        m[2].metric("Neto en el peor 5 %", eur(t["P5_neto"]), help="Percentil 5 de R menos I₀.")
        ver_fig(st, gr.fig_pmf(t))

    with st.expander("🎯 Mecanismo · una tirada de la transformada inversa", expanded=True):
        st.caption("1 · Se genera U ~ Unif(0, 1).  2 · Se busca el menor índice i tal que U ≤ F(i) = p₁ + … + pᵢ (el "
                   "tramo).  3 · Se devuelve el retorno R = vᵢ. Repetido muchas veces, reproduce las probabilidades del plan."
                   + ("  El plan es determinista: cualquier U cae en el único tramo." if t["tramos"] == 1 else ""))
        c1, c2 = st.columns([3, 1])
        if c2.button("🎲 U aleatorio", use_container_width=True):
            ss.u_ejemplo = float(np.round(np.random.random(), 2))
            ss.pop("u_slider", None)
        ss.u_ejemplo = c1.slider("U ~ Unif(0, 1)", 0.0, 1.0, float(ss.u_ejemplo), 0.01, key="u_slider")
        k = int(iv.transformada_inversa(np.array([ss.u_ejemplo]), t["F"])[0])
        m = st.columns(3)
        m[0].metric("U generado", num(ss.u_ejemplo, 2))
        m[1].metric("Retorno simulado R", eur(t["v"][k]), f"tramo {k + 1} de {t['tramos']}", delta_color="off")
        neto = t["v"][k] - com["I0"]
        m[2].metric("Resultado neto R − I₀", eur(neto), "ganancia" if neto >= 0 else "pérdida",
                    delta_color="normal" if neto >= 0 else "inverse")
        ver_fig(st, gr.fig_escalera(t, ss.u_ejemplo))
        mostrar_tabla(iv.tabla_tramos(t), informe.fmt_tramos, "Tramo", "small", iconos=True)

    st.divider()
    c1, c2 = st.columns([3, 1])
    c1.caption("Los cambios se aplican al pulsar «Ejecutar análisis» en la barra lateral.")
    if c2.button("🗑️ Eliminar este plan", use_container_width=True, disabled=len(ss.escenarios) <= 1):
        ss.escenarios.pop(activo); ss.esc_activo = next(iter(ss.escenarios)); ss.version += 1; st.rerun()

# ==============================================================================================
# 3) Resultados
# ==============================================================================================
with tab_res:
    R = ss.resultados
    if R is None:
        st.info("Todavía no hay resultados. Configura los planes y pulsa «▶️ Ejecutar análisis» en la barra lateral.")
    else:
        res = R["res"]
        st.caption(f"Análisis del {R['fecha']} · {R['ajustes']['replicas']} réplicas de {num(R['ajustes']['nsim'], 0)} "
                   f"simulaciones · I₀ = {eur(R['comunes']['I0'])} · semilla {R['ajustes']['semilla']} · "
                   f"planes: {', '.join(res)}")
        st.info("📐 **Teórico**: fórmulas exactas de la distribución discreta del retorno (E(R) = Σ pᵢ·vᵢ). "
                "🎲 **Simulado**: transformada inversa (media de las réplicas, con IC95 %).")
        r_sin, r_det, r_dat = st.tabs(["Síntesis y comparación", "Detalle por plan", "Datos de las réplicas (simulados)"])
        with r_sin:
            st.markdown("#### Síntesis")
            for n, r in res.items():
                st.markdown(f"**{n}.** " + iv.texto_escenario(n, r)[len(n) + 2:])
            st.success("**Recomendación.** " + iv.texto_recomendacion(res))
            c1, c2 = st.columns(2)
            ver_fig(c1, gr.fig_riesgo_retorno(res))
            with c2:
                st.markdown("**Objetivos** (sobre los valores teóricos)")
                mostrar_tabla(iv.tabla_objetivos(res), None, "Objetivo", "large")
            st.markdown("**Comparación de planes** (filas «Teórico» = fórmulas; filas «Simulado» = Monte Carlo)")
            mostrar_tabla(iv.tabla_comparativa(res), iv.fmt_kpi, "KPI", "large")
            ver_fig(st, gr.fig_comparativa(res))
            ver_fig(st, gr.fig_validacion(res))

        with r_det:
            n = st.selectbox("Plan", list(res), index=len(res) - 1, key="det_esc")
            r = res[n]
            t, mc, rm = r["teoria"], r["mc"], r["resumen_mc"]
            d_teo, d_sim, d_dis = st.tabs(["📐 Resultados teóricos", "🎲 Resultados simulados", "🔢 Distribución del retorno"])
            with d_teo:
                st.caption("Valores exactos de la distribución discreta del retorno.")
                m = st.columns(4)
                m[0].metric("Retorno esperado E(R)", eur(t["E"]))
                m[1].metric("Riesgo σ(R)", eur(t["sd"]))
                m[2].metric("P(pérdida) P(R < I₀)", pct(t["Ploss"], 2))
                m[3].metric("Pérdida esperada", eur(t["perdida_media"]), help="E[max(I₀ − R, 0)]")
                m = st.columns(4)
                m[0].metric("Neto esperado E(R) − I₀", eur(t["neto"]))
                m[1].metric("Rentabilidad esperada", pct(t["rent"], 2))
                m[2].metric("Percentil 5 de R", eur(t["P5"]))
                m[3].metric("Coef. de variación", pct(t["CV"], 2))
                st.markdown("**Objetivos** (evaluados sobre los valores teóricos)")
                mostrar_tabla(r["objetivos"], iv.fmt_obj, "Objetivo", "large")
            with d_sim:
                st.caption(f"{R['ajustes']['replicas']} réplicas de {num(R['ajustes']['nsim'], 0)} simulaciones; cada KPI se "
                           "calcula en cada réplica y se muestra la media de las réplicas con su IC95 %.")
                m = st.columns(4)
                for col, (kpi, et, es_pct) in zip(m, [("Retorno medio E(R)", "E(R) simulado", False),
                                                      ("Riesgo σ(R)", "σ(R) simulado", False),
                                                      ("Probabilidad de pérdida P(R < I₀)", "P(pérdida) simulada", True),
                                                      ("Rentabilidad esperada", "Rentabilidad simulada", True)]):
                    vs, vt = rm.loc[kpi, "Simulado · media"], rm.loc[kpi, "Teórico"]
                    f = (lambda x: pct(x, 2)) if es_pct else eur
                    col.metric(et, f(vs), "teórico " + f(vt), delta_color="off")
                st.markdown("**KPIs simulados frente a teóricos**")
                mostrar_tabla(rm, iv.fmt_mc, "KPI", "large")
                c1, c2 = st.columns(2)
                ver_fig(c1, gr.fig_replicas(mc, t))
                ver_fig(c2, gr.fig_traza(mc["traza"], t))
                with st.expander(f"Primeras {len(mc['traza'])} simulaciones (U, tramo, retorno y resultado neto)"):
                    tr = mc["traza"].copy()
                    tr["U"] = tr["U"].map(lambda v: num(v, 4))
                    tr["Retorno"] = tr["Retorno"].map(eur); tr["Neto"] = tr["Neto"].map(eur)
                    st.dataframe(tr.astype(str), use_container_width=True, hide_index=True)
            with d_dis:
                st.markdown("**Tramos del plan** (intervalo de U que produce cada retorno en la transformada inversa)")
                mostrar_tabla(iv.tabla_tramos(t, mc), informe.fmt_tramos, "Tramo", "small", iconos=True)
                c1, c2 = st.columns(2)
                ver_fig(c1, gr.fig_pmf(t, mc))
                ver_fig(c2, gr.fig_cdf(t, mc["R"]))
                st.markdown("**Calculadora de probabilidad del retorno**")
                c1, c2, c3, c4 = st.columns(4)
                tipo = c1.selectbox("Probabilidad", ["<", "≤", "≥", ">"], format_func=lambda x: f"P(R {x} x)", key="prob_tipo")
                x0 = float(c2.number_input("x (€)", -1e12, 1e12, float(t["I0"]), 500.0, key=f"prob_x_{n}"))
                p_teo, p_sim = iv.prob_retorno(t, mc["R"], tipo, x0)
                c3.metric(f"📐 Teórica · P(R {tipo} {num(x0, 0)})", pct(p_teo, 2))
                c4.metric(f"🎲 Simulada · P(R {tipo} {num(x0, 0)})", pct(p_sim, 2))
                st.markdown("**Percentiles del retorno**")
                tq = iv.tabla_cuantiles(t, mc)
                mostrar_tabla(tq, informe.fmt_euros, "Percentil", "small", iconos=True)

        with r_dat:
            st.markdown("Resultados **simulados** de cada réplica (una fila por réplica y plan).")
            largo = pd.concat([r["mc"]["replicas"].assign(Plan=n) for n, r in res.items()]).reset_index()
            st.dataframe(largo, use_container_width=True, hide_index=True)
            buf = io.BytesIO()
            try:
                with pd.ExcelWriter(buf, engine="openpyxl") as xw:
                    largo.to_excel(xw, sheet_name="Replicas", index=False)
                    iv.tabla_comparativa(res).astype(str).to_excel(xw, sheet_name="Comparacion")
                    for i, (n, r) in enumerate(res.items(), 1):
                        iv.tabla_tramos(r["teoria"], r["mc"]).to_excel(xw, sheet_name=f"Tramos {i}")
                        iv.tabla_cuantiles(r["teoria"], r["mc"]).to_excel(xw, sheet_name=f"Percentiles {i}")
                    if ss.sens is not None:
                        informe.tabla_sens(ss.sens["df"]).to_excel(xw, sheet_name="Sensibilidad")
                st.download_button("Descargar Excel", buf.getvalue(), "resultados_inversion.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception:  # openpyxl no disponible
                st.download_button("Descargar CSV", largo.to_csv(index=False).encode("utf-8"), "replicas_inversion.csv", "text/csv")

# ==============================================================================================
# 4) Sensibilidad del estimador
# ==============================================================================================
with tab_sens:
    st.subheader("Sensibilidad del estimador Monte Carlo")
    st.caption("¿Cuánto «tiembla» la estimación del retorno medio según el número de simulaciones? Para cada nsim se "
               "repite la simulación con varias semillas (semilla de la barra lateral + i).")
    sc = ss.sens_cfg
    c = st.columns(5)
    sc["nsim_min"] = int(c[0].number_input("nsim mínimo", 10, 10_000_000, int(sc["nsim_min"]), 50, key=clave("s_nmin")))
    sc["nsim_max"] = int(c[1].number_input("nsim máximo", 20, 10_000_000, int(sc["nsim_max"]), 1000, key=clave("s_nmax")))
    sc["puntos"] = int(c[2].slider("Puntos a evaluar", 3, 10, int(sc["puntos"]), key=clave("s_pts")))
    sc["replicas"] = int(c[3].slider("Réplicas por punto", 2, 20, int(sc["replicas"]), key=clave("s_rep")))
    nombres = list(ss.escenarios)
    pol = c[4].selectbox("Plan", nombres, index=len(nombres) - 1, key=clave("s_pol"))
    c1, c2 = st.columns([3, 1])
    c1.caption("Los puntos se espacian logarítmicamente entre nsim mínimo y máximo. El plan toma sus retornos y "
               "probabilidades de la pestaña Planes.")
    lanzar = c2.button("▶️ Analizar sensibilidad", type="primary", use_container_width=True)
    if lanzar:
        vals = iv.valores_barrido(sc["nsim_min"], sc["nsim_max"], sc["puntos"])
        tp = teoria_plan(ss.escenarios[pol])
        barra = st.progress(0.0, text="Simulando…")
        df = iv.sensibilidad(tp, vals, sc["replicas"], aj["semilla"],
                             lambda k, m: barra.progress(k / m, text=f"Simulación {k} de {m}"))
        texto, red = iv.texto_sensibilidad(df, pol, tp)
        ss.sens = {"df": df, "texto": texto, "reduccion": red,
                   "cfg": {"replicas": sc["replicas"], "plan": pol, "semilla": aj["semilla"]}}
        ss.informe_bytes = None
        barra.progress(1.0, text="Barrido completado")
    S = ss.sens
    if S is None:
        st.info("Pulsa «▶️ Analizar sensibilidad» para ejecutar el barrido.")
    else:
        df, cfg = S["df"], S["cfg"]
        m = st.columns(4)
        m[0].metric("Plan analizado", cfg["plan"])
        m[1].metric("Rango de nsim", f"{num(df.index.min(), 0)} – {num(df.index.max(), 0)}", f"{len(df)} puntos", delta_color="off")
        m[2].metric("Réplicas por punto", str(cfg["replicas"]), "semillas independientes", delta_color="off")
        m[3].metric("Reducción de variabilidad", ("×" + num(S["reduccion"], 1)) if S["reduccion"] else "—",
                    help="σ entre réplicas del primer punto frente a la del último.")
        c1, c2 = st.columns(2)
        ver_fig(c1, gr.fig_sens_convergencia(df))
        ver_fig(c2, gr.fig_sens_variabilidad(df))
        mostrar_tabla(informe.tabla_sens(df), informe.fmt_sens, "nsim", "small")
        st.success(S["texto"])

# ==============================================================================================
# 5) Informe
# ==============================================================================================
with tab_inf:
    R = ss.resultados
    if R is None:
        st.info("Ejecuta primero el análisis para poder generar el informe.")
    else:
        st.markdown("El informe Word incluye los parámetros, la síntesis y la recomendación, la comparación de planes "
                    "(riesgo frente a retorno) y, para cada plan, los resultados teóricos y simulados, la distribución "
                    "del retorno y el funcionamiento de la transformada inversa.")
        titulo = st.text_input("Título del informe", "Riesgo y retorno de planes de inversión", key="tit_inf")
        con_sens = st.checkbox("Incluir el análisis de sensibilidad", value=ss.sens is not None,
                               disabled=ss.sens is None, help="Disponible tras ejecutarlo en la pestaña Sensibilidad.")
        if st.button("📄 Generar informe (.docx)", type="primary"):
            with st.spinner("Generando informe…"):
                ss.informe_bytes = informe.generar_informe(R["res"], R["comunes"], R["ajustes"], titulo,
                                                           ss.sens if con_sens else None)
        if ss.informe_bytes:
            st.download_button("⬇️ Descargar informe", ss.informe_bytes,
                               f"Informe_Riesgo_Retorno_{datetime.date.today():%Y%m%d}.docx",
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
