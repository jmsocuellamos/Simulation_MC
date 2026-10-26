# -*- coding: utf-8 -*-
"""Simulador de colas M/M/s/K (capacidad finita K) · aplicación Streamlit.

Ejecutar con:   streamlit run app.py
"""
import copy
import datetime
import io
import json
import math
import sys

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import mmsk
import graficos as gr
import informe
from mmsk import num, pct, eur, unidad_larga, unidad_sing

st.set_page_config(page_title="Simulador · Colas M/M/s/K", page_icon="📶", layout="wide")
EN_NAVEGADOR = sys.platform == "emscripten"

ESCENARIOS_DEF = {
    "Base · M/M/2/5": {"lambda": 6.0, "mu": 4.0, "s": 2, "K": 5, "cs": None,
                       "descripcion": "Configuración actual: dos servidores (μ = 4 clientes/hora cada uno) y capacidad para "
                                      "5 clientes en total (3 en cola)."},
    "Más capacidad · K = 10": {"lambda": 6.0, "mu": 4.0, "s": 2, "K": 10, "cs": None,
                               "descripcion": "Se amplía la sala de espera: capacidad total K = 10 con los mismos dos servidores."},
    "3 servidores": {"lambda": 6.0, "mu": 4.0, "s": 3, "K": 5, "cs": None,
                     "descripcion": "Se añade un tercer servidor idéntico manteniendo la capacidad K = 5."},
    "Servicio más rápido": {"lambda": 6.0, "mu": 5.0, "s": 2, "K": 5, "cs": 36.0,
                            "descripcion": "Dos servidores más rápidos (μ = 5) pero más caros (c_s = 36 €/hora), con K = 5."},
}
COMUNES_DEF = {"unidad": "h", "cq": 10.0, "cs": 30.0, "s_max": 8, "objetivos": dict(mmsk.OBJETIVOS_DEF)}
AJUSTES_DEF = {"duracion": 5000.0, "replicas": 8, "semilla": 1000}


# ==============================================================================================
# Estado de la sesión
# ==============================================================================================
ss = st.session_state
if "escenarios" not in ss:
    ss.escenarios = copy.deepcopy(ESCENARIOS_DEF)
    ss.comunes = copy.deepcopy(COMUNES_DEF)
    ss.ajustes = copy.deepcopy(AJUSTES_DEF)
    ss.resultados = None
    ss.informe_bytes = None
    ss.version = 0
    ss.esc_activo = next(iter(ss.escenarios))


def ver_fig(contenedor, fig):
    """Muestra una figura de matplotlib y la cierra para no acumular memoria entre ejecuciones."""
    contenedor.pyplot(fig, use_container_width=True)
    plt.close(fig)


def mostrar_probs(df):
    """Tabla de probabilidad de n clientes a todo el ancho, con n como primera columna y columnas compactas."""
    d = fmt_probs(df).reset_index()
    cfg = {"n": st.column_config.NumberColumn("n", width="small"),
           "Región": st.column_config.TextColumn("Región", width="medium"),
           "Clientes en cola": st.column_config.NumberColumn("En cola", width="small", help="Clientes esperando en cola"),
           "Servidores ocupados": st.column_config.NumberColumn("Ocupados", width="small", help="Servidores ocupados")}
    for c in d.columns:
        if c.startswith(("Teórica", "Simulada")):
            ic = "📐 " if c.startswith("Teórica") else "🎲 "
            cfg[c] = st.column_config.TextColumn(ic + c, width="small")
    st.dataframe(d, use_container_width=True, hide_index=True, column_config=cfg,
                 height=min(38 + 35 * len(d), 600))


def fmt_probs(df):
    """Formato de la tabla de probabilidades (porcentajes)."""
    out = df.copy()
    for c in out.columns:
        if c.startswith(("Teórica", "Simulada")):
            out[c] = out[c].map(lambda v: pct(v, 2))
    return out


def clave(*partes):
    """Clave de widget ligada a la versión de la configuración (al cargar o restablecer se regeneran)."""
    return "__".join(str(p) for p in partes) + f"__v{ss.version}"


def configuracion():
    return {"comunes": ss.comunes, "ajustes": ss.ajustes, "escenarios": ss.escenarios}


def cargar_configuracion(d):
    ss.comunes = {**copy.deepcopy(COMUNES_DEF), **d.get("comunes", {})}
    ss.comunes["objetivos"] = {**mmsk.OBJETIVOS_DEF, **d.get("comunes", {}).get("objetivos", {})}
    ss.ajustes = {**copy.deepcopy(AJUSTES_DEF), **{k: v for k, v in d.get("ajustes", {}).items() if k in AJUSTES_DEF}}
    ss.escenarios = d.get("escenarios") or copy.deepcopy(ESCENARIOS_DEF)
    ss.esc_activo = next(iter(ss.escenarios))
    ss.resultados, ss.informe_bytes = None, None
    ss.version += 1


com, aj = ss.comunes, ss.ajustes
if clave("unidad") in ss:            # la unidad elegida en «Parámetros comunes» rige también la barra lateral
    com["unidad"] = ss[clave("unidad")]
U, US, UL = com["unidad"], unidad_sing(com["unidad"]), unidad_larga(com["unidad"])

# ==============================================================================================
# Barra lateral
# ==============================================================================================
with st.sidebar:
    st.header("⚙️ Simulación")
    aj["duracion"] = float(st.number_input("Duración de cada réplica", 1.0, 1e9, float(aj["duracion"]),
                                           step=100.0, key=clave("dur"),
                                           help=f"Tiempo simulado por réplica, en la unidad de tiempo del modelo (ahora, {UL}). Se descarta el "
                                                f"primer {mmsk.CALENTAMIENTO:.0%} como calentamiento. Cuanto más largo, más "
                                                "precisa es la estimación de los KPIs."))
    aj["replicas"] = int(st.number_input("Réplicas (semillas)", 2, 100, int(aj["replicas"]), key=clave("rep"),
                                         help="Cada réplica usa una semilla distinta (semilla + i); el IC95 % se calcula "
                                              "con la variabilidad entre réplicas."))
    aj["semilla"] = int(st.number_input("Semilla de inicialización", 0, 10**9, int(aj["semilla"]), key=clave("sem")))
    if ss.escenarios:
        lam_max = max(float(e["lambda"]) for e in ss.escenarios.values())
        st.caption(f"Con λ = {num(lam_max, 2)} clientes/{US}, cada réplica genera unas "
                   f"{num(lam_max * aj['duracion'], 0)} llegadas.")
    nombres = list(ss.escenarios)
    sel = st.multiselect("Escenarios a simular", nombres, default=nombres, key=clave("sel"))
    if EN_NAVEGADOR:
        st.caption("En el navegador la simulación es más lenta: con réplicas muy largas o muy numerosas puede tardar.")
    if st.button("▶️ Ejecutar análisis", type="primary", use_container_width=True, disabled=not sel):
        barra = st.progress(0.0, text="Simulando…")
        res = {}
        total = len(sel) * aj["replicas"]
        for i, n in enumerate(sel):
            def prog(k, m, i=i, n=n):
                barra.progress((i * aj["replicas"] + k) / total, text=f"{n}: réplica {k} de {m}")
            res[n] = mmsk.analizar_escenario(ss.escenarios[n], com, aj, prog)
        ss.resultados = {"res": res, "comunes": copy.deepcopy(com), "ajustes": copy.deepcopy(aj),
                         "fecha": f"{datetime.datetime.now():%d/%m/%Y %H:%M}"}
        ss.informe_bytes = None
        barra.progress(1.0, text="Análisis completado")

    st.divider()
    st.subheader("💾 Configuración")
    st.download_button("Descargar configuración (JSON)", json.dumps(configuracion(), ensure_ascii=False, indent=2),
                       "configuracion_mmsk.json", "application/json", use_container_width=True)
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
st.title("📶 Simulador de colas M/M/s/K")
st.caption("Teoría de colas con capacidad finita K: distribución límite, probabilidad de bloqueo, validación por "
           "simulación de eventos discretos, costes y dimensionamiento, con comparación de escenarios e informe Word. Basado en «D-190. Sistemas "
           "de Colas», Javier Morales y Asun Martínez Mayoral, equipo IA4LEGOS (CC BY-SA 4.0).")
tab_com, tab_esc, tab_res, tab_inf = st.tabs(["🌐 Parámetros comunes", "🏕️ Escenarios", "📊 Resultados", "📄 Informe"])

# ==============================================================================================
# 1) Parámetros comunes
# ==============================================================================================
with tab_com:
    st.subheader("Parámetros comunes a todos los escenarios")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("#### ⏱️ Unidad de tiempo")
        unidades = list(mmsk.UNIDADES)
        com["unidad"] = st.selectbox("Unidad de tiempo del modelo", unidades, index=unidades.index(com["unidad"]),
                                     format_func=unidad_larga, key=clave("unidad"),
                                     help="Unidad en la que se expresan λ, μ, los tiempos (W, Wq, duración de las réplicas, "
                                          "diagrama de funcionamiento) y los costes por unidad de tiempo (ut).")
        U, US, UL = com["unidad"], unidad_sing(com["unidad"]), unidad_larga(com["unidad"])
        st.markdown("#### 💶 Costes")
        com["cq"] = float(st.number_input("Coste de espera (€/ut)", 0.0, 1e9, float(com["cq"]), 1.0,
                                          key=clave("cq"), help=f"c_q: coste por cliente y unidad de tiempo (ut, ahora {US}) en el sistema, en cola o en servicio."))
        com["cs"] = float(st.number_input("Coste por servidor (€/ut)", 0.0, 1e9,
                                          float(com["cs"]), 1.0, key=clave("cs"),
                                          help=f"c_s: coste por servidor y unidad de tiempo (ut, ahora {US}); valor por defecto, cada escenario puede usar uno "
                                               "propio en la pestaña Escenarios)."))
        com["s_max"] = int(st.number_input("s máximo del dimensionamiento", 1, 60, int(com["s_max"]),
                                           key=clave("smax"), help="Se evalúa el coste total para s = 1 … min(s máximo, K), "
                                                                   "manteniendo la capacidad K de cada escenario."))
    with c2:
        st.markdown("#### 🏆 Objetivos de servicio")
        st.caption("Criterios de calidad que deben cumplir los escenarios (se evalúan sobre los valores teóricos). "
                   "Desmarca un objetivo para no evaluarlo.")
        ob = com["objetivos"]
        def objetivo(clave_obj, etiqueta, defecto, factor, paso, ayuda):
            cc1, cc2 = st.columns([1, 2])
            activo = cc1.checkbox("Evaluar", value=ob.get(clave_obj) is not None, key=clave("obj_on", clave_obj))
            v = ob.get(clave_obj) if ob.get(clave_obj) is not None else defecto
            x = cc2.number_input(etiqueta, 0.0, 1e9, float(v) * factor, paso, key=clave("obj", clave_obj), help=ayuda,
                                 disabled=not activo)
            ob[clave_obj] = (x / factor) if activo else None
        objetivo("RHO_MAX", "ρ̄ máxima (%)", 0.85, 100, 1.0, "Ocupación efectiva máxima admisible de cada servidor, ρ̄ = λ̄/(sμ).")
        objetivo("WQ_MAX", f"Wq máxima ({UL})", 0.25, 1, 0.05, f"Tiempo medio de espera en cola admisible, en {UL}.")
        objetivo("PBLOQ_MAX", "P(bloqueo) máxima (%)", 0.05, 100, 0.5,
                 "Proporción máxima admisible de clientes rechazados por encontrar el sistema lleno (N = K).")
        objetivo("LQ_MAX", "Lq máxima (clientes)", 1.0, 1, 0.5, "Número medio de clientes en cola admisible.")

# ==============================================================================================
# 2) Escenarios
# ==============================================================================================
with tab_esc:
    st.subheader("Escenarios")
    nombres = list(ss.escenarios)
    if ss.esc_activo not in nombres:
        ss.esc_activo = nombres[0]
    c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
    activo = c1.selectbox("Escenario a editar", nombres, index=nombres.index(ss.esc_activo), key=clave("activo"))
    ss.esc_activo = activo
    nuevo = c2.text_input("Nombre para copiar / renombrar", key=clave("nuevo_nombre"), placeholder="nombre del escenario")
    if c3.button("Copiar", use_container_width=True, help="Crea un escenario nuevo copiando el actual"):
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
    st.markdown(f"#### Parámetros de «{activo}»")
    e.setdefault("K", max(int(e.get("s", 1)), 5))
    c1, c2, c3, c5, c4 = st.columns(5)
    e["lambda"] = float(c1.number_input(f"λ · llegadas por {US}", 1e-6, 1e9, float(e["lambda"]), 0.1,
                                        format="%.4f", key=clave("lam", activo),
                                        help=f"Tasa de llegadas (proceso de Poisson), en clientes por {US}."))
    e["mu"] = float(c2.number_input(f"μ · servicios por {US}", 1e-6, 1e9, float(e["mu"]), 0.1,
                                    format="%.4f", key=clave("mu", activo),
                                    help=f"Tasa de servicio de cada servidor, en clientes por {US} (tiempo medio de "
                                         "servicio 1/μ, exponencial)."))
    e["s"] = int(c3.number_input("s · servidores", 1, 60, int(e["s"]), 1, key=clave("s", activo),
                                 help="Número de servidores en paralelo."))
    e["K"] = int(c5.number_input("K · capacidad", int(e["s"]), 10_000, max(int(e["K"]), int(e["s"])), 1,
                                 key=clave("K", activo, e["s"]),
                                 help=f"Clientes máximos en el sistema (en servicio + en cola), K ≥ s. Cola máxima: K − s = "
                                      f"{max(int(e['K']), int(e['s'])) - int(e['s'])}. Las llegadas con el sistema lleno se pierden."))
    propio = c4.checkbox("c_s propio del escenario", value=e.get("cs") is not None, key=clave("cs_on", activo),
                         help="Si se marca, el escenario usa su propio coste por servidor en lugar del común.")
    if propio:
        e["cs"] = float(c4.number_input("Coste por servidor (€/ut)", 0.0, 1e9,
                                        float(e["cs"] if e.get("cs") is not None else com["cs"]), 1.0, key=clave("cs_esc", activo)))
    else:
        e["cs"] = None
    e["descripcion"] = st.text_area("Descripción", e.get("descripcion", ""), key=clave("desc", activo), height=70)

    t = mmsk.teoria(e["lambda"], e["mu"], e["s"], e["K"])
    cs_e = e["cs"] if e["cs"] is not None else com["cs"]
    cst = mmsk.costes(t, com["cq"], cs_e)
    if t["rho"] < 1:
        st.success(f"M/M/{t['s']}/{t['K']} · régimen subcrítico: ρ = {pct(t['rho'])} < 100 %. "
                   f"Se rechaza el {pct(t['Pbloqueo'], 2)} de las llegadas.")
    else:
        st.warning(f"M/M/{t['s']}/{t['K']} · régimen saturado: ρ = {pct(t['rho'])} ≥ 100 %. Con capacidad infinita sería "
                   f"inestable; aquí el sistema se llena y rechaza el {pct(t['Pbloqueo'], 2)} de las llegadas.")
    st.markdown("##### 📐 Vista previa teórica (fórmulas del modelo M/M/s/K, sin simular)")
    m = st.columns(6)
    m[0].metric("ρ̄ (efectiva)", pct(t["rho_eff"]), help="Ocupación efectiva por servidor ρ̄ = λ̄/(sμ).")
    m[1].metric("P(bloqueo)", pct(t["Pbloqueo"], 2), help="Probabilidad de que una llegada encuentre el sistema lleno, p_K.")
    m[2].metric("L (en el sistema)", num(t["L"], 3), help="Número medio de clientes en el sistema.")
    m[3].metric("Lq (en cola)", num(t["Lq"], 3), help="Número medio de clientes en cola.")
    m[4].metric(f"Wq ({UL})", num(t["Wq"], 4), help="Tiempo medio de espera en cola (clientes aceptados).")
    m[5].metric(f"CT (€/{US})", eur(cst["CT"]), help="Coste total por unidad de tiempo.")
    ver_fig(st, gr.fig_pn(t, "Probabilidad de n clientes en el sistema (teórica)"))
    with st.expander("Tabla de probabilidad de n clientes en el sistema (teórica)", expanded=True):
        mostrar_probs(mmsk.tabla_probabilidades(t))

    st.divider()
    c1, c2 = st.columns([3, 1])
    c1.caption("Los cambios se aplican al pulsar «Ejecutar análisis» en la barra lateral.")
    if c2.button("🗑️ Eliminar este escenario", use_container_width=True, disabled=len(ss.escenarios) <= 1):
        ss.escenarios.pop(activo); ss.esc_activo = next(iter(ss.escenarios)); ss.version += 1; st.rerun()


# ==============================================================================================
# 3) Resultados
# ==============================================================================================
def tabla_fmt(df, fn):
    out = df.copy().astype(object)
    for f in df.index:
        for c in df.columns:
            out.loc[f, c] = fn(f, c, df.loc[f, c])
    return out


with tab_res:
    R = ss.resultados
    if R is None:
        st.info("Todavía no hay resultados. Configura los escenarios y pulsa «▶️ Ejecutar análisis» en la barra lateral.")
    else:
        res, Uc = R["res"], R["comunes"]["unidad"]
        USc, ULc = unidad_sing(Uc), unidad_larga(Uc)
        st.caption(f"Análisis del {R['fecha']} · simulación: {R['ajustes']['replicas']} réplicas de "
                   f"{num(R['ajustes']['duracion'], 0)} {ULc} · semilla {R['ajustes']['semilla']} · "
                   f"escenarios: {', '.join(res)}")
        st.info("📐 **Teórico**: calculado con las fórmulas exactas del modelo M/M/s/K. "
                "🎲 **Simulado**: estimado con la simulación de eventos discretos (media de las réplicas, con IC95 %).")
        r_sin, r_det, r_dat = st.tabs(["Síntesis y comparación", "Detalle por escenario", "Datos de las réplicas (simulados)"])
        with r_sin:
            st.markdown("#### Síntesis")
            for n, r in res.items():
                st.markdown(f"**{n}.** " + mmsk.texto_escenario(n, r, Uc)[len(n) + 2:])
            st.success("**Recomendación.** " + mmsk.texto_recomendacion(res, Uc))
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown("**Comparación de escenarios** (filas «Teórico» = fórmulas; filas «Simulado» = Monte Carlo)")
                st.dataframe(tabla_fmt(mmsk.tabla_comparativa(res), mmsk.fmt_kpi), use_container_width=True, height=560)
            with c2:
                st.markdown("**Objetivos de servicio** (sobre los valores teóricos)")
                st.dataframe(mmsk.tabla_objetivos(res), use_container_width=True)
                ver_fig(st, gr.fig_validacion(res))
            ver_fig(st, gr.fig_comparativa(res, ULc))

        with r_det:
            n = st.selectbox("Escenario", list(res), key="det_esc")
            r = res[n]
            t, cst = r["teoria"], r["costes"]
            mc = r["mc"]["replicas"]
            d_teo, d_sim, d_prob, d_cost = st.tabs(["📐 Resultados teóricos", "🎲 Resultados simulados",
                                                   "🔢 Probabilidad de n clientes", "💶 Costes y dimensionamiento"])
            with d_teo:
                st.caption(f"Valores exactos de las fórmulas del modelo M/M/{t['s']}/{t['K']} (distribución límite del "
                           "proceso de nacimiento-muerte truncado en K).")
                m = st.columns(4)
                m[0].metric("ρ (nominal)", pct(t["rho"]), "subcrítico" if t["rho"] < 1 else "saturado",
                            delta_color="normal" if t["rho"] < 1 else "inverse", help="Carga nominal por servidor ρ = λ/(sμ).")
                m[1].metric("ρ̄ (efectiva)", pct(t["rho_eff"]), help="Ocupación efectiva por servidor ρ̄ = λ̄/(sμ).")
                m[2].metric("a (Erlangs)", num(t["a"], 3), help="Carga ofrecida a = λ/μ.")
                m[3].metric("P0 (vacío)", pct(t["P0"]), help="Probabilidad de que el sistema esté vacío.")
                m = st.columns(4)
                m[0].metric("P(bloqueo) = p_K", pct(t["Pbloqueo"], 2), help="Fracción de llegadas rechazadas.")
                m[1].metric(f"λ̄ (aceptados/{USc})", num(t["lambda_bar"], 3), help="Tasa efectiva de entrada λ̄ = λ(1 − p_K).")
                m[2].metric(f"Rechazos/{USc}", num(t["lambda"] - t["lambda_bar"], 3), help="Clientes perdidos por unidad de tiempo, λ·p_K.")
                m[3].metric("P(espera) aceptados", pct(t["Pespera"]), help="Probabilidad de que un cliente aceptado tenga que esperar.")
                m = st.columns(4)
                m[0].metric("L · en el sistema", num(t["L"], 3), help="Número medio de clientes en el sistema.")
                m[1].metric("Lq · en cola", num(t["Lq"], 3), help="Número medio de clientes en cola.")
                m[2].metric(f"W ({ULc})", num(t["W"], 4), help="Tiempo medio en el sistema (W = L/λ̄).")
                m[3].metric(f"Wq ({ULc})", num(t["Wq"], 4), help="Tiempo medio en cola (Wq = W − 1/μ).")
                st.markdown("**Objetivos de servicio** (evaluados sobre los valores teóricos)")
                st.dataframe(tabla_fmt(r["objetivos"], informe.fmt_obj), use_container_width=True)
            with d_sim:
                st.caption(f"Estimaciones de la simulación de eventos discretos: {R['ajustes']['replicas']} réplicas de "
                           f"{num(R['ajustes']['duracion'], 0)} {ULc}, descartando el primer {mmsk.CALENTAMIENTO:.0%} como calentamiento "
                           f"(unas {num(mc['Llegadas'].mean(), 0)} llegadas medidas por réplica, de las que se rechazan unas {num(mc['Rechazos'].mean(), 0)}). Se comparan con el valor teórico y su intervalo de confianza al 95 %.")
                m = st.columns(5)
                for col, (k, d) in zip(m, [("L", 3), ("Lq", 3), ("W", 4), ("Wq", 4), ("P(bloqueo)", None)]):
                    v_sim, v_teo = mc[k].mean(), r["resumen_mc"].loc[k, "Teórico (fórmula)"]
                    et = k + (f" sim. ({ULc})" if k in ("W", "Wq") else " simulado")
                    col.metric(et, pct(v_sim, 2) if d is None else num(v_sim, d),
                               "teórico " + (pct(v_teo, 2) if d is None else num(v_teo, d)),
                               delta_color="off")
                st.markdown("**KPIs simulados frente a teóricos**")
                st.dataframe(tabla_fmt(r["resumen_mc"], informe.fmt_mc), use_container_width=True)
                c1, c2 = st.columns(2)
                ver_fig(c1, gr.fig_pn_comparada(r["mc"]["pn"], t))
                ver_fig(c2, gr.fig_traza(r["mc"]["traza"], t["s"], ULc, K=t["K"]))
                n_rech = int(r["clientes"]["Rechazado"].sum())
                st.markdown(f"**Diagrama de funcionamiento (simulado)** · primeras {len(r['clientes'])} llegadas "
                            f"({n_rech} rechazadas por encontrar el sistema lleno, marcadas con ✕)")
                ver_fig(st, gr.fig_diagrama(r["clientes"], t["s"], None, ULc))
                with st.expander("Clientes del diagrama (llegada, inicio y fin de servicio, servidor, rechazo)"):
                    st.dataframe(r["clientes"].round(4), use_container_width=True, hide_index=True)
            with d_prob:
                st.caption("Probabilidad de que haya exactamente n clientes en el sistema, P(N = n), su acumulada P(N ≤ n) y la "
                           "probabilidad de superar n, P(N > n). Columnas «Teórica»: fórmulas exactas. Columnas «Simulada»: "
                           "fracción del tiempo simulado con n clientes (todas las réplicas). n va de 0 a K; p_K es la probabilidad de bloqueo.")
                st.markdown("**Calculadora**")
                c1, c2, c3, c4 = st.columns(4)
                tipo = c1.selectbox("Probabilidad", ["=", "≤", "≥", ">"], format_func=lambda x: f"P(N {x} k)", key="prob_tipo")
                k = int(c2.number_input("k (clientes)", 0, int(t["K"]), min(int(t["s"]), int(t["K"])), 1, key=f"prob_k_{n}"))
                p_teo, p_sim = mmsk.prob_evento(t, r["mc"]["pn"], tipo, k)
                c3.metric(f"📐 Teórica · P(N {tipo} {k})", pct(p_teo, 2))
                c4.metric(f"🎲 Simulada · P(N {tipo} {k})", pct(p_sim, 2))
                tp = mmsk.tabla_probabilidades(t, r["mc"]["pn"])
                ver_fig(st, gr.fig_pn_comparada(r["mc"]["pn"], t, "Probabilidad de n clientes: simulada (barras) y teórica (línea)"))
                st.markdown("**Tabla de probabilidades**")
                mostrar_probs(tp)
                st.download_button("Descargar la tabla (CSV)", tp.to_csv().encode("utf-8"), f"probabilidades_{n}.csv", "text/csv")
            with d_cost:
                st.caption("Costes por unidad de tiempo calculados con los valores teóricos (CTQ = λ·c_q·Wq con λ nominal). "
                           "El coste de los clientes rechazados no se incluye.")
                m = st.columns(5)
                m[0].metric(f"c_t (€/{USc})", eur(cst["c_t"]), help="c_t = c_q + c_s")
                m[1].metric("CTS", eur(cst["CTS"]), help="Coste de los servidores ocupados.")
                m[2].metric("CTSn", eur(cst["CTSn"]), help="Coste de los servidores libres.")
                m[3].metric("CTQ", eur(cst["CTQ"]), help="Coste de espera de los clientes en cola.")
                m[4].metric("CT (total)", eur(cst["CT"]), help="CT = CTS + CTSn + CTQ.")
                c1, c2 = st.columns(2)
                ver_fig(c1, gr.fig_costes(cst, USc))
                ver_fig(c2, gr.fig_barrido(r["barrido"], t["s"], r["s_opt"], USc))
                if r["s_opt"] is not None:
                    st.info(f"Con K = {t['K']}, número de servidores de menor coste total: **s = {r['s_opt']}** "
                            f"(CT = {eur(r['barrido'].loc[r['s_opt'], 'CT'])}/{USc}).")
                st.dataframe(tabla_fmt(r["barrido"], informe.fmt_barrido), use_container_width=True)

        with r_dat:
            st.markdown("Resultados **simulados** de cada réplica (una fila por réplica y escenario).")
            largo = pd.concat([r["mc"]["replicas"].assign(Escenario=n) for n, r in res.items()]).reset_index()
            st.dataframe(largo, use_container_width=True, hide_index=True)
            buf = io.BytesIO()
            try:
                with pd.ExcelWriter(buf, engine="openpyxl") as xw:
                    largo.to_excel(xw, sheet_name="Replicas", index=False)
                    mmsk.tabla_comparativa(res).astype(str).to_excel(xw, sheet_name="Comparacion")
                    for i, (n, r) in enumerate(res.items(), 1):
                        r["barrido"].to_excel(xw, sheet_name=f"Barrido {i}")
                        mmsk.tabla_probabilidades(r["teoria"], r["mc"]["pn"]).to_excel(xw, sheet_name=f"Probabilidades {i}")
                st.download_button("Descargar Excel", buf.getvalue(), "resultados_mmsk.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as ex:  # openpyxl no disponible
                st.download_button("Descargar CSV", largo.to_csv(index=False).encode("utf-8"), "replicas_mmsk.csv", "text/csv")

# ==============================================================================================
# 4) Informe
# ==============================================================================================
with tab_inf:
    R = ss.resultados
    if R is None:
        st.info("Ejecuta primero el análisis para poder generar el informe.")
    else:
        st.markdown("El informe Word incluye los parámetros, la síntesis y la recomendación, la comparación de escenarios y, "
                    "para cada escenario, la teoría (KPIs, bloqueo y distribución límite), la validación por simulación, el diagrama "
                    "de funcionamiento y el análisis de costes y dimensionamiento.")
        titulo = st.text_input("Título del informe", "Sistema de colas M/M/s/K", key="tit_inf")
        if st.button("📄 Generar informe (.docx)", type="primary"):
            with st.spinner("Generando informe…"):
                ss.informe_bytes = informe.generar_informe(R["res"], R["comunes"], R["ajustes"], titulo)
        if ss.informe_bytes:
            st.download_button("⬇️ Descargar informe", ss.informe_bytes,
                               f"Informe_Colas_MMsK_{datetime.date.today():%Y%m%d}.docx",
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
