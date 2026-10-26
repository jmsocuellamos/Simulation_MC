# -*- coding: utf-8 -*-
"""Beneficio esperado por mixtura (¿compra o no compra?) · aplicación Streamlit.
Mixtura en dos pasos, simulación Monte Carlo del beneficio por cliente, comparación de estrategias comerciales,
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

import mixtura as mx
import graficos as gr
import informe
from mixtura import num, pct, eur

st.set_page_config(page_title="Simulador · Mixtura", page_icon="🛒", layout="wide")
EN_NAVEGADOR = sys.platform == "emscripten"

MODELOS_DEF = ["Sencillo", "Estándar", "Lujo"]
ESCENARIOS_DEF = {
    "Actual": {"pcompra": 50.0, "modelos": MODELOS_DEF, "valores": [30.0, 60.0, 75.0], "probs": [25.0, 50.0, 25.0],
               "coste": 0.0, "descripcion": "Situación actual: compra la mitad de los clientes; sencillo 25 %, estándar 50 % y lujo 25 %."},
    "Captación": {"pcompra": 60.0, "modelos": MODELOS_DEF, "valores": [30.0, 60.0, 75.0], "probs": [25.0, 50.0, 25.0],
                  "coste": 3.0, "descripcion": "Más vendedores en tienda: compra el 60 % de los clientes, con un coste de 3 € por cliente que entra."},
    "Promoción del lujo": {"pcompra": 50.0, "modelos": MODELOS_DEF, "valores": [30.0, 60.0, 75.0],
                           "probs": [15.0, 45.0, 40.0], "coste": 1.5,
                           "descripcion": "Exposición y financiación de la gama alta: el lujo sube al 40 % de las compras, con un coste de 1,5 € por cliente."},
}
COMUNES_DEF = {"visitantes": 200, "objetivos": dict(mx.OBJETIVOS_DEF)}
AJUSTES_DEF = {"nsim": 20000, "replicas": 5, "semilla": 1000}
SENS_DEF = {"nsim_min": 200, "nsim_max": 50000, "puntos": 6, "replicas": 5, "metrica": "compradores"}
COL_M, COL_V, COL_P = "Modelo", "Ganancia (€)", "Probabilidad (%)"

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
    ss.esc_activo = next(iter(ss.escenarios))
    ss.u1, ss.u2 = 0.45, 0.55


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
    ss.comunes["objetivos"] = {**mx.OBJETIVOS_DEF, **d.get("comunes", {}).get("objetivos", {})}
    ss.ajustes = {**copy.deepcopy(AJUSTES_DEF), **{k: v for k, v in d.get("ajustes", {}).items() if k in AJUSTES_DEF}}
    ss.sens_cfg = {**copy.deepcopy(SENS_DEF), **d.get("sensibilidad", {})}
    ss.escenarios = d.get("escenarios") or copy.deepcopy(ESCENARIOS_DEF)
    ss.esc_activo = next(iter(ss.escenarios))
    ss.resultados, ss.sens, ss.informe_bytes = None, None, None
    ss.version += 1


def teoria_plan(e):
    return mx.teoria_escenario(e, ss.comunes)


com, aj = ss.comunes, ss.ajustes

# ==============================================================================================
# Barra lateral
# ==============================================================================================
with st.sidebar:
    st.header("⚙️ Simulación")
    aj["nsim"] = int(st.number_input("Clientes por réplica (nsim)", 10, 10_000_000, int(aj["nsim"]), 1000,
                                     key=clave("nsim"), help="Número de clientes simulados en cada réplica."))
    aj["replicas"] = int(st.number_input("Réplicas (semillas)", 2, 100, int(aj["replicas"]), key=clave("rep"),
                                         help="Cada réplica usa una semilla distinta (semilla + i); el IC95 % se calcula "
                                              "con la variabilidad entre réplicas. Todas las estrategias usan las mismas "
                                              "semillas (números aleatorios comunes)."))
    aj["semilla"] = int(st.number_input("Semilla de inicialización", 0, 10**9, int(aj["semilla"]), key=clave("sem")))
    st.caption(f"Por estrategia: {num(aj['nsim'] * aj['replicas'], 0)} clientes simulados en dos pasos.")
    nombres = list(ss.escenarios)
    sel = st.multiselect("Estrategias a simular", nombres, default=nombres, key=clave("sel"))
    if EN_NAVEGADOR:
        st.caption("En el navegador la simulación es más lenta: con muchas simulaciones o réplicas puede tardar.")
    if st.button("▶️ Ejecutar análisis", type="primary", use_container_width=True, disabled=not sel):
        barra = st.progress(0.0, text="Simulando…")
        res = {}
        total = len(sel) * aj["replicas"]
        for i, n in enumerate(sel):
            def prog(k, m, i=i, n=n):
                barra.progress((i * aj["replicas"] + k) / total, text=f"{n}: réplica {k} de {m}")
            res[n] = mx.analizar_escenario(ss.escenarios[n], com, aj, prog)
        ss.resultados = {"res": res, "comunes": copy.deepcopy(com), "ajustes": copy.deepcopy(aj),
                         "fecha": f"{datetime.datetime.now():%d/%m/%Y %H:%M}"}
        ss.informe_bytes = None
        barra.progress(1.0, text="Análisis completado")

    st.divider()
    st.subheader("💾 Configuración")
    st.download_button("Descargar configuración (JSON)", json.dumps(configuracion(), ensure_ascii=False, indent=2),
                       "configuracion_mixtura.json", "application/json", use_container_width=True)
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
st.title("🛒 Beneficio esperado por mixtura")
st.caption("¿Compra o no compra? Mixtura en dos pasos (indicador de compra y transformada inversa para el modelo), "
           "simulación Monte Carlo del beneficio por cliente, comparación de estrategias comerciales, sensibilidad del "
           "estimador e informe Word. Basado en el Ejemplo 4 de «B-70. Distribuciones derivadas», Javier Morales y Asun "
           "Martínez Mayoral, equipo IA4LEGOS (CC BY-SA 4.0).")
tab_com, tab_esc, tab_res, tab_sens, tab_inf = st.tabs(["🌐 Parámetros comunes", "🏕️ Estrategias", "📊 Resultados",
                                                        "🔬 Sensibilidad", "📄 Informe"])

# ==============================================================================================
# 1) Parámetros comunes
# ==============================================================================================
with tab_com:
    st.subheader("Parámetros comunes a todas las estrategias")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("#### 🏬 Tienda")
        com["visitantes"] = int(st.number_input("Clientes que entran por día", 1, 10_000_000, int(com["visitantes"]), 10,
                                                key=clave("vis"), help="Para expresar el beneficio neto diario."))
    with c2:
        st.markdown("#### 🏆 Objetivos")
        st.caption("Criterios que deben cumplir las estrategias (se evalúan sobre los valores teóricos). "
                   "Desmarca un objetivo para no evaluarlo.")
        ob = com["objetivos"]
        def objetivo(clave_obj, etiqueta, defecto, factor, paso, ayuda):
            cc1, cc2 = st.columns([1, 2])
            activo = cc1.checkbox("Evaluar", value=ob.get(clave_obj) is not None, key=clave("obj_on", clave_obj))
            v = ob.get(clave_obj) if ob.get(clave_obj) is not None else defecto
            x = cc2.number_input(etiqueta, -1e12, 1e12, float(v) * factor, paso, key=clave("obj", clave_obj), help=ayuda,
                                 disabled=not activo)
            ob[clave_obj] = (x / factor) if activo else None
        objetivo("BNETO_MIN", "Beneficio neto mín. (€/cliente)", 29.0, 1, 0.5,
                 "Beneficio neto medio mínimo por cliente que entra (tras el coste comercial).")
        objetivo("PCOMPRA_MIN", "Proporción que compra mín. (%)", 0.55, 100, 1.0,
                 "Proporción mínima de clientes que acaban comprando.")
        objetivo("BCOMPRA_MIN", "Beneficio por comprador mín. (€)", 58.0, 1, 0.5,
                 "Beneficio medio mínimo por cliente que compra, E(B | compra).")
        objetivo("ALTA_MIN", "Compras de gama alta mín. (%)", 0.80, 100, 1.0,
                 "Proporción mínima de compras que no son del primer modelo de la lista (gama básica).")

# ==============================================================================================
# 2) Planes (escenarios)
# ==============================================================================================
with tab_esc:
    st.subheader("Estrategias comerciales")
    nombres = list(ss.escenarios)
    if ss.esc_activo not in nombres:
        ss.esc_activo = nombres[0]
    c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
    activo = c1.selectbox("Estrategia a editar", nombres, index=nombres.index(ss.esc_activo), key=clave("activo"))
    ss.esc_activo = activo
    nuevo = c2.text_input("Nombre para copiar / renombrar", key=clave("nuevo_nombre"), placeholder="nombre de la estrategia")
    if c3.button("Copiar", use_container_width=True, help="Crea una estrategia nueva copiando la actual"):
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
    c1, c2 = st.columns([2, 3])
    with c1:
        cc = st.columns(2)
        e["pcompra"] = float(cc[0].number_input("Pr(compra) (%)", 0.0, 100.0, float(e["pcompra"]), 1.0, key=clave("pc", activo),
                                                help="De cada 100 clientes que entran, cuántos acaban comprando un horno."))
        e["coste"] = float(cc[1].number_input("Coste por cliente (€)", 0.0, 1e9, float(e.get("coste") or 0.0), 0.5,
                                              key=clave("coste", activo),
                                              help="Coste comercial de la estrategia por cliente que entra (personal, "
                                                   "publicidad, financiación…). Se descuenta para el beneficio neto."))
        st.caption(f"Modelos y probabilidades condicionadas a la compra (hasta {mx.MAX_MODELOS}); se normalizan si no "
                   "suman 100 %. Añade o borra filas con los controles de la tabla.")
        k_df = clave("modelos", activo)
        k_base = "base_" + k_df
        if k_base not in ss:          # datos de partida fijos para el editor (evita aplicar dos veces los cambios)
            ss[k_base] = pd.DataFrame({COL_M: list(e["modelos"]), COL_V: [float(x) for x in e["valores"]],
                                       COL_P: [float(x) for x in e["probs"]]})
        ed = st.data_editor(ss[k_base], key=k_df, num_rows="dynamic", use_container_width=True, hide_index=True,
                            column_config={COL_M: st.column_config.TextColumn(COL_M),
                                           COL_V: st.column_config.NumberColumn(COL_V, step=1.0, format="%.2f"),
                                           COL_P: st.column_config.NumberColumn(COL_P, min_value=0.0, max_value=100.0,
                                                                                step=1.0, format="%.1f")})
        ed = ed.dropna(subset=[COL_V, COL_P]).head(mx.MAX_MODELOS)
        if len(ed):
            e["modelos"] = [str(x) if isinstance(x, str) and x.strip() else f"Modelo {i + 1}" for i, x in enumerate(ed[COL_M])]
            e["valores"] = [float(x) for x in ed[COL_V]]
            e["probs"] = [float(x) for x in ed[COL_P]]
        else:
            st.error("La estrategia necesita al menos un modelo con ganancia y probabilidad.")
        suma = sum(e["probs"])
        if abs(suma - 100) > 1e-9:
            st.warning(f"Las probabilidades de los modelos suman {num(suma, 1)} %: se normalizan para sumar 100 %.")
        e["descripcion"] = st.text_area("Descripción", e.get("descripcion", ""), key=clave("desc", activo), height=90)

    t = teoria_plan(e)
    with c2:
        st.markdown("##### 📐 Vista previa teórica (fórmulas de la mixtura, sin simular)")
        m = st.columns(3)
        m[0].metric("E(B | compra)", eur(t["E_compra"]), help="Σ pᵢ·bᵢ: beneficio medio por cliente que compra.")
        m[1].metric("E(B) por cliente", eur(t["E"]), help="Pr(compra)·E(B | compra): beneficio medio por cliente que entra.")
        m[2].metric("σ(B) por cliente", eur(t["sd"]))
        m = st.columns(3)
        m[0].metric("Beneficio neto por cliente", eur(t["E_neto"]), help="E(B) − coste por cliente.")
        m[1].metric("Beneficio neto diario", eur(t["dia_neto"], 0), help=f"{num(com['visitantes'], 0)} clientes × beneficio neto.")
        m[2].metric("Compras de gama alta", pct(t["gama_alta"], 1), help="Todos los modelos salvo el primero de la lista.")
        ver_fig(st, gr.fig_descomposicion(t))

    with st.expander("🎯 Mecanismo · un cliente, dos pasos", expanded=True):
        st.caption("1 · Se genera U₁ ~ Unif(0, 1): si U₁ < Pr(compra) el cliente compra; si no, B = 0 €.  2 · Solo si "
                   "compra, se genera U₂ ~ Unif(0, 1) y la transformada inversa elige el modelo.  3 · B = ganancia del "
                   "modelo. Repetido muchas veces, reproduce la mixtura.")
        c1, c2, c3 = st.columns([2, 2, 1])
        if c3.button("🎲 U₁ y U₂ aleatorios", use_container_width=True):
            ss.u1, ss.u2 = float(np.round(np.random.random(), 2)), float(np.round(np.random.random(), 2))
            ss.pop("u1_slider", None); ss.pop("u2_slider", None)
        ss.u1 = c1.slider("U₁ ~ Unif(0, 1) · ¿compra?", 0.0, 1.0, float(ss.u1), 0.01, key="u1_slider")
        ss.u2 = c2.slider("U₂ ~ Unif(0, 1) · modelo", 0.0, 1.0, float(ss.u2), 0.01, key="u2_slider")
        compra, kmod, ben = mx.un_cliente(t, ss.u1, ss.u2)
        m = st.columns(4)
        m[0].metric("¿Compra?", "Sí" if compra else "No", f"U₁ {'<' if compra else '≥'} {num(t['pc'], 2)}", delta_color="off")
        m[1].metric("U₂ usado", num(ss.u2, 2) if compra else "—", "solo si compra", delta_color="off")
        m[2].metric("Modelo elegido", t["modelos"][kmod] if compra else "—")
        m[3].metric("Beneficio B", eur(ben))
        ver_fig(st, gr.fig_mecanismo(t, ss.u1, ss.u2))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Desenlaces de la mixtura** (teóricos)")
            mostrar_tabla(mx.tabla_desenlaces(t), informe.fmt_tabla, "Desenlace", "medium", iconos=True)
        with c2:
            st.markdown("**Modelos entre los compradores** (teóricos)")
            mostrar_tabla(mx.tabla_modelos(t), informe.fmt_tabla, "Modelo", "small", iconos=True)

    st.divider()
    c1, c2 = st.columns([3, 1])
    c1.caption("Los cambios se aplican al pulsar «Ejecutar análisis» en la barra lateral.")
    if c2.button("🗑️ Eliminar esta estrategia", use_container_width=True, disabled=len(ss.escenarios) <= 1):
        ss.escenarios.pop(activo); ss.esc_activo = next(iter(ss.escenarios)); ss.version += 1; st.rerun()

# ==============================================================================================
# 3) Resultados
# ==============================================================================================
with tab_res:
    R = ss.resultados
    if R is None:
        st.info("Todavía no hay resultados. Configura las estrategias y pulsa «▶️ Ejecutar análisis» en la barra lateral.")
    else:
        res = R["res"]
        st.caption(f"Análisis del {R['fecha']} · {R['ajustes']['replicas']} réplicas de {num(R['ajustes']['nsim'], 0)} "
                   f"clientes · {num(R['comunes']['visitantes'], 0)} clientes/día · semilla {R['ajustes']['semilla']} · "
                   f"estrategias: {', '.join(res)}")
        st.info("📐 **Teórico**: fórmulas de la mixtura (E(B) = Pr(compra)·Σ pᵢ·bᵢ). "
                "🎲 **Simulado**: simulación en dos pasos (media de las réplicas, con IC95 %).")
        r_sin, r_det, r_dat = st.tabs(["Síntesis y comparación", "Detalle por estrategia", "Datos de las réplicas (simulados)"])
        with r_sin:
            st.markdown("#### Síntesis")
            for n, r in res.items():
                st.markdown(f"**{n}.** " + mx.texto_escenario(n, r)[len(n) + 2:])
            st.success("**Recomendación.** " + mx.texto_recomendacion(res))
            st.markdown("**Comparación de estrategias** (filas «Teórico» = fórmulas; filas «Simulado» = Monte Carlo)")
            mostrar_tabla(mx.tabla_comparativa(res), mx.fmt_kpi, "KPI", "large")
            c1, c2 = st.columns([2, 3])
            with c1:
                st.markdown("**Objetivos** (sobre los valores teóricos)")
                mostrar_tabla(mx.tabla_objetivos(res), None, "Objetivo", "large")
            with c2:
                ver_fig(st, gr.fig_validacion(res))
            ver_fig(st, gr.fig_comparativa(res))

        with r_det:
            n = st.selectbox("Estrategia", list(res), key="det_esc")
            r = res[n]
            t, mc, rm = r["teoria"], r["mc"], r["resumen_mc"]
            d_teo, d_sim, d_dis = st.tabs(["📐 Resultados teóricos", "🎲 Resultados simulados", "🔢 Distribución del beneficio"])
            with d_teo:
                st.caption("Valores exactos de la mixtura.")
                m = st.columns(4)
                m[0].metric("Pr(compra)", pct(t["pc"], 1))
                m[1].metric("E(B | compra)", eur(t["E_compra"]))
                m[2].metric("E(B) por cliente", eur(t["E"]))
                m[3].metric("Pérdida por no compra", eur(t["coste_no_compra"]), help="E(B | compra) − E(B)")
                m = st.columns(4)
                m[0].metric("σ(B) por cliente", eur(t["sd"]))
                m[1].metric("Coste por cliente", eur(t["coste"]))
                m[2].metric("Beneficio neto por cliente", eur(t["E_neto"]))
                m[3].metric("Beneficio neto diario", eur(t["dia_neto"], 0))
                st.markdown("**Objetivos** (evaluados sobre los valores teóricos)")
                mostrar_tabla(r["objetivos"], mx.fmt_obj, "Objetivo", "large")
                ver_fig(st, gr.fig_descomposicion(t))
            with d_sim:
                st.caption(f"{R['ajustes']['replicas']} réplicas de {num(R['ajustes']['nsim'], 0)} clientes; cada KPI se "
                           "calcula en cada réplica y se muestra la media de las réplicas con su IC95 %.")
                m = st.columns(4)
                for col, (kpi, et, es_pct) in zip(m, [("Beneficio medio por cliente E(B)", "E(B) simulado", False),
                                                      ("Beneficio medio por comprador E(B | compra)", "E(B | compra) simulado", False),
                                                      ("Proporción que compra P(compra)", "P(compra) simulada", True),
                                                      ("Beneficio neto por cliente E(B) − coste", "Neto simulado", False)]):
                    vs, vt = rm.loc[kpi, "Simulado · media"], rm.loc[kpi, "Teórico"]
                    f = (lambda x: pct(x, 2)) if es_pct else eur
                    col.metric(et, f(vs), "teórico " + f(vt), delta_color="off")
                st.markdown("**KPIs simulados frente a teóricos**")
                mostrar_tabla(rm, mx.fmt_mc, "KPI", "large")
                c1, c2 = st.columns(2)
                ver_fig(c1, gr.fig_replicas(mc, t))
                ver_fig(c2, gr.fig_traza(mc["traza"], t))
                with st.expander(f"Primeros {len(mc['traza'])} clientes (U₁, compra, U₂, modelo y beneficio)"):
                    tr = mc["traza"].copy()
                    tr["U1"] = tr["U1"].map(lambda v: num(v, 4)); tr["U2"] = tr["U2"].map(lambda v: num(v, 4))
                    tr["Compra"] = tr["Compra"].map(lambda v: "Sí" if v else "No")
                    tr["Beneficio"] = tr["Beneficio"].map(eur)
                    st.dataframe(tr.astype(str), use_container_width=True, hide_index=True)
            with d_dis:
                st.markdown("**Desenlaces de la mixtura**")
                mostrar_tabla(mx.tabla_desenlaces(t, mc), informe.fmt_tabla, "Desenlace", "medium", iconos=True)
                st.markdown("**Modelos entre los compradores** (intervalo de U₂ de la transformada inversa)")
                mostrar_tabla(mx.tabla_modelos(t, mc), informe.fmt_tabla, "Modelo", "small", iconos=True)
                ver_fig(st, gr.fig_desenlaces(t, mc))

        with r_dat:
            st.markdown("Resultados **simulados** de cada réplica (una fila por réplica y estrategia).")
            largo = pd.concat([r["mc"]["replicas"].assign(Estrategia=n) for n, r in res.items()]).reset_index()
            st.dataframe(largo, use_container_width=True, hide_index=True)
            buf = io.BytesIO()
            try:
                with pd.ExcelWriter(buf, engine="openpyxl") as xw:
                    largo.to_excel(xw, sheet_name="Replicas", index=False)
                    mx.tabla_comparativa(res).astype(str).to_excel(xw, sheet_name="Comparacion")
                    for i, (n, r) in enumerate(res.items(), 1):
                        mx.tabla_desenlaces(r["teoria"], r["mc"]).to_excel(xw, sheet_name=f"Desenlaces {i}")
                        mx.tabla_modelos(r["teoria"], r["mc"]).to_excel(xw, sheet_name=f"Modelos {i}")
                    if ss.sens is not None:
                        informe.tabla_sens(ss.sens["df"]).to_excel(xw, sheet_name="Sensibilidad")
                st.download_button("Descargar Excel", buf.getvalue(), "resultados_mixtura.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception:  # openpyxl no disponible
                st.download_button("Descargar CSV", largo.to_csv(index=False).encode("utf-8"), "replicas_mixtura.csv", "text/csv")

# ==============================================================================================
# 4) Sensibilidad del estimador
# ==============================================================================================
with tab_sens:
    st.subheader("Sensibilidad del estimador Monte Carlo")
    st.caption("¿Cuánto «tiembla» la estimación según el número de clientes simulados? Para cada nsim se repite la "
               "simulación con varias semillas (semilla de la barra lateral + i).")
    sc = ss.sens_cfg
    c = st.columns(3)
    sc["nsim_min"] = int(c[0].number_input("nsim mínimo", 10, 10_000_000, int(sc["nsim_min"]), 50, key=clave("s_nmin")))
    sc["nsim_max"] = int(c[1].number_input("nsim máximo", 20, 10_000_000, int(sc["nsim_max"]), 1000, key=clave("s_nmax")))
    met = list(mx.METRICAS)
    sc["metrica"] = c[2].selectbox("Métrica a analizar", met, index=met.index(sc.get("metrica", "compradores")),
                                   format_func=lambda k: mx.METRICAS[k], key=clave("s_met"))
    c = st.columns(3)
    sc["puntos"] = int(c[0].slider("Puntos a evaluar", 3, 10, int(sc["puntos"]), key=clave("s_pts")))
    sc["replicas"] = int(c[1].slider("Réplicas por punto", 2, 20, int(sc["replicas"]), key=clave("s_rep")))
    pol = c[2].selectbox("Estrategia", list(ss.escenarios), key=clave("s_pol"))
    c1, c2 = st.columns([3, 1])
    c1.caption("Los puntos se espacian logarítmicamente entre nsim mínimo y máximo. La estrategia toma sus parámetros "
               "de la pestaña Estrategias.")
    lanzar = c2.button("▶️ Analizar sensibilidad", type="primary", use_container_width=True)
    if lanzar:
        vals = mx.valores_barrido(sc["nsim_min"], sc["nsim_max"], sc["puntos"])
        tp = teoria_plan(ss.escenarios[pol])
        barra = st.progress(0.0, text="Simulando…")
        df = mx.sensibilidad(tp, sc["metrica"], vals, sc["replicas"], aj["semilla"],
                             lambda k, m: barra.progress(k / m, text=f"Simulación {k} de {m}"))
        texto, red = mx.texto_sensibilidad(df, sc["metrica"], pol)
        ss.sens = {"df": df, "texto": texto, "reduccion": red,
                   "cfg": {"replicas": sc["replicas"], "estrategia": pol, "semilla": aj["semilla"], "metrica": sc["metrica"]}}
        ss.informe_bytes = None
        barra.progress(1.0, text="Barrido completado")
    S = ss.sens
    if S is None:
        st.info("Pulsa «▶️ Analizar sensibilidad» para ejecutar el barrido.")
    else:
        df, cfg = S["df"], S["cfg"]
        m = st.columns(4)
        m[0].metric("Métrica analizada", mx.METRICAS[cfg["metrica"]].split(" · ")[-1], f"estrategia «{cfg['estrategia']}»",
                    delta_color="off")
        m[1].metric("Rango de nsim", f"{num(df.index.min(), 0)} – {num(df.index.max(), 0)}", f"{len(df)} puntos", delta_color="off")
        m[2].metric("Réplicas por punto", str(cfg["replicas"]), "semillas independientes", delta_color="off")
        m[3].metric("Reducción de variabilidad", ("×" + num(S["reduccion"], 1)) if S["reduccion"] else "—",
                    help="σ entre réplicas del primer punto frente a la del último.")
        c1, c2 = st.columns(2)
        ver_fig(c1, gr.fig_sens_convergencia(df, cfg["metrica"]))
        ver_fig(c2, gr.fig_sens_variabilidad(df, cfg["metrica"]))
        mostrar_tabla(informe.tabla_sens(df), informe.fmt_sens_factory(cfg["metrica"]), "nsim", "small")
        st.success(S["texto"])

# ==============================================================================================
# 5) Informe
# ==============================================================================================
with tab_inf:
    R = ss.resultados
    if R is None:
        st.info("Ejecuta primero el análisis para poder generar el informe.")
    else:
        st.markdown("El informe Word incluye los parámetros, la síntesis y la recomendación, la comparación de "
                    "estrategias y, para cada estrategia, los resultados teóricos y simulados, la distribución del "
                    "beneficio por cliente y el funcionamiento de la mixtura en dos pasos.")
        titulo = st.text_input("Título del informe", "Beneficio esperado por mixtura", key="tit_inf")
        con_sens = st.checkbox("Incluir el análisis de sensibilidad", value=ss.sens is not None,
                               disabled=ss.sens is None, help="Disponible tras ejecutarlo en la pestaña Sensibilidad.")
        if st.button("📄 Generar informe (.docx)", type="primary"):
            with st.spinner("Generando informe…"):
                ss.informe_bytes = informe.generar_informe(R["res"], R["comunes"], R["ajustes"], titulo,
                                                           ss.sens if con_sens else None)
        if ss.informe_bytes:
            st.download_button("⬇️ Descargar informe", ss.informe_bytes,
                               f"Informe_Mixtura_{datetime.date.today():%Y%m%d}.docx",
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
