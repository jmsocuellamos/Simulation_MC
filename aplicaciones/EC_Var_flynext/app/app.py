# -*- coding: utf-8 -*-
"""EC1 · Optimización de costes en «Fly-Next» · aplicación Streamlit.
Coste de una entrega por el método de composición (Bloque A), coste diario con franjas de tráfico (Bloque B),
escenarios con objetivos y ahorros pareados, optimización de la mezcla de rutas, sensibilidad e informe Word.

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

import flynext as fx
import graficos as gr
import informe
from flynext import num, pct, eur

st.set_page_config(page_title="Simulador · Fly-Next", page_icon="🚚", layout="wide")
EN_NAVEGADOR = sys.platform == "emscripten"

ESCENARIOS_DEF = {
    "Actual": fx.escenario(descripcion="Operación actual del enunciado: rutas 60 / 30 / 10 %, P_fuel = 1,5 €/L, "
                                       "C_labor = 25 €/h y peaje fijo K = 50 € si D > 100 km."),
    "Peaje proporcional (O3.2)": fx.escenario(K=0.0, tarifa_km=0.15, descripcion="El peaje deja de ser fijo: 0,15 € por "
                                              "cada km por encima de 100 km."),
    "Peaje K −20 % (O3.3)": fx.escenario(K=40.0, descripcion="Se negocia una reducción del 20 % del recargo fijo de peajes "
                                         "(K = 40 €)."),
    "Proveedor de combustible (O3.4)": fx.escenario(P_fuel=1.35, factor_G=1.05, descripcion="Combustible más barato "
                                                    "(1,35 €/L) pero de menor calidad: el consumo G aumenta un 5 %."),
    "Desplazar urbanos Punta → Valle (O5.1)": fx.escenario(desplazar=0.20, descripcion="El 20 % de los pedidos Urbanos de "
                                                           "la franja Punta se entrega en la franja Valle. El coste de "
                                                           "implantación (incentivos, coste comercial) se puede fijar en el escenario."),
    "Externalizar Premium (O5.2)": fx.escenario(externalizar=True, coste_ext=120.0, descripcion="Las entregas Premium se "
                                                "externalizan a un proveedor especializado por 120 € cada una."),
}
COMUNES_DEF = copy.deepcopy(fx.COMUNES_DEF)
COMUNES_DEF["reduccion_opt"] = 0.10
AJUSTES_DEF = {"nsim": 10000, "replicas": 5, "ndias": 60, "replicas_B": 10, "semilla": 1000}
SENS_DEF = {"metrica": "entrega", "n_min": 200, "n_max": 50000, "dias_min": 5, "dias_max": 200, "puntos": 6, "replicas": 5}

# ==============================================================================================
# Estado de la sesión
# ==============================================================================================
ss = st.session_state
if "escenarios" not in ss:
    ss.escenarios = copy.deepcopy(ESCENARIOS_DEF)
    ss.comunes = copy.deepcopy(COMUNES_DEF)
    ss.ajustes = copy.deepcopy(AJUSTES_DEF)
    ss.sens_cfg = copy.deepcopy(SENS_DEF)
    ss.base = "Actual"
    ss.resultados = None
    ss.opt = None
    ss.sens_econ = None
    ss.sens = None
    ss.informe_bytes = None
    ss.version = 0
    ss.esc_activo = next(iter(ss.escenarios))
    ss.u = [0.85, 0.50, 0.50, 0.50]
    ss.cache_teo = {}


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
    """Tabla a todo el ancho, con los valores ya formateados, la primera columna con el nombre de la fila y una
    altura que muestra todas las filas sin barra de desplazamiento."""
    d = (tabla_fmt(df, fn) if fn else df.astype(object)).copy()
    nombre = primera or df.index.name or " "
    d.index = [num(i, 0) if isinstance(i, (int, np.integer)) else (" · ".join(map(str, i)) if isinstance(i, tuple) else str(i))
               for i in d.index]
    d.index.name = nombre
    d = d.reset_index()
    d.columns = [str(c) for c in d.columns]
    cfg = {nombre: st.column_config.TextColumn(nombre, width=ancho_primera)}
    for c in d.columns[1:]:
        et = c
        if iconos and c.startswith(("Teóric", "Simulad")):
            et = ("📐 " if c.startswith("Teóric") else "🎲 ") + c
        cfg[c] = st.column_config.TextColumn(et)
    st.dataframe(d.astype(str), use_container_width=True, hide_index=True, column_config=cfg, height=38 + 35 * len(d))


def clave(*partes):
    """Clave de widget ligada a la versión de la configuración (al cargar o restablecer se regeneran)."""
    return "__".join(str(p) for p in partes) + f"__v{ss.version}"


def _json(o):
    return json.loads(json.dumps(o, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))


def configuracion():
    return _json({"comunes": ss.comunes, "ajustes": ss.ajustes, "escenarios": ss.escenarios, "base": ss.base,
                  "sensibilidad": ss.sens_cfg})


def _claves_int(d):
    """Restaura las claves numéricas (rutas) de los diccionarios cargados desde JSON."""
    if isinstance(d, dict):
        return {(int(k) if isinstance(k, str) and k.isdigit() else k): _claves_int(v) for k, v in d.items()}
    return d


def cargar_configuracion(d):
    d = _claves_int(d)
    com = copy.deepcopy(COMUNES_DEF)
    for k, v in d.get("comunes", {}).items():
        com[k] = v
    com["objetivos"] = {**COMUNES_DEF["objetivos"], **d.get("comunes", {}).get("objetivos", {})}
    for r in fx.RUTAS:
        com["G"][r] = tuple(com["G"][r])
    ss.comunes = com
    ss.ajustes = {**copy.deepcopy(AJUSTES_DEF), **{k: v for k, v in d.get("ajustes", {}).items() if k in AJUSTES_DEF}}
    ss.sens_cfg = {**copy.deepcopy(SENS_DEF), **d.get("sensibilidad", {})}
    ss.escenarios = {n: {**copy.deepcopy(fx.ESC_BASE), **e} for n, e in (d.get("escenarios") or copy.deepcopy(ESCENARIOS_DEF)).items()}
    ss.base = d.get("base") if d.get("base") in ss.escenarios else next(iter(ss.escenarios))
    ss.esc_activo = next(iter(ss.escenarios))
    ss.resultados, ss.opt, ss.sens_econ, ss.sens, ss.informe_bytes = None, None, None, None, None
    ss.cache_teo = {}
    ss.version += 1


def teoria_cache(com, esc):
    k = json.dumps(_json([com, esc]), sort_keys=True)
    if k not in ss.cache_teo:
        if len(ss.cache_teo) > 40:
            ss.cache_teo = {}
        ss.cache_teo[k] = fx.teoria(com, esc)
    return ss.cache_teo[k]


com, aj = ss.comunes, ss.ajustes

# ==============================================================================================
# Barra lateral
# ==============================================================================================
with st.sidebar:
    st.header("⚙️ Simulación")
    st.markdown("**Bloque A · una entrega**")
    aj["nsim"] = int(st.number_input("Entregas por réplica", 100, 5_000_000, int(aj["nsim"]), 1000, key=clave("nsim")))
    aj["replicas"] = int(st.number_input("Réplicas A (semillas)", 2, 100, int(aj["replicas"]), key=clave("rep")))
    st.markdown("**Bloque B · días con franjas**")
    aj["ndias"] = int(st.number_input("Días por réplica", 1, 100_000, int(aj["ndias"]), 10, key=clave("ndias")))
    aj["replicas_B"] = int(st.number_input("Réplicas B (periodos)", 2, 200, int(aj["replicas_B"]), key=clave("repB"),
                                           help="Macro-réplicas de periodos completos (réplica i → semilla + 1000 + i)."))
    aj["semilla"] = int(st.number_input("Semilla de inicialización", 0, 10**9, int(aj["semilla"]), key=clave("sem"),
                                        help="Réplica A i → semilla + i. Todos los escenarios usan las mismas semillas "
                                             "(números aleatorios comunes), por lo que sus diferencias son pareadas."))
    nombres = list(ss.escenarios)
    ss.base = st.selectbox("Escenario base (para ahorros)", nombres,
                           index=nombres.index(ss.base) if ss.base in nombres else 0, key=clave("base"))
    sel = st.multiselect("Escenarios a simular", nombres, default=nombres, key=clave("sel"))
    lam_dia = sum(float(com["lambda"][fx.HORAS_FRANJA[h]]) for h in range(24))
    st.caption(f"Por escenario: {num(aj['nsim'] * aj['replicas'], 0)} entregas (A) y unos "
               f"{num(lam_dia * aj['ndias'] * aj['replicas_B'], 0)} pedidos (B).")
    if EN_NAVEGADOR:
        st.caption("En el navegador la simulación es más lenta: con muchos escenarios, entregas o días puede tardar.")
    if st.button("▶️ Ejecutar análisis", type="primary", use_container_width=True, disabled=not sel):
        barra = st.progress(0.0, text="Simulando…")
        res = {}
        pasos = len(sel) * (aj["replicas"] + aj["replicas_B"]) + 2
        cont = [0]
        def prog(texto=""):
            cont[0] += 1
            barra.progress(min(1.0, cont[0] / pasos), text=texto or "Simulando…")
        for n in sel:
            res[n] = fx.analizar_escenario(ss.escenarios[n], com, aj, lambda n=n: prog(f"{n}"))
        b = ss.base if ss.base in res else sel[0]
        prog("Optimización de rutas (O3.1)")
        ss.opt = fx.analizar_optimizacion(com, ss.escenarios[b], com["reduccion_opt"], res[b]["mcA"], aj["nsim"],
                                          aj["replicas"], aj["semilla"])
        ss.opt["escenario"] = b
        prog("Sensibilidad económica (O2.2)")
        ss.sens_econ = fx.sensibilidad_economica(com, ss.escenarios[b], [-0.5, -0.25, 0.0, 0.25, 0.5],
                                                 nsim=max(aj["nsim"], 20000), semilla=aj["semilla"]) + (b,)
        ss.resultados = {"res": res, "comunes": copy.deepcopy(com), "ajustes": copy.deepcopy(aj), "base": b,
                         "fecha": f"{datetime.datetime.now():%d/%m/%Y %H:%M}"}
        ss.informe_bytes = None
        barra.progress(1.0, text="Análisis completado")

    st.divider()
    st.subheader("💾 Configuración")
    st.download_button("Descargar configuración (JSON)", json.dumps(configuracion(), ensure_ascii=False, indent=2),
                       "configuracion_flynext.json", "application/json", use_container_width=True)
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
st.title("🚚 Optimización de costes en «Fly-Next»")
st.caption("Coste de una entrega por el método de composición (ruta → distancia, consumo y retraso) y coste diario de la "
           "operación con franjas de tráfico; escenarios con objetivos y ahorros pareados, optimización de la mezcla de "
           "rutas, sensibilidad e informe Word. Estudio de caso EC1 de Javier Morales, equipo IA4LEGOS (CC BY-SA 4.0).")
tab_com, tab_esc, tab_res, tab_opt, tab_sens, tab_inf = st.tabs(["🌐 Parámetros comunes", "🏕️ Escenarios", "📊 Resultados",
                                                                 "🎯 Optimización", "🔬 Sensibilidad", "📄 Informe"])

# ==============================================================================================
# 1) Parámetros comunes
# ==============================================================================================
with tab_com:
    st.subheader("Parámetros comunes a todos los escenarios")
    st.markdown("#### 🛣️ Distribuciones condicionadas a la ruta")
    for r in fx.RUTAS:
        c = st.columns([1.2, 1, 1, 1, 1, 1])
        c[0].markdown(f"**{r} · {fx.NOMBRE_RUTA[r]}**  \nD ~ " + ("Poisson" if com["D"][r]["tipo"] == "poisson" else "Normal"))
        com["D"][r]["mu"] = float(c[1].number_input("D · λ (km)" if com["D"][r]["tipo"] == "poisson" else "D · media (km)",
                                                    0.0, 1e5, float(com["D"][r]["mu"]), 1.0, key=clave("Dmu", r)))
        if com["D"][r]["tipo"] == "norm":
            com["D"][r]["sd"] = float(c[2].number_input("D · desv. típ. (km)", 0.01, 1e5, float(com["D"][r]["sd"]), 1.0,
                                                        key=clave("Dsd", r)))
        else:
            c[2].markdown("&nbsp;")
        g0 = float(c[3].number_input("G · mín. (L/km)", 0.0, 10.0, float(com["G"][r][0]), 0.005, format="%.3f", key=clave("g0", r)))
        g1 = float(c[4].number_input("G · máx. (L/km)", 0.0, 10.0, float(com["G"][r][1]), 0.005, format="%.3f", key=clave("g1", r)))
        com["G"][r] = (g0, max(g1, g0 + 1e-6))
        com["M"][r] = float(c[5].number_input("M · media (h)", 0.001, 1e3, float(com["M"][r]), 0.1, key=clave("M", r)))

    st.markdown("#### 🕒 Franjas horarias y tráfico")
    c = st.columns(3)
    for i, f in enumerate(fx.FRANJAS):
        horas = sum(1 for h in range(24) if fx.HORAS_FRANJA[h] == f)
        com["lambda"][f] = float(c[i].number_input(f"λ {f} (pedidos/hora · {horas} h)", 0.0, 1e5, float(com["lambda"][f]),
                                                   1.0, key=clave("lam", f)))
    c1, c2 = st.columns(2)
    for cont, tabla, tit in ((c1, "f_retraso", "Factor sobre el retraso M"), (c2, "f_consumo", "Factor sobre el consumo G")):
        with cont:
            st.markdown(f"**{tit}** (filas: franja; columnas: ruta)")
            k_df = clave(tabla)
            k_base = "base_" + k_df
            if k_base not in ss:
                ss[k_base] = pd.DataFrame({fx.NOMBRE_RUTA[r]: [float(com[tabla][f][r]) for f in fx.FRANJAS] for r in fx.RUTAS},
                                          index=list(fx.FRANJAS))
            ed = st.data_editor(ss[k_base], key=k_df, use_container_width=True,
                                column_config={fx.NOMBRE_RUTA[r]: st.column_config.NumberColumn(fx.NOMBRE_RUTA[r], min_value=0.0,
                                                                                                step=0.01, format="%.2f")
                                               for r in fx.RUTAS})
            for f in fx.FRANJAS:
                for r in fx.RUTAS:
                    v = ed.loc[f, fx.NOMBRE_RUTA[r]]
                    if v == v:
                        com[tabla][f][r] = float(v)

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("#### 💶 Estudio")
        cc = st.columns(2)
        com["umbral_Y"] = float(cc[0].number_input("Umbral de entrega cara (€)", 0.0, 1e6, float(com["umbral_Y"]), 10.0, key=clave("uY")))
        com["entregas_ano"] = int(cc[1].number_input("Entregas al año", 1, 10**9, int(com["entregas_ano"]), 1000, key=clave("ea"),
                                                     help="Para expresar los ahorros anuales (O3.3)."))
        com["presupuesto"] = float(cc[0].number_input("Presupuesto diario (€)", 0.0, 1e9, float(com["presupuesto"]), 1000.0,
                                                      key=clave("pres"), help="O6.2: P(coste diario > presupuesto)."))
        com["capacidad"] = int(cc[1].number_input("Capacidad (pedidos/hora)", 0, 10**6, int(com.get("capacidad") or 0), 1,
                                                  key=clave("cap"), help="O6.3: 0 = sin límite. Con límite, los pedidos que "
                                                                         "superan la capacidad de la hora se rechazan."))
        com["reduccion_opt"] = float(cc[0].number_input("Reducción objetivo O3.1 (%)", 0.0, 99.0, 100 * float(com["reduccion_opt"]),
                                                        1.0, key=clave("ropt"))) / 100
    with c2:
        st.markdown("#### 🏆 Objetivos")
        st.caption("Se evalúan sobre los valores teóricos. Desmarca un objetivo para no evaluarlo.")
        ob = com["objetivos"]
        def objetivo(clave_obj, etiqueta, defecto, factor, paso, ayuda):
            cc1, cc2 = st.columns([1, 2])
            activo = cc1.checkbox("Evaluar", value=ob.get(clave_obj) is not None, key=clave("obj_on", clave_obj))
            v = ob.get(clave_obj) if ob.get(clave_obj) is not None else defecto
            x = cc2.number_input(etiqueta, -1e12, 1e12, float(v) * factor, paso, key=clave("obj", clave_obj), help=ayuda,
                                 disabled=not activo)
            ob[clave_obj] = (x / factor) if activo else None
        objetivo("EY_MAX", "E(Y) máximo por entrega (€)", 36.0, 1, 0.5, "Coste esperado máximo de una entrega.")
        objetivo("PCOLA_MAX", "P(Y > umbral) máxima (%)", 0.04, 100, 0.5, "Proporción máxima de entregas caras.")
        objetivo("DIA_MAX", "Coste diario máximo (€)", 26300.0, 1, 100.0, "Coste diario esperado máximo con tráfico.")
        objetivo("PPRES_MAX", "P(día > presupuesto) máx. (%)", 0.02, 100, 0.5,
                 "Probabilidad máxima de que el coste de un día supere el presupuesto.")

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
            if ss.base == activo:
                ss.base = nuevo
            ss.esc_activo = nuevo; ss.version += 1; st.rerun()
        else:
            st.warning("Escribe un nombre nuevo que no exista.")

    e = ss.escenarios[activo]
    st.markdown(f"#### Parámetros de «{activo}»")
    st.markdown("**Mezcla de rutas y función de coste**")
    c = st.columns(5)
    p = [float(x) * 100 for x in e["p"]]
    p[0] = c[0].number_input("p Urbana (%)", 0.0, 100.0, p[0], 1.0, key=clave("p1", activo))
    p[1] = c[1].number_input("p Regional (%)", 0.0, 100.0, p[1], 1.0, key=clave("p2", activo))
    p[2] = c[2].number_input("p Premium (%)", 0.0, 100.0, p[2], 1.0, key=clave("p3", activo))
    e["p"] = [x / 100 for x in p]
    e["P_fuel"] = float(c[3].number_input("P_fuel (€/L)", 0.0, 100.0, float(e["P_fuel"]), 0.05, key=clave("pf", activo)))
    e["C_labor"] = float(c[4].number_input("C_labor (€/h)", 0.0, 1e4, float(e["C_labor"]), 1.0, key=clave("cl", activo)))
    c = st.columns(5)
    e["K"] = float(c[0].number_input("Peaje fijo K (€)", 0.0, 1e5, float(e["K"]), 5.0, key=clave("K", activo)))
    e["umbral_D"] = float(c[1].number_input("Umbral de peaje (km)", 0.0, 1e5, float(e["umbral_D"]), 10.0, key=clave("uD", activo)))
    e["tarifa_km"] = float(c[2].number_input("Peaje por km extra (€/km)", 0.0, 100.0, float(e["tarifa_km"]), 0.01,
                                             key=clave("tk", activo), help="O3.2: peaje proporcional a la distancia por encima del umbral."))
    e["factor_G"] = float(c[3].number_input("Consumo del proveedor (×G)", 0.1, 5.0, float(e["factor_G"]), 0.01,
                                            key=clave("fg", activo), help="O3.4: 1,05 = el consumo aumenta un 5 %."))
    e["coste_impl"] = float(c[4].number_input("Coste de implantación (€/día)", 0.0, 1e9, float(e.get("coste_impl") or 0.0), 50.0,
                                              key=clave("ci", activo), help="Coste operativo o comercial diario del cambio; "
                                                                            "se descuenta del ahorro."))
    st.markdown("**Gestión del tráfico y de las rutas Premium**")
    c = st.columns(4)
    e["desplazar"] = float(c[0].number_input("Urbanos de Punta a Valle (%)", 0.0, 100.0, 100 * float(e["desplazar"]), 5.0,
                                             key=clave("dz", activo), help="O5.1")) / 100
    e["reduccion_punta"] = float(c[1].number_input("Reducción del retraso en Punta (%)", 0.0, 100.0,
                                                   100 * float(e["reduccion_punta"]), 5.0, key=clave("rp", activo),
                                                   help="O5.3: el factor de retraso en Punta se multiplica por (1 − r).")) / 100
    e["externalizar"] = bool(c[2].checkbox("Externalizar Premium", value=bool(e["externalizar"]), key=clave("ex", activo),
                                           help="O5.2: coste fijo por entrega Premium."))
    e["coste_ext"] = float(c[3].number_input("Coste por entrega externalizada (€)", 0.0, 1e6, float(e["coste_ext"]), 5.0,
                                             key=clave("ce", activo), disabled=not e["externalizar"]))
    if abs(sum(p) - 100) > 1e-9:
        st.warning(f"Las probabilidades de ruta suman {num(sum(p), 1)} %: se normalizan para sumar 100 %.")
    e["descripcion"] = st.text_area("Descripción", e.get("descripcion", ""), key=clave("desc", activo), height=70)

    t = teoria_cache(com, e)
    A, B = t["A"], t["B"]
    st.markdown("##### 📐 Vista previa teórica (fórmulas exactas e integración numérica, sin simular)")
    m = st.columns(6)
    m[0].metric("E(Y) por entrega", eur(A["EY"], 2), help="Coste esperado de una entrega (sin franjas).")
    m[1].metric(f"P(Y > {num(com['umbral_Y'], 0)} €)", pct(A["Pcola"], 2))
    m[2].metric("Percentil 90 de Y", eur(A["P90"], 2))
    m[3].metric("Coste diario", eur(B["dia_con"]), f"sobrecoste {eur(B['dia_delta'])}", delta_color="off")
    m[4].metric("P95 coste diario", eur(B["P95"]), help="Aproximación normal de la suma compuesta de Poisson.")
    m[5].metric("P(día > presupuesto)", pct(B["Ppres"], 2))

    with st.expander("🎯 Mecanismo · una entrega por composición", expanded=True):
        st.caption("1 · U_T elige la ruta por transformada inversa.  2 · Condicionadas a la ruta, U_D, U_G y U_M dan D "
                   "(inversa de la Poisson o de la normal), G = g₀ + (g₁ − g₀)·U_G y M = −media·ln(1 − U_M).  3 · Coste "
                   "Y = D·G·P_fuel + M·C_labor + peaje(D); con tráfico, G y M se multiplican por los factores de la franja.")
        c = st.columns([1, 1, 1, 1, 1.2, 1])
        if c[5].button("🎲 U aleatorios", use_container_width=True):
            ss.u = [float(np.round(np.random.uniform(0.01, 0.99), 2)) for _ in range(4)]
            for k_ in ("uT_s", "uD_s", "uG_s", "uM_s"):
                ss.pop(k_, None)
        ss.u = [c[i].slider(et, 0.01, 0.99, min(max(float(ss.u[i]), 0.01), 0.99), 0.01, key=k_)
                for i, (et, k_) in enumerate((("U_T · ruta", "uT_s"), ("U_D · distancia", "uD_s"), ("U_G · consumo", "uG_s"),
                                              ("U_M · retraso", "uM_s")))]
        franja = c[4].selectbox("Franja", fx.FRANJAS, index=2, key="mec_franja")
        en = fx.una_entrega(com, e, *ss.u, franja=franja)
        m = st.columns(6)
        m[0].metric("Ruta", fx.NOMBRE_RUTA[en["T"]])
        m[1].metric("Distancia D", f"{num(en['D'], 1)} km")
        m[2].metric("Consumo G", f"{num(en['G'], 4)} L/km", f"×{num(en['fG'], 2)} con tráfico", delta_color="off")
        m[3].metric("Retraso M", f"{num(en['M'], 3)} h", f"×{num(en['fM'], 2)} con tráfico", delta_color="off")
        m[4].metric("Coste sin tráfico", eur(en["Ysin"], 2))
        m[5].metric("Coste con tráfico", eur(en["Ycon"], 2), f"+{eur(en['Ycon'] - en['Ysin'], 2)}", delta_color="inverse")
        ver_fig(st, gr.fig_mecanismo(A["p"], ss.u[0], en["sin"], en["con"]))

    st.divider()
    c1, c2 = st.columns([3, 1])
    c1.caption("Los cambios se aplican al pulsar «Ejecutar análisis» en la barra lateral.")
    if c2.button("🗑️ Eliminar este escenario", use_container_width=True, disabled=len(ss.escenarios) <= 1):
        ss.escenarios.pop(activo); ss.esc_activo = next(iter(ss.escenarios))
        if ss.base not in ss.escenarios:
            ss.base = next(iter(ss.escenarios))
        ss.version += 1; st.rerun()

# ==============================================================================================
# 3) Resultados
# ==============================================================================================
with tab_res:
    R = ss.resultados
    if R is None:
        st.info("Todavía no hay resultados. Configura los escenarios y pulsa «▶️ Ejecutar análisis» en la barra lateral.")
    else:
        res, comR, ajR, base = R["res"], R["comunes"], R["ajustes"], R["base"]
        u = comR["umbral_Y"]
        st.caption(f"Análisis del {R['fecha']} · A: {ajR['replicas']} réplicas de {num(ajR['nsim'], 0)} entregas · B: "
                   f"{ajR['replicas_B']} réplicas de {num(ajR['ndias'], 0)} días · semilla {ajR['semilla']} · base «{base}»")
        st.info("📐 **Teórico**: momentos exactos, distribución de Y por integración numérica y coste diario como suma "
                "compuesta de Poisson (percentiles diarios por aproximación normal). 🎲 **Simulado**: composición por "
                "transformada inversa con macro-réplicas (media de las réplicas con IC95 %).")
        r_sin, r_a, r_b, r_ah, r_dat = st.tabs(["Síntesis y comparación", "Bloque A · una entrega", "Bloque B · un día",
                                                f"Ahorros frente a «{base}»", "Datos de las réplicas (simulados)"])
        with r_sin:
            st.markdown("#### Síntesis")
            for n, r in res.items():
                st.markdown(f"**{n}.** " + fx.texto_escenario(n, r, comR)[len(n) + 2:])
            st.success("**Recomendación.** " + fx.texto_recomendacion(res, base, comR))
            st.markdown("**Comparación de escenarios** (filas «Teórico» = fórmulas; filas «Simulado» = Monte Carlo)")
            mostrar_tabla(fx.tabla_comparativa(res, comR), fx.fmt_kpi, "KPI", "large")
            c1, c2 = st.columns([2, 3])
            with c1:
                st.markdown("**Objetivos** (sobre los valores teóricos)")
                mostrar_tabla(fx.tabla_objetivos(res), None, "Objetivo", "large")
            with c2:
                ver_fig(st, gr.fig_validacion(res, comR))
            ver_fig(st, gr.fig_comparativa(res, comR))
            st.markdown("**Cuadro de mando propuesto (O7.1)**")
            mostrar_tabla(pd.DataFrame({"Por qué es relevante": dict(fx.KPIS_CUADRO)}), None, "KPI", "large")

        with r_a:
            n = st.selectbox("Escenario", list(res), key="det_A")
            r = res[n]; t = r["teoria"]
            st.markdown("**O1.1 – O1.2 · coste esperado, entregas caras y percentil 90** (macro-réplicas)")
            mostrar_tabla(informe._sub(r["resA"], informe.filas_A_global(u)), fx.fmt_res, "KPI", "large")
            st.markdown("**O1.3 · por ruta**: ¿qué ruta pesa más en la cola y cuál tiene mayor incertidumbre relativa?")
            mostrar_tabla(informe.tabla_A_rutas(t, r["resA"], u), informe.fmt_rutas, "Ruta", "small", iconos=True)
            c1, c2 = st.columns(2)
            ver_fig(c1, gr.fig_hist_Y(t["A"], r["mcA"]["Y"], u))
            ver_fig(c2, gr.fig_por_ruta(t["A"], r["resA"], u))
            st.markdown("**O2.1 · descomposición de la varianza**: Cov(término, Y)/Var(Y); las contribuciones suman el 100 %.")
            c1, c2 = st.columns([2, 3])
            with c1:
                mostrar_tabla(informe._sub(r["resA"], [f"Contribución a Var(Y) · {x}" for x in fx.TERMINOS]), fx.fmt_res,
                              "Término", "medium")
            with c2:
                ver_fig(st, gr.fig_descomposicion(t["A"], r["resA"]))
            ver_fig(st, gr.fig_traza_A(r["mcA"]["traza"], r["esc"]))
            with st.expander(f"Primeras {len(r['mcA']['traza'])} entregas simuladas"):
                tr = r["mcA"]["traza"].copy()
                for c_ in tr.columns[1:]:
                    if c_ != "Ruta":
                        tr[c_] = tr[c_].map(lambda v: num(v, 4 if c_ in ("U_T", "G (L/km)") else 2))
                st.dataframe(tr.astype(str), use_container_width=True, hide_index=True)

        with r_b:
            n = st.selectbox("Escenario", list(res), key="det_B")
            r = res[n]; t = r["teoria"]
            st.markdown("**O4.2 – O4.3 y O6.2 · coste diario, sobrecoste, percentil 95, CVaR 5 % y riesgo presupuestario**")
            filas_b = [k for k in r["resB"].index if "por pedido" not in k and "Cuota" not in k]
            mostrar_tabla(informe._sub(r["resB"], filas_b), fx.fmt_res, "KPI", "large")
            st.markdown("**O4.1 y O5.3 · por franja horaria**")
            mostrar_tabla(informe.tabla_franjas(t, r["resB"]), informe.fmt_franjas, "Franja", "small", iconos=True)
            c1, c2 = st.columns(2)
            ver_fig(c1, gr.fig_hist_dias(t["B"], r["mcB"]["dias"]["Coste con tráfico"], comR["presupuesto"]))
            ver_fig(c2, gr.fig_franjas(t["B"], r["resB"]))
            ver_fig(st, gr.fig_horas(comR, t["B"]))
            st.markdown("**O5.3, O6.1, O6.3, O8.1 y O8.4 · equilibrio de franjas, dimensionamiento y diseño pareado**")
            mostrar_tabla(informe.tabla_extras_B(t, r["mcB"], comR), None, "Indicador", "large")

        with r_ah:
            dif = fx.diferencias(res, base, comR)
            if dif is None:
                st.info("Simula al menos dos escenarios, incluido el escenario base, para ver los ahorros.")
            else:
                st.caption("Ahorro = coste del escenario base − coste del escenario (positivo = ahorra). Las diferencias "
                           "simuladas son pareadas: cada réplica usa las mismas semillas en todos los escenarios.")
                st.markdown(f"**O3.3 y O3.4 · ahorro por entrega y anual ({num(comR['entregas_ano'], 0)} entregas al año)**")
                mostrar_tabla(dif[["Ahorro por entrega · teórico (€)", "Ahorro por entrega · simulado (€)",
                                   "IC95 ahorro por entrega (€)", "Ahorro anual · teórico (€)", "Ahorro anual · simulado (€)",
                                   "Cambio en σ(Y) · teórico (€)", "Cambio en P(Y > umbral) · teórico"]], fx.fmt_dif,
                              "Escenario", "large", iconos=True)
                st.markdown("**O5.1 y O5.2 · ahorro diario con franjas, sobrecoste y coste de implantación**")
                mostrar_tabla(dif[["Ahorro diario · teórico (€)", "Ahorro diario · simulado (€)", "IC95 ahorro diario (€)",
                                   "Ahorro en sobrecoste diario · teórico (€)", "Ahorro en sobrecoste diario · simulado (€)",
                                   "Coste de implantación (€/día)", "Ahorro diario neto · teórico (€)"]], fx.fmt_dif,
                              "Escenario", "large", iconos=True)
                ver_fig(st, gr.fig_ahorros(dif))

        with r_dat:
            st.markdown("Resultados **simulados** de cada réplica.")
            la = pd.concat([r["mcA"]["replicas"].assign(Escenario=n) for n, r in res.items()]).reset_index()
            lb = pd.concat([r["mcB"]["replicas"].assign(Escenario=n) for n, r in res.items()]).reset_index()
            st.markdown("**Bloque A** (una fila por réplica de entregas)")
            st.dataframe(la, use_container_width=True, hide_index=True)
            st.markdown("**Bloque B** (una fila por réplica de periodo)")
            st.dataframe(lb, use_container_width=True, hide_index=True)
            buf = io.BytesIO()
            try:
                with pd.ExcelWriter(buf, engine="openpyxl") as xw:
                    la.to_excel(xw, sheet_name="Replicas A", index=False)
                    lb.to_excel(xw, sheet_name="Replicas B", index=False)
                    fx.tabla_comparativa(res, comR).astype(str).to_excel(xw, sheet_name="Comparacion")
                    for i, (n, r) in enumerate(res.items(), 1):
                        r["resA"].astype(str).to_excel(xw, sheet_name=f"KPIs A {i}")
                        r["resB"].astype(str).to_excel(xw, sheet_name=f"KPIs B {i}")
                        r["mcB"]["dias"].to_excel(xw, sheet_name=f"Dias {i}", index=False)
                    dif = fx.diferencias(res, base, comR)
                    if dif is not None:
                        dif.astype(str).to_excel(xw, sheet_name="Ahorros")
                st.download_button("Descargar Excel", buf.getvalue(), "resultados_flynext.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception:  # openpyxl no disponible
                st.download_button("Descargar CSV (A)", la.to_csv(index=False).encode("utf-8"), "replicas_A.csv", "text/csv")

# ==============================================================================================
# 4) Optimización de la mezcla de rutas (O3.1)
# ==============================================================================================
with tab_opt:
    st.subheader("O3.1 · Reducir el coste esperado cambiando solo las probabilidades de ruta")
    st.markdown("**Problema:** minimizar ‖p − p₀‖² sujeto a Σ pⱼ·μⱼ ≤ (1 − r)·Σ p₀ⱼ·μⱼ, Σ pⱼ = 1 y pⱼ ≥ 0, con "
                "μⱼ = E(Y | T = j) y sin tocar las distribuciones condicionadas. Se busca la mezcla de rutas **más "
                "parecida a la actual** que consigue la reducción pedida.")
    c = st.columns([2, 1, 1])
    nombres = list(ss.escenarios)
    eo = c[0].selectbox("Escenario de partida", nombres, index=nombres.index(ss.base) if ss.base in nombres else 0, key=clave("o_esc"))
    red = float(c[1].number_input("Reducción (%)", 0.0, 99.0, 100 * float(com["reduccion_opt"]), 1.0, key=clave("o_red"))) / 100
    c[2].markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
    if c[2].button("▶️ Optimizar", type="primary", use_container_width=True):
        with st.spinner("Resolviendo…"):
            ss.opt = fx.analizar_optimizacion(com, ss.escenarios[eo], red, None, aj["nsim"], aj["replicas"], aj["semilla"])
            ss.opt["escenario"] = eo
            ss.informe_bytes = None
    O = ss.opt
    if O is None:
        st.info("Ejecuta el análisis (barra lateral) o pulsa «Optimizar».")
    else:
        st.caption(f"Escenario de partida «{O['escenario']}» · reducción del {pct(O['reduccion'], 0)}.")
        m = st.columns(4)
        m[0].metric("📐 E(Y) actual", eur(O["teo"]["E0"] if O["teo"] else float(np.dot(O["p0"], O["mu_teo"])), 2))
        if O["teo"] is not None:
            m[1].metric("📐 E(Y) objetivo", eur(O["teo"]["objetivo"], 2))
            m[2].metric("📐 p* (U / R / P)", " / ".join(num(x, 3) for x in O["teo"]["p"]))
        else:
            m[1].metric("📐 Solución", "no factible")
        if O.get("verif"):
            v = O["verif"]
            m[3].metric("🎲 E(Y) re-simulado con p*", eur(v["E"], 2), f"IC95 [{eur(v['lo'], 2)}; {eur(v['hi'], 2)}]", delta_color="off")
        filas = {}
        for k_, et in (("teo", "📐 Con medias teóricas μⱼ"), ("sim", "🎲 Con medias simuladas μⱼ")):
            s = O[k_]
            mu = O["mu_teo"] if k_ == "teo" else O["mu_sim"]
            filas[et] = {"μ (U / R / P)": " / ".join(eur(x, 2) for x in mu),
                         "p* (U / R / P)": " / ".join(num(x, 3) for x in s["p"]) if s else "no factible",
                         "E(Y) con p*": eur(s["E"], 2) if s else "—", "Distancia ‖p* − p₀‖": num(s["distancia"], 3) if s else "—"}
        mostrar_tabla(pd.DataFrame(filas).T, None, "Cálculo", "medium")
        ver_fig(st, gr.fig_optimizacion(O))
        st.caption("Como E(Y) es lineal en p, la región factible es un semiplano del símplex y la solución es la proyección "
                   "de p₀ sobre la recta E(Y) = objetivo: se desplaza peso de la ruta Premium (la más cara) hacia las demás.")
        if O["sim"] is not None:
            nom_new = f"Rutas optimizadas (−{num(100 * O['reduccion'], 0)} %)"
            if st.button(f"➕ Crear el escenario «{nom_new}» con p* simulado"):
                ss.escenarios[nom_new] = dict(copy.deepcopy(ss.escenarios[O["escenario"]]), p=[float(x) for x in O["sim"]["p"]],
                                              descripcion=f"Mezcla de rutas de O3.1 (p* = {' / '.join(num(x, 3) for x in O['sim']['p'])}).")
                ss.version += 1
                st.rerun()

# ==============================================================================================
# 5) Sensibilidad
# ==============================================================================================
with tab_sens:
    st.subheader("Sensibilidad")
    s_ec, s_est = st.tabs(["O2.2 · parámetros económicos", "Estimador Monte Carlo"])
    with s_ec:
        st.caption("Se varía cada parámetro económico (P_fuel, C_labor, K) manteniendo los demás fijos y se mide su "
                   "impacto en Var(Y). La simulación usa una misma muestra de uniformes (números aleatorios comunes).")
        c = st.columns([2, 1, 1])
        nombres = list(ss.escenarios)
        es_ = c[0].selectbox("Escenario", nombres, index=nombres.index(ss.base) if ss.base in nombres else 0, key=clave("se_esc"))
        rango = float(c[1].number_input("Variación máxima (± %)", 1.0, 200.0, 50.0, 5.0, key=clave("se_r"))) / 100
        c[2].markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
        if c[2].button("▶️ Calcular", use_container_width=True):
            with st.spinner("Calculando…"):
                ss.sens_econ = fx.sensibilidad_economica(com, ss.escenarios[es_], list(np.linspace(-rango, rango, 5)),
                                                         nsim=max(aj["nsim"], 20000), semilla=aj["semilla"]) + (es_,)
                ss.informe_bytes = None
        if ss.sens_econ is None:
            st.info("Ejecuta el análisis o pulsa «Calcular».")
        else:
            df, elas, en_ = ss.sens_econ
            mayor = max(elas, key=elas.get)
            st.success(f"Escenario «{en_}»: el parámetro con más impacto en Var(Y) es **{mayor}** (elasticidad "
                       f"{num(elas[mayor], 2)}: un 1 % más de {mayor.split(' ')[0]} aumenta Var(Y) un {num(elas[mayor], 2)} %).")
            ver_fig(st, gr.fig_sens_econ(df, elas))
            mostrar_tabla(informe.tabla_sens_econ(df), informe.fmt_sens_econ, "Parámetro · variación", "medium", iconos=True)
    with s_est:
        st.caption("¿Cuánto «tiembla» la estimación según el tamaño de la simulación? Para cada valor se repite con varias semillas.")
        sc = ss.sens_cfg
        c = st.columns(3)
        met = ["entrega", "dia"]
        sc["metrica"] = c[0].selectbox("Métrica", met, index=met.index(sc.get("metrica", "entrega")), key=clave("s_met"),
                                       format_func=lambda k: "E(Y) por entrega · por nº de entregas" if k == "entrega"
                                       else "Coste diario medio · por nº de días")
        if sc["metrica"] == "entrega":
            sc["n_min"] = int(c[1].number_input("Entregas mínimo", 10, 10**7, int(sc["n_min"]), 50, key=clave("s_nmin")))
            sc["n_max"] = int(c[2].number_input("Entregas máximo", 20, 10**7, int(sc["n_max"]), 1000, key=clave("s_nmax")))
        else:
            sc["dias_min"] = int(c[1].number_input("Días mínimo", 1, 10**5, int(sc["dias_min"]), 5, key=clave("s_dmin")))
            sc["dias_max"] = int(c[2].number_input("Días máximo", 2, 10**5, int(sc["dias_max"]), 10, key=clave("s_dmax")))
        c = st.columns(3)
        sc["puntos"] = int(c[0].slider("Puntos a evaluar", 3, 10, int(sc["puntos"]), key=clave("s_pts")))
        sc["replicas"] = int(c[1].slider("Réplicas por punto", 2, 20, int(sc["replicas"]), key=clave("s_rep")))
        pol = c[2].selectbox("Escenario", list(ss.escenarios), key=clave("s_pol"))
        if st.button("▶️ Analizar sensibilidad", type="primary"):
            vals = (fx.valores_barrido(sc["n_min"], sc["n_max"], sc["puntos"]) if sc["metrica"] == "entrega" else
                    fx.valores_barrido(sc["dias_min"], sc["dias_max"], sc["puntos"]))
            barra = st.progress(0.0, text="Simulando…")
            df = fx.sensibilidad_estimador(com, ss.escenarios[pol], sc["metrica"], vals, sc["replicas"], aj["semilla"],
                                           lambda k, m: barra.progress(k / m, text=f"Simulación {k} de {m}"))
            a, z = df["σ entre réplicas"].iloc[0], df["σ entre réplicas"].iloc[-1]
            red_ = a / z if a > 0 and z > 0 else None
            dentro = int(((df["IC95 inf"] <= df["Teórico"]) & (df["Teórico"] <= df["IC95 sup"])).sum())
            texto = (f"Escenario «{pol}». Al crecer el tamaño de la simulación, la media de las réplicas se acerca al valor "
                     "teórico y la dispersión entre semillas se reduce"
                     + (f" ({num(red_, 1)} veces; la teoría, 1/√n, predice {num((df.index[-1] / df.index[0]) ** 0.5, 1)})" if red_ else "")
                     + f". El valor teórico queda dentro del IC95 % en {dentro} de los {len(df)} puntos.")
            ss.sens = {"df": df, "texto": texto, "cfg": {"metrica": sc["metrica"], "escenario": pol}}
            ss.informe_bytes = None
            barra.progress(1.0, text="Barrido completado")
        S = ss.sens
        if S is None:
            st.info("Pulsa «▶️ Analizar sensibilidad» para ejecutar el barrido.")
        else:
            c1, c2 = st.columns(2)
            ver_fig(c1, gr.fig_sens_convergencia(S["df"], S["cfg"]["metrica"]))
            ver_fig(c2, gr.fig_sens_variabilidad(S["df"], S["cfg"]["metrica"]))
            mostrar_tabla(informe.tabla_sens(S["df"]), informe.fmt_sens, "n", "small")
            st.success(S["texto"])

# ==============================================================================================
# 6) Informe
# ==============================================================================================
with tab_inf:
    R = ss.resultados
    if R is None:
        st.info("Ejecuta primero el análisis para poder generar el informe.")
    else:
        st.markdown("El informe Word se organiza por los objetivos del encargo: planteamiento, síntesis y recomendación "
                    "con el cuadro de mando (O7), comparación de escenarios y ahorros (O3.2–O3.4, O5.1–O5.2) y, para cada "
                    "escenario, el coste de una entrega (O1, O2.1) y el coste diario con franjas (O4, O5.3, O6, O8); "
                    "además, la optimización de rutas (O3.1) y la sensibilidad económica (O2.2).")
        titulo = st.text_input("Título del informe", "Optimización de costes en Fly-Next", key="tit_inf")
        c = st.columns(3)
        con_opt = c[0].checkbox("Incluir la optimización (O3.1)", value=ss.opt is not None, disabled=ss.opt is None)
        con_ec = c[1].checkbox("Incluir la sensibilidad económica (O2.2)", value=ss.sens_econ is not None, disabled=ss.sens_econ is None)
        con_est = c[2].checkbox("Incluir la sensibilidad del estimador", value=ss.sens is not None, disabled=ss.sens is None)
        if st.button("📄 Generar informe (.docx)", type="primary"):
            with st.spinner("Generando informe…"):
                ss.informe_bytes = informe.generar_informe(
                    R["res"], R["comunes"], R["ajustes"], R["base"], ss.opt if con_opt else None,
                    ss.sens_econ[:2] if (con_ec and ss.sens_econ) else None, ss.sens if con_est else None, titulo)
        if ss.informe_bytes:
            st.download_button("⬇️ Descargar informe", ss.informe_bytes, f"Informe_FlyNext_{datetime.date.today():%Y%m%d}.docx",
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
