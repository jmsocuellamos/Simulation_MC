# -*- coding: utf-8 -*-
"""EC2 · Farmacoeconomía de la terapia para la superbacteria «Staph-X» · aplicación Streamlit.
Coste de un paciente por el método de composición (Bloque A), coste de un brote de 90 días con capacidad UCI limitada
(Bloque B), escenarios con objetivos, ahorros pareados e ICER, dimensionamiento de la UCI, sensibilidad, ampliación
optativa e informe Word.

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

import staphx as sx
import graficos as gr
import informe
from staphx import num, pct, eur

st.set_page_config(page_title="Simulador · Staph-X", page_icon="🧫", layout="wide")
EN_NAVEGADOR = sys.platform == "emscripten"

ESCENARIOS_DEF = {
    "Actual": sx.escenario(descripcion="Tratamiento actual del enunciado: C_fijo = 1.200 €, C_día = 600 €, C_UCI = 5.000 € "
                                       "si D > 7 días, 3 camas UCI y clínica externa a 9.000 €."),
    "Genérico (O3.1)": sx.escenario(C_fijo=600.0, factor_L=1.3, descripcion="Tratamiento genérico más barato (C_fijo = "
                                    "600 €) pero menos eficaz: la carga bacteriana L es un 30 % mayor en todos los "
                                    "fenotipos (supuesto)."),
    "Fármaco Persistente (O3.2)": sx.escenario(factor_R3=2.0, coste_extra=800.0, descripcion="Nuevo fármaco que reduce a "
                                               "la mitad la tasa inversa de depuración del Persistente (R × 2) con un "
                                               "coste adicional de 800 € por tratamiento."),
    "Pago por resultados (O3.3)": sx.escenario(devolucion=0.5, descripcion="El fabricante devuelve el 50 % de C_fijo si el "
                                               "paciente desarrolla complicación: Y = C_fijo + C_día·D + "
                                               "(C_UCI − 0,5·C_fijo)·I{D > 7}."),
    "Aislamiento (O5.1)": sx.escenario(lambda_pico=18.0, coste_impl=50000.0, descripcion="Aislamiento preventivo que "
                                       "reduce λ del Pico de 25 a 18 ingresos/día, con un coste de implantación de "
                                       "50.000 € por brote (supuesto)."),
    "Contrato por volumen (O5.2)": sx.escenario(tramo_Q=15, coste_tramo=7000.0, descripcion="Contrato con la clínica "
                                                "privada de coste decreciente con el volumen: 9.000 € los 15 primeros "
                                                "derivados del brote y 7.000 € a partir del 16.º (supuesto)."),
    "5 camas (O6.3a)": sx.escenario(camas=5, descripcion="5 camas UCI fijas todo el año (coste de oportunidad de la "
                                    "reserva incluido)."),
    "3 + 4 de alquiler (O6.3b)": sx.escenario(camas_alquiler=4, coste_alquiler=6500.0, descripcion="3 camas fijas y "
                                              "opción de alquilar hasta 4 más solo cuando se necesitan, a 6.500 € por "
                                              "uso (supuesto: sin cuota fija)."),
    "Diagnóstico precoz (O8.6)": sx.escenario(factor_L3=0.7, coste_impl=108000.0, descripcion="Programa de diagnóstico "
                                              "temprano que reduce un 30 % la carga inicial del Persistente, con un coste "
                                              "de 108.000 € por brote (supuesto)."),
}
COMUNES_DEF = copy.deepcopy(sx.COMUNES_DEF)
AJUSTES_DEF = {"nsim": 10000, "replicas": 5, "brotes": 100 if EN_NAVEGADOR else 200, "semilla": 1000}
SENS_DEF = {"metrica": "paciente", "n_min": 200, "n_max": 50000, "b_min": 10, "b_max": 400, "puntos": 6, "replicas": 5}
AMPL_DEF = {"coste_10": 12000.0, "presupuesto": 250000.0, "cvs": [0.0, 0.2, 0.4, 0.6]}

# ==============================================================================================
# Estado de la sesión
# ==============================================================================================
ss = st.session_state
if "escenarios" not in ss:
    ss.escenarios = copy.deepcopy(ESCENARIOS_DEF)
    ss.comunes = copy.deepcopy(COMUNES_DEF)
    ss.ajustes = copy.deepcopy(AJUSTES_DEF)
    ss.sens_cfg = copy.deepcopy(SENS_DEF)
    ss.ampl_cfg = copy.deepcopy(AMPL_DEF)
    ss.base = "Actual"
    ss.resultados = None
    ss.capacidad = None
    ss.sensib = None
    ss.ampliacion = None
    ss.sens = None
    ss.informe_bytes = None
    ss.version = 0
    ss.esc_activo = next(iter(ss.escenarios))
    ss.u = [0.95, 0.50, 0.50, 0.50]
    ss.cache_teo = {}


def ver_fig(contenedor, fig):
    """Muestra una figura de matplotlib y la cierra para no acumular memoria entre ejecuciones."""
    contenedor.pyplot(fig, use_container_width=True)
    plt.close(fig)


def tabla_fmt(df, fn):
    out = df.copy().astype(object)
    for i, f in enumerate(df.index):
        for j, c in enumerate(df.columns):
            out.iloc[i, j] = fn(f, c, df.iloc[i, j])
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
                  "sensibilidad": ss.sens_cfg, "ampliacion": ss.ampl_cfg})


def _claves_int(d):
    """Restaura las claves numéricas (fenotipos) de los diccionarios cargados desde JSON."""
    if isinstance(d, dict):
        return {(int(k) if isinstance(k, str) and k.isdigit() else k): _claves_int(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_claves_int(v) for v in d]
    return d


def cargar_configuracion(d):
    d = _claves_int(d)
    com = copy.deepcopy(COMUNES_DEF)
    for k, v in d.get("comunes", {}).items():
        com[k] = v
    com["objetivos"] = {**COMUNES_DEF["objetivos"], **d.get("comunes", {}).get("objetivos", {})}
    com["R"] = {f: tuple(com["R"][f]) for f in sx.FENOTIPOS}
    com["alpha"] = {f: float(com["alpha"][f]) for f in sx.FENOTIPOS}
    ss.comunes = com
    ss.ajustes = {**copy.deepcopy(AJUSTES_DEF), **{k: v for k, v in d.get("ajustes", {}).items() if k in AJUSTES_DEF}}
    ss.sens_cfg = {**copy.deepcopy(SENS_DEF), **d.get("sensibilidad", {})}
    ss.ampl_cfg = {**copy.deepcopy(AMPL_DEF), **d.get("ampliacion", {})}
    ss.escenarios = {n: {**copy.deepcopy(sx.ESC_BASE), **e} for n, e in (d.get("escenarios") or copy.deepcopy(ESCENARIOS_DEF)).items()}
    ss.base = d.get("base") if d.get("base") in ss.escenarios else next(iter(ss.escenarios))
    ss.esc_activo = next(iter(ss.escenarios))
    ss.resultados, ss.capacidad, ss.sensib, ss.ampliacion, ss.sens, ss.informe_bytes = None, None, None, None, None, None
    ss.cache_teo = {}
    for k_ in [k for k in ss.keys() if str(k).startswith("base_")]:
        del ss[k_]
    ss.version += 1


def teoria_cache(com, esc):
    k = json.dumps(_json([com, esc]), sort_keys=True)
    if k not in ss.cache_teo:
        if len(ss.cache_teo) > 40:
            ss.cache_teo = {}
        ss.cache_teo[k] = sx.teoria(com, esc)
    return ss.cache_teo[k]


def calcular_capacidad(com, esc, nombre, aj, mcB=None, tB=None):
    df, b = sx.barrido_camas(com, esc)
    out = {"escenario": nombre, "camas": df, "camas_opt": b, "cuota": sx.cuota_maxima_pico(com, esc)}
    if mcB is None:
        mcB = sx.simular_B(com, esc, aj["brotes"], aj["semilla"])
        tB = sx.teoria_B(com, esc)
    out["pareada"] = sx.verificacion_pareada(mcB, tB)
    out["ocupacion"] = sx.simular_ocupacion(com, esc, aj["brotes"], aj["semilla"])
    return out


def calcular_sensib(com, esc, nombre, aj, rango=0.2):
    return {"escenario": nombre,
            "param": sx.sensibilidad_parametros(com, esc, list(np.linspace(-rango, rango, 5))),
            "sobol": sx.indices_sobol(com, esc, n=max(aj["nsim"], 20000), semilla=aj["semilla"]),
            "umbral": sx.experimento_umbral(com, esc, (5.0, float(esc["umbral"]), 10.0) if esc["umbral"] not in (5, 10)
                                            else (5.0, 7.0, 10.0), nsim=max(aj["nsim"], 20000), semilla=aj["semilla"])}


def calcular_ampliacion(com, esc, nombre, aj, cfg):
    return {"escenario": nombre, "cfg": dict(cfg),
            "optim": sx.optimizacion_conjunta(com, esc, coste_10=cfg["coste_10"], presupuesto=cfg["presupuesto"]),
            "diagnostico": sx.ahorro_diagnostico(com, esc, coste_10=cfg["coste_10"]),
            "cox": sx.experimento_cox(com, esc, tuple(cfg["cvs"]), brotes=min(int(aj["brotes"]), 300), semilla=aj["semilla"])}


com, aj = ss.comunes, ss.ajustes

# ==============================================================================================
# Barra lateral
# ==============================================================================================
with st.sidebar:
    st.header("⚙️ Simulación")
    st.markdown("**Bloque A · un paciente**")
    aj["nsim"] = int(st.number_input("Pacientes por réplica", 100, 5_000_000, int(aj["nsim"]), 1000, key=clave("nsim")))
    aj["replicas"] = int(st.number_input("Réplicas A (semillas)", 2, 100, int(aj["replicas"]), key=clave("rep")))
    st.markdown("**Bloque B · brote de 90 días**")
    aj["brotes"] = int(st.number_input("Brotes simulados (réplicas B)", 2, 100_000, int(aj["brotes"]), 50, key=clave("brotes"),
                                       help="Cada brote completo es una réplica (réplica i → semilla + 1000 + i)."))
    aj["semilla"] = int(st.number_input("Semilla de inicialización", 0, 10**9, int(aj["semilla"]), key=clave("sem"),
                                        help="Réplica A i → semilla + i. Todos los escenarios usan las mismas semillas "
                                             "(números aleatorios comunes), por lo que sus diferencias son pareadas."))
    nombres = list(ss.escenarios)
    ss.base = st.selectbox("Escenario base (para ahorros e ICER)", nombres,
                           index=nombres.index(ss.base) if ss.base in nombres else 0, key=clave("base"))
    sel = st.multiselect("Escenarios a simular", nombres, default=nombres, key=clave("sel"))
    pac_brote = sum(float(f["lambda"]) * int(f["dias"]) for f in com["fases"])
    st.caption(f"Por escenario: {num(aj['nsim'] * aj['replicas'], 0)} pacientes (A) y unos "
               f"{num(pac_brote * aj['brotes'], 0)} pacientes en {num(aj['brotes'], 0)} brotes (B).")
    if EN_NAVEGADOR:
        st.caption("En el navegador la simulación es más lenta: con muchos escenarios, pacientes o brotes puede tardar.")
    if st.button("▶️ Ejecutar análisis", type="primary", use_container_width=True, disabled=not sel):
        barra = st.progress(0.0, text="Simulando…")
        res = {}
        pasos = len(sel) * (aj["replicas"] + aj["brotes"]) + 3
        cont = [0]
        def prog(texto=""):
            cont[0] += 1
            barra.progress(min(1.0, cont[0] / pasos), text=texto or "Simulando…")
        for n in sel:
            res[n] = sx.analizar_escenario(ss.escenarios[n], com, aj, lambda n=n: prog(f"{n}"))
        b = ss.base if ss.base in res else sel[0]
        prog("Capacidad UCI (O6, O8.1, O8.2, O8.5)")
        ss.capacidad = calcular_capacidad(com, ss.escenarios[b], b, aj, res[b]["mcB"], res[b]["teoria"]["B"])
        prog("Sensibilidad (O2.2, O3.4)")
        ss.sensib = calcular_sensib(com, ss.escenarios[b], b, aj)
        prog("Ampliación (O8.3, O8.4, O8.6)")
        ss.ampliacion = calcular_ampliacion(com, ss.escenarios[b], b, aj, ss.ampl_cfg)
        ss.resultados = {"res": res, "comunes": copy.deepcopy(com), "ajustes": copy.deepcopy(aj), "base": b,
                         "fecha": f"{datetime.datetime.now():%d/%m/%Y %H:%M}"}
        ss.informe_bytes = None
        barra.progress(1.0, text="Análisis completado")

    st.divider()
    st.subheader("💾 Configuración")
    st.download_button("Descargar configuración (JSON)", json.dumps(configuracion(), ensure_ascii=False, indent=2),
                       "configuracion_staphx.json", "application/json", use_container_width=True)
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
st.title("🧫 Farmacoeconomía de la terapia para «Staph-X»")
st.caption("Coste de un paciente por el método de composición (fenotipo → carga bacteriana, depuración y días de ingreso) "
           "y coste de un brote de 90 días con capacidad UCI limitada; escenarios con objetivos, ahorros pareados e ICER, "
           "dimensionamiento de la UCI, sensibilidad, ampliación e informe Word. Estudio de caso EC2 de Javier Morales, "
           "equipo IA4LEGOS (CC BY-SA 4.0).")
tab_com, tab_esc, tab_res, tab_cap, tab_sens, tab_amp, tab_inf = st.tabs(
    ["🌐 Parámetros comunes", "🏕️ Escenarios", "📊 Resultados", "🛏️ Capacidad UCI", "🔬 Sensibilidad",
     "🧪 Ampliación (O8)", "📄 Informe"])

# ==============================================================================================
# 1) Parámetros comunes
# ==============================================================================================
with tab_com:
    st.subheader("Parámetros comunes a todos los escenarios")
    st.markdown("#### 🧬 Fenotipos y distribuciones condicionadas")
    p = [float(x) for x in com["p"]]
    for f in sx.FENOTIPOS:
        c = st.columns([1.2, 1, 1, 1, 1])
        c[0].markdown(f"**T = {f} · {sx.NOMBRE_FEN[f]}**  \nL ~ Gamma(α; escala), R ~ U(r₀, r₁)")
        p[f - 1] = float(c[1].number_input("Probabilidad p (%)", 0.0, 100.0, 100 * p[f - 1], 1.0, key=clave("p", f))) / 100
        com["alpha"][f] = float(c[2].number_input("α de la Gamma", 0.05, 1e3, float(com["alpha"][f]), 0.5, key=clave("a", f)))
        r0 = float(c[3].number_input("R · r₀ (log/día)", 0.001, 100.0, float(com["R"][f][0]), 0.05, format="%.3f", key=clave("r0", f)))
        r1 = float(c[4].number_input("R · r₁ (log/día)", 0.001, 100.0, float(com["R"][f][1]), 0.05, format="%.3f", key=clave("r1", f)))
        com["R"][f] = (r0, max(r1, r0 + 1e-6))
    com["p"] = p
    if abs(sum(p) - 1) > 1e-9:
        st.warning(f"Las probabilidades de fenotipo suman {num(100 * sum(p), 1)} %: se normalizan para sumar 100 %.")
    c = st.columns(4)
    com["escala_L"] = float(c[0].number_input("Escala de la Gamma (β⁻¹)", 0.01, 100.0, float(com["escala_L"]), 0.1, key=clave("escL")))
    com["eps_mu"] = float(c[1].number_input("ε · media (días)", -10.0, 10.0, float(com["eps_mu"]), 0.1, key=clave("emu")))
    com["eps_sd"] = float(c[2].number_input("ε · desv. típica (días)", 0.001, 10.0, float(com["eps_sd"]), 0.05, key=clave("esd")))
    com["coste_cama_dia"] = float(c[3].number_input("Coste de oportunidad por cama y día (€)", 0.0, 1e5,
                                                    float(com["coste_cama_dia"]), 10.0, key=clave("ccd"),
                                                    help="Supuesto para O6.1 y O6.3: coste de reservar una cama UCI "
                                                         "durante el brote, la use o no."))

    st.markdown("#### 🦠 Fases del brote (Tablas 1 y 2)")
    k_df = clave("fases"); k_base = "base_" + k_df
    if k_base not in ss:
        ss[k_base] = pd.DataFrame({"Días": [int(f["dias"]) for f in com["fases"]],
                                   "λ (ingresos/día)": [float(f["lambda"]) for f in com["fases"]],
                                   "p Rápido": [float(f["p"][0]) for f in com["fases"]],
                                   "p Estándar": [float(f["p"][1]) for f in com["fases"]],
                                   "p Persistente": [float(f["p"][2]) for f in com["fases"]]},
                                  index=[f["nombre"] for f in com["fases"]])
    ed = st.data_editor(ss[k_base], key=k_df, use_container_width=True,
                        column_config={"Días": st.column_config.NumberColumn("Días", min_value=0, step=1),
                                       "λ (ingresos/día)": st.column_config.NumberColumn("λ (ingresos/día)", min_value=0.0, step=0.5, format="%.2f"),
                                       **{c_: st.column_config.NumberColumn(c_, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
                                          for c_ in ("p Rápido", "p Estándar", "p Persistente")}})
    for i, f in enumerate(com["fases"]):
        fila = ed.iloc[i]
        if fila["Días"] == fila["Días"]:
            f["dias"] = int(fila["Días"])
        if fila["λ (ingresos/día)"] == fila["λ (ingresos/día)"]:
            f["lambda"] = float(fila["λ (ingresos/día)"])
        f["p"] = [float(fila[c_]) if fila[c_] == fila[c_] else 0.0 for c_ in ("p Rápido", "p Estándar", "p Persistente")]
    st.caption(f"Duración del brote: {sum(int(f['dias']) for f in com['fases'])} días · las mezclas de cada fase se "
               "normalizan para sumar 1.")

    st.markdown("#### 🏆 Objetivos")
    st.caption("Se evalúan sobre los valores teóricos. Desmarca un objetivo para no evaluarlo.")
    ob = com["objetivos"]
    def objetivo(clave_obj, etiqueta, defecto, factor, paso, ayuda):
        cc1, cc2 = st.columns([1, 3])
        activo = cc1.checkbox("Evaluar", value=ob.get(clave_obj) is not None, key=clave("obj_on", clave_obj))
        v = ob.get(clave_obj) if ob.get(clave_obj) is not None else defecto
        x = cc2.number_input(etiqueta, -1e12, 1e12, float(v) * factor, paso, key=clave("obj", clave_obj), help=ayuda,
                             disabled=not activo)
        ob[clave_obj] = (x / factor) if activo else None
    c1, c2 = st.columns(2)
    with c1:
        objetivo("EY_MAX", "E(Y) máximo por paciente (€)", 3200.0, 1, 50.0, "Coste esperado máximo de un paciente.")
        objetivo("PCOMP_MAX", "P(complicación) máxima (%)", 0.07, 100, 0.5, "Proporción máxima de pacientes complicados.")
    with c2:
        objetivo("TOTAL_MAX", "Coste total del brote máximo (€)", 4_550_000.0, 1, 10000.0,
                 "Coste total esperado del brote con capacidad, reserva de camas e implantación.")
        objetivo("EXT_MAX", "Derivados por brote máximo", 15.0, 1, 1.0, "Pacientes derivados a la clínica externa.")

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
    st.markdown("**Tratamiento y función de coste**")
    c = st.columns(5)
    e["C_fijo"] = float(c[0].number_input("C_fijo (€)", 0.0, 1e6, float(e["C_fijo"]), 50.0, key=clave("cf", activo)))
    e["coste_extra"] = float(c[1].number_input("Coste adicional del fármaco (€)", 0.0, 1e6, float(e["coste_extra"]), 50.0,
                                               key=clave("cx", activo), help="O3.2: coste extra por tratamiento."))
    e["C_dia"] = float(c[2].number_input("C_día (€/día)", 0.0, 1e5, float(e["C_dia"]), 10.0, key=clave("cd", activo)))
    e["C_UCI"] = float(c[3].number_input("C_UCI (€)", 0.0, 1e6, float(e["C_UCI"]), 100.0, key=clave("cu", activo)))
    e["umbral"] = float(c[4].number_input("Umbral de complicación (días)", 0.0, 100.0, float(e["umbral"]), 0.5,
                                          key=clave("um", activo), help="O3.4"))
    c = st.columns(5)
    e["devolucion"] = float(c[0].number_input("Devolución de C_fijo si complica (%)", 0.0, 100.0, 100 * float(e["devolucion"]),
                                              5.0, key=clave("dv", activo), help="O3.3: pago por resultados.")) / 100
    e["factor_L"] = float(c[1].number_input("Carga L · todos (×)", 0.05, 10.0, float(e["factor_L"]), 0.05, key=clave("fl", activo),
                                            help="O3.1: 1,3 = un 30 % más de carga bacteriana (genérico menos eficaz)."))
    e["factor_L3"] = float(c[2].number_input("Carga L · Persistente (×)", 0.05, 10.0, float(e["factor_L3"]), 0.05,
                                             key=clave("fl3", activo), help="O8.6: diagnóstico precoz (< 1 reduce la carga)."))
    e["factor_R3"] = float(c[3].number_input("Depuración R · Persistente (×)", 0.05, 10.0, float(e["factor_R3"]), 0.05,
                                             key=clave("fr3", activo), help="O3.2: 2 = la tasa inversa de depuración se reduce a la mitad."))
    e["coste_impl"] = float(c[4].number_input("Coste de implantación (€/brote)", 0.0, 1e9, float(e.get("coste_impl") or 0.0),
                                              1000.0, key=clave("ci", activo), help="Coste de la medida por brote (O5.1, O8.6)."))
    st.markdown("**Brote y capacidad UCI**")
    c = st.columns(6)
    lp = e.get("lambda_pico")
    lam_pico_def = next((float(f["lambda"]) for f in com["fases"] if f["nombre"] == "Pico"), 25.0)
    e["lambda_pico"] = float(c[0].number_input("λ del Pico (ingresos/día)", 0.0, 1e4, float(lp if lp not in (None, "") else lam_pico_def),
                                               1.0, key=clave("lp", activo), help="O5.1: aislamiento preventivo."))
    if abs(e["lambda_pico"] - lam_pico_def) < 1e-12:
        e["lambda_pico"] = None
    e["camas"] = int(c[1].number_input("Camas UCI propias", 0, 100, int(e["camas"]), 1, key=clave("cm", activo)))
    e["C_ext"] = float(c[2].number_input("C_ext clínica externa (€)", 0.0, 1e6, float(e["C_ext"]), 100.0, key=clave("ce", activo)))
    e["camas_alquiler"] = int(c[3].number_input("Camas de alquiler", 0, 100, int(e["camas_alquiler"]), 1, key=clave("ca", activo),
                                                help="O6.3b: se usan solo si las propias están llenas."))
    e["coste_alquiler"] = float(c[4].number_input("Coste por uso de alquiler (€)", 0.0, 1e6, float(e["coste_alquiler"]), 100.0,
                                                  key=clave("cal", activo), disabled=e["camas_alquiler"] == 0))
    tramo = c[5].checkbox("Contrato por volumen", value=int(e.get("tramo_Q") or 0) > 0, key=clave("tr_on", activo), help="O5.2")
    c = st.columns(6)
    if tramo:
        e["tramo_Q"] = int(c[0].number_input("Derivados a precio normal por brote", 1, 10**5, max(1, int(e.get("tramo_Q") or 15)), 1,
                                             key=clave("tq", activo)))
        e["coste_tramo"] = float(c[1].number_input("Coste a partir de ahí (€)", 0.0, 1e6, float(e["coste_tramo"]), 100.0,
                                                   key=clave("ct", activo)))
    else:
        e["tramo_Q"] = 0
    e["descripcion"] = st.text_area("Descripción", e.get("descripcion", ""), key=clave("desc", activo), height=70)

    t = teoria_cache(com, e)
    A, B = t["A"], t["B"]
    st.markdown("##### 📐 Vista previa teórica (fórmulas exactas, sin simular)")
    m = st.columns(6)
    m[0].metric("E(Y) por paciente", eur(A["EY"]), help="Coste esperado de un paciente.")
    m[1].metric("P(complicación)", pct(A["Pcomp"], 2))
    m[2].metric("Percentil 95 de Y", eur(A["P95"]))
    m[3].metric("Coste del brote", eur(B["total_con"]), f"sobrecoste {eur(B['delta'])}", delta_color="off")
    m[4].metric("Derivados por brote", num(B["derivados"], 1), f"{num(B['dias_desb'], 1)} días con desbordamiento", delta_color="off")
    m[5].metric("Con reserva e implantación", eur(B["total_gestion"]))

    with st.expander("🎯 Mecanismo · un paciente por composición", expanded=True):
        st.caption("1 · U_T elige el fenotipo por transformada inversa.  2 · Condicionadas al fenotipo, U_L, U_R y U_ε dan "
                   "L (inversa de la Gamma), R = r₀ + (r₁ − r₀)·U_R y ε = μ + σ·Φ⁻¹(U_ε); D = máx(0, ln(L + 1)/R + ε).  "
                   "3 · Coste Y = C_fijo + C_día·D + C_UCI·I{D > umbral}.")
        c = st.columns([1, 1, 1, 1, 1])
        if c[4].button("🎲 U aleatorios", use_container_width=True):
            ss.u = [float(np.round(np.random.uniform(0.01, 0.99), 2)) for _ in range(4)]
            for k_ in ("uT_s", "uL_s", "uR_s", "ue_s"):
                ss.pop(k_, None)
        ss.u = [c[i].slider(et, 0.01, 0.99, min(max(float(ss.u[i]), 0.01), 0.99), 0.01, key=k_)
                for i, (et, k_) in enumerate((("U_T · fenotipo", "uT_s"), ("U_L · carga", "uL_s"), ("U_R · depuración", "uR_s"),
                                              ("U_ε · ruido", "ue_s")))]
        pac = sx.un_paciente(com, e, *ss.u)
        m = st.columns(6)
        m[0].metric("Fenotipo", sx.NOMBRE_FEN[pac["T"]])
        m[1].metric("Carga L", f"{num(pac['L'], 2)} UFC/ml")
        m[2].metric("Depuración R", f"{num(pac['R'], 3)} log/día")
        m[3].metric("Ruido ε", f"{num(pac['eps'], 2)} días")
        m[4].metric("Días D", num(pac["D"], 2), "complica" if pac["comp"] else "no complica",
                    delta_color="inverse" if pac["comp"] else "off")
        m[5].metric("Coste Y", eur(pac["Y"]))
        ver_fig(st, gr.fig_mecanismo(A["p"], ss.u[0], pac, e["umbral"]))

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
        st.caption(f"Análisis del {R['fecha']} · A: {ajR['replicas']} réplicas de {num(ajR['nsim'], 0)} pacientes · B: "
                   f"{num(ajR['brotes'], 0)} brotes · semilla {ajR['semilla']} · base «{base}»")
        st.info("📐 **Teórico**: momentos y distribución exactos (discretización fina de Z = ln(L+1)/R y normal truncada) "
                "y brote como suma compuesta de Poisson (percentil 95 por aproximación normal). 🎲 **Simulado**: "
                "composición por transformada inversa con macro-réplicas (media de las réplicas con IC95 %).")
        r_sin, r_a, r_b, r_ah, r_dat = st.tabs(["Síntesis y comparación", "Bloque A · un paciente", "Bloque B · un brote",
                                                f"Ahorros e ICER frente a «{base}»", "Datos de las réplicas (simulados)"])
        cap = ss.capacidad
        with r_sin:
            st.markdown("#### Síntesis")
            for n, r in res.items():
                st.markdown(f"**{n}.** " + sx.texto_escenario(n, r)[len(n) + 2:])
            recs = sx.recomendaciones(res, base, comR, cap["camas"] if cap else None, cap["camas_opt"] if cap else None)
            st.success("**Recomendaciones para el próximo brote (O7.1)**\n\n" + "\n".join(f"{i + 1}. {r_}" for i, r_ in enumerate(recs)))
            st.markdown("**Comparación de escenarios** (filas «Teórico» = fórmulas; filas «Simulado» = Monte Carlo)")
            mostrar_tabla(sx.tabla_comparativa(res), sx.fmt_kpi, "KPI", "large")
            c1, c2 = st.columns([2, 3])
            with c1:
                st.markdown("**Objetivos** (sobre los valores teóricos)")
                mostrar_tabla(sx.tabla_objetivos(res), None, "Objetivo", "large")
            with c2:
                ver_fig(st, gr.fig_validacion(res))
            ver_fig(st, gr.fig_comparativa(res))
            st.markdown("**Cuadro de mando propuesto**")
            mostrar_tabla(pd.DataFrame({"Para qué sirve": dict(sx.KPIS_CUADRO)}), None, "KPI", "large")

        with r_a:
            n = st.selectbox("Escenario", list(res), key="det_A")
            r = res[n]; t = r["teoria"]
            st.markdown("**O1.1 – O1.2 · coste esperado, probabilidad de complicación y percentil 95** (macro-réplicas)")
            mostrar_tabla(informe._sub(r["resA"], informe.FILAS_A), sx.fmt_res, "KPI", "large")
            st.caption("El percentil 95 es el colchón de provisión de fondos: con E(Y) se cubre el gasto medio, pero el 5 % de "
                       "los pacientes (casi todos Persistentes complicados) cuesta más que el P95.")
            st.markdown("**O1.3 · por fenotipo**: ¿qué fenotipo tiene mayor incertidumbre relativa y cuál concentra las complicaciones?")
            mostrar_tabla(informe.tabla_fenotipos(t, r["resA"]), informe.fmt_fen, "Fenotipo", "small", iconos=True)
            c1, c2 = st.columns(2)
            ver_fig(c1, gr.fig_hist_Y(t["A"], r["mcA"]["Y"]))
            ver_fig(c2, gr.fig_por_fen(t["A"], r["resA"]))
            st.markdown("**O2.1 · descomposición de la varianza**: Cov(término, Y)/Var(Y); las contribuciones suman el 100 %.")
            c1, c2 = st.columns([2, 3])
            with c1:
                mostrar_tabla(informe._sub(r["resA"], [f"Contribución a Var(Y) · {x}" for x in sx.TERMINOS]), sx.fmt_res,
                              "Término", "medium")
            with c2:
                ver_fig(st, gr.fig_descomposicion(t["A"], r["resA"]))
            ver_fig(st, gr.fig_traza_A(r["mcA"]["traza"], r["esc"]))
            with st.expander(f"Primeros {len(r['mcA']['traza'])} pacientes simulados"):
                tr = r["mcA"]["traza"].copy()
                for c_ in tr.columns[1:]:
                    if c_ not in ("Fenotipo", "Complicación"):
                        tr[c_] = tr[c_].map(lambda v: num(v, 3 if c_ in ("U_T", "R (log/día)") else 2))
                st.dataframe(tr.astype(str), use_container_width=True, hide_index=True)

        with r_b:
            n = st.selectbox("Escenario", list(res), key="det_B")
            r = res[n]; t = r["teoria"]
            st.markdown("**O4.1 – O4.3 · coste del brote con y sin restricción, sobrecoste, desbordamientos y derivados**")
            mostrar_tabla(informe._sub(r["resB"], informe.FILAS_B), sx.fmt_res, "KPI", "large")
            st.markdown("**O5.3 · sobrecoste por fase del brote**")
            mostrar_tabla(informe.tabla_fases(t, r["resB"]), informe.fmt_fases, "Fase", "small", iconos=True)
            c1, c2 = st.columns(2)
            ver_fig(c1, gr.fig_hist_brote(t["B"], r["mcB"]["replicas"]["Coste total con capacidad"]))
            ver_fig(c2, gr.fig_fases(t["B"], r["resB"]))
            ver_fig(st, gr.fig_dias(r["mcB"]["dias"], t["B"]["camas"], t["B"]["extra"]))
            st.markdown("**O4.1, O6.2 y O8.1 · saturación, diseño pareado y reserva de camas**")
            mostrar_tabla(informe.tabla_extras_B(t, r["mcB"], comR, r["esc"]), None, "Indicador", "large")

        with r_ah:
            dif = sx.diferencias(res, base)
            if dif is None:
                st.info("Simula al menos dos escenarios, incluido el escenario base, para ver los ahorros.")
            else:
                st.caption("Ahorro = coste del escenario base − coste del escenario (positivo = ahorra). Las diferencias "
                           "simuladas son pareadas: cada réplica usa las mismas semillas en todos los escenarios.")
                st.markdown("**O3.1 – O3.3 · coste por paciente y efectividad**")
                mostrar_tabla(informe.tabla_dif(dif, informe.COLS_DIF_A), sx.fmt_col, "Escenario", "large", iconos=True)
                st.markdown("**O3.2 · ICER = Δcoste / Δefecto** (incluye la implantación repartida entre los pacientes del brote)")
                mostrar_tabla(informe.tabla_dif(dif, informe.COLS_DIF_ICER), sx.fmt_col, "Escenario", "large")
                st.markdown("**O5.1, O5.2, O6.3 y O8.6 · ahorro por brote** (con reserva de camas e implantación)")
                mostrar_tabla(informe.tabla_dif(dif, informe.COLS_DIF_B), sx.fmt_col, "Escenario", "large", iconos=True)
                ver_fig(st, gr.fig_ahorros(dif))

        with r_dat:
            st.markdown("Resultados **simulados** de cada réplica.")
            la = pd.concat([r["mcA"]["replicas"].assign(Escenario=n) for n, r in res.items()]).reset_index()
            lb = pd.concat([r["mcB"]["replicas"].assign(Escenario=n) for n, r in res.items()]).reset_index()
            st.markdown("**Bloque A** (una fila por réplica de pacientes)")
            st.dataframe(la, use_container_width=True, hide_index=True)
            st.markdown("**Bloque B** (una fila por brote simulado)")
            st.dataframe(lb, use_container_width=True, hide_index=True)
            buf = io.BytesIO()
            try:
                with pd.ExcelWriter(buf, engine="openpyxl") as xw:
                    la.to_excel(xw, sheet_name="Replicas A", index=False)
                    lb.to_excel(xw, sheet_name="Brotes B", index=False)
                    sx.tabla_comparativa(res).astype(str).to_excel(xw, sheet_name="Comparacion")
                    for i, (n, r) in enumerate(res.items(), 1):
                        r["resA"].astype(str).to_excel(xw, sheet_name=f"KPIs A {i}")
                        r["resB"].astype(str).to_excel(xw, sheet_name=f"KPIs B {i}")
                        d1 = r["mcB"]["dias"]
                        d1[d1["Réplica"] <= 5].to_excel(xw, sheet_name=f"Dias {i}", index=False)
                    dif = sx.diferencias(res, base)
                    if dif is not None:
                        dif.astype(str).to_excel(xw, sheet_name="Ahorros e ICER")
                st.download_button("Descargar Excel", buf.getvalue(), "resultados_staphx.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception:  # openpyxl no disponible
                st.download_button("Descargar CSV (A)", la.to_csv(index=False).encode("utf-8"), "replicas_A.csv", "text/csv")

# ==============================================================================================
# 4) Capacidad UCI (O6.1, O6.2, O6.3, O8.1, O8.2, O8.5)
# ==============================================================================================
with tab_cap:
    st.subheader("Dimensionamiento y robustez de la capacidad UCI")
    c = st.columns([2, 1])
    nombres = list(ss.escenarios)
    ec = c[0].selectbox("Escenario", nombres, index=nombres.index(ss.base) if ss.base in nombres else 0, key=clave("c_esc"))
    c[1].markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
    if c[1].button("▶️ Calcular", type="primary", use_container_width=True, key="b_cap"):
        with st.spinner("Calculando…"):
            ss.capacidad = calcular_capacidad(com, ss.escenarios[ec], ec, aj)
            ss.informe_bytes = None
    C = ss.capacidad
    if C is None:
        st.info("Ejecuta el análisis (barra lateral) o pulsa «Calcular».")
    else:
        df, b = C["camas"], C["camas_opt"]
        st.caption(f"Escenario «{C['escenario']}» · coste de oportunidad {eur(com['coste_cama_dia'])} por cama y día.")
        m = st.columns(4)
        m[0].metric("📐 Camas óptimas", f"{b}")
        m[1].metric("📐 Sobrecoste + reserva con el óptimo", eur(df.loc[b, "Sobrecoste + reserva (€)"]))
        m[2].metric("📐 P(UCI llena 2 días seguidos)", pct(df.loc[b, "P(UCI llena dos días seguidos)"], 1), f"con {b} camas",
                    delta_color="off")
        m[3].metric("📐 Reserva de una cama", eur(df["Coste de reservarla (€)"].iloc[0]), "por brote", delta_color="off")
        st.markdown("**O6.1 · sobrecoste esperado frente al número de camas reservadas** (O6.2: riesgo de saturación sostenida)")
        mostrar_tabla(df, informe.fmt_tabla, "Camas", "small")
        ver_fig(st, gr.fig_camas(df, b))
        st.markdown("**O6.3 · 5 camas fijas frente a 3 + 4 de alquiler**: compáralas en «Resultados → Ahorros e ICER» con "
                    "los escenarios «5 camas (O6.3a)» y «3 + 4 de alquiler (O6.3b)». Supuestos: coste de oportunidad por "
                    "cama reservada y día, coste por uso de la cama de alquiler y sin cuota fija de disponibilidad.")
        st.markdown("**O8.5 · cuota diaria máxima por cama alquilada solo durante el Pico**")
        if len(C["cuota"]):
            mostrar_tabla(C["cuota"], informe.fmt_tabla, "Camas extra en el Pico", "small")
            ver_fig(st, gr.fig_cuota(C["cuota"]))
        dp, fac = C["pareada"]
        st.markdown(f"**O4.1 y O8.1 · diseño pareado frente a independiente** · la varianza del sobrecoste se multiplica por "
                    f"**{num(fac, 0)}** si se emparejan brotes distintos")
        mostrar_tabla(dp, informe.fmt_tabla, "Diseño", "medium", iconos=True)
        do, ocu = C["ocupacion"]
        st.markdown(f"**O8.2 · «un día, una decisión» frente a ocupación de ⌈D − umbral⌉ días** (ocupación media de las camas "
                    f"{pct(ocu, 0)}; mismos pacientes)")
        mostrar_tabla(do, informe.fmt_tabla, "Modelo", "large")
        st.caption("Si los complicados ocupan la cama varios días, las camas se liberan más tarde y el modelo diario del "
                   "caso subestima los derivados y el sobrecoste; un modelo de colas M/M/c/K recoge esa dependencia entre días.")

# ==============================================================================================
# 5) Sensibilidad
# ==============================================================================================
with tab_sens:
    st.subheader("Sensibilidad")
    s_par, s_um, s_est = st.tabs(["O2.2 · parámetros y Sobol", "O3.4 · umbral de complicación", "Estimador Monte Carlo"])
    with s_par:
        st.caption("Se varía cada parámetro manteniendo los demás fijos (Var(Y) teórica) y se calculan los índices de Sobol "
                   "de las entradas aleatorias (T, L, R, ε) por simulación.")
        c = st.columns([2, 1, 1])
        nombres = list(ss.escenarios)
        es_ = c[0].selectbox("Escenario", nombres, index=nombres.index(ss.base) if ss.base in nombres else 0, key=clave("se_esc"))
        rango = float(c[1].number_input("Variación máxima (± %)", 1.0, 90.0, 20.0, 5.0, key=clave("se_r"))) / 100
        c[2].markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
        if c[2].button("▶️ Calcular", use_container_width=True, key="b_sens"):
            with st.spinner("Calculando…"):
                ss.sensib = calcular_sensib(com, ss.escenarios[es_], es_, aj, rango)
                ss.informe_bytes = None
        S = ss.sensib
        if S is None:
            st.info("Ejecuta el análisis o pulsa «Calcular».")
        else:
            df, elas = S["param"]
            mayor = max(elas, key=lambda k: abs(elas[k]))
            st.success(f"Escenario «{S['escenario']}»: el parámetro con más impacto en Var(Y) es **{mayor}** (elasticidad "
                       f"{num(elas[mayor], 2)}: un 1 % más cambia Var(Y) un {num(elas[mayor], 2)} %).")
            ver_fig(st, gr.fig_sens_param(df, elas))
            mostrar_tabla(informe.tabla_sens_param(df), informe.fmt_sens_param, "Parámetro · variación", "medium", iconos=True)
            c1, c2 = st.columns([2, 3])
            with c1:
                mostrar_tabla(S["sobol"], lambda f, c_, v: pct(v, 1), "Entrada", "small")
            with c2:
                ver_fig(st, gr.fig_sobol(S["sobol"]))
    with s_um:
        S = ss.sensib
        if S is None:
            st.info("Ejecuta el análisis o pulsa «Calcular» en la pestaña anterior.")
        else:
            st.caption(f"Escenario «{S['escenario']}» · mismos pacientes (números aleatorios comunes) para cada umbral.")
            mostrar_tabla(informe.tabla_umbral(S["umbral"]), lambda f, c_, v: sx.fmt_val(f, v), "KPI", "large")
            ver_fig(st, gr.fig_umbral(S["umbral"]))
            st.caption("Bajar el umbral a 5 días adelanta la atención intensiva (más complicaciones y presión sobre las camas); "
                       "subirlo a 10 días reduce el coste pero puede retrasar la atención de pacientes graves.")
    with s_est:
        st.caption("¿Cuánto «tiembla» la estimación según el tamaño de la simulación? Para cada valor se repite con varias semillas.")
        sc = ss.sens_cfg
        c = st.columns(3)
        met = ["paciente", "brote"]
        sc["metrica"] = c[0].selectbox("Métrica", met, index=met.index(sc.get("metrica", "paciente")), key=clave("s_met"),
                                       format_func=lambda k: "E(Y) por paciente · por nº de pacientes" if k == "paciente"
                                       else "Coste medio del brote · por nº de brotes")
        if sc["metrica"] == "paciente":
            sc["n_min"] = int(c[1].number_input("Pacientes mínimo", 10, 10**7, int(sc["n_min"]), 50, key=clave("s_nmin")))
            sc["n_max"] = int(c[2].number_input("Pacientes máximo", 20, 10**7, int(sc["n_max"]), 1000, key=clave("s_nmax")))
        else:
            sc["b_min"] = int(c[1].number_input("Brotes mínimo", 2, 10**5, int(sc["b_min"]), 5, key=clave("s_bmin")))
            sc["b_max"] = int(c[2].number_input("Brotes máximo", 3, 10**5, int(sc["b_max"]), 10, key=clave("s_bmax")))
        c = st.columns(3)
        sc["puntos"] = int(c[0].slider("Puntos a evaluar", 3, 10, int(sc["puntos"]), key=clave("s_pts")))
        sc["replicas"] = int(c[1].slider("Réplicas por punto", 2, 20, int(sc["replicas"]), key=clave("s_rep")))
        pol = c[2].selectbox("Escenario", list(ss.escenarios), key=clave("s_pol"))
        if st.button("▶️ Analizar sensibilidad", type="primary"):
            vals = (sx.valores_barrido(sc["n_min"], sc["n_max"], sc["puntos"]) if sc["metrica"] == "paciente" else
                    sx.valores_barrido(sc["b_min"], sc["b_max"], sc["puntos"]))
            barra = st.progress(0.0, text="Simulando…")
            df = sx.sensibilidad_estimador(com, ss.escenarios[pol], sc["metrica"], vals, sc["replicas"], aj["semilla"],
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
# 6) Ampliación optativa (O8.3, O8.4, O8.6)
# ==============================================================================================
with tab_amp:
    st.subheader("Ampliación optativa")
    acf = ss.ampl_cfg
    c = st.columns([2, 1, 1, 1])
    nombres = list(ss.escenarios)
    ea = c[0].selectbox("Escenario", nombres, index=nombres.index(ss.base) if ss.base in nombres else 0, key=clave("a_esc"))
    acf["coste_10"] = float(c[1].number_input("Coste del diagnóstico para −10 % (€/brote)", 0.0, 1e8, float(acf["coste_10"]),
                                              1000.0, key=clave("a_c10"), help="Coste del programa = c·(10·r)²: "
                                                                               "rendimientos decrecientes."))
    acf["presupuesto"] = float(c[2].number_input("Presupuesto reserva + programa (€)", 0.0, 1e9, float(acf["presupuesto"]),
                                                 10000.0, key=clave("a_pres")))
    c[3].markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
    if c[3].button("▶️ Calcular", type="primary", use_container_width=True, key="b_amp"):
        with st.spinner("Calculando…"):
            ss.ampliacion = calcular_ampliacion(com, ss.escenarios[ea], ea, aj, acf)
            ss.informe_bytes = None
    AM = ss.ampliacion
    if AM is None:
        st.info("Ejecuta el análisis (barra lateral) o pulsa «Calcular».")
    else:
        df, opt = AM["optim"]
        st.markdown(f"**O8.3 · camas UCI y diagnóstico precoz con presupuesto limitado** · escenario «{AM['escenario']}»")
        st.caption("Minimizar el coste total esperado del brote (pacientes + sobrecoste + reserva + programa) sujeto a "
                   f"reserva + programa ≤ {eur(AM['cfg']['presupuesto'])}; la función objetivo sale de la teoría del modelo "
                   "(validada por simulación).")
        if opt is not None:
            m = st.columns(4)
            m[0].metric("📐 Camas", int(opt["Camas"]))
            m[1].metric("📐 Reducción de L del Persistente", pct(opt["Reducción de L₃"], 0))
            m[2].metric("📐 Coste total esperado", eur(opt["Coste total esperado (€)"]))
            m[3].metric("📐 Gasto fijo", eur(opt["Gasto fijo (€)"]), f"de {eur(AM['cfg']['presupuesto'])}", delta_color="off")
        ver_fig(st, gr.fig_optim(df, opt))
        st.markdown("**O8.6 · coste-beneficio del diagnóstico precoz**")
        mostrar_tabla(informe.indice_pct(AM["diagnostico"]), informe.fmt_tabla, "Reducción de L₃", "small")
        st.markdown("**O8.4 · proceso de Cox: intensidad diaria aleatoria λ·G, G ~ Gamma(media 1, CV)** (simulado, mismos uniformes)")
        mostrar_tabla(informe.indice_cv(AM["cox"]), informe.fmt_tabla, "CV", "small")
        ver_fig(st, gr.fig_cox(AM["cox"]))

# ==============================================================================================
# 7) Informe
# ==============================================================================================
with tab_inf:
    R = ss.resultados
    if R is None:
        st.info("Ejecuta primero el análisis para poder generar el informe.")
    else:
        st.markdown("El informe Word se organiza por los objetivos del encargo: planteamiento, síntesis y tres "
                    "recomendaciones (O7.1), comparación de escenarios, ahorros e ICER (O3, O5.1, O5.2, O6.3, O8.6) y, para "
                    "cada escenario, el coste de un paciente (O1, O2.1) y el coste del brote (O4, O5.3, O6.2); además, la "
                    "capacidad UCI (O6.1, O8.1, O8.2, O8.5), la sensibilidad (O2.2, O3.4) y la ampliación (O8.3, O8.4, O8.6).")
        titulo = st.text_input("Título del informe", "Farmacoeconomía de la terapia para Staph-X", key="tit_inf")
        c = st.columns(4)
        con_cap = c[0].checkbox("Capacidad UCI", value=ss.capacidad is not None, disabled=ss.capacidad is None)
        con_sen = c[1].checkbox("Sensibilidad (O2.2, O3.4)", value=ss.sensib is not None, disabled=ss.sensib is None)
        con_amp = c[2].checkbox("Ampliación (O8)", value=ss.ampliacion is not None, disabled=ss.ampliacion is None)
        con_est = c[3].checkbox("Sensibilidad del estimador", value=ss.sens is not None, disabled=ss.sens is None)
        if st.button("📄 Generar informe (.docx)", type="primary"):
            with st.spinner("Generando informe…"):
                ss.informe_bytes = informe.generar_informe(
                    R["res"], R["comunes"], R["ajustes"], R["base"], ss.capacidad if con_cap else None,
                    ss.sensib if con_sen else None, ss.ampliacion if con_amp else None, ss.sens if con_est else None, titulo)
        if ss.informe_bytes:
            st.download_button("⬇️ Descargar informe", ss.informe_bytes, f"Informe_StaphX_{datetime.date.today():%Y%m%d}.docx",
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
