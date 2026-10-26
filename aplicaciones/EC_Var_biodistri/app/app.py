# -*- coding: utf-8 -*-
"""EC3 · Rentabilidad de proyectos de calidad en «Bio-Distri S.A.» · aplicación Streamlit.
Beneficio de un proyecto por el método de composición (Bloque A), programa plurianual con aprendizaje y degradación,
con y sin mantenimiento (Bloque B), escenarios con objetivos y diferencias pareadas, análisis del mantenimiento,
sensibilidad, ampliación optativa e informe Word. Todas las cantidades, en miles de € (k€).

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

import biodistri as bd
import graficos as gr
import informe
from biodistri import num, pct, keur

st.set_page_config(page_title="Simulador · Bio-Distri", page_icon="📦", layout="wide")
EN_NAVEGADOR = sys.platform == "emscripten"

ESCENARIOS_DEF = {
    "Actual": bd.escenario(descripcion="Cartera y penalización del enunciado (multa de 20 k€ si R < 10 %); programa de "
                                       "6 años con λ = 0,80, δ = 0,85, C_mant = 3 k€/año y r = 5 %, con mantenimiento."),
    "Penalización continua (O3.1)": bd.escenario(K=0.0, c_pen=200.0, descripcion="La multa deja de ser un escalón: "
                                                 "200·(0,10 − R) k€ si R < 0,10 (máximo 20 k€ cuando R = 0)."),
    "Solo IoT e IA (O3.2)": bd.escenario(p=[0.35 / 0.60, 0.25 / 0.60, 0.0, 0.0], descripcion="Toda la inversión se "
                                         "concentra en los dos proyectos de mayor beneficio esperado, IoT e IA, "
                                         "manteniendo su proporción relativa actual."),
    "Garantía de resultados (O3.4)": bd.escenario(garantia=0.20, descripcion="Si R < 0,10 el proveedor devuelve el 20 % "
                                                  "de la inversión I."),
    "Sin mantenimiento": bd.escenario(politica="sin", descripcion="Mismo programa que «Actual», pero la política elegida "
                                      "es no reinvertir en mantenimiento."),
    "Degradación severa δ = 0,70 (O5.1)": bd.escenario(delta=0.70, descripcion="El ahorro se degrada un 30 % al año si no "
                                                       "se mantiene."),
    "Mantenimiento a 1,5 k€ (O5.2)": bd.escenario(C_mant=1.5, descripcion="Coste de mantenimiento a la mitad."),
    "Horizonte de 10 años (O6.3)": bd.escenario(H=10, descripcion="Programa de 10 años en lugar de 6."),
    "Experiencia previa t₀ = 3 (O8.2)": bd.escenario(t0=3, descripcion="El centro empieza el programa en el punto t = 3 "
                                                     "de la curva de aprendizaje."),
}
COMUNES_DEF = copy.deepcopy(bd.COMUNES_DEF)
AJUSTES_DEF = {"nsim": 10000, "replicas": 5, "centros": 300, "replicas_B": 10 if EN_NAVEGADOR else 20, "semilla": 1000}
SENS_DEF = {"metrica": "proyecto", "n_min": 200, "n_max": 50000, "c_min": 20, "c_max": 3000, "puntos": 6, "replicas": 5}
MANT_DEF = {"delta": [0.60, 0.70, 0.80, 0.85, 0.90, 0.95], "C_mant": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0],
            "H": [3, 6, 8, 10, 12, 15], "subvencion": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "racha_tipo": 3, "racha_inicio": 2, "racha_anyos": 3}
AMPL_DEF = {"umbrales": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0], "lam_min": 0.60, "lam_max": 0.95}

# ==============================================================================================
# Estado de la sesión
# ==============================================================================================
ss = st.session_state
if "escenarios" not in ss:
    ss.escenarios = copy.deepcopy(ESCENARIOS_DEF)
    ss.comunes = copy.deepcopy(COMUNES_DEF)
    ss.ajustes = copy.deepcopy(AJUSTES_DEF)
    ss.sens_cfg = copy.deepcopy(SENS_DEF)
    ss.mant_cfg = copy.deepcopy(MANT_DEF)
    ss.ampl_cfg = copy.deepcopy(AMPL_DEF)
    ss.base = "Actual"
    ss.resultados = None
    ss.mant = None
    ss.sensib = None
    ss.ampliacion = None
    ss.sens = None
    ss.informe_bytes = None
    ss.version = 0
    ss.esc_activo = next(iter(ss.escenarios))
    ss.u = [0.50, 0.50, 0.50, 0.50]
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
                  "sensibilidad": ss.sens_cfg, "mantenimiento": ss.mant_cfg, "ampliacion": ss.ampl_cfg})


def _claves_int(d):
    """Restaura las claves numéricas (tipos de proyecto) de los diccionarios cargados desde JSON."""
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
    com["dist"] = {f: {**COMUNES_DEF["dist"][f], **com["dist"].get(f, {})} for f in bd.PROYECTOS}
    ss.comunes = com
    ss.ajustes = {**copy.deepcopy(AJUSTES_DEF), **{k: v for k, v in d.get("ajustes", {}).items() if k in AJUSTES_DEF}}
    ss.sens_cfg = {**copy.deepcopy(SENS_DEF), **d.get("sensibilidad", {})}
    ss.mant_cfg = {**copy.deepcopy(MANT_DEF), **d.get("mantenimiento", {})}
    ss.ampl_cfg = {**copy.deepcopy(AMPL_DEF), **d.get("ampliacion", {})}
    ss.escenarios = {n: {**copy.deepcopy(bd.ESC_BASE), **e} for n, e in (d.get("escenarios") or copy.deepcopy(ESCENARIOS_DEF)).items()}
    ss.base = d.get("base") if d.get("base") in ss.escenarios else next(iter(ss.escenarios))
    ss.esc_activo = next(iter(ss.escenarios))
    ss.resultados, ss.mant, ss.sensib, ss.ampliacion, ss.sens, ss.informe_bytes = None, None, None, None, None, None
    ss.cache_teo = {}
    for k_ in [k for k in ss.keys() if str(k).startswith("base_")]:
        del ss[k_]
    ss.version += 1


def teoria_cache(com, esc):
    k = json.dumps(_json([com, esc]), sort_keys=True)
    if k not in ss.cache_teo:
        if len(ss.cache_teo) > 40:
            ss.cache_teo = {}
        ss.cache_teo[k] = bd.teoria(com, esc)
    return ss.cache_teo[k]


def calcular_mant(com, esc, nombre, aj, cfg):
    rep = min(int(aj["replicas_B"]), 10)
    barr = {par: bd.barrido(com, esc, par, cfg[par], aj["centros"], rep, aj["semilla"])
            for par in ("delta", "C_mant", "H", "subvencion")}
    tB = bd.teoria_B(com, esc, con_rejilla=False)
    racha = bd.experimento_racha(com, esc, int(cfg["racha_tipo"]), int(cfg["racha_inicio"]), int(cfg["racha_anyos"]),
                                 n=20000, semilla=aj["semilla"])[0]
    return {"escenario": nombre, "barridos": barr, "C_ind": tB["C_ind"], "X_ind": tB["X_ind"],
            "C_ind_tipo": {f: tB["por_tipo"][f]["C_ind"] for f in bd.PROYECTOS},
            "racha": (racha, {"tipo": int(cfg["racha_tipo"]), "inicio": int(cfg["racha_inicio"]),
                              "anyos": int(cfg["racha_anyos"])})}


def calcular_sensib(com, esc, nombre, aj, rango=0.2):
    return {"escenario": nombre, "V": float(esc["V"]),
            "param": bd.sensibilidad_parametros(com, esc, list(np.linspace(-rango, rango, 5))),
            "sobol": bd.indices_sobol(com, esc, n=max(aj["nsim"], 20000), semilla=aj["semilla"]),
            "veq": bd.v_equilibrio(com, esc, 3, 0.0)}


def calcular_ampliacion(com, esc, nombre, aj, cfg, mcB=None, tB=None):
    if mcB is None:
        mcB = bd.simular_B(com, esc, aj["centros"], aj["replicas_B"], aj["semilla"])
        tB = bd.teoria_B(com, esc, con_rejilla=False)
    rep = min(int(aj["replicas_B"]), 10)
    return {"escenario": nombre, "contraste": bd.contraste_ganancia(mcB, tB),
            "experiencia": bd.comparar_experiencia(com, esc, (1, 3), aj["centros"], rep, aj["semilla"]),
            "adaptativa": bd.politica_adaptativa(com, esc, cfg["umbrales"], aj["centros"], rep, aj["semilla"]),
            "bayes": bd.actualizacion_bayesiana(com, esc, cfg["lam_min"], cfg["lam_max"], semilla=aj["semilla"]),
            "sustitucion": bd.sustitucion(com, esc, aj["centros"], rep, aj["semilla"])}


def lista_numeros(texto, defecto, entero=False):
    try:
        v = [float(x.replace(",", ".")) for x in texto.replace(";", " ").split()]
        v = [int(round(x)) for x in v] if entero else v
        return sorted(set(v)) or defecto
    except Exception:
        return defecto


def texto_lista(v, porcentaje=False):
    return "  ".join(num(100 * x if porcentaje else x, 2).rstrip("0").rstrip(",") for x in v)


com, aj = ss.comunes, ss.ajustes

# ==============================================================================================
# Barra lateral
# ==============================================================================================
with st.sidebar:
    st.header("⚙️ Simulación")
    st.markdown("**Bloque A · un proyecto**")
    aj["nsim"] = int(st.number_input("Proyectos por réplica", 100, 5_000_000, int(aj["nsim"]), 1000, key=clave("nsim")))
    aj["replicas"] = int(st.number_input("Réplicas A (semillas)", 2, 100, int(aj["replicas"]), key=clave("rep")))
    st.markdown("**Bloque B · programa plurianual**")
    aj["centros"] = int(st.number_input("Centros por réplica", 20, 1_000_000, int(aj["centros"]), 50, key=clave("centros")))
    aj["replicas_B"] = int(st.number_input("Réplicas B (semillas)", 2, 200, int(aj["replicas_B"]), key=clave("repB"),
                                           help="Réplica i → semilla + 1000 + i. Las dos políticas usan la misma base."))
    aj["semilla"] = int(st.number_input("Semilla de inicialización", 0, 10**9, int(aj["semilla"]), key=clave("sem"),
                                        help="Réplica A i → semilla + i. Todos los escenarios usan las mismas semillas "
                                             "(números aleatorios comunes), por lo que sus diferencias son pareadas."))
    nombres = list(ss.escenarios)
    ss.base = st.selectbox("Escenario base (para diferencias)", nombres,
                           index=nombres.index(ss.base) if ss.base in nombres else 0, key=clave("base"))
    sel = st.multiselect("Escenarios a simular", nombres, default=nombres, key=clave("sel"))
    st.caption(f"Por escenario: {num(aj['nsim'] * aj['replicas'], 0)} proyectos (A) y "
               f"{num(aj['centros'] * aj['replicas_B'], 0)} centros (B).")
    if EN_NAVEGADOR:
        st.caption("En el navegador la simulación es más lenta: con muchos escenarios, proyectos o centros puede tardar.")
    if st.button("▶️ Ejecutar análisis", type="primary", use_container_width=True, disabled=not sel):
        barra = st.progress(0.0, text="Simulando…")
        res = {}
        pasos = len(sel) * (aj["replicas"] + aj["replicas_B"]) + 3
        cont = [0]
        def prog(texto=""):
            cont[0] += 1
            barra.progress(min(1.0, cont[0] / pasos), text=texto or "Simulando…")
        for n in sel:
            res[n] = bd.analizar_escenario(ss.escenarios[n], com, aj, lambda n=n: prog(f"{n}"))
        b = ss.base if ss.base in res else sel[0]
        prog("Mantenimiento (O5, O6, O8.5)")
        ss.mant = calcular_mant(com, ss.escenarios[b], b, aj, ss.mant_cfg)
        prog("Sensibilidad (O2.2, O3.3)")
        ss.sensib = calcular_sensib(com, ss.escenarios[b], b, aj)
        prog("Ampliación (O8)")
        ss.ampliacion = calcular_ampliacion(com, ss.escenarios[b], b, aj, ss.ampl_cfg, res[b]["mcB"], res[b]["teoria"]["B"])
        ss.resultados = {"res": res, "comunes": copy.deepcopy(com), "ajustes": copy.deepcopy(aj), "base": b,
                         "fecha": f"{datetime.datetime.now():%d/%m/%Y %H:%M}"}
        ss.informe_bytes = None
        barra.progress(1.0, text="Análisis completado")

    st.divider()
    st.subheader("💾 Configuración")
    st.download_button("Descargar configuración (JSON)", json.dumps(configuracion(), ensure_ascii=False, indent=2),
                       "configuracion_biodistri.json", "application/json", use_container_width=True)
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
st.title("📦 Rentabilidad de los proyectos de calidad en «Bio-Distri»")
st.caption("Beneficio de un proyecto de mejora por el método de composición (tipo → inversión, reducción de merma y "
           "ahorro) y programa plurianual con curva de aprendizaje y degradación, con y sin mantenimiento; escenarios con "
           "objetivos y diferencias pareadas, análisis del mantenimiento, sensibilidad, ampliación e informe Word. "
           "Cantidades en miles de € (k€). Estudio de caso EC3 de Javier Morales, equipo IA4LEGOS (CC BY-SA 4.0).")
tab_com, tab_esc, tab_res, tab_man, tab_sens, tab_amp, tab_inf = st.tabs(
    ["🌐 Parámetros comunes", "🏕️ Escenarios", "📊 Resultados", "🛠️ Mantenimiento", "🔬 Sensibilidad",
     "🧪 Ampliación (O8)", "📄 Informe"])

# ==============================================================================================
# 1) Parámetros comunes
# ==============================================================================================
with tab_com:
    st.subheader("Parámetros comunes a todos los escenarios")
    st.markdown("#### 🧪 Distribuciones condicionadas al tipo de proyecto")
    st.caption("I ~ LN(μ, σ) (μ y σ del logaritmo, en k€) · R ~ Beta(α, β) · S ~ Gamma(k, θ) (k€).")
    k_df = clave("dist"); k_base = "base_" + k_df
    COLS = {"mu_I": "μ de ln I", "sigma_I": "σ de ln I", "a_R": "α de R", "b_R": "β de R", "k_S": "k de S", "theta_S": "θ de S"}
    if k_base not in ss:
        ss[k_base] = pd.DataFrame({et: [float(com["dist"][f][c]) for f in bd.PROYECTOS] for c, et in COLS.items()},
                                  index=[bd.NOMBRE_LARGO[f] for f in bd.PROYECTOS])
    ed = st.data_editor(ss[k_base], key=k_df, use_container_width=True,
                        column_config={et: st.column_config.NumberColumn(et, min_value=0.001 if c != "mu_I" else -50.0,
                                                                         step=0.05, format="%.3f")
                                       for c, et in COLS.items()})
    for i, f in enumerate(bd.PROYECTOS):
        for c, et in COLS.items():
            v = ed.iloc[i][et]
            if v == v:
                com["dist"][f][c] = float(v)
    tabla_m = pd.DataFrame({bd.NOMBRE[f]: {"E(I) (k€)": np.exp(com["dist"][f]["mu_I"] + com["dist"][f]["sigma_I"] ** 2 / 2),
                                           "E(R)": com["dist"][f]["a_R"] / (com["dist"][f]["a_R"] + com["dist"][f]["b_R"]),
                                           "E(S) (k€)": com["dist"][f]["k_S"] * com["dist"][f]["theta_S"]}
                            for f in bd.PROYECTOS}).T
    st.caption("📐 Medias teóricas de cada variable por tipo")
    mostrar_tabla(tabla_m, lambda f, c, v: pct(v, 1) if c == "E(R)" else keur(v, 2), "Tipo", "small")

    st.markdown("#### 🏆 Objetivos")
    st.caption("Se evalúan sobre los valores teóricos; los del programa, con la política elegida en cada escenario. "
               "Desmarca un objetivo para no evaluarlo.")
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
        objetivo("EY_MIN", "E(Y) mínimo por proyecto (k€)", 225.0, 1, 5.0, "Beneficio esperado mínimo de un proyecto.")
        objetivo("PPERD_MAX", "P(Y < 0) máxima (%)", 0.021, 100, 0.1, "Probabilidad máxima de que un proyecto pierda dinero.")
    with c2:
        objetivo("VAN_MIN", "E(VAN) mínimo por centro (k€)", 1665.0, 1, 5.0, "VAN esperado del programa con la política elegida.")
        objetivo("CVAR_MIN", "CVaR 5 % mínimo del VAN (k€)", 1175.0, 1, 5.0,
                 "VAN medio del 5 % de centros con peor resultado, con la política elegida.")

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
    st.markdown("**Cartera de proyectos y función de beneficio**")
    c = st.columns(4)
    p = [float(x) * 100 for x in e["p"]]
    for i, f in enumerate(bd.PROYECTOS):
        p[i] = c[i].number_input(f"p {bd.NOMBRE[f]} (%)", 0.0, 100.0, p[i], 1.0, key=clave("p", f, activo))
    e["p"] = [x / 100 for x in p]
    if abs(sum(p) - 100) > 1e-9:
        st.warning(f"Las probabilidades suman {num(sum(p), 1)} %: se normalizan para sumar 100 %.")
    c = st.columns(5)
    e["V"] = float(c[0].number_input("V_stock (k€)", 0.0, 1e7, float(e["V"]), 10.0, key=clave("V", activo),
                                     help="Valor del stock en riesgo (O3.3)."))
    e["K"] = float(c[1].number_input("Multa fija K (k€)", 0.0, 1e6, float(e["K"]), 1.0, key=clave("K", activo)))
    e["c_pen"] = float(c[2].number_input("Multa proporcional c (k€ por unidad)", 0.0, 1e7, float(e["c_pen"]), 10.0,
                                         key=clave("c", activo), help="O3.1: c·(umbral − R) si R < umbral."))
    e["umbral_R"] = float(c[3].number_input("Umbral de R (%)", 0.0, 100.0, 100 * float(e["umbral_R"]), 1.0,
                                            key=clave("u", activo))) / 100
    e["garantia"] = float(c[4].number_input("Garantía (% de I si R < umbral)", 0.0, 100.0, 100 * float(e["garantia"]), 5.0,
                                            key=clave("g", activo), help="O3.4")) / 100
    st.markdown("**Programa plurianual**")
    c = st.columns(8)
    e["H"] = int(c[0].number_input("Horizonte H (años)", 1, 50, int(e["H"]), 1, key=clave("H", activo)))
    e["lam"] = float(c[1].number_input("Aprendizaje λ", 0.01, 1.0, float(e["lam"]), 0.01, key=clave("lam", activo)))
    e["delta"] = float(c[2].number_input("Degradación δ", 0.01, 1.0, float(e["delta"]), 0.01, key=clave("dl", activo)))
    e["C_mant"] = float(c[3].number_input("C_mant (k€/año)", 0.0, 1e5, float(e["C_mant"]), 0.5, key=clave("cm", activo)))
    e["r"] = float(c[4].number_input("Descuento r (%)", 0.0, 100.0, 100 * float(e["r"]), 0.5, key=clave("r", activo))) / 100
    e["subvencion"] = float(c[5].number_input("Subvención X (%)", 0.0, 100.0, 100 * float(e.get("subvencion") or 0), 5.0,
                                              key=clave("x", activo), help="O8.5: cofinanciación del mantenimiento.")) / 100
    e["t0"] = int(c[6].number_input("Partida t₀ en la curva", 1, 50, int(e.get("t0") or 1), 1, key=clave("t0", activo),
                                    help="O8.2: 1 = sin experiencia; 3 = empieza como en el tercer año."))
    pols = list(bd.POLITICAS)
    e["politica"] = c[7].selectbox("Política elegida", pols, index=pols.index(e.get("politica", "con")),
                                   format_func=lambda k: bd.POLITICAS[k], key=clave("pol", activo))
    e["descripcion"] = st.text_area("Descripción", e.get("descripcion", ""), key=clave("desc", activo), height=70)

    t = teoria_cache(com, e)
    A, B = t["A"], t["B"]
    st.markdown("##### 📐 Vista previa teórica (fórmulas exactas y convolución, sin simular)")
    m = st.columns(6)
    m[0].metric("E(Y) por proyecto", keur(A["EY"]))
    m[1].metric("P(Y < 0)", pct(A["Pperd"], 2))
    m[2].metric("VaR 95 % / CVaR 5 %", f"{num(A['VaR'], 1)} / {num(A['CVaR'], 1)}", help="k€")
    m[3].metric("E(VAN) con mantenimiento", keur(B["E_con"]), f"sin: {keur(B['E_sin'])}", delta_color="off")
    m[4].metric("Ganancia E(Δ)", keur(B["E_gan"], 2), f"P(Δ > 0) = {pct(B['P_gan'], 1)}", delta_color="off")
    m[5].metric("C_mant de indiferencia", keur(B["C_ind"], 2), "por año", delta_color="off")

    with st.expander("🎯 Mecanismo · un proyecto por composición y su programa", expanded=True):
        st.caption("1 · U_T elige el tipo por transformada inversa.  2 · Condicionadas al tipo, I = exp(μ + σ·Φ⁻¹(U_I)), "
                   "R = F⁻¹_Beta(U_R) y S = θ·F⁻¹_Gamma(U_S).  3 · Y = V·R + S − I − Pen(R).  En el programa se repiten "
                   "los mismos U cada año para ver el aprendizaje de R y la degradación de S.")
        c = st.columns(5)
        if c[4].button("🎲 U aleatorios", use_container_width=True):
            ss.u = [float(np.round(np.random.uniform(0.01, 0.99), 2)) for _ in range(4)]
            for k_ in ("uT_s", "uI_s", "uR_s", "uS_s"):
                ss.pop(k_, None)
        ss.u = [c[i].slider(et, 0.01, 0.99, min(max(float(ss.u[i]), 0.01), 0.99), 0.01, key=k_)
                for i, (et, k_) in enumerate((("U_T · tipo", "uT_s"), ("U_I · inversión", "uI_s"), ("U_R · reducción", "uR_s"),
                                              ("U_S · ahorro", "uS_s")))]
        pr = bd.un_proyecto(com, e, *ss.u)
        m = st.columns(6)
        m[0].metric("Tipo", bd.NOMBRE[pr["T"]])
        m[1].metric("Inversión I", keur(pr["I"], 2))
        m[2].metric("Reducción R", pct(pr["R"], 1), "penalización" if pr["R"] < e["umbral_R"] else "sin penalización",
                    delta_color="inverse" if pr["R"] < e["umbral_R"] else "off")
        m[3].metric("Ahorro S", keur(pr["S"], 2))
        m[4].metric("Penalización", keur(-pr["terminos"][3], 2))
        m[5].metric("Beneficio Y", keur(pr["Y"], 2))
        ver_fig(st, gr.fig_mecanismo(A["p"], ss.u[0], pr))
        ver_fig(st, gr.fig_trayectoria(bd.trayectoria(com, e, *ss.u), e))

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
        st.caption(f"Análisis del {R['fecha']} · A: {ajR['replicas']} réplicas de {num(ajR['nsim'], 0)} proyectos · B: "
                   f"{ajR['replicas_B']} réplicas de {num(ajR['centros'], 0)} centros · semilla {ajR['semilla']} · base «{base}»")
        st.info("📐 **Teórico**: momentos exactos y distribución de Y y del VAN por convolución en rejilla de las "
                "distribuciones exactas de cada término (P(Y < 0), VaR y CVaR). 🎲 **Simulado**: composición por "
                "transformada inversa con macro-réplicas (media de las réplicas con IC95 %).")
        r_sin, r_a, r_b, r_dif, r_dat = st.tabs(["Síntesis y comparación", "Bloque A · un proyecto", "Bloque B · programa",
                                                 f"Diferencias frente a «{base}»", "Datos de las réplicas (simulados)"])
        with r_sin:
            st.markdown("#### Síntesis")
            for n, r in res.items():
                st.markdown(f"**{n}.** " + bd.texto_escenario(n, r)[len(n) + 2:])
            recs = bd.recomendaciones(res, base, comR)
            st.success("**Recomendación (O7.1)**\n\n" + "\n".join(f"{i + 1}. {r_}" for i, r_ in enumerate(recs)))
            st.markdown("**Comparación de escenarios** (filas «Teórico» = fórmulas; filas «Simulado» = Monte Carlo)")
            mostrar_tabla(bd.tabla_comparativa(res), bd.fmt_kpi, "KPI", "large")
            c1, c2 = st.columns([2, 3])
            with c1:
                st.markdown("**Objetivos** (sobre los valores teóricos)")
                mostrar_tabla(bd.tabla_objetivos(res), None, "Objetivo", "large")
            with c2:
                ver_fig(st, gr.fig_validacion(res))
            ver_fig(st, gr.fig_comparativa(res))
            st.markdown("**Cuadro de mando propuesto**")
            mostrar_tabla(pd.DataFrame({"Para qué sirve": dict(bd.KPIS_CUADRO)}), None, "KPI", "large")

        with r_a:
            n = st.selectbox("Escenario", list(res), key="det_A")
            r = res[n]; t = r["teoria"]
            st.markdown("**O1.1 – O1.2 · beneficio esperado, probabilidad de pérdida, VaR 95 % y CVaR 5 %** (macro-réplicas)")
            mostrar_tabla(informe._sub(r["resA"], informe.FILAS_A), bd.fmt_res, "KPI", "large")
            st.caption("El VaR dice cuánto gana como mínimo el 95 % de los proyectos; el CVaR, cuánto se gana (o pierde) en "
                       "media en el 5 % peor: mide la gravedad de la cola, no solo dónde empieza.")
            st.markdown("**O1.3 · por tipo de proyecto**")
            mostrar_tabla(informe.tabla_tipos(t, r["resA"]), informe.fmt_tipos, "Tipo", "small", iconos=True)
            c1, c2 = st.columns(2)
            ver_fig(c1, gr.fig_hist_Y(t["A"], r["mcA"]["Y"]))
            ver_fig(c2, gr.fig_por_tipo(t["A"], r["resA"]))
            st.markdown("**O2.1 · descomposición de la varianza**: Cov(término, Y)/Var(Y); las contribuciones suman el 100 %.")
            c1, c2 = st.columns([2, 3])
            with c1:
                mostrar_tabla(informe._sub(r["resA"], [f"Contribución a Var(Y) · {x}" for x in bd.TERMINOS]), bd.fmt_res,
                              "Término", "medium")
            with c2:
                ver_fig(st, gr.fig_descomposicion(t["A"], r["resA"]))
            ver_fig(st, gr.fig_traza_A(r["mcA"]["traza"]))
            with st.expander(f"Primeros {len(r['mcA']['traza'])} proyectos simulados"):
                tr = r["mcA"]["traza"].copy()
                for c_ in tr.columns[1:]:
                    if c_ != "Tipo":
                        tr[c_] = tr[c_].map(lambda v: num(v, 3 if c_ in ("U_T", "R") else 2))
                st.dataframe(tr.astype(str), use_container_width=True, hide_index=True)

        with r_b:
            n = st.selectbox("Escenario", list(res), key="det_B")
            r = res[n]; t = r["teoria"]
            st.markdown("**O4.1 – O4.3 · VAN con y sin mantenimiento, ganancia pareada, VaR y CVaR**")
            mostrar_tabla(informe._sub(r["resB"], informe.FILAS_B), bd.fmt_res, "KPI", "large")
            st.markdown("**O5.3 · ganancia del mantenimiento por tipo de proyecto**")
            mostrar_tabla(informe.tabla_tipos_B(t, r["mcB"]), informe.fmt_tipos_B, "Tipo", "small", iconos=True)
            c1, c2 = st.columns(2)
            ver_fig(c1, gr.fig_hist_van(t["B"], r["mcB"]))
            ver_fig(c2, gr.fig_ganancia(t["B"], r["mcB"]))
            ver_fig(st, gr.fig_ejemplo_B(r["mcB"]["ejemplo"], r["esc"]))
            st.markdown("**O4.1, O6.1 y O8.5 · descuento, coste de indiferencia y diseño pareado**")
            mostrar_tabla(informe.tabla_extras_B(t, r["mcB"], r["esc"]), None, "Indicador", "large")

        with r_dif:
            dif = bd.diferencias(res, base)
            if dif is None:
                st.info("Simula al menos dos escenarios, incluido el escenario base, para ver las diferencias.")
            else:
                st.caption("Diferencia = escenario − base (positivo = mejora). Las diferencias simuladas son pareadas: cada "
                           "réplica usa las mismas semillas en todos los escenarios.")
                st.markdown("**O3.1 – O3.4 · beneficio y riesgo de un proyecto**")
                mostrar_tabla(informe.tabla_dif(dif, informe.COLS_DIF_A), bd.fmt_col, "Escenario", "large", iconos=True)
                st.markdown("**Programa plurianual · E(VAN) con la política elegida y ganancia del mantenimiento**")
                mostrar_tabla(informe.tabla_dif(dif, informe.COLS_DIF_B), bd.fmt_col, "Escenario", "large", iconos=True)
                ver_fig(st, gr.fig_diferencias(dif))

        with r_dat:
            st.markdown("Resultados **simulados** de cada réplica.")
            la = pd.concat([r["mcA"]["replicas"].assign(Escenario=n) for n, r in res.items()]).reset_index()
            lb = pd.concat([r["mcB"]["replicas"].assign(Escenario=n) for n, r in res.items()]).reset_index()
            st.markdown("**Bloque A** (una fila por réplica de proyectos)")
            st.dataframe(la, use_container_width=True, hide_index=True)
            st.markdown("**Bloque B** (una fila por réplica de centros)")
            st.dataframe(lb, use_container_width=True, hide_index=True)
            buf = io.BytesIO()
            try:
                with pd.ExcelWriter(buf, engine="openpyxl") as xw:
                    la.to_excel(xw, sheet_name="Replicas A", index=False)
                    lb.to_excel(xw, sheet_name="Replicas B", index=False)
                    bd.tabla_comparativa(res).astype(str).to_excel(xw, sheet_name="Comparacion")
                    for i, (n, r) in enumerate(res.items(), 1):
                        r["resA"].astype(str).to_excel(xw, sheet_name=f"KPIs A {i}")
                        r["resB"].astype(str).to_excel(xw, sheet_name=f"KPIs B {i}")
                        pd.DataFrame({"Tipo": r["mcB"]["T"], "VAN con": r["mcB"]["VAN_con"], "VAN sin": r["mcB"]["VAN_sin"]}) \
                            .to_excel(xw, sheet_name=f"Centros {i}", index=False)
                    dif = bd.diferencias(res, base)
                    if dif is not None:
                        dif.astype(str).to_excel(xw, sheet_name="Diferencias")
                st.download_button("Descargar Excel", buf.getvalue(), "resultados_biodistri.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception:  # openpyxl no disponible
                st.download_button("Descargar CSV (A)", la.to_csv(index=False).encode("utf-8"), "replicas_A.csv", "text/csv")

# ==============================================================================================
# 4) Mantenimiento (O5.1, O5.2, O6.1, O6.2, O6.3, O8.5)
# ==============================================================================================
with tab_man:
    st.subheader("¿De qué depende el valor del mantenimiento?")
    mc = ss.mant_cfg
    c = st.columns([2, 1])
    nombres = list(ss.escenarios)
    em = c[0].selectbox("Escenario", nombres, index=nombres.index(ss.base) if ss.base in nombres else 0, key=clave("m_esc"))
    c = st.columns(4)
    mc["delta"] = lista_numeros(c[0].text_input("Valores de δ", texto_lista(mc["delta"]), key=clave("m_d")), mc["delta"])
    mc["C_mant"] = lista_numeros(c[1].text_input("Valores de C_mant (k€/año)", texto_lista(mc["C_mant"]), key=clave("m_c")), mc["C_mant"])
    mc["H"] = lista_numeros(c[2].text_input("Valores de H (años)", texto_lista(mc["H"]), key=clave("m_h")), mc["H"], entero=True)
    mc["subvencion"] = [x / 100 for x in lista_numeros(c[3].text_input("Subvenciones X (%)", texto_lista(mc["subvencion"], True),
                                                                       key=clave("m_x")), [100 * v for v in mc["subvencion"]])]
    c = st.columns([1, 1, 1, 1])
    tipos = list(bd.PROYECTOS)
    mc["racha_tipo"] = c[0].selectbox("Racha O6.2 · tipo", tipos, index=tipos.index(int(mc["racha_tipo"])),
                                      format_func=lambda f: bd.NOMBRE[f], key=clave("m_rt"))
    mc["racha_inicio"] = int(c[1].number_input("Año de inicio de la racha", 1, 50, int(mc["racha_inicio"]), key=clave("m_ri")))
    mc["racha_anyos"] = int(c[2].number_input("Años de racha", 1, 50, int(mc["racha_anyos"]), key=clave("m_ra")))
    c[3].markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
    if c[3].button("▶️ Calcular", type="primary", use_container_width=True, key="b_man"):
        with st.spinner("Calculando…"):
            ss.mant = calcular_mant(com, ss.escenarios[em], em, aj, mc)
            ss.informe_bytes = None
    M = ss.mant
    if M is None:
        st.info("Ejecuta el análisis (barra lateral) o pulsa «Calcular».")
    else:
        st.caption(f"Escenario «{M['escenario']}». Δ = Σ d_t·S⁰_t·(1 − δ^(t−1)) − C_mant·Σ d_t: el mantenimiento solo "
                   "protege el ahorro S, así que su valor no depende de R ni de I.")
        m = st.columns(4)
        m[0].metric("📐 C_mant de indiferencia (O6.1)", keur(M["C_ind"], 3), "por año", delta_color="off")
        m[1].metric("📐 Subvención de indiferencia (O8.5)", pct(M["X_ind"], 1) if M["X_ind"] > 0 else "no hace falta")
        best = max(M["C_ind_tipo"], key=M["C_ind_tipo"].get)
        m[2].metric("📐 Indiferencia del mejor tipo", keur(M["C_ind_tipo"][best], 2), bd.NOMBRE[best], delta_color="off")
        m[3].metric("📐 Indiferencia por tipo", " / ".join(num(v, 2) for v in M["C_ind_tipo"].values()), "IoT / IA / Emp. / Aud.",
                    delta_color="off")
        titulos = {"delta": "O5.1 · tasa de degradación δ", "C_mant": "O5.2 y O6.1 · coste de mantenimiento",
                   "H": "O6.3 · horizonte del programa", "subvencion": "O8.5 · subvención del mantenimiento"}
        for par, df in M["barridos"].items():
            st.markdown(f"**{titulos[par]}**")
            ver_fig(st, gr.fig_barrido(df))
            mostrar_tabla(informe.indice_fmt(df, informe.FMT_IDX[par]), informe.fmt_tabla, bd.PARAM_BARRIDO[par], "small",
                          iconos=True)
        if M.get("racha") is not None:
            df, cfg = M["racha"]
            st.markdown(f"**O6.2 · racha de {cfg['anyos']} años con R⁰ en el 10 % inferior desde el año {cfg['inicio']} · "
                        f"{bd.NOMBRE[cfg['tipo']]}** (🎲 20.000 centros, mismos números aleatorios con y sin racha)")
            ver_fig(st, gr.fig_racha(df))
            mostrar_tabla(informe.tabla_racha(df), informe.fmt_racha, "Año", "small")
            st.caption("La racha reduce R y el VAN acumulado (las penalizaciones solo pueden darse el primer año, porque "
                       "el aprendizaje garantiza R_t ≥ 1 − λ^(t−1)), pero no toca el ahorro S: la ganancia del mantenimiento "
                       "es la misma con o sin racha. Tras la racha, el aprendizaje devuelve R a su senda.")

# ==============================================================================================
# 5) Sensibilidad
# ==============================================================================================
with tab_sens:
    st.subheader("Sensibilidad")
    s_par, s_v, s_est = st.tabs(["O2.2 · parámetros y Sobol", "O3.3 · V_stock del Empaque", "Estimador Monte Carlo"])
    with s_par:
        st.caption("Se varía cada parámetro manteniendo los demás fijos (Var(Y) teórica) y se calculan los índices de Sobol "
                   "de las entradas aleatorias (T, I, R, S) por simulación.")
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
    with s_v:
        S = ss.sensib
        if S is None:
            st.info("Ejecuta el análisis o pulsa «Calcular» en la pestaña anterior.")
        else:
            dv, ey3, er3 = S["veq"]
            st.markdown(f"E(Y | Empaque) = {keur(ey3)} es lineal en V_stock con pendiente E(R | Empaque) = {num(er3, 4)}. "
                        f"Con V_stock = {keur(S['V'], 0)} el Empaque ya es rentable en media; su umbral de rentabilidad es "
                        f"**V_stock = {keur(dv.iloc[0]['V_stock necesario (k€)'])}**.")
            mostrar_tabla(dv, informe.fmt_tabla, "Meta", "large")
            ver_fig(st, gr.fig_vequilibrio(dv, S["V"]))
    with s_est:
        st.caption("¿Cuánto «tiembla» la estimación según el tamaño de la simulación? Para cada valor se repite con varias semillas.")
        sc = ss.sens_cfg
        c = st.columns(3)
        met = ["proyecto", "centro"]
        sc["metrica"] = c[0].selectbox("Métrica", met, index=met.index(sc.get("metrica", "proyecto")), key=clave("s_met"),
                                       format_func=lambda k: "E(Y) por proyecto · por nº de proyectos" if k == "proyecto"
                                       else "E(VAN) con mantenimiento · por nº de centros")
        if sc["metrica"] == "proyecto":
            sc["n_min"] = int(c[1].number_input("Proyectos mínimo", 10, 10**7, int(sc["n_min"]), 50, key=clave("s_nmin")))
            sc["n_max"] = int(c[2].number_input("Proyectos máximo", 20, 10**7, int(sc["n_max"]), 1000, key=clave("s_nmax")))
        else:
            sc["c_min"] = int(c[1].number_input("Centros mínimo", 2, 10**6, int(sc["c_min"]), 10, key=clave("s_cmin")))
            sc["c_max"] = int(c[2].number_input("Centros máximo", 3, 10**6, int(sc["c_max"]), 100, key=clave("s_cmax")))
        c = st.columns(3)
        sc["puntos"] = int(c[0].slider("Puntos a evaluar", 3, 10, int(sc["puntos"]), key=clave("s_pts")))
        sc["replicas"] = int(c[1].slider("Réplicas por punto", 2, 20, int(sc["replicas"]), key=clave("s_rep")))
        pol = c[2].selectbox("Escenario", list(ss.escenarios), key=clave("s_pol"))
        if st.button("▶️ Analizar sensibilidad", type="primary"):
            vals = (bd.valores_barrido(sc["n_min"], sc["n_max"], sc["puntos"]) if sc["metrica"] == "proyecto" else
                    bd.valores_barrido(sc["c_min"], sc["c_max"], sc["puntos"]))
            barra = st.progress(0.0, text="Simulando…")
            df = bd.sensibilidad_estimador(com, ss.escenarios[pol], sc["metrica"], vals, sc["replicas"], aj["semilla"],
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
# 6) Ampliación optativa (O8)
# ==============================================================================================
with tab_amp:
    st.subheader("Ampliación optativa")
    acf = ss.ampl_cfg
    c = st.columns([2, 2, 1, 1, 1])
    nombres = list(ss.escenarios)
    ea = c[0].selectbox("Escenario", nombres, index=nombres.index(ss.base) if ss.base in nombres else 0, key=clave("a_esc"))
    acf["umbrales"] = lista_numeros(c[1].text_input("Umbrales de ahorro O8.3 (k€)", texto_lista(acf["umbrales"]),
                                                    key=clave("a_u")), acf["umbrales"])
    acf["lam_min"] = float(c[2].number_input("λ mín. (O8.4)", 0.01, 0.99, float(acf["lam_min"]), 0.05, key=clave("a_l0")))
    acf["lam_max"] = float(c[3].number_input("λ máx. (O8.4)", 0.02, 1.0, max(float(acf["lam_max"]), acf["lam_min"] + 0.01), 0.05,
                                             key=clave("a_l1")))
    c[4].markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
    if c[4].button("▶️ Calcular", type="primary", use_container_width=True, key="b_amp"):
        with st.spinner("Calculando…"):
            ss.ampliacion = calcular_ampliacion(com, ss.escenarios[ea], ea, aj, acf)
            ss.informe_bytes = None
    AM = ss.ampliacion
    if AM is None:
        st.info("Ejecuta el análisis (barra lateral) o pulsa «Calcular».")
    else:
        st.caption(f"Escenario «{AM['escenario']}».")
        st.markdown("**O8.1 · contraste H₀: E(Δ) = 0** (t de Student sobre las diferencias pareadas de todos los centros)")
        mostrar_tabla(AM["contraste"], informe.fmt_tabla, "Grupo", "small", iconos=True)
        st.markdown("**O8.2 · experiencia previa en la curva de aprendizaje**")
        mostrar_tabla(AM["experiencia"], informe.fmt_tabla, "Partida", "small", iconos=True)
        da, ec, es = AM["adaptativa"]
        st.markdown("**O8.3 · política adaptativa**: mantener el año t solo si el ahorro que se observaría sin mantener "
                    "(S⁰_t·δ^n, n = años desde el último mantenimiento) cae por debajo del umbral")
        ver_fig(st, gr.fig_adaptativa(da, ec, es))
        mostrar_tabla(informe.indice_fmt(da, lambda v: keur(v, 1)), informe.fmt_tabla, "Umbral", "small", iconos=True)
        db, dv = AM["bayes"]
        st.markdown("**O8.4 · actualización bayesiana de λ tras los dos primeros años** (λ de cada centro ~ U(λ mín., λ máx.); "
                    "R₂ = 1 − (1 − R⁰₂)·λ aporta la información)")
        mostrar_tabla(db, informe.fmt_tabla, "Tipo", "small")
        mostrar_tabla(informe.indice_fmt(dv, lambda v: num(v, 3)), informe.fmt_tabla, "λ supuesta", "small")
        st.markdown("**O8.6 · mantener frente a sustituir el proyecto a mitad del horizonte** (🎲)")
        mostrar_tabla(AM["sustitucion"], informe.fmt_tabla, "Alternativa", "large", iconos=True)
        st.caption(informe.TEXTO_O86)

# ==============================================================================================
# 7) Informe
# ==============================================================================================
with tab_inf:
    R = ss.resultados
    if R is None:
        st.info("Ejecuta primero el análisis para poder generar el informe.")
    else:
        st.markdown("El informe Word se organiza por los objetivos del encargo: planteamiento, síntesis y recomendación "
                    "(O7.1), comparación de escenarios (O3) y, para cada escenario, el beneficio de un proyecto (O1, O2.1) y "
                    "el programa con y sin mantenimiento (O4, O5.3); además, el análisis del mantenimiento (O5.1, O5.2, O6, "
                    "O8.5), la sensibilidad (O2.2, O3.3) y la ampliación (O8).")
        titulo = st.text_input("Título del informe", "Rentabilidad de los proyectos de calidad en Bio-Distri", key="tit_inf")
        c = st.columns(4)
        con_man = c[0].checkbox("Mantenimiento", value=ss.mant is not None, disabled=ss.mant is None)
        con_sen = c[1].checkbox("Sensibilidad (O2.2, O3.3)", value=ss.sensib is not None, disabled=ss.sensib is None)
        con_amp = c[2].checkbox("Ampliación (O8)", value=ss.ampliacion is not None, disabled=ss.ampliacion is None)
        con_est = c[3].checkbox("Sensibilidad del estimador", value=ss.sens is not None, disabled=ss.sens is None)
        if st.button("📄 Generar informe (.docx)", type="primary"):
            with st.spinner("Generando informe…"):
                ss.informe_bytes = informe.generar_informe(
                    R["res"], R["comunes"], R["ajustes"], R["base"], ss.mant if con_man else None,
                    ss.sensib if con_sen else None, ss.ampliacion if con_amp else None, ss.sens if con_est else None, titulo)
        if ss.informe_bytes:
            st.download_button("⬇️ Descargar informe", ss.informe_bytes, f"Informe_BioDistri_{datetime.date.today():%Y%m%d}.docx",
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
