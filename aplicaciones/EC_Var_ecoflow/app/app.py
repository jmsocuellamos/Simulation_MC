# -*- coding: utf-8 -*-
"""EC4 · Gestión de riesgos en la micro-red de «Eco-Flow Renewables» · aplicación Streamlit.
Beneficio de un día por el método de composición (Bloque A), batería de almacenamiento recorrida día a día durante un
año (Bloque B), escenarios con objetivos y diferencias pareadas, dimensionamiento de la batería, sensibilidad,
ampliación optativa e informe Word.

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

import ecoflow as ef
import graficos as gr
import informe
from ecoflow import num, pct, eur

st.set_page_config(page_title="Simulador · Eco-Flow", page_icon="⚡", layout="wide")
EN_NAVEGADOR = sys.platform == "emscripten"

ESCENARIOS_DEF = {
    "Actual": ef.escenario(descripcion="Operación del enunciado: P_exp = 40 €/MWh, P_pen = 80 €/MWh, C_fijo = 1.000 €/día, "
                                       "c = 2 €/MWh; batería de 20 MWh con η = 0,98 que empieza vacía."),
    "Spot volátil (O3.1)": ef.escenario(cv_pen=0.5, descripcion="El precio de penalización es una lognormal de media "
                                        "80 €/MWh y coeficiente de variación del 50 %."),
    "Solar +30 % (O3.2)": ef.escenario(solar_factor=1.3, descripcion="Se amplía un 30 % la capacidad solar instalada "
                                       "(G_s escalada); el coste de la ampliación no se incluye en Y."),
    "Contrato a plazo (O3.2)": ef.escenario(contrato=True, P_contrato=60.0, prima=150.0, descripcion="El déficit se compra "
                                            "a un precio pactado de 60 €/MWh a cambio de una prima fija de 150 €/día."),
    "Respuesta de demanda (O3.4)": ef.escenario(dr_red=0.10, dr_umbral=20.0, dr_desc=20.0, descripcion="Si el déficit "
                                                "supera 20 MWh, los clientes reducen su consumo un 10 % a cambio de un "
                                                "descuento de 20 € por MWh no consumido."),
    "Batería de 10 MWh (O5.1)": ef.escenario(cap=10.0, descripcion="Batería con la mitad de capacidad."),
    "Batería de 40 MWh (O5.1)": ef.escenario(cap=40.0, descripcion="Batería con el doble de capacidad."),
    "Batería con η = 0,90 (O5.2)": ef.escenario(eta=0.90, descripcion="Batería menos eficiente (se pierde un 10 % en el ciclo)."),
    "Servicios de red (O8.5)": ef.escenario(tarifa=1.0, descripcion="El regulador paga 1 € por MWh de capacidad y día por "
                                            "tener almacenamiento disponible."),
}
COMUNES_DEF = copy.deepcopy(ef.COMUNES_DEF)
AJUSTES_DEF = {"nsim": 10000, "replicas": 5, "anyos": 50 if EN_NAVEGADOR else 100, "semilla": 1000}
SENS_DEF = {"metrica": "dia", "n_min": 200, "n_max": 50000, "a_min": 2, "a_max": 200, "puntos": 6, "replicas": 5}
BAT_DEF = {"cap": [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0], "eta": [0.80, 0.90, 0.95, 0.98, 1.00],
           "racha_inicio": 100, "racha_dias": 20}
AMPL_DEF = {"cv_exp": 0.30, "umbrales": [60.0, 70.0, 80.0, 90.0, 100.0], "coste_solar": 9000.0}

# ==============================================================================================
# Estado de la sesión
# ==============================================================================================
ss = st.session_state
if "escenarios" not in ss:
    ss.escenarios = copy.deepcopy(ESCENARIOS_DEF)
    ss.comunes = copy.deepcopy(COMUNES_DEF)
    ss.ajustes = copy.deepcopy(AJUSTES_DEF)
    ss.sens_cfg = copy.deepcopy(SENS_DEF)
    ss.bat_cfg = copy.deepcopy(BAT_DEF)
    ss.ampl_cfg = copy.deepcopy(AMPL_DEF)
    ss.base = "Actual"
    ss.resultados = None
    ss.bat = None
    ss.sensib = None
    ss.ampliacion = None
    ss.sens = None
    ss.informe_bytes = None
    ss.version = 0
    ss.esc_activo = next(iter(ss.escenarios))
    ss.u = [0.50, 0.50, 0.50, 0.50, 0.50]
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
                  "sensibilidad": ss.sens_cfg, "bateria": ss.bat_cfg, "ampliacion": ss.ampl_cfg})


def _claves_int(d):
    """Restaura las claves numéricas (escenarios climáticos) de los diccionarios cargados desde JSON."""
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
    com["dist"] = {f: {**COMUNES_DEF["dist"][f], **com["dist"].get(f, {})} for f in ef.ESCENARIOS}
    ss.comunes = com
    ss.ajustes = {**copy.deepcopy(AJUSTES_DEF), **{k: v for k, v in d.get("ajustes", {}).items() if k in AJUSTES_DEF}}
    ss.sens_cfg = {**copy.deepcopy(SENS_DEF), **d.get("sensibilidad", {})}
    ss.bat_cfg = {**copy.deepcopy(BAT_DEF), **d.get("bateria", {})}
    ss.ampl_cfg = {**copy.deepcopy(AMPL_DEF), **d.get("ampliacion", {})}
    ss.escenarios = {n: {**copy.deepcopy(ef.ESC_BASE), **e} for n, e in (d.get("escenarios") or copy.deepcopy(ESCENARIOS_DEF)).items()}
    ss.base = d.get("base") if d.get("base") in ss.escenarios else next(iter(ss.escenarios))
    ss.esc_activo = next(iter(ss.escenarios))
    ss.resultados, ss.bat, ss.sensib, ss.ampliacion, ss.sens, ss.informe_bytes = None, None, None, None, None, None
    ss.cache_teo = {}
    for k_ in [k for k in ss.keys() if str(k).startswith("base_")]:
        del ss[k_]
    ss.version += 1


def teoria_cache(com, esc):
    k = json.dumps(_json([com, esc]), sort_keys=True)
    if k not in ss.cache_teo:
        if len(ss.cache_teo) > 30:
            ss.cache_teo = {}
        ss.cache_teo[k] = ef.teoria(com, esc)
    return ss.cache_teo[k]


def calcular_bat(com, esc, nombre, aj, cfg, tA=None):
    tA = tA or ef.teoria_A(com, esc)
    anyos = min(int(aj["anyos"]), 100)
    dc = ef.barrido(com, esc, "cap", cfg["cap"], anyos, aj["semilla"], tA)
    de = ef.barrido(com, esc, "eta", cfg["eta"], anyos, aj["semilla"], tA)
    dr, rs = ef.experimento_racha(com, esc, int(cfg["racha_inicio"]), int(cfg["racha_dias"]), anyos=max(anyos, 100),
                                  semilla=aj["semilla"], tA=tA)
    return {"escenario": nombre, "cap": dc, "cap_opt": ef.capacidad_optima(dc), "eta": de, "amort": float(esc["amort"]),
            "racha": (dr, rs, {"inicio": int(cfg["racha_inicio"]), "dias": int(cfg["racha_dias"])})}


def calcular_sensib(com, esc, nombre, aj, rango=0.2):
    return {"escenario": nombre,
            "param": ef.sensibilidad_parametros(com, esc, list(np.linspace(-rango, rango, 5))),
            "sobol": ef.indices_sobol(com, esc, n=max(aj["nsim"], 20000), semilla=aj["semilla"]),
            "solar": ef.solar_necesaria(com, esc, 0.0)}


def calcular_ampliacion(com, esc, nombre, aj, cfg, mcB=None, tB=None, tA=None):
    tA = tA or ef.teoria_A(com, esc)
    if mcB is None:
        mcB = ef.simular_B(com, esc, aj["anyos"], aj["semilla"])
        tB = ef.teoria_bateria(com, esc, tA)
    anyos = min(int(aj["anyos"]), 100)
    return {"escenario": nombre, "contraste": ef.contraste_ganancia(mcB, tB),
            "inicial": ef.estado_inicial(com, esc, (0.0, 0.5, 1.0), anyos, aj["semilla"], tA),
            "activa": ef.gestion_activa(com, esc, float(cfg["cv_exp"]), cfg["umbrales"], anyos, aj["semilla"]),
            "cv_exp": float(cfg["cv_exp"]), "coste_solar": float(cfg["coste_solar"]),
            "tarifa": {"sola": tB["tarifa_sola"], "equilibrio": tB["tarifa_equilibrio"]},
            "solar_vs": ef.bateria_vs_solar(com, esc, float(cfg["coste_solar"]), tA)}


def lista_numeros(texto, defecto):
    try:
        v = [float(x.replace(",", ".")) for x in texto.replace(";", " ").split()]
        return sorted(set(v)) or defecto
    except Exception:
        return defecto


def texto_lista(v):
    return "  ".join(num(x, 2).rstrip("0").rstrip(",") for x in v)


com, aj = ss.comunes, ss.ajustes

# ==============================================================================================
# Barra lateral
# ==============================================================================================
with st.sidebar:
    st.header("⚙️ Simulación")
    st.markdown("**Bloque A · un día**")
    aj["nsim"] = int(st.number_input("Días por réplica", 100, 5_000_000, int(aj["nsim"]), 1000, key=clave("nsim")))
    aj["replicas"] = int(st.number_input("Réplicas A (semillas)", 2, 100, int(aj["replicas"]), key=clave("rep")))
    st.markdown("**Bloque B · batería (años de 365 días)**")
    aj["anyos"] = int(st.number_input("Años simulados", 2, 100_000, int(aj["anyos"]), 10, key=clave("anyos"),
                                      help="Cada año es una macro-réplica (año i → semilla + 1000 + i). Las estimaciones son "
                                           "medias por día; los años solo sirven para el IC95 %."))
    aj["semilla"] = int(st.number_input("Semilla de inicialización", 0, 10**9, int(aj["semilla"]), key=clave("sem"),
                                        help="Réplica A i → semilla + i. Todos los escenarios usan las mismas semillas "
                                             "(números aleatorios comunes), por lo que sus diferencias son pareadas."))
    nombres = list(ss.escenarios)
    ss.base = st.selectbox("Escenario base (para diferencias)", nombres,
                           index=nombres.index(ss.base) if ss.base in nombres else 0, key=clave("base"))
    sel = st.multiselect("Escenarios a simular", nombres, default=nombres, key=clave("sel"))
    st.caption(f"Por escenario: {num(aj['nsim'] * aj['replicas'], 0)} días independientes (A) y "
               f"{num(aj['anyos'] * ef.N_DIAS, 0)} días encadenados (B).")
    if EN_NAVEGADOR:
        st.caption("En el navegador la simulación es más lenta: con muchos escenarios, días o años puede tardar.")
    if st.button("▶️ Ejecutar análisis", type="primary", use_container_width=True, disabled=not sel):
        barra = st.progress(0.0, text="Simulando…")
        res = {}
        pasos = len(sel) * (aj["replicas"] + aj["anyos"]) + 3
        cont = [0]
        def prog(texto=""):
            cont[0] += 1
            barra.progress(min(1.0, cont[0] / pasos), text=texto or "Simulando…")
        for n in sel:
            res[n] = ef.analizar_escenario(ss.escenarios[n], com, aj, lambda n=n: prog(f"{n}"))
        b = ss.base if ss.base in res else sel[0]
        tA = res[b]["teoria"]["A"]
        prog("Batería (O5.1, O5.2, O6.1, O6.2)")
        ss.bat = calcular_bat(com, ss.escenarios[b], b, aj, ss.bat_cfg, tA)
        prog("Sensibilidad (O2.2, O3.3)")
        ss.sensib = calcular_sensib(com, ss.escenarios[b], b, aj)
        prog("Ampliación (O8)")
        ss.ampliacion = calcular_ampliacion(com, ss.escenarios[b], b, aj, ss.ampl_cfg, res[b]["mcB"], res[b]["teoria"]["B"], tA)
        ss.resultados = {"res": res, "comunes": copy.deepcopy(com), "ajustes": copy.deepcopy(aj), "base": b,
                         "fecha": f"{datetime.datetime.now():%d/%m/%Y %H:%M}"}
        ss.informe_bytes = None
        barra.progress(1.0, text="Análisis completado")

    st.divider()
    st.subheader("💾 Configuración")
    st.download_button("Descargar configuración (JSON)", json.dumps(configuracion(), ensure_ascii=False, indent=2),
                       "configuracion_ecoflow.json", "application/json", use_container_width=True)
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
st.title("⚡ Gestión de riesgos en la micro-red de «Eco-Flow Renewables»")
st.caption("Beneficio de un día por el método de composición (escenario climático → solar, eólica, hidráulica y demanda) "
           "y batería de almacenamiento recorrida día a día durante un año; escenarios con objetivos y diferencias "
           "pareadas, dimensionamiento de la batería, sensibilidad, ampliación e informe Word. Estudio de caso EC4 de "
           "Javier Morales, equipo IA4LEGOS (CC BY-SA 4.0).")
tab_com, tab_esc, tab_res, tab_bat, tab_sens, tab_amp, tab_inf = st.tabs(
    ["🌐 Parámetros comunes", "🏕️ Escenarios", "📊 Resultados", "🔋 Batería", "🔬 Sensibilidad", "🧪 Ampliación (O8)",
     "📄 Informe"])

# ==============================================================================================
# 1) Parámetros comunes
# ==============================================================================================
with tab_com:
    st.subheader("Parámetros comunes a todos los escenarios")
    st.markdown("#### 🌦️ Distribuciones condicionadas al escenario climático")
    st.caption("Solar ~ Tri(mín., moda, máx.) · Eólica ~ Weibull(k, λ) · Hidráulica ~ LN(μ, σ) (del logaritmo) · "
               "Demanda ~ N(μ, σ). Todo en MWh.")
    k_df = clave("dist"); k_base = "base_" + k_df
    COLS = [("tri", 0, "Solar · mín."), ("tri", 1, "Solar · moda"), ("tri", 2, "Solar · máx."), ("weib", 0, "Eólica · k"),
            ("weib", 1, "Eólica · λ"), ("ln", 0, "Hidro · μ"), ("ln", 1, "Hidro · σ"), ("norm", 0, "Demanda · μ"),
            ("norm", 1, "Demanda · σ")]
    if k_base not in ss:
        ss[k_base] = pd.DataFrame({et: [float(com["dist"][f][c][i]) for f in ef.ESCENARIOS] for c, i, et in COLS},
                                  index=[ef.NOMBRE[f] for f in ef.ESCENARIOS])
    ed = st.data_editor(ss[k_base], key=k_df, use_container_width=True,
                        column_config={et: st.column_config.NumberColumn(et, step=0.1, format="%.2f") for _, _, et in COLS})
    for j, f in enumerate(ef.ESCENARIOS):
        for c, i, et in COLS:
            v = ed.iloc[j][et]
            if v == v:
                com["dist"][f][c][i] = float(v)
        a, m, b = com["dist"][f]["tri"]
        if not (a <= m <= b and b > a):
            st.warning(f"{ef.NOMBRE[f]}: la triangular necesita mín. ≤ moda ≤ máx. y mín. < máx.")
    st.caption("📐 Medias y desviaciones típicas teóricas por escenario (con la solar del escenario «Actual»)")
    mostrar_tabla(informe.tabla_fuentes(com, ss.escenarios.get("Actual", ef.ESC_BASE)), None, "Escenario", "small")

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
        objetivo("EY_MIN", "E(Y) diario mínimo con batería (€)", -3000.0, 1, 50.0, "Beneficio diario esperado con batería.")
        objetivo("PDEF_MAX", "P(compra en el mercado) máxima con batería (%)", 0.65, 100, 1.0,
                 "Proporción máxima de días en los que hay que comprar en el mercado spot.")
    with c2:
        objetivo("CVAR_MIN", "CVaR 5 % diario mínimo sin batería (€)", -11000.0, 1, 100.0,
                 "Beneficio medio del 5 % de los peores días.")
        objetivo("NETO_MIN", "Ganancia neta anual mínima de la batería (€)", 0.0, 1, 1000.0,
                 "365·ganancia diaria + servicios de red − amortización.")

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
    st.markdown("**Clima y mercado**")
    c = st.columns(8)
    p = [float(x) * 100 for x in e["p"]]
    for i, f in enumerate(ef.ESCENARIOS):
        p[i] = c[i].number_input(f"p {ef.NOMBRE[f]} (%)", 0.0, 100.0, p[i], 1.0, key=clave("p", f, activo))
    e["p"] = [x / 100 for x in p]
    e["P_exp"] = float(c[4].number_input("P_exp (€/MWh)", 0.0, 1e4, float(e["P_exp"]), 1.0, key=clave("pe", activo)))
    e["P_pen"] = float(c[5].number_input("P_pen (€/MWh)", 0.0, 1e4, float(e["P_pen"]), 1.0, key=clave("pp", activo)))
    e["cv_pen"] = float(c[6].number_input("CV del spot (%)", 0.0, 500.0, 100 * float(e.get("cv_pen") or 0), 5.0,
                                          key=clave("cv", activo), help="O3.1: 0 = precio constante; > 0 = lognormal de media P_pen.")) / 100
    e["C_fijo"] = float(c[7].number_input("C_fijo (€/día)", 0.0, 1e7, float(e["C_fijo"]), 50.0, key=clave("cf", activo)))
    if abs(sum(p) - 100) > 1e-9:
        st.warning(f"Las probabilidades suman {num(sum(p), 1)} %: se normalizan para sumar 100 %.")
    st.markdown("**Palancas a corto plazo (O3)**")
    c = st.columns(8)
    e["c_var"] = float(c[0].number_input("c (€/MWh generado)", 0.0, 1e4, float(e["c_var"]), 0.5, key=clave("cv2", activo)))
    e["solar_factor"] = float(c[1].number_input("Solar · factor de capacidad", 0.0, 20.0, float(e["solar_factor"]), 0.05,
                                                key=clave("sf", activo), help="O3.2: 1,3 = +30 % de capacidad solar."))
    e["solar_extra"] = float(c[2].number_input("Solar · +MWh fijos al día", 0.0, 1e4, float(e["solar_extra"]), 1.0,
                                               key=clave("se", activo), help="O3.3: desplazamiento de G_s."))
    e["contrato"] = bool(c[3].checkbox("Contrato a plazo", value=bool(e.get("contrato")), key=clave("ct", activo), help="O3.2"))
    e["P_contrato"] = float(c[4].number_input("Precio pactado (€/MWh)", 0.0, 1e4, float(e["P_contrato"]), 1.0,
                                              key=clave("pc", activo), disabled=not e["contrato"]))
    e["prima"] = float(c[5].number_input("Prima del contrato (€/día)", 0.0, 1e7, float(e["prima"]), 10.0,
                                         key=clave("pr", activo), disabled=not e["contrato"]))
    e["dr_red"] = float(c[6].number_input("Respuesta de demanda (% de D)", 0.0, 100.0, 100 * float(e["dr_red"]), 1.0,
                                          key=clave("dr", activo), help="O3.4: 0 = sin programa.")) / 100
    e["dr_umbral"] = float(c[7].number_input("… si el déficit supera (MWh)", 0.0, 1e4, float(e["dr_umbral"]), 1.0,
                                             key=clave("du", activo), disabled=e["dr_red"] <= 0))
    st.markdown("**Batería (Bloque B)**")
    c = st.columns(8)
    e["dr_desc"] = float(c[0].number_input("Descuento (€/MWh no consumido)", 0.0, 1e4, float(e["dr_desc"]), 1.0,
                                           key=clave("dd", activo), disabled=e["dr_red"] <= 0))
    e["cap"] = float(c[1].number_input("Capacidad Cap (MWh)", 0.0, 1e4, float(e["cap"]), 1.0, key=clave("cap", activo)))
    e["eta"] = float(c[2].number_input("Eficiencia η", 0.01, 1.0, float(e["eta"]), 0.01, key=clave("eta", activo)))
    e["E1"] = float(c[3].number_input("Carga inicial E₁ (% de Cap)", 0.0, 100.0, 100 * float(e["E1"]), 10.0,
                                      key=clave("e1", activo), help="O8.2")) / 100
    e["amort"] = float(c[4].number_input("Amortización (€/MWh·año)", 0.0, 1e7, float(e["amort"]), 100.0, key=clave("am", activo),
                                         help="O6.1: coste anualizado de la batería por MWh de capacidad (supuesto)."))
    e["tarifa"] = float(c[5].number_input("Servicios de red (€/MWh·día)", 0.0, 1e4, float(e.get("tarifa") or 0), 0.5,
                                          key=clave("ta", activo), help="O8.5"))
    e["descripcion"] = st.text_area("Descripción", e.get("descripcion", ""), key=clave("desc", activo), height=70)

    t = teoria_cache(com, e)
    A, B = t["A"], t["B"]
    st.markdown("##### 📐 Vista previa teórica (fórmulas exactas, convolución y cadena de Markov, sin simular)")
    m = st.columns(6)
    m[0].metric("E(Y) diario", eur(A["EY"]), f"con batería: {eur(B['EY_con'])}", delta_color="off")
    m[1].metric("P(B < 0)", pct(A["Pdef"], 1), f"compra con batería: {pct(B['Pspot_con'], 1)}", delta_color="off")
    m[2].metric("VaR 95 % / CVaR 5 %", f"{num(A['VaR'], 0)} / {num(A['CVaR'], 0)}", help="€")
    m[3].metric("Ganancia de la batería", f"{eur(B['E_gan'], 1)}/día", f"{eur(B['gan_anual'])}/año", delta_color="off")
    m[4].metric("Ganancia neta anual", eur(B["neto_anual"]), f"amortización {eur(B['amortizacion'])}", delta_color="off")
    m[5].metric("LOLE", f"{num(B['LOLE_con'], 0)} días", f"sin batería {num(B['LOLE_sin'], 0)}", delta_color="off")

    with st.expander("🎯 Mecanismo · un día por composición", expanded=True):
        st.caption("1 · U_T elige el escenario por transformada inversa.  2 · Condicionadas al escenario: G_s con la inversa "
                   "de la triangular, G_w = λ·(−ln(1 − U))^(1/k), G_h = exp(μ + σ·Φ⁻¹(U)) y D = μ + σ·Φ⁻¹(U).  3 · B = G − D "
                   "y Y = P_exp·B⁺ − P_pen·B⁻ − C_fijo − c·G.")
        c = st.columns(6)
        if c[5].button("🎲 U aleatorios", use_container_width=True):
            ss.u = [float(np.round(np.random.uniform(0.01, 0.99), 2)) for _ in range(5)]
            for k_ in ("uT_s", "us_s", "uw_s", "uh_s", "ud_s"):
                ss.pop(k_, None)
        ss.u = [c[i].slider(et, 0.01, 0.99, min(max(float(ss.u[i]), 0.01), 0.99), 0.01, key=k_)
                for i, (et, k_) in enumerate((("U_T · clima", "uT_s"), ("U_s · solar", "us_s"), ("U_w · eólica", "uw_s"),
                                              ("U_h · hidro", "uh_s"), ("U_D · demanda", "ud_s")))]
        dia = ef.un_dia(com, e, *ss.u)
        m = st.columns(6)
        m[0].metric("Escenario", ef.NOMBRE[int(dia["T"])])
        m[1].metric("Solar / eólica / hidro", f"{num(dia['Gs'], 1)} / {num(dia['Gw'], 1)} / {num(dia['Gh'], 1)}", help="MWh")
        m[2].metric("Generación G", f"{num(dia['G'], 1)} MWh")
        m[3].metric("Demanda D", f"{num(dia['D'], 1)} MWh")
        m[4].metric("Balance B", f"{num(dia['Bp'], 1)} MWh", "superávit" if dia["Bp"] >= 0 else "déficit",
                    delta_color="normal" if dia["Bp"] >= 0 else "inverse")
        m[5].metric("Beneficio Y", eur(dia["Y"]))
        ver_fig(st, gr.fig_mecanismo(A["p"], ss.u[0], dia))

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
        st.caption(f"Análisis del {R['fecha']} · A: {ajR['replicas']} réplicas de {num(ajR['nsim'], 0)} días · B: "
                   f"{num(ajR['anyos'], 0)} años de 365 días · semilla {ajR['semilla']} · base «{base}»")
        st.info("📐 **Teórico**: distribución exacta de G por convolución, integración en D (E, Var, P(B < 0), VaR y CVaR) y "
                "batería como cadena de Markov en el estado de carga. 🎲 **Simulado**: composición por transformada inversa "
                "con macro-réplicas (media con IC95 %).")
        r_sin, r_a, r_b, r_dif, r_dat = st.tabs(["Síntesis y comparación", "Bloque A · un día", "Bloque B · batería",
                                                 f"Diferencias frente a «{base}»", "Datos de las réplicas (simulados)"])
        with r_sin:
            st.markdown("#### Síntesis")
            for n, r in res.items():
                st.markdown(f"**{n}.** " + ef.texto_escenario(n, r)[len(n) + 2:])
            recs = ef.recomendaciones(res, base, comR, ss.bat["cap"] if ss.bat else None)
            st.success("**Recomendación (O7.1)**\n\n" + "\n".join(f"{i + 1}. {r_}" for i, r_ in enumerate(recs)))
            st.markdown("**Comparación de escenarios** (filas «Teórico» = fórmulas; filas «Simulado» = Monte Carlo)")
            mostrar_tabla(ef.tabla_comparativa(res), ef.fmt_kpi, "KPI", "large")
            c1, c2 = st.columns([2, 3])
            with c1:
                st.markdown("**Objetivos** (sobre los valores teóricos)")
                mostrar_tabla(ef.tabla_objetivos(res), None, "Objetivo", "large")
            with c2:
                ver_fig(st, gr.fig_validacion(res))
            ver_fig(st, gr.fig_comparativa(res))
            st.markdown("**Cuadro de mando propuesto**")
            mostrar_tabla(pd.DataFrame({"Para qué sirve": dict(ef.KPIS_CUADRO)}), None, "KPI", "large")

        with r_a:
            n = st.selectbox("Escenario", list(res), key="det_A")
            r = res[n]; t = r["teoria"]
            st.markdown("**O1.1 – O1.2 · beneficio esperado, probabilidad de déficit, VaR 95 % y CVaR 5 %** (macro-réplicas)")
            mostrar_tabla(informe._sub(r["resA"], informe.FILAS_A), ef.fmt_res, "KPI", "large")
            st.caption("El VaR dice cuánto se gana (o pierde) como mínimo en el 95 % de los días; el CVaR, la media del 5 % peor: "
                       "mide la gravedad de la cola, no solo dónde empieza.")
            st.markdown("**O1.3 · por escenario climático**")
            mostrar_tabla(informe.tabla_tipos(t, r["resA"]), informe.fmt_tipos, "Escenario", "small", iconos=True)
            c1, c2 = st.columns(2)
            ver_fig(c1, gr.fig_hist_Y(t["A"], r["mcA"]["Y"]))
            ver_fig(c2, gr.fig_por_tipo(t["A"], r["resA"]))
            st.markdown("**O2.1 · descomposición de la varianza**: Cov(término, Y)/Var(Y); las contribuciones suman el 100 %.")
            c1, c2 = st.columns([2, 3])
            with c1:
                mostrar_tabla(informe._sub(r["resA"], [f"Contribución a Var(Y) · {x}" for x in ef.TERMINOS]), ef.fmt_res,
                              "Término", "medium")
            with c2:
                ver_fig(st, gr.fig_descomposicion(t["A"], r["resA"]))
            ver_fig(st, gr.fig_traza_A(r["mcA"]["traza"]))
            with st.expander(f"Primeros {len(r['mcA']['traza'])} días simulados"):
                tr = r["mcA"]["traza"].copy()
                for c_ in tr.columns[1:]:
                    if c_ != "Escenario":
                        tr[c_] = tr[c_].map(lambda v: num(v, 2))
                st.dataframe(tr.astype(str), use_container_width=True, hide_index=True)

        with r_b:
            n = st.selectbox("Escenario", list(res), key="det_B")
            r = res[n]; t = r["teoria"]
            st.markdown("**O4.1 – O4.3 y O6.3 · beneficio con y sin batería, ganancia pareada, compras en el mercado, CVaR y LOLE**")
            mostrar_tabla(informe._sub(r["resB"], informe.FILAS_B), ef.fmt_res, "KPI", "large")
            st.markdown("**O5.3 · ganancia de la batería por escenario climático**")
            mostrar_tabla(informe.tabla_tipos_B(t, r["mcB"]), informe.fmt_tipos_B, "Escenario", "small", iconos=True)
            ver_fig(st, gr.fig_soc(t["B"], r["mcB"], r["esc"]))
            c1, c2 = st.columns(2)
            ver_fig(c1, gr.fig_hist_bat(t["A"], r["mcB"]))
            ver_fig(c2, gr.fig_ganancia_tipo(t["B"], r["mcB"]))
            st.markdown("**O4.1, O6.1, O6.3 y O8.5 · rentabilidad anual, LOLE y diseño pareado**")
            mostrar_tabla(informe.tabla_extras_B(t, r["mcB"], r["esc"]), None, "Indicador", "large")
            with st.expander("Registro día a día del primer año simulado"):
                bb, bt = r["mcB"]["base"], r["mcB"]["bat"]
                reg = pd.DataFrame({"Día": np.arange(1, ef.N_DIAS + 1), "Escenario": [ef.NOMBRE[x] for x in bb["T"][0]],
                                    "Generación (MWh)": bb["G"][0], "Demanda (MWh)": bb["D"][0], "Balance (MWh)": bb["Bp"][0],
                                    "Carga al inicio (MWh)": bt["E_ini"][0], "Carga del día (MWh)": bt["carga"][0],
                                    "Cobertura (MWh)": bt["cob"][0], "Excedente vendido (MWh)": bt["vend"][0],
                                    "Déficit residual (MWh)": bt["res"][0], "Y sin batería (€)": bb["Y"][0],
                                    "Y con batería (€)": bt["Y"][0]})
                for c_ in reg.columns[2:]:
                    reg[c_] = reg[c_].map(lambda v: num(v, 2))
                st.dataframe(reg.astype(str), use_container_width=True, hide_index=True)

        with r_dif:
            dif = ef.diferencias(res, base)
            if dif is None:
                st.info("Simula al menos dos escenarios, incluido el escenario base, para ver las diferencias.")
            else:
                st.caption("Diferencia = escenario − base (positivo = mejora). Las diferencias simuladas son pareadas: cada "
                           "réplica usa las mismas semillas en todos los escenarios.")
                st.markdown("**O3.1 – O3.4 · beneficio y riesgo de un día sin batería**")
                mostrar_tabla(informe.tabla_dif(dif, informe.COLS_DIF_A), ef.fmt_col, "Escenario", "large", iconos=True)
                st.markdown("**Con batería · beneficio diario y rentabilidad de la batería**")
                mostrar_tabla(informe.tabla_dif(dif, informe.COLS_DIF_B), ef.fmt_col, "Escenario", "large", iconos=True)
                ver_fig(st, gr.fig_diferencias(dif))

        with r_dat:
            st.markdown("Resultados **simulados** de cada réplica.")
            la = pd.concat([r["mcA"]["replicas"].assign(Escenario=n) for n, r in res.items()]).reset_index()
            lb = pd.concat([r["mcB"]["replicas"].assign(Escenario=n) for n, r in res.items()]).reset_index()
            st.markdown("**Bloque A** (una fila por réplica de días independientes)")
            st.dataframe(la, use_container_width=True, hide_index=True)
            st.markdown("**Bloque B** (una fila por año simulado: medias por día de ese año)")
            st.dataframe(lb, use_container_width=True, hide_index=True)
            buf = io.BytesIO()
            try:
                with pd.ExcelWriter(buf, engine="openpyxl") as xw:
                    la.to_excel(xw, sheet_name="Replicas A", index=False)
                    lb.to_excel(xw, sheet_name="Anyos B", index=False)
                    ef.tabla_comparativa(res).astype(str).to_excel(xw, sheet_name="Comparacion")
                    for i, (n, r) in enumerate(res.items(), 1):
                        r["resA"].astype(str).to_excel(xw, sheet_name=f"KPIs A {i}")
                        r["resB"].astype(str).to_excel(xw, sheet_name=f"KPIs B {i}")
                    dif = ef.diferencias(res, base)
                    if dif is not None:
                        dif.astype(str).to_excel(xw, sheet_name="Diferencias")
                st.download_button("Descargar Excel", buf.getvalue(), "resultados_ecoflow.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception:  # openpyxl no disponible
                st.download_button("Descargar CSV (A)", la.to_csv(index=False).encode("utf-8"), "replicas_A.csv", "text/csv")

# ==============================================================================================
# 4) Batería (O5.1, O5.2, O6.1, O6.2)
# ==============================================================================================
with tab_bat:
    st.subheader("Dimensionamiento y robustez de la batería")
    bc = ss.bat_cfg
    c = st.columns([2, 2, 2, 1, 1, 1])
    nombres = list(ss.escenarios)
    eb = c[0].selectbox("Escenario", nombres, index=nombres.index(ss.base) if ss.base in nombres else 0, key=clave("b_esc"))
    bc["cap"] = lista_numeros(c[1].text_input("Capacidades (MWh)", texto_lista(bc["cap"]), key=clave("b_cap")), bc["cap"])
    bc["eta"] = lista_numeros(c[2].text_input("Eficiencias η", texto_lista(bc["eta"]), key=clave("b_eta")), bc["eta"])
    bc["racha_inicio"] = int(c[3].number_input("Racha · día inicial", 1, 365, int(bc["racha_inicio"]), key=clave("b_ri")))
    bc["racha_dias"] = int(c[4].number_input("Racha · días", 1, 365, int(bc["racha_dias"]), key=clave("b_rd")))
    c[5].markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
    if c[5].button("▶️ Calcular", type="primary", use_container_width=True, key="b_bat"):
        with st.spinner("Calculando…"):
            ss.bat = calcular_bat(com, ss.escenarios[eb], eb, aj, bc)
            ss.informe_bytes = None
    BT = ss.bat
    if BT is None:
        st.info("Ejecuta el análisis (barra lateral) o pulsa «Calcular».")
    else:
        df, opt = BT["cap"], BT["cap_opt"]
        st.caption(f"Escenario «{BT['escenario']}» · amortización {eur(BT['amort'])} por MWh y año.")
        m = st.columns(4)
        m[0].metric("📐 Capacidad óptima Cap*", f"{num(opt, 0)} MWh")
        m[1].metric("📐 Ganancia neta con Cap*", eur(df.loc[opt, "Teórico · ganancia neta anual (€)"]), "al año", delta_color="off")
        m[2].metric("📐 Ganancia diaria con Cap*", eur(df.loc[opt, "Teórico · E(Δ) (€/día)"], 1))
        dr, rs, cfg = BT["racha"]
        m[3].metric("📐 Recuperación tras la racha", f"{num(rs['Días hasta recuperar el 95 % de la carga habitual'], 0)} días")
        st.markdown("**O5.1 y O6.1 · ganancia según la capacidad y capacidad óptima** (¿rendimientos decrecientes?)")
        ver_fig(st, gr.fig_barrido(df, "cap", opt))
        mostrar_tabla(informe.indice_fmt(df, informe.FMT_IDX["cap"]), informe.fmt_tabla, "Capacidad", "small", iconos=True)
        st.markdown("**O5.2 · ganancia según la eficiencia η**")
        ver_fig(st, gr.fig_barrido(BT["eta"], "eta"))
        mostrar_tabla(informe.indice_fmt(BT["eta"], informe.FMT_IDX["eta"]), informe.fmt_tabla, "η", "small", iconos=True)
        st.markdown(f"**O6.2 · racha de {cfg['dias']} días Anticiclónicos desde el día {cfg['inicio']}** (📐 cadena de Markov y 🎲 simulación)")
        ver_fig(st, gr.fig_racha(dr, cfg["inicio"], cfg["dias"]))
        mostrar_tabla(pd.DataFrame({"Valor": rs}), lambda f, c_, v: ef.fmt_val(f, v), "Indicador", "large")

# ==============================================================================================
# 5) Sensibilidad
# ==============================================================================================
with tab_sens:
    st.subheader("Sensibilidad")
    s_par, s_sol, s_est = st.tabs(["O2.2 · parámetros y Sobol", "O3.3 · solar para E(Y) ≥ 0", "Estimador Monte Carlo"])
    with s_par:
        st.caption("Se varía cada parámetro manteniendo los demás fijos (Var(Y) teórica) y se calculan los índices de Sobol "
                   "de las entradas aleatorias (T, G_s, G_w, G_h, D) por simulación.")
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
    with s_sol:
        S = ss.sensib
        if S is None:
            st.info("Ejecuta el análisis o pulsa «Calcular» en la pestaña anterior.")
        else:
            ds, ms = S["solar"]
            st.markdown(f"Solar media actual: **{num(ms, 1)} MWh/día**. MWh/día de media que hay que añadir para que E(Y) = 0 "
                        "(📐 bisección sobre la teoría exacta):")
            mostrar_tabla(ds, informe.fmt_tabla, "Forma de la ampliación", "large", iconos=True)
            if st.button("➕ Crear el escenario «Solar para E(Y) = 0 (O3.3)»"):
                f_ = 1 + ds.iloc[1]["MWh/día de media adicionales"] / ms
                ss.escenarios["Solar para E(Y) = 0 (O3.3)"] = dict(copy.deepcopy(ss.escenarios[S["escenario"]]),
                                                                    solar_factor=float(f_),
                                                                    descripcion=f"Capacidad solar ×{num(f_, 3)} (O3.3).")
                ss.version += 1
                st.rerun()
    with s_est:
        st.caption("¿Cuánto «tiembla» la estimación según el tamaño de la simulación? Para cada valor se repite con varias semillas.")
        sc = ss.sens_cfg
        c = st.columns(3)
        met = ["dia", "anyo"]
        sc["metrica"] = c[0].selectbox("Métrica", met, index=met.index(sc.get("metrica", "dia")), key=clave("s_met"),
                                       format_func=lambda k: "E(Y) diario · por nº de días" if k == "dia"
                                       else "Ganancia de la batería · por nº de años")
        if sc["metrica"] == "dia":
            sc["n_min"] = int(c[1].number_input("Días mínimo", 10, 10**7, int(sc["n_min"]), 50, key=clave("s_nmin")))
            sc["n_max"] = int(c[2].number_input("Días máximo", 20, 10**7, int(sc["n_max"]), 1000, key=clave("s_nmax")))
        else:
            sc["a_min"] = int(c[1].number_input("Años mínimo", 1, 10**5, int(sc["a_min"]), 1, key=clave("s_amin")))
            sc["a_max"] = int(c[2].number_input("Años máximo", 2, 10**5, int(sc["a_max"]), 10, key=clave("s_amax")))
        c = st.columns(3)
        sc["puntos"] = int(c[0].slider("Puntos a evaluar", 3, 10, int(sc["puntos"]), key=clave("s_pts")))
        sc["replicas"] = int(c[1].slider("Réplicas por punto", 2, 20, int(sc["replicas"]), key=clave("s_rep")))
        pol = c[2].selectbox("Escenario", list(ss.escenarios), key=clave("s_pol"))
        if st.button("▶️ Analizar sensibilidad", type="primary"):
            vals = (ef.valores_barrido(sc["n_min"], sc["n_max"], sc["puntos"]) if sc["metrica"] == "dia" else
                    ef.valores_barrido(sc["a_min"], sc["a_max"], sc["puntos"]))
            barra = st.progress(0.0, text="Simulando…")
            df = ef.sensibilidad_estimador(com, ss.escenarios[pol], sc["metrica"], vals, sc["replicas"], aj["semilla"],
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
    c = st.columns([2, 1, 2, 1, 1])
    nombres = list(ss.escenarios)
    ea = c[0].selectbox("Escenario", nombres, index=nombres.index(ss.base) if ss.base in nombres else 0, key=clave("a_esc"))
    acf["cv_exp"] = float(c[1].number_input("CV de P_exp (O8.3, %)", 0.0, 300.0, 100 * float(acf["cv_exp"]), 5.0,
                                            key=clave("a_cv"))) / 100
    acf["umbrales"] = lista_numeros(c[2].text_input("Umbrales de precio para vender lo almacenado (€/MWh)",
                                                    texto_lista(acf["umbrales"]), key=clave("a_u")), acf["umbrales"])
    acf["coste_solar"] = float(c[3].number_input("Coste solar (€ por MWh/día·año)", 1.0, 1e8, float(acf["coste_solar"]),
                                                 500.0, key=clave("a_cs"), help="O8.6 (supuesto)"))
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
        dc, rho, teo = AM["contraste"]
        st.markdown(f"**O8.1 · contraste H₀: E(Δ) = 0** · los días de un año están correlacionados (ρ entre días consecutivos "
                    f"= {num(rho, 2)}), así que el contraste válido usa las medias anuales como macro-réplicas")
        mostrar_tabla(dc, informe.fmt_tabla, "Procedimiento", "medium")
        st.markdown("**O8.2 · estado de carga inicial**")
        mostrar_tabla(informe.indice_fmt(AM["inicial"], lambda v: pct(v, 0)), informe.fmt_tabla, "Carga inicial", "small", iconos=True)
        st.markdown(f"**O8.3 · gestión activa** con precio de exportación aleatorio (CV {pct(AM['cv_exp'], 0)}; 🎲)")
        mostrar_tabla(AM["activa"], informe.fmt_tabla, "Regla", "large", iconos=True)
        st.caption(informe.TEXTO_O83)
        st.markdown("**O8.4 · valor de la información meteorológica**")
        st.caption(informe.TEXTO_O84)
        tb = AM["tarifa"]
        st.markdown(f"**O8.5 · servicios de red**: la batería se paga sola con τ ≥ **{eur(tb['sola'], 2)} por MWh y día**; "
                    f"contando la ganancia operativa basta τ ≥ **{eur(max(tb['equilibrio'], 0), 2)}**"
                    + (" (ya es rentable sin tarifa)." if tb["equilibrio"] <= 0 else "."))
        dv, extra = AM["solar_vs"]
        st.markdown(f"**O8.6 · batería frente a solar del mismo coste** (≈ {num(extra, 1)} MWh/día de solar media)")
        mostrar_tabla(dv, informe.fmt_tabla, "Alternativa", "medium")
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
                    "(O7.1), comparación de escenarios (O3) y, para cada escenario, el beneficio de un día (O1, O2.1) y la "
                    "batería (O4, O5.3, O6.3); además, el dimensionamiento y la racha (O5.1, O5.2, O6.1, O6.2), la "
                    "sensibilidad (O2.2, O3.3) y la ampliación (O8).")
        titulo = st.text_input("Título del informe", "Gestión de riesgos en la micro-red de Eco-Flow Renewables", key="tit_inf")
        c = st.columns(4)
        con_bat = c[0].checkbox("Batería (dimensionamiento y racha)", value=ss.bat is not None, disabled=ss.bat is None)
        con_sen = c[1].checkbox("Sensibilidad (O2.2, O3.3)", value=ss.sensib is not None, disabled=ss.sensib is None)
        con_amp = c[2].checkbox("Ampliación (O8)", value=ss.ampliacion is not None, disabled=ss.ampliacion is None)
        con_est = c[3].checkbox("Sensibilidad del estimador", value=ss.sens is not None, disabled=ss.sens is None)
        if st.button("📄 Generar informe (.docx)", type="primary"):
            with st.spinner("Generando informe…"):
                ss.informe_bytes = informe.generar_informe(
                    R["res"], R["comunes"], R["ajustes"], R["base"], ss.bat if con_bat else None,
                    ss.sensib if con_sen else None, ss.ampliacion if con_amp else None, ss.sens if con_est else None, titulo)
        if ss.informe_bytes:
            st.download_button("⬇️ Descargar informe", ss.informe_bytes, f"Informe_EcoFlow_{datetime.date.today():%Y%m%d}.docx",
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
