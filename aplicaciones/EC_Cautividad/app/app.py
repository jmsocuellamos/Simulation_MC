# -*- coding: utf-8 -*-
"""Simulador del centro de cría en cautividad y reintroducción · aplicación Streamlit.

Ejecutar con:   streamlit run app.py
"""
import copy
import datetime
import io
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motor                      # noqa: E402
import kpis                       # noqa: E402
import validacion                 # noqa: E402
import graficos as gr             # noqa: E402
import informe                    # noqa: E402

st.set_page_config(page_title="Simulador · Centro de cría y reintroducción", page_icon="🦊", layout="wide")

TIPOS = motor.TIPOS
# En el navegador (stlite/Pyodide, p. ej. GitHub Pages) no hay procesos paralelos y el cálculo es más lento
EN_NAVEGADOR = sys.platform == "emscripten"
DESCRIPCIONES = {
    "Línea base": "Protocolo actualmente vigente.",
    "A · IA genómica": "Genotipado asistido por IA: −65% de tiempo en Pruebas Genéticas Avanzadas, −40% en Consulta "
                       "con el Genetista y mejores emparejamientos para Fundador/Cautivo (menos cautividad "
                       "permanente y menos prórrogas en el Módulo de Cría). Licencia de 12.000 €/año.",
    "B · Rescate rápido": "Equipo veterinario dedicado en exclusiva a los RescateLeve, con sus propias valoraciones "
                          "inicial y preliminar (en la mitad de tiempo). Coste de personal de 12.000 €/año.",
    "C · UCI compartida": "Las 5 plazas de UCI Veterinaria pasan a una única bolsa con prioridad RescateCritico > "
                          "RescateLeve > Fundador/Cautivo. Formación con coste único de 4.500 €.",
}
NOMBRES_ESTACION = {(335, 59): "Invierno", (59, 151): "Primavera", (151, 243): "Verano", (243, 335): "Otoño"}


# ==============================================================================================
# Estado de la sesión
# ==============================================================================================
def escenarios_por_defecto():
    out = {}
    for nombre, constructor in motor.ESCENARIOS_POR_DEFECTO.items():
        cfg = constructor()
        cfg.nombre = nombre
        cfg.descripcion = DESCRIPCIONES.get(nombre, "")
        out[nombre] = cfg.to_dict()
    return out


def comunes_por_defecto():
    param = copy.deepcopy(kpis.PARAM)
    lleg = motor.llegadas_a_lista(motor.ESTACIONES_ANIO)
    for r in lleg:
        r["Nombre"] = NOMBRES_ESTACION.get((r["Inicio"], r["Fin"]), "")
    return {"llegadas": lleg, "param": param}


def ajustes_por_defecto():
    return {"n_sim": 30, "dias": 3650, "semilla": 1000, "warmup": 365, "cooldown": 730,
            "paralelo": not EN_NAVEGADOR, "n_procesos": max(1, (os.cpu_count() or 2) - 1)}


def iniciar_estado(forzar=False):
    ss = st.session_state
    if forzar or "escenarios" not in ss:
        ss.escenarios = escenarios_por_defecto()
        ss.comunes = comunes_por_defecto()
        ss.ajustes = ajustes_por_defecto()
        ss.version = ss.get("version", 0) + 1
        ss.esc_activo = "Línea base"
    ss.setdefault("resultados", None)
    ss.setdefault("informe_bytes", None)


iniciar_estado()
ss = st.session_state


def clave(*partes):
    """Clave de widget ligada a la versión del estado (al restablecer/importar, los widgets se reinician)."""
    return f"v{ss.version}__" + "__".join(str(p) for p in partes)


def editor(key, factoria, **kw):
    """st.data_editor con datos iniciales estables (se crean una sola vez por clave)."""
    k0 = "ini__" + key
    if k0 not in ss:
        ss[k0] = factoria()
    return st.data_editor(ss[k0], key=key, **kw)


def _f(x, defecto=None):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return defecto
        return float(x)
    except (TypeError, ValueError):
        return defecto


# ==============================================================================================
# Barra lateral: ajustes de simulación, ejecución y gestión de la configuración
# ==============================================================================================
with st.sidebar:
    st.header("⚙️ Simulación")
    aj = ss.ajustes
    aj["n_sim"] = st.number_input("Nº de réplicas por escenario", 1, 1000, aj["n_sim"], key=clave("n_sim"))
    aj["dias"] = st.number_input("Días por réplica", 365, 36500, aj["dias"], step=365, key=clave("dias"),
                                 help="Con unos 60 ingresos al año y estancias de cientos de días en el Módulo de Cría, "
                                      "cada réplica debe cubrir varios años (por defecto 10 años = 3.650 días).")
    aj["semilla"] = st.number_input("Semilla de inicialización", 0, 10**9, aj["semilla"], key=clave("semilla"),
                                    help="La réplica i de todos los escenarios usa la semilla + i (números aleatorios "
                                         "comunes), lo que permite comparaciones pareadas.")
    c1, c2 = st.columns(2)
    aj["warmup"] = c1.number_input("Calentamiento (días)", 0, 3650, aj["warmup"], step=30, key=clave("warmup"),
                                   help="Días iniciales descartados (el centro arranca vacío).")
    aj["cooldown"] = c2.number_input("Cierre (días)", 0, 3650, aj["cooldown"], step=30, key=clave("cooldown"),
                                     help="Días finales descartados: los individuos que ingresan en ese periodo no "
                                          "suelen terminar su episodio antes del final de la réplica.")
    if EN_NAVEGADOR:
        aj["paralelo"] = False
        st.caption("🌐 Versión web: las réplicas se ejecutan en tu navegador, una tras otra. Cada réplica de 10 "
                   "años tarda unos segundos; no cierres la pestaña mientras se simula.")
    else:
        aj["paralelo"] = st.checkbox("Ejecutar réplicas en paralelo", aj["paralelo"], key=clave("paralelo"))
    if aj["paralelo"]:
        aj["n_procesos"] = st.slider("Procesos", 1, max(1, os.cpu_count() or 1), min(aj["n_procesos"], os.cpu_count() or 1),
                                     key=clave("nproc"))

    st.divider()
    nombres = list(ss.escenarios)
    seleccion = st.multiselect("Escenarios a simular", nombres, default=nombres, key=clave("sel", len(nombres)))
    ref = st.selectbox("Escenario de referencia", seleccion or nombres,
                       index=(seleccion or nombres).index("Línea base") if "Línea base" in (seleccion or nombres) else 0,
                       key=clave("ref", len(nombres)), help="Base de todas las comparaciones pareadas.")
    total_rep = aj["n_sim"] * len(seleccion)
    st.caption(f"Se simularán {total_rep} réplicas de {aj['dias']} días en total.")
    ejecutar = st.button("▶️ Ejecutar simulación", type="primary", use_container_width=True, disabled=not seleccion)

    st.divider()
    st.header("💾 Configuración")
    conf_json = json.dumps({"version_app": 1, "ajustes": ss.ajustes, "comunes": ss.comunes,
                            "escenarios": ss.escenarios}, ensure_ascii=False, indent=1)
    st.download_button("Descargar configuración (JSON)", conf_json, "configuracion_urgencias.json",
                       "application/json", use_container_width=True)
    fichero = st.file_uploader("Cargar configuración (JSON)", type="json", key=clave("upload"))
    if fichero is not None and ss.get("ultimo_json") != fichero.file_id:
        try:
            datos = json.load(fichero)
            ss.escenarios = datos["escenarios"]
            ss.comunes = datos["comunes"]
            ss.ajustes = {**ajustes_por_defecto(), **datos.get("ajustes", {})}
            ss.ultimo_json = fichero.file_id
            ss.version += 1
            ss.esc_activo = next(iter(ss.escenarios))
            st.rerun()
        except Exception as e:
            st.error(f"No se pudo cargar el fichero: {e}")
    if st.button("Restablecer valores por defecto", use_container_width=True):
        iniciar_estado(forzar=True)
        st.rerun()


# ==============================================================================================
# Cabecera
# ==============================================================================================
st.title("🦊 Simulador del centro de cría en cautividad y reintroducción")
st.caption("Centro Vega del Robledal · gemelo digital (SimPy) del protocolo de conservación del zorro venteño y de "
           "sus escenarios de mejora (tiempo en días). Configura los parámetros comunes y los de cada escenario, "
           "ejecuta la simulación Monte Carlo desde la barra lateral y genera el informe Word.")

tab_com, tab_esc, tab_res, tab_inf = st.tabs(["🌐 Parámetros comunes", "🏕️ Escenarios", "📊 Resultados", "📄 Informe"])


# ==============================================================================================
# 1) Parámetros comunes
# ==============================================================================================
with tab_com:
    st.subheader("Parámetros comunes a todos los escenarios")
    st.caption("Estos parámetros (ingresos de individuos, subvenciones, costes y SLAs) son los mismos para todos los "
               "escenarios: así las diferencias entre escenarios se deben únicamente a sus cambios de protocolo.")
    com = ss.comunes
    par = com["param"]

    st.markdown("#### 🍂 Ingresos por estación del año")
    st.caption("Tasa de ingresos (individuos/día) y reparto por perfil (%) en cada periodo del año, indicado por "
               "días del año (0–365). Los periodos deben cubrir el año completo sin solaparse; un periodo con "
               "Inicio > Fin cruza el fin de año (p. ej., Invierno 335 → 59).")
    lleg = editor(clave("llegadas"), lambda: pd.DataFrame(
        [{"Nombre": r.get("Nombre", ""), **{k: r[k] for k in ("Inicio", "Fin", "Tasa")},
          **{f"% {t}": 100 * r[t] for t in TIPOS}} for r in com["llegadas"]]),
        num_rows="dynamic", use_container_width=True, hide_index=True,
        column_config={"Inicio": st.column_config.NumberColumn("Inicio (día)", min_value=0, max_value=364, step=1),
                       "Fin": st.column_config.NumberColumn("Fin (día)", min_value=0, max_value=365, step=1),
                       "Tasa": st.column_config.NumberColumn("Tasa (ind/día)", min_value=0.0, step=0.01, format="%.3f"),
                       **{f"% {t}": st.column_config.NumberColumn(f"% {t}", min_value=0.0, max_value=100.0, format="%.1f")
                          for t in TIPOS}})
    nueva = []
    for _, r in lleg.dropna(subset=["Inicio", "Fin", "Tasa"]).iterrows():
        nueva.append({"Nombre": str(r.get("Nombre") or ""), "Inicio": int(r["Inicio"]), "Fin": int(r["Fin"]),
                      "Tasa": float(r["Tasa"]), **{t: _f(r[f"% {t}"], 0.0) / 100 for t in TIPOS}})
    com["llegadas"] = nueva
    err_lleg = validacion.validar_llegadas(nueva)
    for e in err_lleg:
        st.error(e)
    if not err_lleg:
        media = sum(r["Tasa"] * ((r["Fin"] - r["Inicio"]) % 365) for r in nueva) / 365
        st.success(f"Periodos correctos · ingreso medio de {media:.3f} individuos/día ({media * 365:.0f} al año).")

    st.markdown("#### 💶 Parámetros económicos")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Subvención por desenlace y penalización por fallecimiento (€ por individuo)**")
        sub = editor(clave("subvencion_v2"), lambda: pd.DataFrame({
            "Perfil": TIPOS,
            "Liberado": [float(par["SUBVENCION"]["Liberado"][t]) for t in TIPOS],
            "Cautividad permanente": [float(par["SUBVENCION"]["Cautividad_Permanente"][t]) for t in TIPOS],
            "Fallece": [float(par["PEN_FALLECE"][t]) for t in TIPOS]}),
            hide_index=True, use_container_width=True, disabled=["Perfil"],
            column_config={
                "Perfil": st.column_config.TextColumn("Perfil", width="medium"),
                "Liberado": st.column_config.NumberColumn("Liberado (€)", min_value=0.0, step=50.0, format="%.0f", help="Subvención si el individuo es liberado."),
                "Cautividad permanente": st.column_config.NumberColumn("Cautividad permanente (€)", min_value=0.0, step=50.0, format="%.0f", help="Subvención si el individuo pasa a cautividad permanente."),
                "Fallece": st.column_config.NumberColumn("Fallece (€)", min_value=0.0, step=100.0, format="%.0f", help="Penalización por fallecimiento."),
            })
        par["SUBVENCION"] = {"Liberado": {r["Perfil"]: _f(r["Liberado"], 0.0) for _, r in sub.iterrows()},
                       "Cautividad_Permanente": {r["Perfil"]: _f(r["Cautividad permanente"], 0.0) for _, r in sub.iterrows()}}
        par["PEN_FALLECE"] = {r["Perfil"]: _f(r["Fallece"], 0.0) for _, r in sub.iterrows()}
    with c2:
        st.markdown("**Margen operativo y penalizaciones adicionales**")
        par["MARGEN_EXIGIDO"] = st.number_input("Margen exigido (%)", 0.0, 100.0, 100 * float(par["MARGEN_EXIGIDO"]), 0.5,
                                                key=clave("mexig"), help="Límite inferior del objetivo del patronato (12–15%).") / 100
        par["PEN_SIN_UCI"] = st.number_input("Fallecimiento por falta de plaza de UCI (€, adicional)", 0.0,
                                             value=float(par.get("PEN_SIN_UCI", 0.0)), step=500.0, key=clave("pen_sinuci"))
        par["PEN_DEMORA"] = st.number_input("Penalización por demora (€ por individuo)", 0.0,
                                            value=float(par.get("PEN_DEMORA", 0.0)), step=50.0, key=clave("pen_demora"),
                                            help="Se aplica a cada RescateCritico cuya espera hasta la Evaluación Veterinaria "
                                                 "Inicial supera la latencia máxima del SLA1.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Coste por procedimiento realizado (€)**")
        cs = editor(clave("coste_serv"), lambda: pd.DataFrame({"Estación": list(par["COSTE_SERVICIO"]),
                                                                "Coste (€)": list(par["COSTE_SERVICIO"].values())}),
                    hide_index=True, use_container_width=True, num_rows="dynamic",
                    column_config={"Estación": st.column_config.TextColumn("Estación", width="large"),
                                   "Coste (€)": st.column_config.NumberColumn("Coste (€)", min_value=0.0, format="%.0f")})
        par["COSTE_SERVICIO"] = {r["Estación"]: _f(r["Coste (€)"], 0.0) for _, r in cs.iterrows() if r["Estación"]}
    with c2:
        st.markdown("**Coste por día de estancia (€/día)**")
        cd = editor(clave("coste_dia"), lambda: pd.DataFrame({"Estación": list(par["COSTE_DIA"]),
                                                               "Coste (€/día)": list(par["COSTE_DIA"].values())}),
                    hide_index=True, use_container_width=True, num_rows="dynamic",
                    column_config={"Estación": st.column_config.TextColumn("Estación", width="large"),
                                   "Coste (€/día)": st.column_config.NumberColumn("Coste (€/día)", min_value=0.0, format="%.0f")})
        par["COSTE_DIA"] = {r["Estación"]: _f(r["Coste (€/día)"], 0.0) for _, r in cd.iterrows() if r["Estación"]}

    st.markdown("#### 🏆 Acuerdos de nivel de servicio (SLAs)")
    c1, c2, c3, c4, c5 = st.columns(5)
    par["SLA_LATENCIA_D"] = c1.number_input("SLA1 · latencia máx. RescateCritico (horas)", 0.5, 240.0,
                                            par["SLA_LATENCIA_D"] * 24, 0.5, key=clave("sla1")) / 24
    par["SLA_LATENCIA_CUMPL"] = c1.number_input("SLA1 · % individuos que deben cumplirla", 1.0, 100.0,
                                                par["SLA_LATENCIA_CUMPL"] * 100, 1.0, key=clave("sla1b")) / 100
    par["SLA_RATIO_ESPERA"] = c2.number_input("SLA2 · ratio de espera máx. Fundador/Cautivo (%)", 1.0, 100.0,
                                              par["SLA_RATIO_ESPERA"] * 100, 1.0, key=clave("sla2")) / 100
    par["SLA_OCUP_UCI"] = c3.number_input("SLA3 · ocupación máx. UCI (%)", 1.0, 100.0, par["SLA_OCUP_UCI"] * 100, 1.0,
                                          key=clave("sla3")) / 100
    par["SLA_SIN_UCI"] = c4.number_input("SLA4 · mortalidad Sin_UCI máx. rescates (%)", 0.0, 100.0, par["SLA_SIN_UCI"] * 100,
                                         0.1, key=clave("sla4")) / 100
    par["SLA_LIBERACION"] = c5.number_input("SLA5 · tasa de liberación mínima (%)", 0.0, 100.0, par["SLA_LIBERACION"] * 100,
                                            1.0, key=clave("sla5")) / 100
    st.caption("SLA6 · margen neto ≥ margen exigido (se define en los parámetros económicos).")


# ==============================================================================================
# 2) Escenarios
# ==============================================================================================
def tabla_tiempos(cfg_d, horas):
    """Tabla editable de tiempos de servicio. horas=True → estancias (días); False → atención (horas)."""
    camas = set(cfg_d["JAULAS_CAPACIDAD"]) | set(cfg_d["JAULA_ALIAS"]) | {"UCI_Veterinaria"}
    filas = []
    for est, por_tipo in cfg_d["TIEMPOS_SERVICIO"].items():
        larga = est in camas or motor.es_observacion(est)
        if larga != horas:
            continue
        filas.append({"Estación": est, **{t: (por_tipo.get(t) if horas else (None if por_tipo.get(t) is None else por_tipo[t] * 24))
                                          for t in TIPOS}})
    return pd.DataFrame(filas, columns=["Estación"] + TIPOS)


def tabla_derivacion(cfg_d, estacion):
    destinos = {}
    for t, fila in cfg_d["PROB_DERIVACION"].get(estacion, {}).items():
        for dest, p in fila.items():
            destinos.setdefault(dest, {})[t] = 100 * p
    return pd.DataFrame([{"Destino": d, **{t: v.get(t) for t in TIPOS}} for d, v in destinos.items()],
                        columns=["Destino"] + TIPOS)


def editar_escenario(nombre):
    d = ss.escenarios[nombre]
    k = lambda *p: clave("esc", nombre, *p)
    errores, avisos = validacion.validar_config(motor.Config.from_dict(d))
    if errores:
        st.error(f"La configuración tiene {len(errores)} error(es): revisa la pestaña «Validación».")
    elif avisos:
        st.warning(f"Configuración válida con {len(avisos)} aviso(s).")
    else:
        st.success("Configuración válida.")

    t_gen, t_rec, t_uci, t_tiem, t_der, t_val = st.tabs(
        ["General y costes", "Recursos", "UCI y Recinto", "Tiempos de servicio", "Derivaciones", "Validación"])

    with t_gen:
        d["descripcion"] = st.text_area("Descripción del escenario", d.get("descripcion", ""), key=k("desc"))
        st.markdown("**Costes fijos del escenario**")
        st.caption("Marca «Anual» para los costes recurrentes (se prorratean a los días simulados); los no anuales "
                   "son costes únicos (se imputan íntegros y se calcula su periodo de amortización).")
        cf = editor(k("costes"), lambda: pd.DataFrame(
            [{"Concepto": c.replace("_anual", ""), "Importe (€)": v, "Anual": "anual" in c} for c, v in d.get("COSTES_FIJOS", {}).items()],
            columns=["Concepto", "Importe (€)", "Anual"]), num_rows="dynamic", hide_index=True, use_container_width=True,
            column_config={"Anual": st.column_config.CheckboxColumn("Anual", default=True),
                           "Importe (€)": st.column_config.NumberColumn(min_value=0.0, step=1000.0, format="%.0f")})
        costes = {}
        for _, r in cf.iterrows():
            if r["Concepto"] and _f(r["Importe (€)"]) is not None:
                base = str(r["Concepto"]).strip().replace(" ", "_")
                costes[base + ("_anual" if bool(r["Anual"]) else "")] = float(r["Importe (€)"])
        d["COSTES_FIJOS"] = costes

    with t_rec:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Servidores** (el recurso se libera tras el servicio)")
            srv = editor(k("srv"), lambda: pd.DataFrame({"Servidor": list(d["SERVIDORES_CAPACIDAD"]),
                                                          "Capacidad": list(d["SERVIDORES_CAPACIDAD"].values())}),
                         num_rows="dynamic", hide_index=True, use_container_width=True,
                         column_config={"Capacidad": st.column_config.NumberColumn(min_value=1, step=1)})
            d["SERVIDORES_CAPACIDAD"] = {str(r["Servidor"]): int(r["Capacidad"]) for _, r in srv.iterrows()
                                         if r["Servidor"] and _f(r["Capacidad"]) is not None}
            st.markdown("**Asignación estación → servidor** (si una estación no aparece, usa el servidor de su mismo nombre)")
            pool = editor(k("pool"), lambda: pd.DataFrame({"Estación": list(d["POOL_ESTACION"]),
                                                            "Servidor": list(d["POOL_ESTACION"].values())}),
                          num_rows="dynamic", hide_index=True, use_container_width=True)
            d["POOL_ESTACION"] = {str(r["Estación"]): str(r["Servidor"]) for _, r in pool.iterrows()
                                  if r["Estación"] and r["Servidor"]}
        with c2:
            st.markdown("**Jaulas y recintos** (se retienen durante toda la estancia)")
            cam = editor(k("camas"), lambda: pd.DataFrame({"Jaula": list(d["JAULAS_CAPACIDAD"]),
                                                            "Capacidad": list(d["JAULAS_CAPACIDAD"].values())}),
                         num_rows="dynamic", hide_index=True, use_container_width=True,
                         column_config={"Capacidad": st.column_config.NumberColumn(min_value=1, step=1)})
            d["JAULAS_CAPACIDAD"] = {str(r["Jaula"]): int(r["Capacidad"]) for _, r in cam.iterrows()
                                     if r["Jaula"] and _f(r["Capacidad"]) is not None}
            st.markdown("**Estaciones que comparten jaula** (alias → jaula física)")
            ali = editor(k("alias"), lambda: pd.DataFrame({"Estación": list(d["JAULA_ALIAS"]), "Jaula": list(d["JAULA_ALIAS"].values())}),
                         num_rows="dynamic", hide_index=True, use_container_width=True)
            d["JAULA_ALIAS"] = {str(r["Estación"]): str(r["Jaula"]) for _, r in ali.iterrows() if r["Estación"] and r["Jaula"]}

    with t_uci:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Recinto de Recuperación**")
            d["RECINTO_CAPACIDAD"] = int(st.number_input("Plazas del Recinto", 1, 1000, int(d["RECINTO_CAPACIDAD"]), key=k("box")))
            st.caption("Estación en la que cada perfil captura una plaza del Recinto (la retiene durante la fase de "
                       "manejo agudo, hasta pasar a una jaula, a la UCI o a un desenlace final).")
            estaciones = sorted(set(d["PROB_DERIVACION"]) - set(d["JAULAS_CAPACIDAD"]) - set(d["JAULA_ALIAS"]) - {"UCI_Veterinaria"})
            opciones = ["(no captura Recinto)"] + estaciones
            captura = {}
            for t in TIPOS:
                actual = d["RECINTO_CAPTURA"].get(t)
                sel = st.selectbox(f"{t}", opciones, index=opciones.index(actual) if actual in opciones else 0, key=k("cap", t))
                if sel != "(no captura Recinto)":
                    captura[t] = sel
            d["RECINTO_CAPTURA"] = captura
        with c2:
            st.markdown("**UCI Veterinaria**")
            modo = st.radio("Gestión de plazas", ["Bolsas reservadas por criticidad", "Bolsa compartida con prioridad"],
                            index=0 if d["UCI_MODE"] != "pooled" else 1, key=k("ucimodo"),
                            help="Reservadas: cada perfil solo puede usar su bolsa. Compartida: una única cola con "
                                 "prioridad RescateCritico > RescateLeve > Fundador/Cautivo.")
            d["UCI_MODE"] = "pooled" if modo.startswith("Bolsa compartida") else "partitioned"
            cc1, cc2, cc3 = st.columns(3)
            d["UCI_CRITICO_CAPACIDAD"] = int(cc1.number_input("Plazas RescateCritico", 1, 1000, int(d["UCI_CRITICO_CAPACIDAD"]), key=k("ur")))
            d["UCI_LEVE_CAPACIDAD"] = int(cc2.number_input("Plazas RescateLeve", 1, 1000, int(d["UCI_LEVE_CAPACIDAD"]), key=k("un")))
            d["UCI_BASE_CAPACIDAD"] = int(cc3.number_input("Plazas Fundador/Cautivo", 1, 1000, int(d["UCI_BASE_CAPACIDAD"]), key=k("ub")))
            if d["UCI_MODE"] == "pooled":
                st.caption("En modo compartido solo se usa el total de plazas (las bolsas se guardan por si se vuelve al modo reservado).")
                d["UCI_TOTAL_CAPACIDAD"] = int(st.number_input("Plazas totales de UCI", 1, 3000, int(d["UCI_TOTAL_CAPACIDAD"]), key=k("ut")))
            else:
                d["UCI_TOTAL_CAPACIDAD"] = d["UCI_CRITICO_CAPACIDAD"] + d["UCI_LEVE_CAPACIDAD"] + d["UCI_BASE_CAPACIDAD"]
                st.metric("Plazas totales de UCI", d["UCI_TOTAL_CAPACIDAD"])
            d["LIMITE_ESPERA_UCI"] = float(st.number_input("Límite de espera de plaza UCI (días)", 0.1, 60.0,
                                                           float(d["LIMITE_ESPERA_UCI"]), 0.5, key=k("lim")))

    with t_tiem:
        st.caption("Tiempo nominal de servicio por estación y perfil. El motor genera cada tiempo como "
                   "U(0,75; 1,25) × valor nominal. Deja la celda vacía si el perfil no pasa por esa estación.")
        st.markdown("**Estaciones de atención (horas)**")
        ta = editor(k("t_min"), lambda: tabla_tiempos(d, horas=False), num_rows="dynamic", hide_index=True, use_container_width=True,
                    column_config={"Estación": st.column_config.TextColumn("Estación", width="large"), **{t: st.column_config.NumberColumn(t, min_value=0.0, format="%.2f") for t in TIPOS}})
        st.markdown("**Estancias, jaulas y observación (días)**")
        th = editor(k("t_h"), lambda: tabla_tiempos(d, horas=True), num_rows="dynamic", hide_index=True, use_container_width=True,
                    column_config={"Estación": st.column_config.TextColumn("Estación", width="large"), **{t: st.column_config.NumberColumn(t, min_value=0.0, format="%.2f") for t in TIPOS}})
        tiempos = {}
        for tabla, factor in ((ta, 1 / 24), (th, 1.0)):
            for _, r in tabla.iterrows():
                if not r["Estación"]:
                    continue
                est = str(r["Estación"])
                previo = d["TIEMPOS_SERVICIO"].get(est, {})
                fila = {}
                for t in TIPOS:
                    v = _f(r[t])
                    if v is not None:
                        v = v * factor
                        fila[t] = previo[t] if t in previo and math.isclose(previo[t], v, rel_tol=1e-9) else v
                if fila:
                    tiempos[est] = fila
        d["TIEMPOS_SERVICIO"] = tiempos

    with t_der:
        st.caption("Probabilidades (%) de pasar de la estación de origen a cada destino, por perfil. Cada columna "
                   "debe sumar 100 % (o quedar vacía si ese perfil no pasa por la estación). Destinos finales: "
                   "«Alta» y «Fallece».")
        origenes = list(d["PROB_DERIVACION"])
        c1, c2 = st.columns([2, 1])
        origen = c1.selectbox("Estación de origen", origenes, key=k("origen"))
        nueva_est = c2.text_input("Añadir estación de origen", key=k("nueva_est"), placeholder="nombre y pulsa Intro")
        if nueva_est and nueva_est not in d["PROB_DERIVACION"]:
            d["PROB_DERIVACION"][nueva_est] = {}
            st.rerun()
        der = editor(k("der", origen), lambda: tabla_derivacion(d, origen), num_rows="dynamic", hide_index=True,
                     use_container_width=True,
                     column_config={t: st.column_config.NumberColumn(f"{t} (%)", min_value=0.0, max_value=100.0, format="%.1f")
                                    for t in TIPOS})
        filas = {}
        sumas = {}
        previo_est = d["PROB_DERIVACION"].get(origen, {})
        for t in TIPOS:
            col = {str(r["Destino"]): _f(r[t]) for _, r in der.iterrows() if r["Destino"] and _f(r[t]) is not None}
            if col and sum(col.values()) > 0:
                previo = previo_est.get(t, {})
                filas[t] = {dest: (previo[dest] if dest in previo and math.isclose(previo[dest], p / 100, rel_tol=1e-9)
                                   else p / 100) for dest, p in col.items() if p > 0}
                sumas[t] = sum(col.values())
        d["PROB_DERIVACION"][origen] = filas
        if sumas:
            st.write("Suma por perfil: " + " · ".join(
                f"{'✅' if abs(s - 100) < 1e-6 else '❌'} {t}: {s:.1f} %" for t, s in sumas.items()))
        if st.button("Eliminar esta estación de origen", key=k("del_origen")):
            d["PROB_DERIVACION"].pop(origen, None)
            st.rerun()

    with t_val:
        cfg = motor.Config.from_dict(d)
        errores, avisos = validacion.validar_config(cfg)
        if not errores and not avisos:
            st.success("Sin errores ni avisos.")
        for e in errores:
            st.error(e)
        for a in avisos:
            st.warning(a)
        st.markdown("**Estaciones que puede recorrer cada perfil**")
        st.dataframe(pd.DataFrame({t: [", ".join(sorted(validacion.estaciones_alcanzables(cfg, t)))] for t in TIPOS},
                                  index=["Estaciones"]).T, use_container_width=True)
        if nombre != "Línea base" and "Línea base" in ss.escenarios:
            st.markdown("**Diferencias frente a la Línea base**")
            dif = informe.diferencias_config(motor.Config.from_dict(ss.escenarios["Línea base"]), cfg)
            st.dataframe(dif, hide_index=True, use_container_width=True) if len(dif) else st.info("Sin diferencias.")


with tab_esc:
    st.subheader("Configuración de los escenarios")
    nombres = list(ss.escenarios)
    if ss.esc_activo not in nombres:
        ss.esc_activo = nombres[0]
    c1, c2 = st.columns([3, 2])
    activo = c1.selectbox("Escenario a configurar", nombres, index=nombres.index(ss.esc_activo), key=clave("activo", len(nombres)))
    ss.esc_activo = activo

    with c2.expander("➕ Crear, combinar, renombrar o eliminar escenarios"):
        st.markdown("**Nuevo escenario (copia de otro)**")
        n_nombre = st.text_input("Nombre del nuevo escenario", key=clave("nuevo_nombre"))
        origen_copia = st.selectbox("Copiar de", nombres, key=clave("copiar_de", len(nombres)))
        if st.button("Crear copia", key=clave("crear")):
            if not n_nombre or n_nombre in ss.escenarios:
                st.error("Indica un nombre nuevo y distinto de los existentes.")
            else:
                ss.escenarios[n_nombre] = copy.deepcopy(ss.escenarios[origen_copia])
                ss.escenarios[n_nombre]["nombre"] = n_nombre
                ss.esc_activo = n_nombre
                st.rerun()
        st.markdown("**Combinación de escenarios**")
        st.caption("Aplica sobre la base elegida todos los cambios que cada escenario seleccionado introduce respecto a ella.")
        base_comb = st.selectbox("Base", nombres, index=nombres.index("Línea base") if "Línea base" in nombres else 0,
                                 key=clave("base_comb", len(nombres)))
        a_combinar = st.multiselect("Escenarios a combinar", [n for n in nombres if n != base_comb], key=clave("comb", len(nombres)))
        nombre_comb = st.text_input("Nombre de la combinación", value=" + ".join(n.split(" ")[0] for n in a_combinar),
                                    key=clave("nombre_comb", "-".join(a_combinar)))
        if st.button("Crear combinación", key=clave("crear_comb"), disabled=len(a_combinar) < 2):
            if not nombre_comb or nombre_comb in ss.escenarios:
                st.error("Indica un nombre nuevo y distinto de los existentes.")
            else:
                base_cfg = motor.Config.from_dict(ss.escenarios[base_comb])
                cfgs = [motor.Config.from_dict(ss.escenarios[n]) for n in a_combinar]
                comb = motor.combinar_configs(*cfgs, nombre=nombre_comb, base=base_cfg)
                comb.descripcion = f"Combinación de {', '.join(a_combinar)} sobre {base_comb}."
                ss.escenarios[nombre_comb] = comb.to_dict()
                ss.esc_activo = nombre_comb
                st.rerun()
        st.markdown("**Renombrar / eliminar el escenario activo**")
        renombre = st.text_input("Nuevo nombre", value=activo, key=clave("renombrar", activo))
        cc1, cc2, cc3 = st.columns(3)
        if cc1.button("Renombrar", key=clave("btn_ren", activo)) and renombre and renombre != activo:
            if renombre in ss.escenarios:
                st.error("Ya existe un escenario con ese nombre.")
            else:
                ss.escenarios = {(renombre if n == activo else n): v for n, v in ss.escenarios.items()}
                ss.escenarios[renombre]["nombre"] = renombre
                ss.esc_activo = renombre
                st.rerun()
        if cc2.button("Eliminar", key=clave("btn_del", activo), disabled=len(nombres) <= 1):
            ss.escenarios.pop(activo)
            ss.esc_activo = next(iter(ss.escenarios))
            st.rerun()
        if cc3.button("Restablecer", key=clave("btn_rst", activo), disabled=activo not in motor.ESCENARIOS_POR_DEFECTO,
                      help="Vuelve a los valores por defecto (solo escenarios predefinidos)."):
            ss.escenarios[activo] = escenarios_por_defecto()[activo]
            ss.version += 1
            st.rerun()

    # Estado de validación de todos los escenarios
    estado = []
    for n, dd in ss.escenarios.items():
        e, a = validacion.validar_config(motor.Config.from_dict(dd))
        estado.append(f"{'❌' if e else ('⚠️' if a else '✅')} {n}")
    st.caption("Estado: " + " · ".join(estado))
    st.divider()
    editar_escenario(activo)


# ==============================================================================================
# Ejecución de la simulación
# ==============================================================================================
def parametros_kpi():
    p = copy.deepcopy(ss.comunes["param"])
    p["T_WARMUP"] = ss.ajustes["warmup"]
    p["T_COOLDOWN"] = ss.ajustes["cooldown"]
    return p


if ejecutar:
    aj = ss.ajustes
    problemas = list(validacion.validar_llegadas(ss.comunes["llegadas"]))
    if aj["dias"] <= aj["warmup"] + aj["cooldown"] + 5:
        problemas.append("Los días por réplica deben superar ampliamente el calentamiento + el cierre.")
    for n in seleccion:
        e, _ = validacion.validar_config(motor.Config.from_dict(ss.escenarios[n]))
        problemas += [f"{n}: {x}" for x in e]
    if problemas:
        with tab_res:
            st.error("No se puede simular: corrige primero estos problemas.")
            for p in problemas:
                st.write("• " + p)
    else:
        param = parametros_kpi()
        llegadas = motor.lista_a_llegadas(ss.comunes["llegadas"])
        protocolos = {}
        barra = st.sidebar.progress(0.0, text="Preparando…")
        total = aj["n_sim"] * len(seleccion)
        for j, n in enumerate(seleccion):
            cfg = motor.Config.from_dict(ss.escenarios[n])
            cfg.nombre = n
            cfg.ESTACIONES_ANIO = llegadas

            def prog(i, n_sim, j=j, n=n):
                barra.progress(min(1.0, (j * n_sim + i) / total), text=f"{n}: réplica {i}/{n_sim}")

            protocolos[n] = kpis.montecarlo(cfg, n_sim=aj["n_sim"], dias=aj["dias"], semilla_base=aj["semilla"],
                                            param=param, progreso=prog, paralelo=aj["paralelo"],
                                            n_procesos=aj.get("n_procesos"))
        barra.progress(1.0, text="Simulación terminada")
        ss.resultados = {"protocolos": protocolos, "ref": ref if ref in protocolos else seleccion[0],
                         "ajustes": copy.deepcopy(aj), "comunes": {"llegadas": copy.deepcopy(ss.comunes["llegadas"]),
                                                                   "param": param},
                         "fecha": datetime.datetime.now().strftime("%d/%m/%Y %H:%M")}
        ss.informe_bytes = None
        st.toast("Simulación terminada: revisa la pestaña «Resultados».", icon="✅")


# ==============================================================================================
# 3) Resultados
# ==============================================================================================
def tabla_fmt_filas(df, cols=None):
    """Formatea una tabla con KPIs en filas (mismo formato que el informe)."""
    cols = cols or list(df.columns)
    out = pd.DataFrame(index=df.index, columns=cols, dtype=object)
    for f in df.index:
        for c in cols:
            out.loc[f, c] = informe.fmt_por_fila(f, c, df.loc[f, c])
    return out


def negrita_inicial(p):
    """Pone en negrita el texto hasta el primer ': ' o '. '."""
    cortes = [x for x in (p.find(": "), p.find(". ")) if x > 0]
    if not cortes or min(cortes) > 60:
        return p
    c = min(cortes) + 1
    return f"**{p[:c]}**{p[c:]}"


def mostrar(fig):
    if fig is not None:
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def tabla_fmt(df, fn):
    out = df.copy().astype(object)
    for f in df.index:
        for c in df.columns:
            out.loc[f, c] = fn(f, c, df.loc[f, c])
    return out


with tab_res:
    R = ss.resultados
    if R is None:
        st.info("Todavía no hay resultados. Configura los escenarios y pulsa «▶️ Ejecutar simulación» en la barra lateral.")
    else:
        prot, ref_r, aj_r = R["protocolos"], R["ref"], R["ajustes"]
        st.caption(f"Simulación del {R['fecha']} · {aj_r['n_sim']} réplicas de {aj_r['dias']} días · semilla {aj_r['semilla']} "
                   f"· referencia: **{ref_r}** · escenarios: {', '.join(prot)}")
        r_res, r_graf, r_det, r_dat = st.tabs(["Resumen comparativo", "Gráficos", "Detalle por escenario", "Datos"])

        with r_res:
            recomendacion, ranking = kpis.texto_recomendacion(prot, ref_r)
            with st.container(border=True):
                st.markdown("##### Síntesis automática")
                for p in recomendacion:
                    st.markdown(negrita_inicial(p))
            medias, marcas = kpis.tabla_sintesis(prot, ref=ref_r)
            sint = medias.copy().astype(object)
            for c in sint.columns:
                sint[c] = [kpis.fmt_kpi(k, medias.loc[k, c]) + (f" {marcas.loc[k, c]}" if marcas.loc[k, c] else "")
                           for k in medias.index]
            st.markdown(f"**KPIs principales** (media de las réplicas · ▲ mejora / ▼ empeora significativamente frente a {ref_r})")
            st.dataframe(sint.rename(index=kpis.ETIQUETAS_KPI), use_container_width=True, height=36 * (len(sint) + 1))
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Cumplimiento de SLAs** (% de réplicas)")
                st.dataframe(tabla_fmt(kpis.tabla_slas(prot).rename(index=kpis.slas_mc(prot[ref_r])["Descripción"]),
                                       lambda f, c, v: kpis.pct(v, 0)), use_container_width=True)
            with c2:
                st.markdown("**Ranking de escenarios**")
                rk = ranking[["SLAs cumplidos", "Resultado neto anual (€)", "Liberación global", "Mortalidad global",
                              "Mortalidad Sin_UCI rescates", "Margen neto", "Descartado"]]
                st.dataframe(tabla_fmt(rk, lambda f, c, v: kpis.eur(v) if "€" in c else (
                    kpis.pct(v, 2) if ("Mortalidad" in c or "Margen" in c or "Liberación" in c) else ("Sí" if v is True else ("No" if v is False else str(v))))),
                    use_container_width=True)
            if len(prot) > 1:
                st.markdown(f"**Resultado económico frente a {ref_r}**")
                eco = kpis.tabla_economica(prot, ref=ref_r)
                st.dataframe(tabla_fmt(eco, lambda f, c, v: kpis.eur(v)), use_container_width=True)

        with r_graf:
            todas = sorted(set.intersection(*[set(m["resumen"].columns) for m in prot.values()]) -
                           {c for c in prot[ref_r]["resumen"].columns if c.startswith("Cumple_")})
            elegidos = st.multiselect("KPIs a comparar", todas, default=[k for k in informe.KPIS_GRAFICOS if k in todas],
                                      format_func=lambda k: kpis.ETIQUETAS_KPI.get(k, k), key="kpis_graf")
            if elegidos:
                mostrar(gr.fig_comparativo(prot, elegidos, 3))
            c1, c2 = st.columns(2)
            with c1:
                mostrar(gr.fig_utilizacion_comparada(prot))
            with c2:
                mostrar(gr.fig_sintesis(ranking, list(prot), ref_r))
            mostrar(gr.fig_tsys_perfil(prot))
            mostrar(gr.fig_ocupacion_uci(prot, R["comunes"]["param"]["SLA_OCUP_UCI"]))

        with r_det:
            esc = st.selectbox("Escenario", list(prot), key="det_esc")
            mc = prot[esc]
            if esc != ref_r:
                st.info(kpis.texto_escenario(esc, prot[ref_r], mc, ref=ref_r))
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown("**KPIs principales (Monte Carlo)**")
                st.dataframe(tabla_fmt_filas(kpis.resumen_mc(mc, kpis.KPIS_SINTESIS).rename(index=kpis.ETIQUETAS_KPI),
                                             ["Media", "SD", "IC95_inf", "IC95_sup", "P2.5", "P97.5"]), use_container_width=True)
            with c2:
                st.markdown("**SLAs**")
                s = kpis.slas_mc(mc).set_index("Descripción")
                st.dataframe(tabla_fmt(s[["Valor_medio", "%Réplicas_cumple", "Veredicto"]],
                                       lambda f, c, v: v if isinstance(v, str) else (kpis.pct(v, 0) if c == "%Réplicas_cumple"
                                                                                     else kpis.num(v, 4))),
                             use_container_width=True)
            st.markdown("**KPIs temporales por perfil (días)**")
            st.dataframe(kpis.tabla_media_mc(mc, "operativos").round(3), use_container_width=True)
            st.markdown("**KPIs biológicos y de conservación por perfil**")
            st.dataframe(tabla_fmt(kpis.tabla_media_mc(mc, "biologicos"),
                                   lambda f, c, v: kpis.num(v, 1) if c in ("Individuos", "Liberados_anio") else kpis.pct(v, 2)),
                         use_container_width=True)
            rec = kpis.tabla_media_mc(mc, "recursos")
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown("**Recursos**")
                st.dataframe(rec[["Tipo", "Capacidad", "Utilizacion", "Lq_media", "%Bloqueo", "Diagnostico", "%Réplicas_ρ≥0.85"]]
                             .round(3), use_container_width=True)
            with c2:
                mostrar(gr.fig_utilizacion(rec, f"Utilización media · {esc}"))
            st.markdown("**Carga y espera por estación**")
            st.dataframe(kpis.tabla_media_mc(mc, "estaciones").round(3), use_container_width=True)
            st.markdown("**Resultado económico por perfil (media por réplica)**")
            st.dataframe(tabla_fmt(kpis.tabla_media_mc(mc, "economicos"),
                                   lambda f, c, v: kpis.num(v, 1) if c == "Individuos" else (kpis.pct(v) if c == "Margen" else kpis.eur(v))),
                         use_container_width=True)
            if esc != ref_r:
                c1, c2 = st.columns([3, 2])
                with c1:
                    st.markdown(f"**Comparación pareada con {ref_r}**")
                    cmp_ = kpis.comparar_escenarios(prot[ref_r], mc, kpis.KPIS_SINTESIS)
                    st.dataframe(tabla_fmt_filas(cmp_.rename(index=kpis.ETIQUETAS_KPI),
                                                 ["Base", "Escenario", "Δ%", "p_valor", "Significativo"])
                                 .rename(columns={"Base": ref_r, "Escenario": esc}), use_container_width=True)
                with c2:
                    st.markdown("**Viabilidad económica**")
                    viab = kpis.viabilidad_economica(prot[ref_r], mc)
                    st.dataframe(tabla_fmt(viab, lambda f, c, v: kpis.pct(v) if ("Margen" in f or "%" in f) else (
                        kpis.num(v, 2) if "Cobertura" in f else (kpis.num(v, 0) if "días" in f else kpis.eur(v)))),
                        use_container_width=True)

        with r_dat:
            st.markdown("Valores de cada réplica (una fila por réplica y escenario).")
            largo = pd.concat([m["resumen"].copy().assign(Escenario=n) for n, m in prot.items()]).reset_index()
            st.dataframe(largo, use_container_width=True, height=350)
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as xw:
                for n, m in prot.items():
                    m["resumen"].to_excel(xw, sheet_name=("".join(ch for ch in n if ch.isalnum() or ch in " +-_"))[:31] or "esc")
                kpis.resumen_mc(prot[ref_r], kpis.KPIS_SINTESIS).to_excel(xw, sheet_name="Resumen_referencia")
            st.download_button("Descargar réplicas (Excel)", buf.getvalue(), "replicas_simulacion.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ==============================================================================================
# 4) Informe
# ==============================================================================================
with tab_inf:
    R = ss.resultados
    if R is None:
        st.info("Ejecuta primero una simulación para poder generar el informe.")
    else:
        st.markdown("El informe Word incluye: resumen ejecutivo, parámetros comunes y de cada escenario (con sus "
                    "diferencias frente a la referencia), tablas comparativas de KPIs, SLAs y resultado económico, "
                    "gráficas comparativas y un apartado de detalle por escenario con todas sus tablas y gráficas.")
        titulo = st.text_input("Título del informe", "Centro de cría en cautividad y reintroducción", key="tit_inf")
        if st.button("📄 Generar informe (.docx)", type="primary"):
            barra = st.progress(0.0, text="Generando informe…")
            ss.informe_bytes = informe.generar_informe(
                R["protocolos"], R["ref"],
                {k: R["ajustes"][k] for k in ("n_sim", "dias", "semilla", "warmup", "cooldown")},
                R["comunes"], titulo=titulo, progreso=lambda f, t: barra.progress(min(f, 1.0), text=t))
            barra.progress(1.0, text="Informe listo")
        if ss.informe_bytes:
            st.download_button("⬇️ Descargar informe", ss.informe_bytes,
                               f"Informe_Centro_Cria_{datetime.date.today():%Y%m%d}.docx",
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                               type="primary")
