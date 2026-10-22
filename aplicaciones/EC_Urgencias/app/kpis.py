# -*- coding: utf-8 -*-
"""KPIs, SLAs, análisis Monte Carlo y comparación de escenarios de la red de colas hospitalaria.

Funciones del cuaderno '01 Red de Colas Hospitalaria' (Tareas 1, 2 y 6), reorganizadas como módulo
para la aplicación Streamlit."""
import math
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

from motor import TIPOS, simular

PARAM = {
    # Reembolso por alta (€) según perfil
    "REEMBOLSO": {"Verde": 420, "Amarillo": 1250, "Naranja": 6100, "Rojo": 15200},
    # Coste por servicio/procedimiento realizado (€)
    "COSTE_SERVICIO": {"Pruebas_Basicas": 40, "Pruebas_Avanzadas": 220,
                       "Consulta_Especialista": 95, "Quirofano": 3500},
    # Coste por día de estancia (€/día)
    "COSTE_DIA": {"Planta": 550, "Planta_PostUCI": 550, "UCI": 2100, "Rehabilitacion": 380},
    # Penalizaciones
    "PEN_FALLECE": 3000,
    "PEN_SIN_UCI": 10000,      # adicional a la anterior
    # Tramos de margen exigido según tasa de altas: (umbral, margen)
    "TRAMOS_MARGEN": [(0.90, 0.15), (0.80, 0.10), (0.75, 0.075)],
    "MARGEN_MINIMO": 0.05,
    # SLAs
    "SLA_LATENCIA_H": 15 / 60,          # 15 minutos hasta Diag_Inicial (Naranja/Rojo)
    "SLA_LATENCIA_CUMPL": 0.95,         # se exige que el 95% de los pacientes lo cumpla
    "SLA_RATIO_ESPERA": 0.25,           # Verde/Amarillo
    "SLA_OCUP_UCI": 0.90,
    "SLA_SIN_UCI": 0.02,                # sobre pacientes Naranja/Rojo
    # Ventanas de análisis (horas)
    "T_WARMUP": 10 * 24,
    "T_COOLDOWN": 10 * 24,
}


ESTADOS_ABS = ["Alta", "Fallece"]


ESTACIONES_CAMA = {"Planta", "Planta_PostUCI", "Quirofano", "Rehabilitacion", "UCI"}


def estaciones_cama(cfg):
    """Estaciones que retienen una cama: camas físicas, sus alias y la UCI."""
    return set(cfg.CAMAS_CAPACIDAD) | set(getattr(cfg, "CAMA_ALIAS", {})) | {"UCI"}


FRANJAS = [("22-07", 22, 7), ("07-10", 7, 10), ("10-14", 10, 14), ("14-17", 14, 17), ("17-22", 17, 22)]


def franja_de_hora(h):
    """Devuelve la etiqueta de la franja horaria a la que pertenece la hora del día `h`."""
    for nombre, ini, fin in FRANJAS:
        if (ini > fin and (h >= ini or h < fin)) or (ini < fin and ini <= h < fin):
            return nombre


def margen_exigido(tasa_alta, param=PARAM):
    """Margen operativo exigido por el acuerdo de financiación según la tasa de altas."""
    for umbral, margen in param["TRAMOS_MARGEN"]:
        if tasa_alta > umbral:
            return margen
    return param["MARGEN_MINIMO"]


def preparar_traza(df):
    """Ordena la traza por paciente y tiempo y añade columnas auxiliares:
    * `T_end_real`: instante real de salida de la estación (= T_in de la fila siguiente).
    * `T_hueco`: espera no registrada en T_wait (espera de Box).
    * `Sala_prev`: sala física que ocupaba el paciente en la fila anterior.
    * `Fin`: True si la fila está cortada por el final de la simulación."""
    d = df.sort_values(["Id_pat", "T_in", "T_out"], kind="mergesort").reset_index(drop=True).copy()
    g = d.groupby("Id_pat", sort=False)
    sig = g["T_in"].shift(-1)
    d["T_end_real"] = sig.fillna(d["T_out"])
    d["T_end_real"] = np.maximum(d["T_end_real"], d["T_out"])
    d["T_hueco"] = (d["T_end_real"] - d["T_out"]).clip(lower=0)
    d.loc[d["T_hueco"] < 1e-3, "T_hueco"] = 0.0
    d["Sala_prev"] = g["Wait_room"].shift(1).fillna("Sala_Espera_Urgencias")
    d["Fin"] = d["Station_out"].eq("Fin_Simulacion")
    d["Sin_UCI"] = d["ID_server"].eq("Sin_UCI_Disponible")
    return d


def horizonte(df):
    """Duración total (horas) de la traza, redondeada al día."""
    return math.ceil(df["T_out"].max() / 24) * 24


def tasa_llegadas_franja(df, dias=None):
    """Tasa de llegadas (pacientes/hora) y reparto por perfil en cada franja horaria."""
    adm = df.loc[df["Station_in"] == "Admision", ["Id_pat", "Pat_type", "T_in"]].drop_duplicates("Id_pat")
    if dias is None:
        dias = horizonte(df) / 24
    adm["Franja"] = (adm["T_in"].astype(int) % 24).map(franja_de_hora)
    horas_franja = {n: ((fin - ini) % 24) * dias for n, ini, fin in FRANJAS}
    tab = adm.groupby("Franja")["Id_pat"].count().rename("Llegadas").to_frame()
    tab["Horas"] = tab.index.map(horas_franja)
    tab["Tasa_obs (pac/h)"] = tab["Llegadas"] / tab["Horas"]
    tab["Tasa_teórica"] = tab.index.map(TASAS_TEORICAS)
    mix = pd.crosstab(adm["Franja"], adm["Pat_type"], normalize="index").reindex(columns=TIPOS)
    tab = tab.join(mix.add_prefix("%"))
    tab = tab.reindex([n for n, _, _ in FRANJAS])
    total = pd.Series({"Llegadas": len(adm), "Horas": 24 * dias,
                       "Tasa_obs (pac/h)": len(adm) / (24 * dias)}, name="Total")
    mix_tot = adm["Pat_type"].value_counts(normalize=True).reindex(TIPOS).add_prefix("%")
    total = pd.concat([total, mix_tot])
    return pd.concat([tab, pd.DataFrame([total], index=["Total"])])


def tiempos_servicio(df, cfg=None):
    """Estadísticos del tiempo de servicio (horas) por estación y perfil.
    Si se pasa `cfg`, añade el valor teórico de la configuración para comparar."""
    d = df.loc[~df["Station_out"].eq("Fin_Simulacion") & ~df["ID_server"].eq("Sin_UCI_Disponible")]
    tab = (d.groupby(["Station_in", "Pat_type"])["T_server"]
             .agg(n="count", media="mean", sd="std", minimo="min", maximo="max").reset_index())
    if cfg is not None:
        tab["teorico"] = [cfg.TIEMPOS_SERVICIO.get(e, {}).get(t, np.nan)
                          for e, t in zip(tab["Station_in"], tab["Pat_type"])]
        tab["media/teorico"] = tab["media"] / tab["teorico"]
    return tab


def tabla_tiempos_servicio(df):
    """Tabla pivote: media del tiempo de servicio (minutos) por estación (filas) y perfil."""
    t = tiempos_servicio(df)
    piv = t.pivot(index="Station_in", columns="Pat_type", values="media").reindex(columns=TIPOS) * 60
    return piv.round(2)


def probabilidades_derivacion(df, separar_sin_uci=True):
    """Matriz de probabilidades de derivación estimada (Station_in -> Station_out) por perfil.
    Excluye las filas cortadas por el final de la simulación. Si `separar_sin_uci`, los
    fallecimientos por falta de cama de UCI se muestran como destino 'Fallece_Sin_UCI'."""
    d = df.loc[~df["Station_out"].eq("Fin_Simulacion"), ["Pat_type", "Station_in", "Station_out", "ID_server"]].copy()
    if separar_sin_uci:
        d.loc[d["ID_server"].eq("Sin_UCI_Disponible"), "Station_out"] = "Fallece_Sin_UCI"
    cnt = d.groupby(["Pat_type", "Station_in", "Station_out"]).size().rename("n").reset_index()
    cnt["p"] = cnt["n"] / cnt.groupby(["Pat_type", "Station_in"])["n"].transform("sum")
    return cnt


def matriz_derivacion(df, tipo):
    """Matriz de transición (filas: origen, columnas: destino) para un perfil concreto."""
    p = probabilidades_derivacion(df)
    p = p[p["Pat_type"] == tipo]
    return p.pivot(index="Station_in", columns="Station_out", values="p").fillna(0).round(3)


def tabla_pacientes(d, param=PARAM):
    """Una fila por paciente con sus tiempos, desenlace, ruta y resultado económico.
    `d` debe venir de `preparar_traza`."""
    g = d.groupby("Id_pat", sort=False)
    pac = pd.DataFrame({
        "Pat_type": g["Pat_type"].first(),
        "T_llegada": g["T_in"].min(),
        "T_salida": g["T_out"].max(),
        "T_ser": g["T_server"].sum(),
        "Desenlace": g["Station_out"].last(),
    })
    pac["T_sys"] = pac["T_salida"] - pac["T_llegada"]
    pac["T_q"] = (pac["T_sys"] - pac["T_ser"]).clip(lower=0)
    pac["Ratio_espera"] = np.where(pac["T_sys"] > 0, pac["T_q"] / pac["T_sys"], 0.0)
    pac["Finalizado"] = pac["Desenlace"].isin(ESTADOS_ABS)

    # Rutas clínicas
    visita = lambda est: d.loc[d["Station_in"].eq(est)].groupby("Id_pat").size().reindex(pac.index).fillna(0) > 0
    pac["Planta"] = visita("Planta")
    pac["Quirofano"] = visita("Quirofano")
    pac["UCI"] = visita("UCI")
    pac["Sin_UCI"] = d.loc[d["Sin_UCI"]].groupby("Id_pat").size().reindex(pac.index).fillna(0) > 0
    # Origen de la derivación a UCI (estación previa a la primera petición de UCI)
    d_uci = d.assign(Prev=g["Station_in"].shift(1)).loc[d["Station_in"].eq("UCI")]
    prev = d_uci.groupby("Id_pat")["Prev"].first().reindex(pac.index)
    pac["UCI_desde"] = np.select(
        [prev.eq("Planta"), prev.eq("Quirofano"), prev.notna()],
        ["Planta", "Quirofano", "Urgencias"], default="")
    pac["Hospitalizado"] = pac["Planta"] | pac["UCI"] | pac["Quirofano"]

    # Latencia Admisión -> primer contacto en Diagnóstico Inicial
    di = d.loc[d["Station_in"].eq("Diag_Inicial") & ~d["Fin"]].groupby("Id_pat").first()
    contacto = di["T_end_real"] - di["T_server"]
    pac["Latencia_DI"] = (contacto.reindex(pac.index) - pac["T_llegada"])

    # Economía por paciente
    ok = d.loc[~d["Fin"]]
    coste = ok["Station_in"].map(param["COSTE_SERVICIO"]).fillna(0) \
        + ok["Station_in"].map(param["COSTE_DIA"]).fillna(0) * ok["T_server"] / 24
    pac["Coste_var"] = coste.groupby(ok["Id_pat"]).sum().reindex(pac.index).fillna(0)
    pac["Penalizacion"] = np.where(pac["Desenlace"].eq("Fallece"), param["PEN_FALLECE"], 0) \
        + np.where(pac["Sin_UCI"], param["PEN_SIN_UCI"], 0)
    pac["Ingreso"] = np.where(pac["Desenlace"].eq("Alta"), pac["Pat_type"].map(param["REEMBOLSO"]), 0)
    return pac


def ventana_pacientes(pac, t_fin, param=PARAM):
    """Selecciona los pacientes llegados en [warm-up, fin - cool-down] y que han finalizado."""
    t0, t1 = param["T_WARMUP"], t_fin - param["T_COOLDOWN"]
    sel = pac[(pac["T_llegada"] >= t0) & (pac["T_llegada"] < t1)]
    return sel[sel["Finalizado"]], sel, (t1 - t0) / 24


def kpis_operativos(pac_fin):
    """KPIs temporales (horas) globales y por perfil sobre pacientes finalizados."""
    def resumen(x):
        return pd.Series({
            "Pacientes": len(x),
            "T_sys_media": x["T_sys"].mean(), "T_sys_mediana": x["T_sys"].median(),
            "T_sys_P90": x["T_sys"].quantile(0.9),
            "T_ser_media": x["T_ser"].mean(), "T_q_media": x["T_q"].mean(),
            "T_q_P90": x["T_q"].quantile(0.9),
            "Ratio_espera_medio": x["Ratio_espera"].mean(),
            "Ratio_espera_agregado": x["T_q"].sum() / x["T_sys"].sum(),
        })
    tab = pac_fin.groupby("Pat_type").apply(resumen, include_groups=False).reindex(TIPOS)
    tab.loc["Global"] = resumen(pac_fin)
    return tab


def kpis_estaciones(d, t0, t1):
    """Carga y espera por estación (visitas con llegada en [t0, t1)): visitas, espera y servicio
    medios, carga total (h de servicio), espera total (h) y nº medio de pacientes en cola al llegar."""
    x = d[(d["T_in"] >= t0) & (d["T_in"] < t1) & ~d["Fin"]].copy()
    x["Espera"] = x["T_wait"] + x["T_hueco"]
    tab = x.groupby("Station_in").agg(
        Visitas=("Id_pat", "count"),
        Espera_media_h=("Espera", "mean"),
        Espera_P90_h=("Espera", lambda s: s.quantile(0.9)),
        Servicio_medio_h=("T_server", "mean"),
        Carga_total_h=("T_server", "sum"),
        Espera_total_h=("Espera", "sum"),
        Cola_media_llegada=("Pat_in_queue", "mean"),
        Cola_max_llegada=("Pat_in_queue", "max"),
    )
    tab["%Espera_total"] = tab["Espera_total_h"] / tab["Espera_total_h"].sum()
    return tab.sort_values("Espera_total_h", ascending=False)


def _clip_dur(ini, fin, t0, t1):
    """Duración del solape de los intervalos [ini, fin] con la ventana [t0, t1]."""
    return (np.minimum(fin, t1) - np.maximum(ini, t0)).clip(lower=0)


def capacidades(cfg):
    """Capacidad de cada recurso (servidores, camas, Box y bolsas de UCI) según `cfg`."""
    cap = dict(cfg.SERVIDORES_CAPACIDAD)
    cap.update(cfg.CAMAS_CAPACIDAD)
    cap["Box_Urgencias"] = cfg.BOX_CAPACIDAD
    if cfg.UCI_MODE == "pooled":
        cap["UCI"] = cfg.UCI_TOTAL_CAPACIDAD
    else:
        cap["UCI_Rojo"] = cfg.UCI_ROJO_CAPACIDAD
        cap["UCI_Naranja"] = cfg.UCI_NARANJA_CAPACIDAD
        cap["UCI_Base"] = cfg.UCI_BASE_CAPACIDAD
    return cap


def _nombre_sala(sala, tipo, cfg):
    """Traduce la sala física de la traza al nombre del recurso (bolsas de UCI)."""
    if cfg.UCI_MODE == "pooled":
        return sala
    bolsa = np.select([tipo.eq("Rojo"), tipo.eq("Naranja")], ["UCI_Rojo", "UCI_Naranja"], "UCI_Base")
    return np.where(sala.eq("UCI"), bolsa, sala)


def segmentos_ocupacion(d, cfg):
    """Segmentos de ocupación de recursos físicos (Box, Quirófano, Planta, UCI, Rehabilitación).
    Tipo 'Estancia': el paciente ocupa la cama/box en su estación.
    Tipo 'Bloqueo'  : el paciente ocupa la cama/box anterior esperando cama en otra unidad."""
    es_cama = d["Station_in"].isin(estaciones_cama(cfg)) & ~d["Fin"]
    # Estancia en la sala registrada en la fila (excluida la espera de cama, que se ocupa en la sala previa)
    # (en la fila de captura del Box, el hueco es la espera de Box, al principio de la fila)
    ini = np.where(d["Fin"], d["T_in"], d["T_in"] + np.where(es_cama, d["T_wait"], d["T_hueco"]))
    est = pd.DataFrame({"Recurso": _nombre_sala(d["Wait_room"], d["Pat_type"], cfg),
                        "Ini": ini, "Fin": d["T_end_real"].values, "Tipo": "Estancia",
                        "Pat_type": d["Pat_type"].values, "Id_pat": d["Id_pat"].values})
    # Bloqueo: espera de cama mientras se retiene la sala anterior
    b = d[es_cama & (d["T_wait"] > 0)]
    blo = pd.DataFrame({"Recurso": _nombre_sala(b["Sala_prev"], b["Pat_type"], cfg),
                        "Ini": b["T_in"].values, "Fin": (b["T_in"] + b["T_wait"]).values,
                        "Tipo": "Bloqueo", "Pat_type": b["Pat_type"].values, "Id_pat": b["Id_pat"].values})
    seg = pd.concat([est, blo], ignore_index=True)
    validos = set(capacidades(cfg))
    return seg[seg["Recurso"].isin(validos) & (seg["Fin"] > seg["Ini"])]


def kpis_recursos(d, cfg, t0, t1):
    """Utilización, cola media (Little), bed-blocking y diagnóstico de cuello de botella por recurso."""
    T = t1 - t0
    cap = capacidades(cfg)
    filas = {}
    # --- Servidores: ocupados durante T_server; espera en cola = T_wait
    srv = d[~d["Fin"] & d["ID_server"].isin(cfg.SERVIDORES_CAPACIDAD.keys())]
    fin_s = srv["T_end_real"]; ini_s = fin_s - srv["T_server"]
    ocup = _clip_dur(ini_s, fin_s, t0, t1).groupby(srv["ID_server"]).sum()
    cola = _clip_dur(ini_s - srv["T_wait"], ini_s, t0, t1).groupby(srv["ID_server"]).sum()
    for r in cfg.SERVIDORES_CAPACIDAD:
        filas[r] = {"Tipo": "Servidor", "Capacidad": cap[r], "Ocupacion_h": ocup.get(r, 0.0),
                    "Cola_h": cola.get(r, 0.0), "Bloqueo_h": 0.0}
    # --- Camas / Box / UCI
    seg = segmentos_ocupacion(d, cfg)
    dur = _clip_dur(seg["Ini"], seg["Fin"], t0, t1)
    ocup_c = dur.groupby(seg["Recurso"]).sum()
    bloq_c = dur[seg["Tipo"].eq("Bloqueo")].groupby(seg["Recurso"]).sum()
    # Espera en cola de camas: T_wait en estaciones cama (+ hueco del Box)
    ec = d[d["Station_in"].isin(estaciones_cama(cfg))]
    rec_ec = pd.Series(_nombre_sala(ec["Station_in"].replace(getattr(cfg, "CAMA_ALIAS", {})), ec["Pat_type"], cfg), index=ec.index)
    cola_c = _clip_dur(ec["T_in"], ec["T_in"] + ec["T_wait"], t0, t1).groupby(rec_ec).sum()
    hb = d[d["T_hueco"] > 0]
    cola_c.loc["Box_Urgencias"] = _clip_dur(hb["T_in"], hb["T_in"] + hb["T_hueco"], t0, t1).sum()
    for r in cap:
        if r in cfg.SERVIDORES_CAPACIDAD:
            continue
        filas[r] = {"Tipo": "Cama", "Capacidad": cap[r], "Ocupacion_h": ocup_c.get(r, 0.0),
                    "Cola_h": cola_c.get(r, 0.0), "Bloqueo_h": bloq_c.get(r, 0.0)}
    tab = pd.DataFrame(filas).T
    for c in ["Capacidad", "Ocupacion_h", "Cola_h", "Bloqueo_h"]:
        tab[c] = tab[c].astype(float)
    tab["Utilizacion"] = tab["Ocupacion_h"] / (tab["Capacidad"] * T)
    tab["Lq_media"] = tab["Cola_h"] / T                  # Ley de Little: nº medio en cola
    tab["L_media"] = tab["Ocupacion_h"] / T + tab["Lq_media"]
    tab["%Bloqueo"] = np.where(tab["Ocupacion_h"] > 0, tab["Bloqueo_h"] / tab["Ocupacion_h"], 0)
    tab["Diagnostico"] = np.select([tab["Utilizacion"] >= 1, tab["Utilizacion"] >= 0.85],
                                   ["INESTABLE (ρ≥1)", "Alta congestión"], "OK")
    return tab


def ocupacion_diaria(seg, recursos, capacidad, t0, t1):
    """Ocupación media diaria (fracción) del conjunto de `recursos` con capacidad total `capacidad`."""
    s = seg[seg["Recurso"].isin(recursos)]
    ini, fin = s["Ini"].values, s["Fin"].values
    dias = np.arange(t0, t1, 24.0)
    occ = [np.clip(np.minimum(fin, a + 24) - np.maximum(ini, a), 0, None).sum() / (24 * capacidad) for a in dias]
    return pd.Series(occ, index=(dias / 24).astype(int), name="Ocupacion")


def kpis_clinicos(pac_fin, dias_ventana):
    """Mortalidad, altas, derivación a UCI, hospitalización y muertes Sin_UCI (global y por perfil)."""
    def resumen(x):
        n = len(x)
        return pd.Series({
            "Pacientes": n,
            "Tasa_alta": (x["Desenlace"] == "Alta").mean(),
            "Tasa_mortalidad": (x["Desenlace"] == "Fallece").mean(),
            "Tasa_mort_Sin_UCI": x["Sin_UCI"].mean(),
            "Altas_dia": (x["Desenlace"] == "Alta").sum() / dias_ventana,
            "Tasa_hospitalizacion": x["Hospitalizado"].mean(),
            "Tasa_UCI": x["UCI"].mean(),
            "Tasa_UCI_desde_Urgencias": (x["UCI_desde"] == "Urgencias").mean(),
            "Tasa_UCI_desde_Quirofano": (x["UCI_desde"] == "Quirofano").mean(),
            "Tasa_UCI_desde_Planta": (x["UCI_desde"] == "Planta").mean(),
            "Tasa_Quirofano": x["Quirofano"].mean(),
        })
    tab = pac_fin.groupby("Pat_type").apply(resumen, include_groups=False).reindex(TIPOS)
    tab.loc["Global"] = resumen(pac_fin)
    return tab


def coste_fijo_periodo(cfg, dias_ventana):
    """Coste fijo del escenario imputable a la ventana analizada: los costes anuales se
    prorratean (días/365) y los costes únicos se imputan íntegramente (criterio prudente)."""
    total = 0.0
    for k, v in getattr(cfg, "COSTES_FIJOS", {}).items():
        total += v * dias_ventana / 365 if "anual" in k else v
    return total


def kpis_economicos(pac_fin, cfg, dias_ventana, param=PARAM):
    """Ingresos, costes, margen operativo bruto y neto (tras coste fijo) y margen exigido."""
    def resumen(x):
        ing, cv, pen = x["Ingreso"].sum(), x["Coste_var"].sum(), x["Penalizacion"].sum()
        return pd.Series({"Pacientes": len(x), "Ingresos": ing, "Coste_variable": cv,
                          "Penalizaciones": pen, "Coste_total": cv + pen,
                          "Coste_medio_paciente": (cv + pen) / max(len(x), 1),
                          "Ingreso_medio_paciente": ing / max(len(x), 1),
                          "Beneficio": ing - cv - pen,
                          "Margen": (ing - cv - pen) / ing if ing > 0 else np.nan})
    tab = pac_fin.groupby("Pat_type").apply(resumen, include_groups=False).reindex(TIPOS)
    tab.loc["Global"] = resumen(pac_fin)
    g = tab.loc["Global"]
    cf = coste_fijo_periodo(cfg, dias_ventana)
    tasa_alta = (pac_fin["Desenlace"] == "Alta").mean()
    glob = {
        "Ingresos_dia": g["Ingresos"] / dias_ventana,
        "Coste_total_dia": g["Coste_total"] / dias_ventana,
        "Coste_medio_paciente": g["Coste_medio_paciente"],
        "Beneficio_operativo_dia": g["Beneficio"] / dias_ventana,
        "Margen_operativo": g["Margen"],
        "Coste_fijo_periodo": cf,
        "Margen_neto": (g["Beneficio"] - cf) / g["Ingresos"],
        "Beneficio_neto_dia": (g["Beneficio"] - cf) / dias_ventana,
        "Margen_exigido": margen_exigido(tasa_alta, param),
    }
    return tab, glob


def evaluar_slas(pac_fin, seg, cfg, t0, t1, econ_glob, param=PARAM):
    """Evalúa los cinco SLAs de la Sección A.08. Devuelve un DataFrame con valor, umbral y cumplimiento."""
    filas = []
    # 1) Latencia Admisión -> Diag_Inicial (Naranja/Rojo)
    nr = pac_fin[pac_fin["Pat_type"].isin(["Naranja", "Rojo"])]
    lat = nr["Latencia_DI"].dropna()
    pct = (lat <= param["SLA_LATENCIA_H"]).mean()
    filas.append(("SLA1 Latencia DI ≤15 min (Naranja/Rojo)", "% pacientes ≤15 min", pct,
                  f"≥ {param['SLA_LATENCIA_CUMPL']:.0%}", pct >= param["SLA_LATENCIA_CUMPL"]))
    for t in ["Naranja", "Rojo"]:
        l = nr.loc[nr["Pat_type"] == t, "Latencia_DI"].dropna()
        filas.append((f"   · {t}: latencia media (min) / % ≤15 min", "min", l.mean() * 60,
                      f"{(l <= param['SLA_LATENCIA_H']).mean():.1%}", np.nan))
    # 2) Ratio de espera < 25% (Verde y Amarillo)
    for t in ["Verde", "Amarillo"]:
        x = pac_fin[pac_fin["Pat_type"] == t]
        r = x["Ratio_espera"].mean()
        filas.append((f"SLA2 Ratio espera < 25% ({t})", "T_q/T_sys medio", r,
                      f"< {param['SLA_RATIO_ESPERA']:.0%}", r < param["SLA_RATIO_ESPERA"]))
    # 3) Ocupación estructural de UCI ≤ 90%
    rec_uci = [r for r in capacidades(cfg) if r.startswith("UCI")]
    occ = ocupacion_diaria(seg, rec_uci, cfg.UCI_TOTAL_CAPACIDAD, t0, t1)
    filas.append(("SLA3 Ocupación media UCI ≤ 90%", "ocupación media", occ.mean(),
                  f"≤ {param['SLA_OCUP_UCI']:.0%}", occ.mean() <= param["SLA_OCUP_UCI"]))
    filas.append(("   · % días con ocupación UCI > 90%", "% días", (occ > param["SLA_OCUP_UCI"]).mean(), "info", np.nan))
    # 4) Mortalidad evitable Sin_UCI < 2% de Naranja/Rojo
    m = nr["Sin_UCI"].mean()
    filas.append(("SLA4 Muertes Sin_UCI < 2% (Naranja/Rojo)", "tasa", m,
                  f"< {param['SLA_SIN_UCI']:.0%}", m < param["SLA_SIN_UCI"]))
    # 5) Margen operativo neto ≥ margen exigido
    filas.append(("SLA5 Margen neto ≥ margen exigido", "margen neto", econ_glob["Margen_neto"],
                  f"≥ {econ_glob['Margen_exigido']:.1%}", econ_glob["Margen_neto"] >= econ_glob["Margen_exigido"]))
    return pd.DataFrame(filas, columns=["SLA", "Métrica", "Valor", "Umbral", "Cumple"]).set_index("SLA")


def calcular_kpis(df, cfg, dias=None, param=PARAM):
    """Calcula TODOS los KPIs y SLAs de una traza (histórica o simulada).
    Devuelve un diccionario con las tablas y un resumen plano `resumen` (una cifra por KPI),
    pensado para acumular réplicas en el análisis Monte Carlo."""
    d = preparar_traza(df)
    t_fin = dias * 24 if dias is not None else horizonte(df)
    # Ventana de análisis común: [warm-up, fin - cool-down]
    t0, t1 = param["T_WARMUP"], t_fin - param["T_COOLDOWN"]
    pac = tabla_pacientes(d, param)
    pac_fin, pac_vent, dias_v = ventana_pacientes(pac, t_fin, param)
    op = kpis_operativos(pac_fin)
    est = kpis_estaciones(d, t0, t1)
    rec = kpis_recursos(d, cfg, t0, t1)
    seg = segmentos_ocupacion(d, cfg)
    cli = kpis_clinicos(pac_fin, dias_v)
    eco_tab, eco = kpis_economicos(pac_fin, cfg, dias_v, param)
    sla = evaluar_slas(pac_fin, seg, cfg, t0, t1, eco, param)

    # Resumen plano
    res = {"Pacientes_ventana": len(pac_vent), "Pacientes_finalizados": len(pac_fin),
           "Llegadas_dia": len(pac_vent) / dias_v}
    for perfil in TIPOS + ["Global"]:
        for k in ["T_sys_media", "T_ser_media", "T_q_media", "Ratio_espera_medio"]:
            res[f"{k}_{perfil}"] = op.loc[perfil, k]
        for k in ["Tasa_alta", "Tasa_mortalidad", "Tasa_mort_Sin_UCI", "Tasa_hospitalizacion", "Tasa_UCI"]:
            res[f"{k}_{perfil}"] = cli.loc[perfil, k]
    for k in ["Altas_dia", "Tasa_UCI_desde_Urgencias", "Tasa_UCI_desde_Quirofano", "Tasa_UCI_desde_Planta", "Tasa_Quirofano"]:
        res[k] = cli.loc["Global", k]
    for r in rec.index:
        res[f"Util_{r}"] = rec.loc[r, "Utilizacion"]
        res[f"Lq_{r}"] = rec.loc[r, "Lq_media"]
        if rec.loc[r, "Tipo"] == "Cama":
            res[f"Bloqueo_h_dia_{r}"] = rec.loc[r, "Bloqueo_h"] / ((t1 - t0) / 24)
    res.update(eco)
    nr = pac_fin["Pat_type"].isin(["Naranja", "Rojo"])
    res["Latencia_DI_media_NR_min"] = pac_fin.loc[nr, "Latencia_DI"].mean() * 60
    res["Mort_Sin_UCI_NR"] = pac_fin.loc[nr, "Sin_UCI"].mean()
    claves_sla = {"SLA1": "SLA1 Latencia", "SLA2_Verde": "SLA2 Ratio espera < 25% (Verde)",
                  "SLA2_Amarillo": "SLA2 Ratio espera < 25% (Amarillo)", "SLA3": "SLA3",
                  "SLA4": "SLA4", "SLA5": "SLA5"}
    for k, pref in claves_sla.items():
        fila = sla.loc[sla.index.str.startswith(pref)].iloc[0]
        res[f"SLA_valor_{k}"] = fila["Valor"]
        res[f"Cumple_{k}"] = bool(fila["Cumple"])
    res["Ocup_media_UCI"] = res["SLA_valor_SLA3"]
    # Acceso a UCI por perfil (espera de cama y fallos por falta de cama sobre las peticiones)
    uci = d[d["Station_in"].eq("UCI") & (d["T_in"] >= t0) & (d["T_in"] < t1)]
    for perfil in ["Amarillo", "Naranja", "Rojo"]:
        u = uci[uci["Pat_type"] == perfil]
        res[f"Peticiones_UCI_dia_{perfil}"] = len(u) / dias_v
        res[f"Espera_UCI_media_{perfil}"] = u["T_wait"].mean() if len(u) else np.nan
        res[f"Espera_UCI_P95_{perfil}"] = u["T_wait"].quantile(0.95) if len(u) else np.nan
        res[f"Pct_peticiones_UCI_fallidas_{perfil}"] = u["Sin_UCI"].mean() if len(u) else np.nan
    res["Pct_dias_UCI_90"] = sla.loc["   · % días con ocupación UCI > 90%", "Valor"]

    return {"traza": d, "pacientes": pac, "pacientes_fin": pac_fin, "operativos": op,
            "estaciones": est, "recursos": rec, "segmentos": seg, "clinicos": cli,
            "economicos": eco_tab, "economicos_glob": eco, "slas": sla, "resumen": res,
            "dias_ventana": dias_v, "t0": t0, "t1": t1, "t_fin": t_fin}


TABLAS_MC = ["operativos", "clinicos", "estaciones", "recursos", "economicos"]


def ejecutar_replica(cfg, dias, semilla, param=None, guardar_detalle=False):
    """Simula una réplica y calcula sus KPIs. Devuelve (resumen, tablas, detalle)."""
    param = PARAM if param is None else param
    df_sim = simular(cfg, dias=dias, semilla=semilla)
    k = calcular_kpis(df_sim, cfg, dias=dias, param=param)
    tablas = {t: k[t] for t in TABLAS_MC}
    detalle = None
    if guardar_detalle:
        rec_uci = [r for r in capacidades(cfg) if r.startswith("UCI")]
        detalle = {
            "pacientes_fin": k["pacientes_fin"][["Pat_type", "T_sys", "T_ser", "T_q", "Ratio_espera", "Desenlace",
                                                 "Latencia_DI"]].copy(),
            "ocupacion_uci": ocupacion_diaria(k["segmentos"], rec_uci, cfg.UCI_TOTAL_CAPACIDAD, k["t0"], k["t1"]),
            "ocupacion_uci_bolsas": {r: ocupacion_diaria(k["segmentos"], [r], capacidades(cfg)[r], k["t0"], k["t1"])
                                     for r in rec_uci},
        }
    return dict(k["resumen"]), tablas, detalle


def _replica_worker(args):
    cfg, dias, semilla, param, i = args
    res, tablas, det = ejecutar_replica(cfg, dias, semilla, param, guardar_detalle=(i == 0))
    return i, res, tablas, det


def montecarlo(cfg, n_sim=50, dias=180, semilla_base=1000, param=None, progreso=None,
               paralelo=False, n_procesos=None, verbose=False):
    """Ejecuta n_sim réplicas de `cfg` (réplica i -> semilla semilla_base + i).
    `progreso(i_completadas, n_sim)` se llama tras cada réplica (barra de progreso de la app).
    Con `paralelo=True` reparte las réplicas entre procesos; si falla, continúa en serie."""
    param = PARAM if param is None else param
    resultados = {}
    args = [(cfg, dias, semilla_base + i, param, i) for i in range(n_sim)]
    hecho = False
    if sys.platform == "emscripten":       # navegador (Pyodide): sin multiproceso
        paralelo = False
    if paralelo and n_sim > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=n_procesos) as ex:
                futuros = [ex.submit(_replica_worker, a) for a in args]
                for f in as_completed(futuros):
                    i, res, tablas, det = f.result()
                    resultados[i] = (res, tablas, det)
                    if progreso:
                        progreso(len(resultados), n_sim)
            hecho = True
        except Exception as e:          # p. ej. entornos que no permiten procesos
            if verbose:
                print("Ejecución paralela no disponible, se continúa en serie:", e)
            resultados = {}
    if not hecho:
        t_ini = time.time()
        for a in args:
            i, res, tablas, det = _replica_worker(a)
            resultados[i] = (res, tablas, det)
            if progreso:
                progreso(len(resultados), n_sim)
            if verbose and ((i + 1) % 10 == 0 or i + 1 == n_sim):
                print(f"  [{cfg.nombre}] réplica {i + 1}/{n_sim} · {time.time() - t_ini:.0f} s")
    filas, tablas = [], {t: [] for t in TABLAS_MC}
    for i in sorted(resultados):
        res, tb, _ = resultados[i]
        r = dict(res); r["Replica"] = i
        filas.append(r)
        for t in TABLAS_MC:
            tablas[t].append(tb[t].assign(Replica=i))
    return {"cfg": cfg, "nombre": cfg.nombre, "dias": dias, "n_sim": n_sim, "semilla_base": semilla_base,
            "resumen": pd.DataFrame(filas).set_index("Replica"),
            "detalle_replica0": resultados[0][2],
            **{t: pd.concat(v) for t, v in tablas.items()}}


def resumen_mc(mc, kpis=None, alpha=0.05):
    """Media, sd, IC al 95% de la media y percentiles 2.5/97.5 de cada KPI escalar."""
    r = mc["resumen"]
    cols = kpis if kpis is not None else [c for c in r.columns if not c.startswith("Cumple_")]
    r = r[cols].astype(float)
    n = r.notna().sum()
    tq = stats.t.ppf(1 - alpha / 2, np.maximum(n - 1, 1))
    se = r.std() / np.sqrt(n)
    return pd.DataFrame({"Media": r.mean(), "SD": r.std(),
                         "IC95_inf": r.mean() - tq * se, "IC95_sup": r.mean() + tq * se,
                         "P2.5": r.quantile(0.025), "P97.5": r.quantile(0.975)})


def tabla_media_mc(mc, tabla):
    """Promedio entre réplicas de una tabla de KPIs (operativos, clinicos, estaciones, recursos...)."""
    t = mc[tabla]
    num = t.select_dtypes("number").drop(columns="Replica")
    media = num.groupby(level=0, sort=False).mean()
    if tabla == "recursos":
        media.insert(0, "Tipo", t.groupby(level=0, sort=False)["Tipo"].first())
        media["Diagnostico"] = np.select([media["Utilizacion"] >= 1, media["Utilizacion"] >= 0.85],
                                         ["INESTABLE (ρ≥1)", "Alta congestión"], "OK")
        media["%Réplicas_ρ≥0.85"] = (t["Utilizacion"] >= 0.85).groupby(level=0, sort=False).mean()
    if tabla in ("operativos", "clinicos", "economicos"):
        media = media.reindex(TIPOS + ["Global"])
    return media


def slas_mc(mc):
    """Para cada SLA: valor medio de la métrica, IC95 y % de réplicas que lo cumplen."""
    r = mc["resumen"]
    nombres = {"SLA1": "Latencia DI ≤15 min (N/R): % pacientes que cumplen (≥95%)",
               "SLA2_Verde": "Ratio espera Verde (<25%)",
               "SLA2_Amarillo": "Ratio espera Amarillo (<25%)",
               "SLA3": "Ocupación media UCI (≤90%)",
               "SLA4": "Mortalidad Sin_UCI en N/R (<2%)",
               "SLA5": "Margen neto ≥ margen exigido"}
    filas = []
    for k, txt in nombres.items():
        v = r[f"SLA_valor_{k}"].astype(float)
        se = v.std() / np.sqrt(len(v))
        filas.append({"SLA": k, "Descripción": txt, "Valor_medio": v.mean(),
                      "IC95_inf": v.mean() - 1.96 * se, "IC95_sup": v.mean() + 1.96 * se,
                      "%Réplicas_cumple": r[f"Cumple_{k}"].astype(float).mean()})
    out = pd.DataFrame(filas).set_index("SLA")
    out["Veredicto"] = np.select([out["%Réplicas_cumple"] >= 0.95, out["%Réplicas_cumple"] <= 0.05],
                                 ["Cumple", "No cumple"], "Cumplimiento incierto")
    return out


def comparar_escenarios(mc_base, mc_esc, kpis, alpha=0.05):
    """Comparación pareada (misma semilla por réplica) de los KPIs del escenario frente a la base."""
    b = mc_base["resumen"][kpis].astype(float)
    e = mc_esc["resumen"][kpis].astype(float)
    comunes = b.index.intersection(e.index)
    b, e = b.loc[comunes], e.loc[comunes]
    dif = e - b
    n = dif.notna().sum()
    tq = stats.t.ppf(1 - alpha / 2, np.maximum(n - 1, 1))
    se = dif.std() / np.sqrt(n)
    pval = {}
    for k in kpis:
        x = dif[k].dropna()
        pval[k] = stats.ttest_1samp(x, 0).pvalue if len(x) > 1 and x.std() > 0 else np.nan
    out = pd.DataFrame({"Base": b.mean(), "Escenario": e.mean(), "Δ": dif.mean(),
                        "Δ%": dif.mean() / b.mean().abs().replace(0, np.nan),
                        "IC95_Δ_inf": dif.mean() - tq * se, "IC95_Δ_sup": dif.mean() + tq * se,
                        "p_valor": pd.Series(pval)})
    out["Significativo"] = out["p_valor"] < alpha
    return out


def viabilidad_economica(mc_base, mc_esc, alpha=0.05):
    """Ahorro/ingreso adicional del escenario frente a la base y su relación con el coste fijo."""
    cf = getattr(mc_esc["cfg"], "COSTES_FIJOS", {})
    anual = sum(v for k, v in cf.items() if "anual" in k)
    unico = sum(v for k, v in cf.items() if "anual" not in k)
    b, e = mc_base["resumen"], mc_esc["resumen"]
    idx = b.index.intersection(e.index)
    d_dia = (e.loc[idx, "Beneficio_operativo_dia"] - b.loc[idx, "Beneficio_operativo_dia"]).astype(float)
    m, se = d_dia.mean(), d_dia.std() / np.sqrt(len(d_dia))
    tq = stats.t.ppf(1 - alpha / 2, len(d_dia) - 1)
    res = {
        "Δ beneficio operativo (€/día, antes de coste fijo)": m,
        "IC95 inf (€/día)": m - tq * se, "IC95 sup (€/día)": m + tq * se,
        "Δ beneficio operativo anualizado (€/año)": m * 365,
        "Coste fijo anual (€/año)": anual,
        "Coste fijo único (€)": unico,
        "Resultado neto anual (€/año)": m * 365 - anual,
        "Cobertura del coste anual (Δ/coste)": (m * 365 / anual) if anual else np.nan,
        "Periodo de amortización del coste único (días)": (unico / m) if (unico and m > 0) else np.nan,
        "Margen operativo base": b["Margen_operativo"].mean(),
        "Margen operativo escenario (bruto)": e["Margen_operativo"].mean(),
        "Margen neto escenario (tras coste fijo)": e["Margen_neto"].mean(),
        "Margen exigido medio escenario": e["Margen_exigido"].mean(),
        "% réplicas margen neto ≥ exigido": e["Cumple_SLA5"].astype(float).mean(),
        "% réplicas margen neto ≥ margen base": (e.loc[idx, "Margen_neto"] >= b.loc[idx, "Margen_neto"]).mean(),
    }
    return pd.Series(res).to_frame("Valor")


KPIS_CLAVE = (
    ["Llegadas_dia"]
    + [f"T_sys_media_{t}" for t in TIPOS + ["Global"]]
    + [f"T_q_media_{t}" for t in TIPOS + ["Global"]]
    + [f"Ratio_espera_medio_{t}" for t in TIPOS]
    + ["Latencia_DI_media_NR_min", "SLA_valor_SLA1"]
    + [f"Tasa_mortalidad_{t}" for t in TIPOS + ["Global"]]
    + ["Tasa_alta_Global", "Tasa_mort_Sin_UCI_Global", "Mort_Sin_UCI_NR",
       "Tasa_hospitalizacion_Global", "Tasa_UCI_Global", "Tasa_UCI_desde_Urgencias",
       "Tasa_UCI_desde_Quirofano", "Tasa_UCI_desde_Planta"]
    + ["Util_Medicos", "Util_Pruebas_Avanzadas", "Util_Box_Urgencias", "Util_Quirofano",
       "Util_Planta", "Util_Rehabilitacion", "Ocup_media_UCI", "Pct_dias_UCI_90",
       "Bloqueo_h_dia_Box_Urgencias", "Bloqueo_h_dia_Quirofano", "Bloqueo_h_dia_Planta"]
    + ["Coste_medio_paciente", "Ingresos_dia", "Beneficio_operativo_dia", "Margen_operativo",
       "Margen_neto", "Margen_exigido"]
)


def sentido_kpi(k):
    if k.startswith(("Tasa_alta", "SLA_valor_SLA1", "Margen", "Beneficio", "Ingresos", "Altas_dia")):
        return +1
    if k.startswith(("T_sys", "T_q", "Ratio_espera", "Latencia", "Tasa_mortalidad", "Tasa_mort_Sin_UCI",
                     "Mort_Sin_UCI", "Coste", "Bloqueo", "Espera_UCI", "Pct_peticiones_UCI_fallidas",
                     "Pct_dias_UCI_90", "Lq_")):
        return -1
    return 0


ETIQUETAS_KPI = {
    "T_sys_media_Verde": "T. medio en el sistema · Verde (h)",
    "T_sys_media_Amarillo": "T. medio en el sistema · Amarillo (h)",
    "T_sys_media_Naranja": "T. medio en el sistema · Naranja (h)",
    "T_sys_media_Rojo": "T. medio en el sistema · Rojo (h)",
    "T_sys_media_Global": "T. medio en el sistema · Global (h)",
    "T_q_media_Global": "T. medio en cola · Global (h)",
    "Latencia_DI_media_NR_min": "Latencia hasta Diag. Inicial N/R (min)",
    "SLA_valor_SLA1": "% N/R atendidos en ≤15 min",
    "Tasa_mortalidad_Verde": "Mortalidad · Verde",
    "Tasa_mortalidad_Amarillo": "Mortalidad · Amarillo",
    "Tasa_mortalidad_Naranja": "Mortalidad · Naranja",
    "Tasa_mortalidad_Rojo": "Mortalidad · Rojo",
    "Tasa_mortalidad_Global": "Mortalidad · Global",
    "Tasa_mort_Sin_UCI_Global": "Mortalidad Sin_UCI (s/ total pacientes)",
    "Mort_Sin_UCI_NR": "Mortalidad Sin_UCI (s/ Naranja+Rojo)",
    "Tasa_alta_Global": "Tasa de altas",
    "Ocup_media_UCI": "Ocupación media UCI",
    "Util_Quirofano": "Utilización Quirófano",
    "Util_Planta": "Utilización Planta",
    "Util_Box_Urgencias": "Utilización Box",
    "Util_Medicos": "Utilización Médicos",
    "Coste_medio_paciente": "Coste medio por paciente (€)",
    "Beneficio_operativo_dia": "Beneficio operativo (€/día)",
    "Margen_operativo": "Margen operativo bruto",
    "Margen_neto": "Margen operativo neto (tras coste fijo)",
    "Margen_exigido": "Margen exigido por el convenio",
}


KPIS_SINTESIS = list(ETIQUETAS_KPI)


def tabla_sintesis(protocolos, kpis=KPIS_SINTESIS, ref="Línea base"):
    """Tabla única: media de cada KPI por protocolo y marca de cambio significativo frente a la base
    (▲ mejora / ▼ empeora, comparación pareada al 5%)."""
    base = protocolos[ref]
    medias = pd.DataFrame({et: m["resumen"][kpis].astype(float).mean() for et, m in protocolos.items()})
    marcas = pd.DataFrame("", index=kpis, columns=medias.columns)
    for et, m in protocolos.items():
        if et == ref:
            continue
        c = comparar_escenarios(base, m, kpis)
        for k in kpis:
            s = sentido_kpi(k)
            if c.loc[k, "Significativo"] and s != 0:
                marcas.loc[k, et] = "▲" if np.sign(c.loc[k, "Δ"]) == s else "▼"
    return medias, marcas


def resultado_economico(mc_base, mc):
    """Resultado económico anualizado del protocolo frente a la línea base."""
    cf = getattr(mc["cfg"], "COSTES_FIJOS", {})
    anual = sum(v for k, v in cf.items() if "anual" in k)
    unico = sum(v for k, v in cf.items() if "anual" not in k)
    idx = mc_base["resumen"].index.intersection(mc["resumen"].index)
    d = (mc["resumen"].loc[idx, "Beneficio_operativo_dia"] - mc_base["resumen"].loc[idx, "Beneficio_operativo_dia"]).astype(float)
    se = d.std() / np.sqrt(len(d)) if len(d) > 1 else np.nan
    return {"Δ beneficio operativo anual (€)": d.mean() * 365,
            "IC95 inf (€)": (d.mean() - 1.96 * se) * 365, "IC95 sup (€)": (d.mean() + 1.96 * se) * 365,
            "Coste fijo anual (€)": anual, "Coste fijo único (€)": unico,
            "Resultado neto 1er año (€)": d.mean() * 365 - anual - unico,
            "Resultado neto años sucesivos (€/año)": d.mean() * 365 - anual}


def tabla_economica(protocolos, ref="Línea base"):
    filas = {et: resultado_economico(protocolos[ref], m) for et, m in protocolos.items() if et != ref}
    return pd.DataFrame(filas).T


def tabla_slas(protocolos):
    """% de réplicas que cumplen cada SLA, por protocolo."""
    return pd.DataFrame({et: slas_mc(m)["%Réplicas_cumple"] for et, m in protocolos.items()})


def num(x, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, d=1):
    return "—" if x is None or pd.isna(x) else num(100 * x, d) + " %"


def eur(x, d=0):
    return "—" if x is None or pd.isna(x) else num(x, d) + " €"


def fmt_kpi(k, v):
    """Formatea un valor según el tipo de KPI."""
    if pd.isna(v):
        return "—"
    if k.startswith(("Tasa_", "Util_", "Ratio_", "Margen", "Ocup", "Pct_", "Mort_", "SLA_valor_SLA1", "Cumple", "%")):
        return pct(v, 2 if k.startswith(("Tasa_mort_Sin", "Mort_Sin")) else 1)
    if any(t in k for t in ("Coste", "Ingreso", "Beneficio", "€", "Resultado")):
        return eur(v)
    return num(v, 2)


def ranking_protocolos(protocolos, ref):
    """Ordena los protocolos: primero los que no empeoran la mortalidad, luego por nº de SLAs
    cumplidos y, a igualdad, por resultado económico neto anual frente a la referencia."""
    base = protocolos[ref]
    filas = []
    for et, m in protocolos.items():
        s = slas_mc(m)
        fila = {"Protocolo": et, "SLAs cumplidos": int((s["Veredicto"] == "Cumple").sum()),
                "SLAs cumplidos (lista)": ", ".join(s.index[s["Veredicto"] == "Cumple"]),
                "Mortalidad global": m["resumen"]["Tasa_mortalidad_Global"].mean(),
                "Mortalidad Sin_UCI N/R": m["resumen"]["Mort_Sin_UCI_NR"].mean(),
                "T_sys global (h)": m["resumen"]["T_sys_media_Global"].mean(),
                "Margen neto": m["resumen"]["Margen_neto"].mean()}
        if et == ref:
            fila.update({"Resultado neto anual (€)": 0.0, "Resultado neto 1er año (€)": 0.0, "Descartado": False})
        else:
            e = resultado_economico(base, m)
            c = comparar_escenarios(base, m, ["Tasa_mortalidad_Global", "Mort_Sin_UCI_NR"])
            peor = any(c.loc[k, "Significativo"] and c.loc[k, "Δ"] > 0 for k in c.index)
            fila.update({"Resultado neto anual (€)": e["Resultado neto años sucesivos (€/año)"],
                         "Resultado neto 1er año (€)": e["Resultado neto 1er año (€)"], "Descartado": peor})
        filas.append(fila)
    r = pd.DataFrame(filas).set_index("Protocolo")
    return r.sort_values(["Descartado", "SLAs cumplidos", "Resultado neto anual (€)"], ascending=[True, False, False])


def _cambios_significativos(mc_base, mc, kpis):
    """Listas de textos con las mejoras y empeoramientos significativos de un protocolo."""
    c = comparar_escenarios(mc_base, mc, kpis)
    mejoras, empeora = [], []
    for k in kpis:
        s = sentido_kpi(k)
        if s == 0 or not c.loc[k, "Significativo"]:
            continue
        txt = f"{ETIQUETAS_KPI.get(k, k)} ({fmt_kpi(k, c.loc[k, 'Base'])} → {fmt_kpi(k, c.loc[k, 'Escenario'])})"
        (mejoras if np.sign(c.loc[k, "Δ"]) == s else empeora).append(txt)
    return mejoras, empeora


def texto_escenario(et, mc_base, mc, ref="la referencia", kpis=None):
    """Párrafo descriptivo de un escenario frente a la referencia."""
    kpis = KPIS_SINTESIS if kpis is None else kpis
    mejoras, empeora = _cambios_significativos(mc_base, mc, kpis)
    e = resultado_economico(mc_base, mc)
    s = slas_mc(mc)
    t = f"{et}: "
    t += ("mejora de forma significativa " + "; ".join(mejoras) + ". ") if mejoras else \
        "no produce mejoras estadísticamente significativas en los KPIs principales. "
    if empeora:
        t += "Empeora " + "; ".join(empeora) + ". "
    signif = e["IC95 inf (€)"] > 0 or e["IC95 sup (€)"] < 0
    t += (f"Frente a {ref} genera {eur(e['Δ beneficio operativo anual (€)'])} de beneficio operativo "
          f"adicional al año (IC95: {eur(e['IC95 inf (€)'])} a {eur(e['IC95 sup (€)'])}"
          + ("" if signif else "; diferencia no significativa") + ")")
    if e["Coste fijo anual (€)"]:
        t += (f", que debe cubrir un coste fijo de {eur(e['Coste fijo anual (€)'])}/año: resultado neto de "
              f"{eur(e['Resultado neto años sucesivos (€/año)'])}/año")
        if e["Coste fijo único (€)"]:
            t += f" (más un coste único de {eur(e['Coste fijo único (€)'])})"
        t += ". "
    elif e["Coste fijo único (€)"]:
        dif = e["Δ beneficio operativo anual (€)"]
        dias = e["Coste fijo único (€)"] / (dif / 365) if dif > 0 else np.nan
        t += (f", con un coste único de {eur(e['Coste fijo único (€)'])}"
              + (f", que se amortiza en unos {num(dias, 0)} días. " if not pd.isna(dias) else ", que no llega a amortizarse. "))
    else:
        t += ". "
    t += f"Cumple {int((s['Veredicto'] == 'Cumple').sum())} de los 6 SLAs evaluados."
    return t


def texto_recomendacion(protocolos, ref):
    """Redacta la recomendación ejecutiva (lista de párrafos) y devuelve también el ranking."""
    rk = ranking_protocolos(protocolos, ref)
    base = protocolos[ref]
    par = []
    sb = slas_mc(base)
    tensos = tabla_media_mc(base, "recursos")["Utilizacion"].sort_values(ascending=False).head(3)
    par.append(
        f"Diagnóstico de partida ({ref}). La simulación estima una mortalidad global del "
        f"{pct(base['resumen']['Tasa_mortalidad_Global'].mean())}, una tasa de altas del "
        f"{pct(base['resumen']['Tasa_alta_Global'].mean())} y un margen operativo del "
        f"{pct(base['resumen']['Margen_operativo'].mean())} (exigido: {pct(base['resumen']['Margen_exigido'].mean())}). "
        f"Se cumplen {int((sb['Veredicto'] == 'Cumple').sum())} de los 6 SLAs; no se cumplen: "
        f"{', '.join(sb.index[sb['Veredicto'] != 'Cumple']) or 'ninguno'}. Los recursos más tensionados son "
        + ", ".join(f"{r} (ρ = {num(u, 2)})" for r, u in tensos.items()) + ".")
    for et, m in protocolos.items():
        if et != ref:
            par.append(texto_escenario(et, base, m, ref=ref))
    candidatos = rk[(~rk["Descartado"]) & (rk.index != ref)]
    if len(candidatos):
        mejor = candidatos.index[0]
        m = rk.loc[mejor]
        if m["SLAs cumplidos"] >= rk.loc[ref, "SLAs cumplidos"] and m["Resultado neto anual (€)"] > 0:
            par.append(
                f"Recomendación. Se recomienda implantar {mejor}. Es la alternativa que, sin empeorar la mortalidad, "
                f"cumple más SLAs ({int(m['SLAs cumplidos'])}: {m['SLAs cumplidos (lista)'] or '—'}) con el mejor resultado "
                f"económico neto ({eur(m['Resultado neto anual (€)'])}/año frente a {ref}; "
                f"{eur(m['Resultado neto 1er año (€)'])} el primer año) y un margen neto del {pct(m['Margen neto'])}.")
        else:
            mejor = ref
            par.append(f"Recomendación. Ninguna alternativa mejora a {ref} en SLAs y resultado económico a la vez; "
                       f"se recomienda mantener {ref} o revisar el diseño de los escenarios.")
    else:
        mejor = ref
        par.append(f"Recomendación. Todas las alternativas empeoran la mortalidad frente a {ref}; se recomienda mantenerlo.")
    pend = slas_mc(protocolos[mejor])
    pend = pend.index[pend["Veredicto"] != "Cumple"]
    if len(pend):
        cuellos = tabla_media_mc(protocolos[mejor], "recursos")
        cuellos = cuellos[cuellos["Utilizacion"] >= 0.85].index.tolist()
        par.append(f"Aun con la opción recomendada siguen sin cumplirse {', '.join(pend)}."
                   + (f" Los recursos con utilización ≥ 0,85 ({', '.join(cuellos)}) son la siguiente palanca a estudiar."
                      if cuellos else ""))
    return par, rk
