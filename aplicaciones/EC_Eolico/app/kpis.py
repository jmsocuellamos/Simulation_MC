# -*- coding: utf-8 -*-
"""KPIs, SLAs, análisis Monte Carlo y comparación de escenarios del taller de mantenimiento eólico.

Funciones del cuaderno '03 Taller Eólico' (Tareas 1 y 2), con las herramientas de síntesis y
redacción automática de la aplicación. Todos los tiempos en días."""
import math
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

from motor import TIPOS, simular

DESENLACES = ["Operativo", "Baja_Definitiva", "Perdida_Total"]


PARAM = {
    # Tarifa por desenlace (€) según perfil (Sección A.05)
    "TARIFA": {
        "Operativo":       {"Programado": 800, "Leve": 2200, "Grave": 9500, "Critico": 18000},
        "Baja_Definitiva": {"Programado": 150, "Leve": 300, "Grave": 900, "Critico": 1200},
    },
    # Penalización por pérdida total (€) según perfil: coste de sustituir el componente
    "PEN_PERDIDA": {"Programado": 3000, "Leve": 8000, "Grave": 35000, "Critico": 70000},
    # Coste por procedimiento realizado (€)
    "COSTE_SERVICIO": {"Banco_Ensayo_Sobrecarga": 500, "Diagnostico_END": 300, "Consulta_Ingeniero_Fiabilidad": 200,
                       "Diagnostico_Electrico": 60, "Valoracion_Preliminar": 30, "Inspeccion_Tecnica": 25,
                       "Inspeccion_Tecnica_RR": 25, "Recepcion": 10, "Puesta_En_Marcha": 80},
    # Coste por día de estancia (€/día)
    "COSTE_DIA": {"Espera_Repuestos": 20, "Bahia_Reparacion": 60, "Bahia_Reparacion_PostIncidente": 60,
                  "Bahia_Reconstruccion_PostIncidente": 80, "Contencion_Emergencia": 250},
    # Margen operativo objetivo de la dirección (15-20%): se exige el límite inferior
    "MARGEN_EXIGIDO": 0.15,
    "MARGEN_OBJETIVO_SUP": 0.20,
    # SLAs
    "SLA_LATENCIA_D": 4 / 24,           # 4 horas hasta Inspección Técnica (Critico)
    "SLA_LATENCIA_CUMPL": 0.95,         # se exige que el 95% de las órdenes Critico lo cumpla
    "SLA_RATIO_ESPERA": 0.25,           # Programado/Leve
    "SLA_OCUP_CONT": 0.90,
    "SLA_SIN_CONT": 0.02,               # sobre Grave/Critico
    "SLA_OPERATIVO": 0.75,              # tasa de retorno a servicio global mínima
    # Ventanas de análisis (días)
    "T_WARMUP": 60,
    "T_COOLDOWN": 90,
}


PERFILES_SLA_LATENCIA = ["Critico"]


PERFILES_SLA_ESPERA = ["Programado", "Leve"]


PERFILES_GRAVES = ["Grave", "Critico"]


ESTACION_PRIMER_CONTACTO = ("Inspeccion_Tecnica", "Inspeccion_Tecnica_RR")


CONT = "Contencion_Emergencia"


BAHIA_URG = "Bahia_Urgente"


ESTACIONES_DEL_ANIO = [("Invierno", 335, 59), ("Primavera", 59, 151), ("Verano", 151, 243), ("Otoño", 243, 335)]


def estacion_del_anio(dia):
    """Etiqueta de la estación del año a la que pertenece el día del año `dia` (0-364)."""
    for nombre, ini, fin in ESTACIONES_DEL_ANIO:
        if (ini > fin and (dia >= ini or dia < fin)) or (ini < fin and ini <= dia < fin):
            return nombre


def preparar_traza(df):
    """Ordena la traza por orden de trabajo y tiempo y añade columnas auxiliares:
    * `T_end_real`: instante real de salida de la estación (= T_in de la fila siguiente).
    * `T_hueco`: espera no registrada en T_wait (espera de plaza en la Bahía Urgente).
    * `Bahia_prev`: bahía física que ocupaba la orden en la fila anterior.
    * `Fin`: True si la fila está cortada por el final de la simulación."""
    d = df.sort_values(["Id_orden", "T_in", "T_out"], kind="mergesort").reset_index(drop=True).copy()
    g = d.groupby("Id_orden", sort=False)
    sig = g["T_in"].shift(-1)
    d["T_end_real"] = np.maximum(sig.fillna(d["T_out"]), d["T_out"])
    d["T_hueco"] = (d["T_end_real"] - d["T_out"]).clip(lower=0)
    d.loc[d["T_hueco"] < 1e-3, "T_hueco"] = 0.0
    d["Bahia_prev"] = g["Bahia"].shift(1).fillna("Taller_Espera")
    d["Fin"] = d["Station_out"].eq("Fin_Simulacion")
    d["Sin_Cont"] = d["ID_server"].eq("Sin_Contencion_Disponible")
    return d


def horizonte(df):
    """Duración total (días) de la traza, redondeada al año."""
    return math.ceil(df["T_out"].max() / 365) * 365


def tasa_ingresos_estacion(df, dias=None):
    """Tasa de ingresos (órdenes/día) y reparto por perfil en cada estación del año."""
    adm = df.loc[df["Station_in"] == "Recepcion", ["Id_orden", "Perfil", "T_in"]].drop_duplicates("Id_orden")
    if dias is None:
        dias = horizonte(df)
    anios = dias / 365
    adm["Estacion_anio"] = (adm["T_in"].astype(int) % 365).map(estacion_del_anio)
    dias_est = {n: ((fin - ini) % 365) * anios for n, ini, fin in ESTACIONES_DEL_ANIO}
    tab = adm.groupby("Estacion_anio")["Id_orden"].count().rename("Ingresos").to_frame()
    tab["Días"] = tab.index.map(dias_est)
    tab["Tasa_obs (ord/día)"] = tab["Ingresos"] / tab["Días"]
    tab["Tasa_teórica"] = tab.index.map(TASAS_TEORICAS)
    mix = pd.crosstab(adm["Estacion_anio"], adm["Perfil"], normalize="index").reindex(columns=TIPOS)
    tab = tab.join(mix.add_prefix("%")).reindex([n for n, _, _ in ESTACIONES_DEL_ANIO])
    total = pd.Series({"Ingresos": len(adm), "Días": dias, "Tasa_obs (ord/día)": len(adm) / dias})
    total = pd.concat([total, adm["Perfil"].value_counts(normalize=True).reindex(TIPOS).add_prefix("%")])
    out = pd.concat([tab, pd.DataFrame([total], index=["Total"])])
    out["Ingresos/año"] = out["Ingresos"] / anios
    return out


def tiempos_servicio(df, cfg=None):
    """Estadísticos del tiempo de servicio (días) por estación y perfil.
    Si se pasa `cfg`, añade el valor nominal de la configuración para comparar."""
    d = df.loc[~df["Station_out"].eq("Fin_Simulacion") & ~df["ID_server"].eq("Sin_Contencion_Disponible")]
    tab = (d.groupby(["Station_in", "Perfil"])["T_server"]
             .agg(n="count", media="mean", sd="std", minimo="min", maximo="max").reset_index())
    if cfg is not None:
        tab["teorico"] = [cfg.TIEMPOS_SERVICIO.get(e, {}).get(t, np.nan) for e, t in zip(tab["Station_in"], tab["Perfil"])]
        tab["media/teorico"] = tab["media"] / tab["teorico"]
    return tab


def tabla_tiempos_servicio(df):
    """Tabla pivote: media del tiempo de servicio (días) por estación (filas) y perfil."""
    t = tiempos_servicio(df)
    return t.pivot(index="Station_in", columns="Perfil", values="media").reindex(columns=TIPOS).round(3)


def probabilidades_derivacion(df, separar_sin_cont=True):
    """Probabilidades de derivación estimadas (Station_in -> Station_out) por perfil.
    Excluye las filas cortadas por el final de la simulación. Si `separar_sin_cont`, los
    pérdidas totales por falta de plaza de Contención se muestran como destino 'Perdida_Sin_Cont'."""
    d = df.loc[~df["Station_out"].eq("Fin_Simulacion"), ["Perfil", "Station_in", "Station_out", "ID_server"]].copy()
    if separar_sin_cont:
        d.loc[d["ID_server"].eq("Sin_Contencion_Disponible"), "Station_out"] = "Perdida_Sin_Cont"
    cnt = d.groupby(["Perfil", "Station_in", "Station_out"]).size().rename("n").reset_index()
    cnt["p"] = cnt["n"] / cnt.groupby(["Perfil", "Station_in"])["n"].transform("sum")
    return cnt


def matriz_derivacion(df, tipo):
    """Matriz de transición (filas: origen, columnas: destino) para un perfil."""
    p = probabilidades_derivacion(df)
    p = p[p["Perfil"] == tipo]
    return p.pivot(index="Station_in", columns="Station_out", values="p").fillna(0).round(3)


def tabla_ordenes(d, param=PARAM):
    """Una fila por orden de trabajo con sus tiempos, desenlace, ruta y resultado económico.
    `d` debe venir de `preparar_traza`."""
    g = d.groupby("Id_orden", sort=False)
    ind = pd.DataFrame({
        "Perfil": g["Perfil"].first(),
        "T_llegada": g["T_in"].min(),
        "T_salida": g["T_out"].max(),
        "T_ser": g["T_server"].sum(),
        "Desenlace": g["Station_out"].last(),
    })
    ind["T_sys"] = ind["T_salida"] - ind["T_llegada"]
    ind["T_q"] = (ind["T_sys"] - ind["T_ser"]).clip(lower=0)
    ind["Ratio_espera"] = np.where(ind["T_sys"] > 0, ind["T_q"] / ind["T_sys"], 0.0)
    ind["Finalizado"] = ind["Desenlace"].isin(DESENLACES)

    visita = lambda est: d.loc[d["Station_in"].eq(est)].groupby("Id_orden").size().reindex(ind.index).fillna(0) > 0
    ind["Bahia_Reparacion"] = visita("Bahia_Reparacion")
    ind["Banco"] = visita("Banco_Ensayo_Sobrecarga")
    ind["Espera_Repuestos"] = visita("Espera_Repuestos")
    ind["Contencion"] = visita(CONT)
    ind["Sin_Cont"] = d.loc[d["Sin_Cont"]].groupby("Id_orden").size().reindex(ind.index).fillna(0) > 0
    # Origen de la (primera) derivación a Contención de Emergencia
    d_cont = d.assign(Prev=g["Station_in"].shift(1)).loc[d["Station_in"].eq(CONT)]
    prev = d_cont.groupby("Id_orden")["Prev"].first().reindex(ind.index)
    ind["Cont_desde"] = np.select(
        [prev.isin(["Bahia_Reparacion", "Bahia_Reparacion_PostIncidente"]), prev.eq("Banco_Ensayo_Sobrecarga"), prev.notna()],
        ["Bahia_Reparacion", "Banco", "Diagnostico"], default="")

    # Latencia Recepción -> primer contacto en Inspección Técnica
    it = d.loc[d["Station_in"].isin(ESTACION_PRIMER_CONTACTO) & ~d["Fin"]].groupby("Id_orden").first()
    contacto = it["T_end_real"] - it["T_server"]
    ind["Latencia_IT"] = contacto.reindex(ind.index) - ind["T_llegada"]

    # Economía por orden de trabajo
    ok = d.loc[~d["Fin"]]
    coste = ok["Station_in"].map(param["COSTE_SERVICIO"]).fillna(0) \
        + ok["Station_in"].map(param["COSTE_DIA"]).fillna(0) * ok["T_server"]
    ind["Coste_var"] = coste.groupby(ok["Id_orden"]).sum().reindex(ind.index).fillna(0)
    ind["Penalizacion"] = np.where(ind["Desenlace"].eq("Perdida_Total"), ind["Perfil"].map(param["PEN_PERDIDA"]), 0)
    ind["Ingreso"] = 0.0
    for des, tabla in param["TARIFA"].items():
        m = ind["Desenlace"].eq(des)
        ind.loc[m, "Ingreso"] = ind.loc[m, "Perfil"].map(tabla).astype(float)
    return ind


def ventana_ordenes(ind, t_fin, param=PARAM):
    """Selecciona las órdenes ingresadas en [warm-up, fin - cool-down]; devuelve las finalizadas,
    todas las de la ventana y la duración de la ventana (días)."""
    t0, t1 = param["T_WARMUP"], t_fin - param["T_COOLDOWN"]
    sel = ind[(ind["T_llegada"] >= t0) & (ind["T_llegada"] < t1)]
    return sel[sel["Finalizado"]], sel, (t1 - t0)


def kpis_operativos(ord_fin):
    """KPIs temporales (días) globales y por perfil sobre órdenes finalizadas."""
    def resumen(x):
        return pd.Series({
            "Ordenes": len(x),
            "T_sys_media": x["T_sys"].mean(), "T_sys_mediana": x["T_sys"].median(),
            "T_sys_P90": x["T_sys"].quantile(0.9),
            "T_ser_media": x["T_ser"].mean(), "T_q_media": x["T_q"].mean(),
            "T_q_P90": x["T_q"].quantile(0.9),
            "Ratio_espera_medio": x["Ratio_espera"].mean(),
            "Ratio_espera_agregado": x["T_q"].sum() / x["T_sys"].sum() if x["T_sys"].sum() > 0 else np.nan,
        })
    tab = ord_fin.groupby("Perfil").apply(resumen, include_groups=False).reindex(TIPOS)
    tab.loc["Global"] = resumen(ord_fin)
    return tab


def kpis_estaciones(d, t0, t1):
    """Carga y espera por estación (visitas con llegada en [t0, t1))."""
    x = d[(d["T_in"] >= t0) & (d["T_in"] < t1) & ~d["Fin"]].copy()
    x["Espera"] = x["T_wait"] + x["T_hueco"]
    tab = x.groupby("Station_in").agg(
        Visitas=("Id_orden", "count"),
        Espera_media_d=("Espera", "mean"),
        Espera_P90_d=("Espera", lambda s: s.quantile(0.9)),
        Servicio_medio_d=("T_server", "mean"),
        Carga_total_d=("T_server", "sum"),
        Espera_total_d=("Espera", "sum"),
        Cola_media_llegada=("Ord_en_cola", "mean"),
        Cola_max_llegada=("Ord_en_cola", "max"),
    )
    tab["%Espera_total"] = tab["Espera_total_d"] / tab["Espera_total_d"].sum()
    return tab.sort_values("Espera_total_d", ascending=False)


def _clip_dur(ini, fin, t0, t1):
    """Duración del solape de los intervalos [ini, fin] con la ventana [t0, t1]."""
    return (np.minimum(fin, t1) - np.maximum(ini, t0)).clip(lower=0)


def capacidades(cfg):
    """Capacidad de cada recurso (servidores, bahías, Bahía Urgente y bolsas de Contención de Emergencia)."""
    cap = dict(cfg.SERVIDORES_CAPACIDAD)
    cap.update(cfg.BAHIAS_CAPACIDAD)
    cap[BAHIA_URG] = cfg.BAHIA_URGENTE_CAPACIDAD
    if cfg.CONTENCION_MODE == "pooled":
        cap[CONT] = cfg.CONTENCION_TOTAL_CAPACIDAD
    else:
        cap["Cont_Critico"] = cfg.CONTENCION_CRITICO_CAPACIDAD
        cap["Cont_Grave"] = cfg.CONTENCION_GRAVE_CAPACIDAD
        cap["Cont_Base"] = cfg.CONTENCION_BASE_CAPACIDAD
    return cap


def estaciones_bahia(cfg):
    """Estaciones que retienen una bahía/plaza: bahías físicas, sus alias y la Contención de Emergencia."""
    return set(cfg.BAHIAS_CAPACIDAD) | set(getattr(cfg, "BAHIA_ALIAS", {})) | {CONT}


def _nombre_recurso(sala, tipo, cfg):
    """Traduce la bahía física de la traza al nombre del recurso (bolsas de Contención)."""
    if cfg.CONTENCION_MODE == "pooled":
        return np.asarray(sala)
    bolsa = np.select([tipo.eq("Critico"), tipo.eq("Grave")], ["Cont_Critico", "Cont_Grave"], "Cont_Base")
    return np.where(sala.eq(CONT), bolsa, sala)


def segmentos_ocupacion(d, cfg):
    """Segmentos de ocupación de recursos físicos (Bahía Urgente, bahías, Contención).
    Tipo 'Estancia': la orden ocupa la bahía en su estación.
    Tipo 'Bloqueo'  : la orden ocupa la bahía anterior esperando plaza en otra unidad."""
    es_bahia = d["Station_in"].isin(estaciones_bahia(cfg)) & ~d["Fin"]
    # En la fila de captura de la Bahía Urgente, el hueco es la espera de plaza (al principio de la fila)
    ini = np.where(d["Fin"], d["T_in"], d["T_in"] + np.where(es_bahia, d["T_wait"], d["T_hueco"]))
    est = pd.DataFrame({"Recurso": _nombre_recurso(d["Bahia"], d["Perfil"], cfg),
                        "Ini": ini, "Fin": d["T_end_real"].values, "Tipo": "Estancia",
                        "Perfil": d["Perfil"].values, "Id_orden": d["Id_orden"].values})
    b = d[es_bahia & (d["T_wait"] > 0)]
    blo = pd.DataFrame({"Recurso": _nombre_recurso(b["Bahia_prev"], b["Perfil"], cfg),
                        "Ini": b["T_in"].values, "Fin": (b["T_in"] + b["T_wait"]).values,
                        "Tipo": "Bloqueo", "Perfil": b["Perfil"].values, "Id_orden": b["Id_orden"].values})
    seg = pd.concat([est, blo], ignore_index=True)
    return seg[seg["Recurso"].isin(set(capacidades(cfg))) & (seg["Fin"] > seg["Ini"])]


def kpis_recursos(d, cfg, t0, t1):
    """Utilización, cola media (Little), bed-blocking y diagnóstico de cuello de botella por recurso."""
    T = t1 - t0
    cap = capacidades(cfg)
    filas = {}
    srv = d[~d["Fin"] & d["ID_server"].isin(cfg.SERVIDORES_CAPACIDAD.keys())]
    fin_s = srv["T_end_real"]; ini_s = fin_s - srv["T_server"]
    ocup = _clip_dur(ini_s, fin_s, t0, t1).groupby(srv["ID_server"]).sum()
    cola = _clip_dur(ini_s - srv["T_wait"], ini_s, t0, t1).groupby(srv["ID_server"]).sum()
    for r in cfg.SERVIDORES_CAPACIDAD:
        filas[r] = {"Tipo": "Servidor", "Capacidad": cap[r], "Ocupacion_d": ocup.get(r, 0.0),
                    "Cola_d": cola.get(r, 0.0), "Bloqueo_d": 0.0}
    seg = segmentos_ocupacion(d, cfg)
    dur = _clip_dur(seg["Ini"], seg["Fin"], t0, t1)
    ocup_c = dur.groupby(seg["Recurso"]).sum()
    bloq_c = dur[seg["Tipo"].eq("Bloqueo")].groupby(seg["Recurso"]).sum()
    ec = d[d["Station_in"].isin(estaciones_bahia(cfg))]
    rec_ec = pd.Series(_nombre_recurso(ec["Station_in"].replace(getattr(cfg, "BAHIA_ALIAS", {})), ec["Perfil"], cfg), index=ec.index)
    cola_c = _clip_dur(ec["T_in"], ec["T_in"] + ec["T_wait"], t0, t1).groupby(rec_ec).sum()
    hb = d[d["T_hueco"] > 0]
    cola_c.loc[BAHIA_URG] = _clip_dur(hb["T_in"], hb["T_in"] + hb["T_hueco"], t0, t1).sum()
    for r in cap:
        if r in cfg.SERVIDORES_CAPACIDAD:
            continue
        filas[r] = {"Tipo": "Bahía", "Capacidad": cap[r], "Ocupacion_d": ocup_c.get(r, 0.0),
                    "Cola_d": cola_c.get(r, 0.0), "Bloqueo_d": bloq_c.get(r, 0.0)}
    tab = pd.DataFrame(filas).T
    for c in ["Capacidad", "Ocupacion_d", "Cola_d", "Bloqueo_d"]:
        tab[c] = tab[c].astype(float)
    tab["Utilizacion"] = tab["Ocupacion_d"] / (tab["Capacidad"] * T)
    tab["Lq_media"] = tab["Cola_d"] / T                  # Ley de Little: nº medio en cola
    tab["L_media"] = tab["Ocupacion_d"] / T + tab["Lq_media"]
    tab["%Bloqueo"] = np.where(tab["Ocupacion_d"] > 0, tab["Bloqueo_d"] / tab["Ocupacion_d"], 0)
    tab["Diagnostico"] = np.select([tab["Utilizacion"] >= 1, tab["Utilizacion"] >= 0.85],
                                   ["INESTABLE (ρ≥1)", "Alta congestión"], "OK")
    return tab


def ocupacion_periodica(seg, recursos, capacidad, t0, t1, periodo=30):
    """Ocupación media (fracción) del conjunto de `recursos` en periodos de `periodo` días."""
    s = seg[seg["Recurso"].isin(recursos)]
    ini, fin = s["Ini"].values, s["Fin"].values
    inicios = np.arange(t0, t1, periodo)
    occ = [np.clip(np.minimum(fin, a + periodo) - np.maximum(ini, a), 0, None).sum() / (periodo * capacidad)
           for a in inicios]
    return pd.Series(occ, index=inicios.astype(int), name="Ocupacion")


def kpis_fiabilidad(ord_fin, dias_ventana):
    """Retorno a servicio, baja definitiva, pérdida total, derivación a Contención y al Banco de Ensayo (global y por perfil)."""
    def resumen(x):
        return pd.Series({
            "Ordenes": len(x),
            "Tasa_operativo": (x["Desenlace"] == "Operativo").mean(),
            "Tasa_perdida": (x["Desenlace"] == "Perdida_Total").mean(),
            "Tasa_baja": (x["Desenlace"] == "Baja_Definitiva").mean(),
            "Tasa_perdida_Sin_Cont": x["Sin_Cont"].mean(),
            "Operativos_anio": (x["Desenlace"] == "Operativo").sum() / dias_ventana * 365,
            "Tasa_Reparacion": x["Bahia_Reparacion"].mean(),
            "Tasa_Cont": x["Contencion"].mean(),
            "Tasa_Cont_desde_Diagnostico": (x["Cont_desde"] == "Diagnostico").mean(),
            "Tasa_Cont_desde_Banco": (x["Cont_desde"] == "Banco").mean(),
            "Tasa_Cont_desde_Reparacion": (x["Cont_desde"] == "Bahia_Reparacion").mean(),
            "Tasa_Banco": x["Banco"].mean(),
        })
    tab = ord_fin.groupby("Perfil").apply(resumen, include_groups=False).reindex(TIPOS)
    tab.loc["Global"] = resumen(ord_fin)
    return tab


def coste_fijo_periodo(cfg, dias_ventana):
    """Coste fijo imputable a la ventana: los anuales se prorratean (días/365) y los únicos se
    imputan íntegramente (criterio prudente)."""
    return sum(v * dias_ventana / 365 if "anual" in k else v for k, v in getattr(cfg, "COSTES_FIJOS", {}).items())


def kpis_economicos(ord_fin, cfg, dias_ventana, param=PARAM):
    """Facturación, costes, penalizaciones, margen operativo bruto y neto (tras coste fijo)."""
    def resumen(x):
        ing, cv, pen = x["Ingreso"].sum(), x["Coste_var"].sum(), x["Penalizacion"].sum()
        sub = {f"Facturacion_{des}": x.loc[x["Desenlace"] == des, "Ingreso"].sum() for des in param["TARIFA"]}
        return pd.Series({"Ordenes": len(x), "Facturacion_total": ing, **sub, "Coste_variable": cv,
                          "Penalizaciones": pen, "Coste_total": cv + pen,
                          "Coste_medio_orden": (cv + pen) / max(len(x), 1),
                          "Facturacion_media_orden": ing / max(len(x), 1),
                          "Beneficio": ing - cv - pen,
                          "Margen": (ing - cv - pen) / ing if ing > 0 else np.nan})
    tab = ord_fin.groupby("Perfil").apply(resumen, include_groups=False).reindex(TIPOS)
    tab.loc["Global"] = resumen(ord_fin)
    g = tab.loc["Global"]
    cf = coste_fijo_periodo(cfg, dias_ventana)
    anios = dias_ventana / 365
    glob = {
        "Facturacion_anio": g["Facturacion_total"] / anios,
        "Coste_total_anio": g["Coste_total"] / anios,
        "Coste_medio_orden": g["Coste_medio_orden"],
        "Beneficio_operativo_anio": g["Beneficio"] / anios,
        "Margen_operativo": g["Margen"],
        "Coste_fijo_periodo": cf,
        "Margen_neto": (g["Beneficio"] - cf) / g["Facturacion_total"] if g["Facturacion_total"] > 0 else np.nan,
        "Beneficio_neto_anio": (g["Beneficio"] - cf) / anios,
        "Margen_exigido": param["MARGEN_EXIGIDO"],
    }
    return tab, glob


def evaluar_slas(ord_fin, seg, cfg, t0, t1, econ_glob, param=PARAM):
    """Evalúa los seis SLAs de la Sección A.08. Devuelve un DataFrame con valor, umbral y cumplimiento."""
    filas = []
    # 1) Latencia Recepción -> Inspección Técnica (Critico) ≤ 4 horas
    rc = ord_fin[ord_fin["Perfil"].isin(PERFILES_SLA_LATENCIA)]
    lat = rc["Latencia_IT"].dropna()
    pct_ok = (lat <= param["SLA_LATENCIA_D"]).mean() if len(lat) else np.nan
    filas.append(("SLA1 Latencia IT ≤ 4 h (Critico)", "% órdenes ≤ 4 h", pct_ok,
                  f"≥ {param['SLA_LATENCIA_CUMPL']:.0%}", bool(pct_ok >= param["SLA_LATENCIA_CUMPL"]) if len(lat) else np.nan))
    filas.append(("   · latencia media (días)", "días", lat.mean(), "info", np.nan))
    # 2) Ratio de espera < 25% (Programado y Leve)
    for t in PERFILES_SLA_ESPERA:
        r = ord_fin.loc[ord_fin["Perfil"] == t, "Ratio_espera"].mean()
        filas.append((f"SLA2 Ratio espera < 25% ({t})", "T_q/T_sys medio", r,
                      f"< {param['SLA_RATIO_ESPERA']:.0%}", bool(r < param["SLA_RATIO_ESPERA"])))
    # 3) Ocupación estructural de Contención de Emergencia ≤ 90%
    rec_cont = [r for r in capacidades(cfg) if r.startswith("Cont")]
    occ = ocupacion_periodica(seg, rec_cont, cfg.CONTENCION_TOTAL_CAPACIDAD, t0, t1)
    filas.append(("SLA3 Ocupación media Contención ≤ 90%", "ocupación media", occ.mean(),
                  f"≤ {param['SLA_OCUP_CONT']:.0%}", bool(occ.mean() <= param["SLA_OCUP_CONT"])))
    filas.append(("   · % meses con ocupación Contención > 90%", "% meses", (occ > param["SLA_OCUP_CONT"]).mean(), "info", np.nan))
    # 4) Pérdida evitable Sin_Cont < 2% de Grave/Critico
    m = ord_fin.loc[ord_fin["Perfil"].isin(PERFILES_GRAVES), "Sin_Cont"].mean()
    filas.append(("SLA4 Pérdidas Sin_Cont < 2% (Grave/Critico)", "tasa", m, f"< {param['SLA_SIN_CONT']:.0%}",
                  bool(m < param["SLA_SIN_CONT"])))
    # 5) Tasa de retorno a servicio global ≥ 75%
    lib = (ord_fin["Desenlace"] == "Operativo").mean()
    filas.append(("SLA5 Retorno a servicio ≥ 75%", "tasa", lib, f"≥ {param['SLA_OPERATIVO']:.0%}",
                  bool(lib >= param["SLA_OPERATIVO"])))
    # 6) Margen operativo neto ≥ margen exigido
    filas.append(("SLA6 Margen neto ≥ margen exigido", "margen neto", econ_glob["Margen_neto"],
                  f"≥ {econ_glob['Margen_exigido']:.0%}", bool(econ_glob["Margen_neto"] >= econ_glob["Margen_exigido"])))
    return pd.DataFrame(filas, columns=["SLA", "Métrica", "Valor", "Umbral", "Cumple"]).set_index("SLA")


CLAVES_SLA = {"SLA1": "SLA1", "SLA2_Programado": "SLA2 Ratio espera < 25% (Programado)",
              "SLA2_Leve": "SLA2 Ratio espera < 25% (Leve)", "SLA3": "SLA3", "SLA4": "SLA4",
              "SLA5": "SLA5", "SLA6": "SLA6"}


def calcular_kpis(df, cfg, dias=None, param=PARAM):
    """Calcula TODOS los KPIs y SLAs de una traza (histórica o simulada).
    Devuelve un diccionario con las tablas y un resumen plano `resumen` (una cifra por KPI),
    pensado para acumular réplicas en el análisis Monte Carlo."""
    d = preparar_traza(df)
    t_fin = dias if dias is not None else horizonte(df)
    t0, t1 = param["T_WARMUP"], t_fin - param["T_COOLDOWN"]
    ind = tabla_ordenes(d, param)
    ord_fin, ord_vent, dias_v = ventana_ordenes(ind, t_fin, param)
    op = kpis_operativos(ord_fin)
    est = kpis_estaciones(d, t0, t1)
    rec = kpis_recursos(d, cfg, t0, t1)
    seg = segmentos_ocupacion(d, cfg)
    bio = kpis_fiabilidad(ord_fin, dias_v)
    eco_tab, eco = kpis_economicos(ord_fin, cfg, dias_v, param)
    sla = evaluar_slas(ord_fin, seg, cfg, t0, t1, eco, param)

    res = {"Ordenes_ventana": len(ord_vent), "Ordenes_finalizadas": len(ord_fin),
           "Pct_finalizados": len(ord_fin) / max(len(ord_vent), 1), "Ingresos_anio": len(ord_vent) / dias_v * 365}
    for perfil in TIPOS + ["Global"]:
        for k in ["T_sys_media", "T_ser_media", "T_q_media", "Ratio_espera_medio"]:
            res[f"{k}_{perfil}"] = op.loc[perfil, k]
        for k in ["Tasa_operativo", "Tasa_perdida", "Tasa_baja", "Tasa_perdida_Sin_Cont", "Tasa_Reparacion", "Tasa_Banco", "Tasa_Cont"]:
            res[f"{k}_{perfil}"] = bio.loc[perfil, k]
    for k in ["Operativos_anio", "Tasa_Cont_desde_Diagnostico", "Tasa_Cont_desde_Banco", "Tasa_Cont_desde_Reparacion", "Tasa_Banco"]:
        res[k] = bio.loc["Global", k]
    for r in rec.index:
        res[f"Util_{r}"] = rec.loc[r, "Utilizacion"]
        res[f"Lq_{r}"] = rec.loc[r, "Lq_media"]
        if rec.loc[r, "Tipo"] != "Servidor":
            res[f"Bloqueo_d_anio_{r}"] = rec.loc[r, "Bloqueo_d"] / ((t1 - t0) / 365)
    res.update(eco)
    rc = ord_fin["Perfil"].isin(PERFILES_SLA_LATENCIA)
    res["Latencia_IT_media_Critico_d"] = ord_fin.loc[rc, "Latencia_IT"].mean()
    res["Perdida_Sin_Cont_GC"] = ord_fin.loc[ord_fin["Perfil"].isin(PERFILES_GRAVES), "Sin_Cont"].mean()
    for k, pref in CLAVES_SLA.items():
        fila = sla.loc[sla.index.str.startswith(pref)].iloc[0]
        res[f"SLA_valor_{k}"] = fila["Valor"]
        res[f"Cumple_{k}"] = bool(fila["Cumple"]) if not pd.isna(fila["Cumple"]) else False
    res["Ocup_media_Cont"] = res["SLA_valor_SLA3"]
    res["Pct_meses_Cont_90"] = sla.loc["   · % meses con ocupación Contención > 90%", "Valor"]
    # Acceso a Contención por perfil (espera de plaza y fallos sobre las peticiones)
    cont = d[d["Station_in"].eq(CONT) & (d["T_in"] >= t0) & (d["T_in"] < t1)]
    for perfil in TIPOS:
        u = cont[cont["Perfil"] == perfil]
        res[f"Peticiones_Cont_anio_{perfil}"] = len(u) / dias_v * 365
        res[f"Espera_Cont_media_{perfil}"] = u["T_wait"].mean() if len(u) else np.nan
        res[f"Espera_Cont_P95_{perfil}"] = u["T_wait"].quantile(0.95) if len(u) else np.nan
        res[f"Pct_peticiones_Cont_fallidas_{perfil}"] = u["Sin_Cont"].mean() if len(u) else np.nan
    # Uso del banco de Ensayos No Destructivos (recurso escaso)
    pg = d[d["Station_in"].eq("Diagnostico_END") & (d["T_in"] >= t0) & (d["T_in"] < t1) & ~d["Fin"]]
    res["Espera_media_END"] = pg["T_wait"].mean() if len(pg) else np.nan

    return {"traza": d, "ordenes": ind, "ordenes_fin": ord_fin, "operativos": op,
            "estaciones": est, "recursos": rec, "segmentos": seg, "fiabilidad": bio,
            "economicos": eco_tab, "economicos_glob": eco, "slas": sla, "resumen": res,
            "dias_ventana": dias_v, "t0": t0, "t1": t1, "t_fin": t_fin}


TABLAS_MC = ["operativos", "fiabilidad", "estaciones", "recursos", "economicos"]


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
    """Promedio entre réplicas de una tabla de KPIs (operativos, biologicos, estaciones, recursos...)."""
    t = mc[tabla]
    num = t.select_dtypes("number").drop(columns="Replica")
    media = num.groupby(level=0, sort=False).mean()
    if tabla == "recursos":
        media.insert(0, "Tipo", t.groupby(level=0, sort=False)["Tipo"].first())
        media["Diagnostico"] = np.select([media["Utilizacion"] >= 1, media["Utilizacion"] >= 0.85],
                                         ["INESTABLE (ρ≥1)", "Alta congestión"], "OK")
        media["%Réplicas_ρ≥0.85"] = (t["Utilizacion"] >= 0.85).groupby(level=0, sort=False).mean()
    if tabla in ("operativos", "fiabilidad", "economicos"):
        media = media.reindex(TIPOS + ["Global"])
    return media


def slas_mc(mc):
    """Para cada SLA: valor medio de la métrica, IC95 y % de réplicas que lo cumplen."""
    r = mc["resumen"]
    nombres = {"SLA1": "Latencia IT ≤ 4 h (Critico): % que cumplen (≥95%)",
               "SLA2_Programado": "Ratio espera Programado (<25%)",
               "SLA2_Leve": "Ratio espera Leve (<25%)",
               "SLA3": "Ocupación media Contención de Emergencia (≤90%)",
               "SLA4": "Pérdida total Sin_Cont en Grave/Critico (<2%)",
               "SLA5": "Tasa de retorno a servicio global (≥75%)",
               "SLA6": "Margen neto ≥ margen exigido (15%)"}
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
    d_anio = (e.loc[idx, "Beneficio_operativo_anio"] - b.loc[idx, "Beneficio_operativo_anio"]).astype(float)
    m, se = d_anio.mean(), d_anio.std() / np.sqrt(len(d_anio))
    tq = stats.t.ppf(1 - alpha / 2, len(d_anio) - 1)
    res = {
        "Δ beneficio operativo (€/año, antes de coste fijo)": m,
        "IC95 inf (€/año)": m - tq * se, "IC95 sup (€/año)": m + tq * se,
        "Coste fijo anual (€/año)": anual,
        "Coste fijo único (€)": unico,
        "Resultado neto anual (€/año)": m - anual,
        "Cobertura del coste anual (Δ/coste)": (m / anual) if anual else np.nan,
        "Periodo de amortización del coste único (años)": (unico / m) if (unico and m > 0) else np.nan,
        "Margen operativo base": b["Margen_operativo"].mean(),
        "Margen operativo escenario (bruto)": e["Margen_operativo"].mean(),
        "Margen neto escenario (tras coste fijo)": e["Margen_neto"].mean(),
        "Margen exigido medio escenario": e["Margen_exigido"].mean(),
        "% réplicas margen neto ≥ exigido": e["Cumple_SLA6"].astype(float).mean(),
        "% réplicas margen neto ≥ margen base": (e.loc[idx, "Margen_neto"] >= b.loc[idx, "Margen_neto"]).mean(),
    }
    return pd.Series(res).to_frame("Valor")


KPIS_CLAVE = (
    ["Ingresos_anio"]
    + [f"T_sys_media_{t}" for t in TIPOS + ["Global"]]
    + [f"T_q_media_{t}" for t in TIPOS + ["Global"]]
    + [f"Ratio_espera_medio_{t}" for t in TIPOS]
    + ["Latencia_IT_media_Critico_d", "SLA_valor_SLA1"]
    + [f"Tasa_operativo_{t}" for t in TIPOS + ["Global"]]
    + [f"Tasa_perdida_{t}" for t in TIPOS + ["Global"]]
    + [f"Tasa_baja_{t}" for t in TIPOS + ["Global"]]
    + ["Operativos_anio", "Tasa_perdida_Sin_Cont_Global", "Perdida_Sin_Cont_GC", "Tasa_Reparacion_Global",
       "Tasa_Banco_Global", "Tasa_Cont_Global", "Tasa_Cont_desde_Diagnostico", "Tasa_Cont_desde_Banco", "Tasa_Cont_desde_Reparacion"]
    + ["Util_Equipo_Tecnico", "Util_Diagnostico_END", "Espera_media_END",
       "Util_Bahia_Urgente", "Util_Espera_Repuestos", "Util_Banco_Ensayo_Sobrecarga", "Util_Bahia_Reparacion",
       "Util_Bahia_Reconstruccion_PostIncidente", "Ocup_media_Cont", "Pct_meses_Cont_90",
       "Bloqueo_d_anio_Bahia_Urgente", "Bloqueo_d_anio_Banco_Ensayo_Sobrecarga", "Bloqueo_d_anio_Espera_Repuestos",
       "Bloqueo_d_anio_Bahia_Reparacion"]
    + ["Coste_medio_orden", "Facturacion_anio", "Beneficio_operativo_anio", "Margen_operativo",
       "Margen_neto", "Margen_exigido"]
)


def ejecutar_replica(cfg, dias, semilla, param=None, guardar_detalle=False):
    """Simula una réplica y calcula sus KPIs. Devuelve (resumen, tablas, detalle)."""
    param = PARAM if param is None else param
    df_sim = simular(cfg, dias=dias, semilla=semilla)
    k = calcular_kpis(df_sim, cfg, dias=dias, param=param)
    tablas = {t: k[t] for t in TABLAS_MC}
    detalle = None
    if guardar_detalle:
        rec_cont = [r for r in capacidades(cfg) if r.startswith("Cont")]
        detalle = {
            "ordenes_fin": k["ordenes_fin"][["Perfil", "T_sys", "T_ser", "T_q", "Ratio_espera", "Desenlace",
                                                   "Latencia_IT"]].copy(),
            "ocupacion_cont": ocupacion_periodica(k["segmentos"], rec_cont, cfg.CONTENCION_TOTAL_CAPACIDAD, k["t0"], k["t1"]),
            "ocupacion_cont_bolsas": {r: ocupacion_periodica(k["segmentos"], [r], capacidades(cfg)[r], k["t0"], k["t1"])
                                     for r in rec_cont},
        }
    return dict(k["resumen"]), tablas, detalle


def _replica_worker(args):
    cfg, dias, semilla, param, i = args
    res, tablas, det = ejecutar_replica(cfg, dias, semilla, param, guardar_detalle=(i == 0))
    return i, res, tablas, det


def montecarlo(cfg, n_sim=30, dias=1825, semilla_base=1000, param=None, progreso=None,
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


def sentido_kpi(k):
    """Sentido de mejora de cada KPI: -1 = menor es mejor, +1 = mayor es mejor, 0 = neutro."""
    if k.startswith(("Tasa_operativo", "SLA_valor_SLA1", "Margen", "Beneficio", "Facturacion", "Operativos_anio")):
        return +1
    if k.startswith(("T_sys", "T_q", "Ratio_espera", "Latencia", "Tasa_perdida", "Perdida_Sin_Cont", "Tasa_baja",
                     "Coste", "Bloqueo", "Espera_", "Pct_peticiones_Cont_fallidas", "Pct_meses_Cont_90", "Lq_")):
        return -1
    return 0


ETIQUETAS_KPI = {
    "T_sys_media_Programado": "T. medio en el sistema · Programado (días)",
    "T_sys_media_Leve": "T. medio en el sistema · Leve (días)",
    "T_sys_media_Grave": "T. medio en el sistema · Grave (días)",
    "T_sys_media_Critico": "T. medio en el sistema · Critico (días)",
    "T_sys_media_Global": "T. medio en el sistema · Global (días)",
    "T_q_media_Global": "T. medio en cola · Global (días)",
    "Latencia_IT_media_Critico_d": "Latencia hasta Inspección Técnica · Critico (días)",
    "SLA_valor_SLA1": "% Critico inspeccionadas en ≤ 4 horas",
    "Tasa_operativo_Programado": "Retorno a servicio · Programado",
    "Tasa_operativo_Leve": "Retorno a servicio · Leve",
    "Tasa_operativo_Grave": "Retorno a servicio · Grave",
    "Tasa_operativo_Critico": "Retorno a servicio · Critico",
    "Tasa_operativo_Global": "Tasa de retorno a servicio global",
    "Tasa_baja_Global": "Tasa de baja definitiva",
    "Tasa_perdida_Programado": "Pérdida total · Programado",
    "Tasa_perdida_Leve": "Pérdida total · Leve",
    "Tasa_perdida_Grave": "Pérdida total · Grave",
    "Tasa_perdida_Critico": "Pérdida total · Critico",
    "Tasa_perdida_Global": "Pérdida total · Global",
    "Tasa_perdida_Sin_Cont_Global": "Pérdida Sin_Cont (s/ total órdenes)",
    "Perdida_Sin_Cont_GC": "Pérdida Sin_Cont (s/ Grave+Critico)",
    "Operativos_anio": "Órdenes devueltas a servicio al año",
    "Ocup_media_Cont": "Ocupación media Contención de Emergencia",
    "Util_Bahia_Reparacion": "Utilización Bahía de Reparación",
    "Util_Espera_Repuestos": "Utilización Espera de Repuestos",
    "Util_Diagnostico_END": "Utilización Ensayos No Destructivos",
    "Util_Bahia_Urgente": "Utilización Bahía Urgente",
    "Util_Equipo_Tecnico": "Utilización Equipo Técnico",
    "Coste_medio_orden": "Coste medio por orden (€)",
    "Facturacion_anio": "Facturación del contrato (€/año)",
    "Beneficio_operativo_anio": "Beneficio operativo (€/año)",
    "Margen_operativo": "Margen operativo bruto",
    "Margen_neto": "Margen operativo neto (tras coste fijo)",
    "Margen_exigido": "Margen exigido por la dirección",
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
    d = (mc["resumen"].loc[idx, "Beneficio_operativo_anio"] - mc_base["resumen"].loc[idx, "Beneficio_operativo_anio"]).astype(float)
    se = d.std() / np.sqrt(len(d)) if len(d) > 1 else np.nan
    return {"Δ beneficio operativo anual (€)": d.mean(),
            "IC95 inf (€)": d.mean() - 1.96 * se, "IC95 sup (€)": d.mean() + 1.96 * se,
            "Coste fijo anual (€)": anual, "Coste fijo único (€)": unico,
            "Resultado neto 1er año (€)": d.mean() - anual - unico,
            "Resultado neto años sucesivos (€/año)": d.mean() - anual}


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
    if k.startswith(("Tasa_", "Util_", "Ratio_", "Margen", "Ocup", "Pct_", "Perdida_", "SLA_valor_SLA1", "Cumple", "%")):
        return pct(v, 2 if k.startswith(("Tasa_perdida_Sin", "Perdida_Sin")) else 1)
    if k in ("Ingresos_anio", "Operativos_anio") or k.startswith("Peticiones_"):
        return num(v, 1)
    if any(t in k for t in ("Coste", "Facturacion", "Beneficio", "€", "Resultado")):
        return eur(v)
    return num(v, 2)


def ranking_protocolos(protocolos, ref):
    """Ordena los protocolos: primero los que no empeoran la pérdida total, luego por nº de SLAs
    cumplidos y, a igualdad, por resultado económico neto anual frente a la referencia."""
    base = protocolos[ref]
    filas = []
    for et, m in protocolos.items():
        s = slas_mc(m)
        fila = {"Protocolo": et, "SLAs cumplidos": int((s["Veredicto"] == "Cumple").sum()),
                "SLAs cumplidos (lista)": ", ".join(s.index[s["Veredicto"] == "Cumple"]),
                "Retorno a servicio": m["resumen"]["Tasa_operativo_Global"].mean(),
                "Pérdida total global": m["resumen"]["Tasa_perdida_Global"].mean(),
                "Pérdida Sin_Cont G/C": m["resumen"]["Perdida_Sin_Cont_GC"].mean(),
                "T_sys global (días)": m["resumen"]["T_sys_media_Global"].mean(),
                "Margen neto": m["resumen"]["Margen_neto"].mean()}
        if et == ref:
            fila.update({"Resultado neto anual (€)": 0.0, "Resultado neto 1er año (€)": 0.0, "Descartado": False})
        else:
            e = resultado_economico(base, m)
            c = comparar_escenarios(base, m, ["Tasa_perdida_Global", "Perdida_Sin_Cont_GC"])
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
    t += f"Cumple {int((s['Veredicto'] == 'Cumple').sum())} de los {len(s)} SLAs evaluados."
    return t


def texto_recomendacion(protocolos, ref):
    """Redacta la recomendación ejecutiva (lista de párrafos) y devuelve también el ranking."""
    rk = ranking_protocolos(protocolos, ref)
    base = protocolos[ref]
    par = []
    sb = slas_mc(base)
    tensos = tabla_media_mc(base, "recursos")["Utilizacion"].sort_values(ascending=False).head(3)
    par.append(
        f"Diagnóstico de partida ({ref}). La simulación estima una tasa de retorno a servicio del "
        f"{pct(base['resumen']['Tasa_operativo_Global'].mean())}, una tasa de pérdida total del "
        f"{pct(base['resumen']['Tasa_perdida_Global'].mean())}, una tasa de baja definitiva del "
        f"{pct(base['resumen']['Tasa_baja_Global'].mean())} y un margen operativo del "
        f"{pct(base['resumen']['Margen_operativo'].mean())} (exigido: {pct(base['resumen']['Margen_exigido'].mean())}). "
        f"Se cumplen {int((sb['Veredicto'] == 'Cumple').sum())} de los {len(sb)} SLAs; no se cumplen: "
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
                f"Recomendación. Se recomienda implantar {mejor}. Es la alternativa que, sin empeorar la pérdida total, "
                f"cumple más SLAs ({int(m['SLAs cumplidos'])}: {m['SLAs cumplidos (lista)'] or '—'}) con el mejor resultado "
                f"económico neto ({eur(m['Resultado neto anual (€)'])}/año frente a {ref}; "
                f"{eur(m['Resultado neto 1er año (€)'])} el primer año) y un margen neto del {pct(m['Margen neto'])}.")
        else:
            mejor = ref
            par.append(f"Recomendación. Ninguna alternativa mejora a {ref} en SLAs y resultado económico a la vez; "
                       f"se recomienda mantener {ref} o revisar el diseño de los escenarios.")
    else:
        mejor = ref
        par.append(f"Recomendación. Todas las alternativas empeoran la pérdida total frente a {ref}; se recomienda mantenerlo.")
    pend = slas_mc(protocolos[mejor])
    pend = pend.index[pend["Veredicto"] != "Cumple"]
    if len(pend):
        cuellos = tabla_media_mc(protocolos[mejor], "recursos")
        cuellos = cuellos[cuellos["Utilizacion"] >= 0.85].index.tolist()
        par.append(f"Aun con la opción recomendada siguen sin cumplirse {', '.join(pend)}."
                   + (f" Los recursos con utilización ≥ 0,85 ({', '.join(cuellos)}) son la siguiente palanca a estudiar."
                      if cuellos else ""))
    return par, rk
