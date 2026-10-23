# -*- coding: utf-8 -*-
"""KPIs, SLAs, análisis Monte Carlo y comparación de escenarios del centro de cría y reintroducción.

Funciones del cuaderno '02 Centro de Cría en Cautividad y Reintroducción' (Tareas 1 y 2), con las
herramientas de síntesis y redacción automática de la aplicación. Todos los tiempos en días."""
import math
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

from motor import TIPOS, simular

DESENLACES = ["Liberado", "Cautividad_Permanente", "Fallece"]


PARAM = {
    # Subvención por desenlace (€) según perfil (Sección A.05)
    "SUBVENCION": {
        "Liberado":              {"Fundador": 5300, "Cautivo": 3300, "RescateLeve": 2300, "RescateCritico": 4000},
        "Cautividad_Permanente": {"Fundador": 1000, "Cautivo": 800, "RescateLeve": 550, "RescateCritico": 600},
    },
    # Penalización por fallecimiento (€) según perfil
    "PEN_FALLECE": {"Fundador": 500, "Cautivo": 400, "RescateLeve": 300, "RescateCritico": 350},
    # Coste por procedimiento realizado (€)
    "COSTE_SERVICIO": {"Cirugia_Veterinaria": 900, "Pruebas_Geneticas_Avanzadas": 250, "Consulta_Genetista": 150,
                       "Analitica_Basica": 40, "Evaluacion_Veterinaria_Inicial": 25, "Evaluacion_Preliminar": 25,
                       "Evaluacion_Veterinaria_Inicial_RL": 25, "Evaluacion_Preliminar_RL": 25,
                       "Revision_Ingreso": 10, "Admision": 5, "Preparacion_Suelta": 60},
    # Coste por día de estancia (€/día)
    "COSTE_DIA": {"Cuarentena": 18, "Modulo_Cria": 8, "Modulo_Cria_PostUCI": 8, "UCI_Veterinaria": 130,
                  "Recuperacion_PostCritica": 50},
    # Penalizaciones adicionales (Sección A.05)
    "PEN_SIN_UCI": 5000,                # fallecimiento por falta de plaza de UCI (adicional)
    "PEN_DEMORA": 300,                  # RescateCritico con latencia hasta la EVI > SLA_LATENCIA_D
    # Margen operativo objetivo del patronato (12-15%): se exige el límite inferior
    "MARGEN_EXIGIDO": 0.12,
    "MARGEN_OBJETIVO_SUP": 0.15,
    # SLAs
    "SLA_LATENCIA_D": 0.25,             # 6 horas hasta Evaluación Veterinaria Inicial (RescateCritico)
    "SLA_LATENCIA_CUMPL": 0.90,         # se exige que el 90% de los RescateCritico lo cumpla
    "SLA_RATIO_ESPERA": 0.05,           # Fundador/Cautivo
    "SLA_OCUP_UCI": 0.90,
    "SLA_SIN_UCI": 0.02,                # sobre RescateLeve/RescateCritico
    "SLA_LIBERACION": 0.60,             # tasa de liberación global mínima
    # Ventanas de análisis (días)
    "T_WARMUP": 365,
    "T_COOLDOWN": 730,
}


PERFILES_SLA_LATENCIA = ["RescateCritico"]


PERFILES_SLA_ESPERA = ["Fundador", "Cautivo"]


PERFILES_RESCATE = ["RescateLeve", "RescateCritico"]


ESTACION_PRIMER_CONTACTO = ("Evaluacion_Veterinaria_Inicial", "Evaluacion_Veterinaria_Inicial_RL")


UCI = "UCI_Veterinaria"


RECINTO = "Recinto_Recuperacion"


ESTACIONES_DEL_ANIO = [("Invierno", 335, 59), ("Primavera", 59, 151), ("Verano", 151, 243), ("Otoño", 243, 335)]


def ver(df, dec=4):
    """Redondea y sustituye los valores no definidos (NaN) por «—» para mostrar las tablas.
    Un «—» indica que el valor no aplica (p. ej., un perfil que no pasa por una estación)."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    out = df.copy()
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(dec)
    for c in out.columns.difference(num):   # columnas mixtas (números y textos)
        out[c] = out[c].map(lambda v: round(v, dec) if isinstance(v, float) else v)
    return out.astype(object).where(out.notna(), "—")


def _div(a, b):
    """División segura: 0 si a = 0 y b = 0; NaN solo si b = 0 y a ≠ 0."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(b != 0, a / np.where(b != 0, b, 1), np.where(a == 0, 0.0, np.nan))
    return r


def estacion_del_anio(dia):
    """Etiqueta de la estación del año a la que pertenece el día del año `dia` (0-364)."""
    for nombre, ini, fin in ESTACIONES_DEL_ANIO:
        if (ini > fin and (dia >= ini or dia < fin)) or (ini < fin and ini <= dia < fin):
            return nombre


def preparar_traza(df):
    """Ordena la traza por individuo y tiempo y añade columnas auxiliares:
    * `T_end_real`: instante real de salida de la estación (= T_in de la fila siguiente).
    * `T_hueco`: espera no registrada en T_wait (espera de plaza en el Recinto de Recuperación).
    * `Recinto_prev`: recinto físico que ocupaba el individuo en la fila anterior.
    * `Fin`: True si la fila está cortada por el final de la simulación."""
    d = df.sort_values(["Id_ind", "T_in", "T_out"], kind="mergesort").reset_index(drop=True).copy()
    g = d.groupby("Id_ind", sort=False)
    sig = g["T_in"].shift(-1)
    d["T_end_real"] = np.maximum(sig.fillna(d["T_out"]), d["T_out"])
    d["T_hueco"] = (d["T_end_real"] - d["T_out"]).clip(lower=0)
    d.loc[d["T_hueco"] < 1e-3, "T_hueco"] = 0.0
    d["Recinto_prev"] = g["Recinto"].shift(1).fillna("Recinto_Espera")
    d["Fin"] = d["Station_out"].eq("Fin_Simulacion")
    d["Sin_UCI"] = d["ID_server"].eq("Sin_UCI_Disponible")
    return d


def horizonte(df):
    """Duración total (días) de la traza, redondeada al año."""
    return math.ceil(df["T_out"].max() / 365) * 365


def tasa_ingresos_estacion(df, dias=None):
    """Tasa de ingresos (individuos/día) y reparto por perfil en cada estación del año."""
    adm = df.loc[df["Station_in"] == "Admision", ["Id_ind", "Perfil", "T_in"]].drop_duplicates("Id_ind")
    if dias is None:
        dias = horizonte(df)
    anios = dias / 365
    adm["Estacion_anio"] = (adm["T_in"].astype(int) % 365).map(estacion_del_anio)
    dias_est = {n: ((fin - ini) % 365) * anios for n, ini, fin in ESTACIONES_DEL_ANIO}
    tab = adm.groupby("Estacion_anio")["Id_ind"].count().rename("Ingresos").to_frame()
    tab["Días"] = tab.index.map(dias_est)
    tab["Tasa_obs (ind/día)"] = tab["Ingresos"] / tab["Días"]
    tab["Tasa_teórica"] = tab.index.map(TASAS_TEORICAS)
    mix = pd.crosstab(adm["Estacion_anio"], adm["Perfil"], normalize="index").reindex(columns=TIPOS)
    tab = tab.join(mix.add_prefix("%")).reindex([n for n, _, _ in ESTACIONES_DEL_ANIO])
    total = pd.Series({"Ingresos": len(adm), "Días": dias, "Tasa_obs (ind/día)": len(adm) / dias,
                       "Tasa_teórica": sum(TASAS_TEORICAS[n] * ((fin - ini) % 365) for n, ini, fin in ESTACIONES_DEL_ANIO) / 365})
    total = pd.concat([total, adm["Perfil"].value_counts(normalize=True).reindex(TIPOS).add_prefix("%")])
    out = pd.concat([tab, pd.DataFrame([total], index=["Total"])])
    out["Ingresos/año"] = out["Ingresos"] / anios
    return out


def tiempos_servicio(df, cfg=None):
    """Estadísticos del tiempo de servicio (días) por estación y perfil.
    Si se pasa `cfg`, añade el valor nominal de la configuración para comparar."""
    d = df.loc[~df["Station_out"].eq("Fin_Simulacion") & ~df["ID_server"].eq("Sin_UCI_Disponible")]
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


def probabilidades_derivacion(df, separar_sin_uci=True):
    """Probabilidades de derivación estimadas (Station_in -> Station_out) por perfil.
    Excluye las filas cortadas por el final de la simulación. Si `separar_sin_uci`, los
    fallecimientos por falta de plaza de UCI se muestran como destino 'Fallece_Sin_UCI'."""
    d = df.loc[~df["Station_out"].eq("Fin_Simulacion"), ["Perfil", "Station_in", "Station_out", "ID_server"]].copy()
    if separar_sin_uci:
        d.loc[d["ID_server"].eq("Sin_UCI_Disponible"), "Station_out"] = "Fallece_Sin_UCI"
    cnt = d.groupby(["Perfil", "Station_in", "Station_out"]).size().rename("n").reset_index()
    cnt["p"] = cnt["n"] / cnt.groupby(["Perfil", "Station_in"])["n"].transform("sum")
    return cnt


def matriz_derivacion(df, tipo):
    """Matriz de transición (filas: origen, columnas: destino) para un perfil."""
    p = probabilidades_derivacion(df)
    p = p[p["Perfil"] == tipo]
    return p.pivot(index="Station_in", columns="Station_out", values="p").fillna(0).round(3)


def tabla_individuos(d, param=PARAM):
    """Una fila por individuo con sus tiempos, desenlace, ruta y resultado económico.
    `d` debe venir de `preparar_traza`."""
    g = d.groupby("Id_ind", sort=False)
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

    visita = lambda est: d.loc[d["Station_in"].eq(est)].groupby("Id_ind").size().reindex(ind.index).fillna(0) > 0
    ind["Modulo_Cria"] = visita("Modulo_Cria")
    ind["Cirugia"] = visita("Cirugia_Veterinaria")
    ind["Cuarentena"] = visita("Cuarentena")
    ind["UCI"] = visita(UCI)
    ind["Sin_UCI"] = d.loc[d["Sin_UCI"]].groupby("Id_ind").size().reindex(ind.index).fillna(0) > 0
    # Origen de la (primera) derivación a UCI
    d_uci = d.assign(Prev=g["Station_in"].shift(1)).loc[d["Station_in"].eq(UCI)]
    prev = d_uci.groupby("Id_ind")["Prev"].first().reindex(ind.index)
    ind["UCI_desde"] = np.select(
        [prev.isin(["Modulo_Cria", "Modulo_Cria_PostUCI"]), prev.eq("Cirugia_Veterinaria"), prev.notna()],
        ["Modulo_Cria", "Cirugia", "Ingreso"], default="")

    # Latencia Admisión -> primer contacto en Evaluación Veterinaria Inicial
    evi = d.loc[d["Station_in"].isin(ESTACION_PRIMER_CONTACTO) & ~d["Fin"]].groupby("Id_ind").first()
    contacto = evi["T_end_real"] - evi["T_server"]
    ind["Latencia_EVI"] = contacto.reindex(ind.index) - ind["T_llegada"]

    # Economía por individuo
    ok = d.loc[~d["Fin"]]
    coste = ok["Station_in"].map(param["COSTE_SERVICIO"]).fillna(0) \
        + ok["Station_in"].map(param["COSTE_DIA"]).fillna(0) * ok["T_server"]
    ind["Coste_var"] = coste.groupby(ok["Id_ind"]).sum().reindex(ind.index).fillna(0)
    # Demora en atención crítica: RescateCritico con latencia hasta la EVI superior al SLA
    ind["Demora_critica"] = ind["Perfil"].isin(PERFILES_SLA_LATENCIA) & (ind["Latencia_EVI"].fillna(0) > param["SLA_LATENCIA_D"])
    ind["Penalizacion"] = np.where(ind["Desenlace"].eq("Fallece"), ind["Perfil"].map(param["PEN_FALLECE"]), 0) \
        + np.where(ind["Sin_UCI"], param["PEN_SIN_UCI"], 0) \
        + np.where(ind["Demora_critica"], param["PEN_DEMORA"], 0)
    ind["Ingreso"] = 0.0
    for des, tabla in param["SUBVENCION"].items():
        m = ind["Desenlace"].eq(des)
        ind.loc[m, "Ingreso"] = ind.loc[m, "Perfil"].map(tabla).astype(float)
    return ind


def ventana_individuos(ind, t_fin, param=PARAM):
    """Selecciona los individuos ingresados en [warm-up, fin - cool-down]; devuelve los finalizados,
    todos los de la ventana y la duración de la ventana (días)."""
    t0, t1 = param["T_WARMUP"], t_fin - param["T_COOLDOWN"]
    sel = ind[(ind["T_llegada"] >= t0) & (ind["T_llegada"] < t1)]
    return sel[sel["Finalizado"]], sel, (t1 - t0)


def kpis_operativos(ind_fin):
    """KPIs temporales (días) globales y por perfil sobre individuos finalizados."""
    def resumen(x):
        return pd.Series({
            "Individuos": len(x),
            "T_sys_media": x["T_sys"].mean(), "T_sys_mediana": x["T_sys"].median(),
            "T_sys_P90": x["T_sys"].quantile(0.9),
            "T_ser_media": x["T_ser"].mean(), "T_q_media": x["T_q"].mean(),
            "T_q_P90": x["T_q"].quantile(0.9),
            "Ratio_espera_medio": x["Ratio_espera"].mean(),
            "Ratio_espera_agregado": x["T_q"].sum() / x["T_sys"].sum() if x["T_sys"].sum() > 0 else np.nan,
        })
    tab = ind_fin.groupby("Perfil").apply(resumen, include_groups=False).reindex(TIPOS)
    tab.loc["Global"] = resumen(ind_fin)
    return tab


def kpis_estaciones(d, t0, t1):
    """Carga y espera por estación (visitas con llegada en [t0, t1))."""
    x = d[(d["T_in"] >= t0) & (d["T_in"] < t1) & ~d["Fin"]].copy()
    x["Espera"] = x["T_wait"] + x["T_hueco"]
    tab = x.groupby("Station_in").agg(
        Visitas=("Id_ind", "count"),
        Espera_media_d=("Espera", "mean"),
        Espera_P90_d=("Espera", lambda s: s.quantile(0.9)),
        Servicio_medio_d=("T_server", "mean"),
        Carga_total_d=("T_server", "sum"),
        Espera_total_d=("Espera", "sum"),
        Cola_media_llegada=("Ind_en_cola", "mean"),
        Cola_max_llegada=("Ind_en_cola", "max"),
    )
    tab["%Espera_total"] = tab["Espera_total_d"] / tab["Espera_total_d"].sum()
    return tab.sort_values("Espera_total_d", ascending=False)


def _clip_dur(ini, fin, t0, t1):
    """Duración del solape de los intervalos [ini, fin] con la ventana [t0, t1]."""
    return (np.minimum(fin, t1) - np.maximum(ini, t0)).clip(lower=0)


def capacidades(cfg):
    """Capacidad de cada recurso (servidores, jaulas, Recinto de Recuperación y bolsas de UCI)."""
    cap = dict(cfg.SERVIDORES_CAPACIDAD)
    cap.update(cfg.JAULAS_CAPACIDAD)
    cap[RECINTO] = cfg.RECINTO_CAPACIDAD
    if cfg.UCI_MODE == "pooled":
        cap[UCI] = cfg.UCI_TOTAL_CAPACIDAD
    else:
        cap["UCI_Critico"] = cfg.UCI_CRITICO_CAPACIDAD
        cap["UCI_Leve"] = cfg.UCI_LEVE_CAPACIDAD
        cap["UCI_Base"] = cfg.UCI_BASE_CAPACIDAD
    return cap


def estaciones_jaula(cfg):
    """Estaciones que retienen una jaula/plaza: jaulas físicas, sus alias y la UCI."""
    return set(cfg.JAULAS_CAPACIDAD) | set(getattr(cfg, "JAULA_ALIAS", {})) | {UCI}


def _nombre_recurso(sala, tipo, cfg):
    """Traduce el recinto físico de la traza al nombre del recurso (bolsas de UCI)."""
    if cfg.UCI_MODE == "pooled":
        return np.asarray(sala)
    bolsa = np.select([tipo.eq("RescateCritico"), tipo.eq("RescateLeve")], ["UCI_Critico", "UCI_Leve"], "UCI_Base")
    return np.where(sala.eq(UCI), bolsa, sala)


def segmentos_ocupacion(d, cfg):
    """Segmentos de ocupación de recursos físicos (Recinto, jaulas, UCI).
    Tipo 'Estancia': el individuo ocupa la jaula/recinto en su estación.
    Tipo 'Bloqueo'  : el individuo ocupa la jaula/recinto anterior esperando plaza en otra unidad."""
    es_jaula = d["Station_in"].isin(estaciones_jaula(cfg)) & ~d["Fin"]
    # En la fila de captura del Recinto, el hueco es la espera de plaza (al principio de la fila)
    ini = np.where(d["Fin"], d["T_in"], d["T_in"] + np.where(es_jaula, d["T_wait"], d["T_hueco"]))
    est = pd.DataFrame({"Recurso": _nombre_recurso(d["Recinto"], d["Perfil"], cfg),
                        "Ini": ini, "Fin": d["T_end_real"].values, "Tipo": "Estancia",
                        "Perfil": d["Perfil"].values, "Id_ind": d["Id_ind"].values})
    b = d[es_jaula & (d["T_wait"] > 0)]
    blo = pd.DataFrame({"Recurso": _nombre_recurso(b["Recinto_prev"], b["Perfil"], cfg),
                        "Ini": b["T_in"].values, "Fin": (b["T_in"] + b["T_wait"]).values,
                        "Tipo": "Bloqueo", "Perfil": b["Perfil"].values, "Id_ind": b["Id_ind"].values})
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
    ec = d[d["Station_in"].isin(estaciones_jaula(cfg))]
    rec_ec = pd.Series(_nombre_recurso(ec["Station_in"].replace(getattr(cfg, "JAULA_ALIAS", {})), ec["Perfil"], cfg), index=ec.index)
    cola_c = _clip_dur(ec["T_in"], ec["T_in"] + ec["T_wait"], t0, t1).groupby(rec_ec).sum()
    hb = d[d["T_hueco"] > 0]
    cola_c.loc[RECINTO] = _clip_dur(hb["T_in"], hb["T_in"] + hb["T_hueco"], t0, t1).sum()
    for r in cap:
        if r in cfg.SERVIDORES_CAPACIDAD:
            continue
        filas[r] = {"Tipo": "Jaula/plaza", "Capacidad": cap[r], "Ocupacion_d": ocup_c.get(r, 0.0),
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


def kpis_biologicos(ind_fin, dias_ventana):
    """Liberación, mortalidad, cautividad permanente, derivación a UCI y al Módulo de Cría (global y por perfil)."""
    def resumen(x):
        return pd.Series({
            "Individuos": len(x),
            "Tasa_liberacion": (x["Desenlace"] == "Liberado").mean(),
            "Tasa_mortalidad": (x["Desenlace"] == "Fallece").mean(),
            "Tasa_cautividad": (x["Desenlace"] == "Cautividad_Permanente").mean(),
            "Tasa_mort_Sin_UCI": x["Sin_UCI"].mean(),
            "Liberados_anio": (x["Desenlace"] == "Liberado").sum() / dias_ventana * 365,
            "Tasa_Modulo_Cria": x["Modulo_Cria"].mean(),
            "Tasa_UCI": x["UCI"].mean(),
            "Tasa_UCI_desde_Ingreso": (x["UCI_desde"] == "Ingreso").mean(),
            "Tasa_UCI_desde_Cirugia": (x["UCI_desde"] == "Cirugia").mean(),
            "Tasa_UCI_desde_Modulo_Cria": (x["UCI_desde"] == "Modulo_Cria").mean(),
            "Tasa_Cirugia": x["Cirugia"].mean(),
        })
    tab = ind_fin.groupby("Perfil").apply(resumen, include_groups=False).reindex(TIPOS)
    tab.loc["Global"] = resumen(ind_fin)
    return tab


def coste_fijo_periodo(cfg, dias_ventana):
    """Coste fijo imputable a la ventana: los anuales se prorratean (días/365) y los únicos se
    imputan íntegramente (criterio prudente)."""
    return sum(v * dias_ventana / 365 if "anual" in k else v for k, v in getattr(cfg, "COSTES_FIJOS", {}).items())


def kpis_economicos(ind_fin, cfg, dias_ventana, param=PARAM):
    """Subvención, costes, penalizaciones, margen operativo bruto y neto (tras coste fijo)."""
    def resumen(x):
        ing, cv, pen = x["Ingreso"].sum(), x["Coste_var"].sum(), x["Penalizacion"].sum()
        sub = {f"Subvencion_{des}": x.loc[x["Desenlace"] == des, "Ingreso"].sum() for des in param["SUBVENCION"]}
        return pd.Series({"Individuos": len(x), "Subvencion_total": ing, **sub, "Coste_variable": cv,
                          "Penalizaciones": pen, "Coste_total": cv + pen,
                          "Coste_medio_individuo": (cv + pen) / max(len(x), 1),
                          "Subvencion_media_individuo": ing / max(len(x), 1),
                          "Beneficio": ing - cv - pen,
                          "Margen": (ing - cv - pen) / ing if ing > 0 else np.nan})
    tab = ind_fin.groupby("Perfil").apply(resumen, include_groups=False).reindex(TIPOS)
    tab.loc["Global"] = resumen(ind_fin)
    g = tab.loc["Global"]
    cf = coste_fijo_periodo(cfg, dias_ventana)
    anios = dias_ventana / 365
    glob = {
        "Subvencion_anio": g["Subvencion_total"] / anios,
        "Coste_total_anio": g["Coste_total"] / anios,
        "Coste_medio_individuo": g["Coste_medio_individuo"],
        "Beneficio_operativo_anio": g["Beneficio"] / anios,
        "Margen_operativo": g["Margen"],
        "Coste_fijo_periodo": cf,
        "Margen_neto": (g["Beneficio"] - cf) / g["Subvencion_total"] if g["Subvencion_total"] > 0 else np.nan,
        "Beneficio_neto_anio": (g["Beneficio"] - cf) / anios,
        "Margen_exigido": param["MARGEN_EXIGIDO"],
    }
    return tab, glob


def evaluar_slas(ind_fin, seg, cfg, t0, t1, econ_glob, param=PARAM):
    """Evalúa los seis SLAs de la Sección A.08 (el SLA2 se evalúa por separado para Fundador y Cautivo).
    Devuelve un DataFrame con la métrica, su valor, el umbral, si se cumple («Sí»/«No») y un detalle."""
    filas = []
    lim = param["SLA_LATENCIA_D"]
    # 1) Latencia Admisión -> Evaluación Veterinaria Inicial (RescateCritico)
    rc = ind_fin[ind_fin["Perfil"].isin(PERFILES_SLA_LATENCIA)]
    lat = rc["Latencia_EVI"].dropna()
    pct_ok = (lat <= lim).mean() if len(lat) else 0.0
    filas.append((f"SLA1 Latencia EVI ≤ {lim:.2f} d (RescateCritico)", f"% individuos ≤ {lim:.2f} d", pct_ok,
                  f"≥ {param['SLA_LATENCIA_CUMPL']:.0%}", pct_ok >= param["SLA_LATENCIA_CUMPL"],
                  f"latencia media {lat.mean() if len(lat) else 0:.3f} d · P95 {lat.quantile(0.95) if len(lat) else 0:.3f} d"))
    # 2) Ratio de espera (Fundador y Cautivo)
    for t in PERFILES_SLA_ESPERA:
        x = ind_fin[ind_fin["Perfil"] == t]
        r = x["Ratio_espera"].mean() if len(x) else 0.0
        filas.append((f"SLA2 Ratio espera ({t})", "T_q/T_sys medio", r,
                      f"< {param['SLA_RATIO_ESPERA']:.0%}", r < param["SLA_RATIO_ESPERA"],
                      f"T_q medio {x['T_q'].mean() if len(x) else 0:.1f} d de {x['T_sys'].mean() if len(x) else 0:.1f} d en el sistema"))
    # 3) Ocupación estructural de UCI ≤ 90%
    rec_uci = [r for r in capacidades(cfg) if r.startswith("UCI")]
    occ = ocupacion_periodica(seg, rec_uci, cfg.UCI_TOTAL_CAPACIDAD, t0, t1)
    filas.append((f"SLA3 Ocupación media UCI ≤ {param['SLA_OCUP_UCI']:.0%}", "ocupación media", occ.mean(),
                  f"≤ {param['SLA_OCUP_UCI']:.0%}", occ.mean() <= param["SLA_OCUP_UCI"],
                  f"{(occ > param['SLA_OCUP_UCI']).mean():.1%} de los meses por encima del {param['SLA_OCUP_UCI']:.0%}"))
    # 4) Mortalidad evitable Sin_UCI en RescateLeve/RescateCritico
    resc = ind_fin[ind_fin["Perfil"].isin(PERFILES_RESCATE)]
    m = resc["Sin_UCI"].mean() if len(resc) else 0.0
    filas.append((f"SLA4 Muertes Sin_UCI < {param['SLA_SIN_UCI']:.0%} (rescates)", "tasa", m,
                  f"< {param['SLA_SIN_UCI']:.0%}", m < param["SLA_SIN_UCI"],
                  f"{int(resc['Sin_UCI'].sum())} fallecimientos por falta de plaza de UCI"))
    # 5) Tasa de liberación global
    lib = (ind_fin["Desenlace"] == "Liberado").mean() if len(ind_fin) else 0.0
    filas.append((f"SLA5 Tasa de liberación ≥ {param['SLA_LIBERACION']:.0%}", "tasa", lib,
                  f"≥ {param['SLA_LIBERACION']:.0%}", lib >= param["SLA_LIBERACION"],
                  f"{int((ind_fin['Desenlace'] == 'Liberado').sum())} liberados de {len(ind_fin)} individuos finalizados"))
    # 6) Margen operativo neto ≥ margen exigido
    filas.append(("SLA6 Margen neto ≥ margen exigido", "margen neto", econ_glob["Margen_neto"],
                  f"≥ {econ_glob['Margen_exigido']:.0%}", econ_glob["Margen_neto"] >= econ_glob["Margen_exigido"],
                  f"margen operativo antes de coste fijo {econ_glob['Margen_operativo']:.1%}"))
    out = pd.DataFrame(filas, columns=["SLA", "Métrica", "Valor", "Umbral", "Cumple", "Detalle"]).set_index("SLA")
    out["Cumple"] = out["Cumple"].map({True: "Sí", False: "No"})
    return out


CLAVES_SLA = {"SLA1": "SLA1", "SLA2_Fundador": "SLA2 Ratio espera (Fundador)",
              "SLA2_Cautivo": "SLA2 Ratio espera (Cautivo)", "SLA3": "SLA3", "SLA4": "SLA4",
              "SLA5": "SLA5", "SLA6": "SLA6"}


def calcular_kpis(df, cfg, dias=None, param=PARAM):
    """Calcula TODOS los KPIs y SLAs de una traza (histórica o simulada).
    Devuelve un diccionario con las tablas y un resumen plano `resumen` (una cifra por KPI),
    pensado para acumular réplicas en el análisis Monte Carlo."""
    d = preparar_traza(df)
    t_fin = dias if dias is not None else horizonte(df)
    t0, t1 = param["T_WARMUP"], t_fin - param["T_COOLDOWN"]
    ind = tabla_individuos(d, param)
    ind_fin, ind_vent, dias_v = ventana_individuos(ind, t_fin, param)
    op = kpis_operativos(ind_fin)
    est = kpis_estaciones(d, t0, t1)
    rec = kpis_recursos(d, cfg, t0, t1)
    seg = segmentos_ocupacion(d, cfg)
    bio = kpis_biologicos(ind_fin, dias_v)
    eco_tab, eco = kpis_economicos(ind_fin, cfg, dias_v, param)
    sla = evaluar_slas(ind_fin, seg, cfg, t0, t1, eco, param)

    res = {"Individuos_ventana": len(ind_vent), "Individuos_finalizados": len(ind_fin),
           "Pct_finalizados": len(ind_fin) / max(len(ind_vent), 1), "Ingresos_anio": len(ind_vent) / dias_v * 365}
    for perfil in TIPOS + ["Global"]:
        for k in ["T_sys_media", "T_ser_media", "T_q_media", "Ratio_espera_medio"]:
            res[f"{k}_{perfil}"] = op.loc[perfil, k]
        for k in ["Tasa_liberacion", "Tasa_mortalidad", "Tasa_cautividad", "Tasa_mort_Sin_UCI", "Tasa_Modulo_Cria", "Tasa_UCI"]:
            res[f"{k}_{perfil}"] = bio.loc[perfil, k]
    for k in ["Liberados_anio", "Tasa_UCI_desde_Ingreso", "Tasa_UCI_desde_Cirugia", "Tasa_UCI_desde_Modulo_Cria", "Tasa_Cirugia"]:
        res[k] = bio.loc["Global", k]
    for r in rec.index:
        res[f"Util_{r}"] = rec.loc[r, "Utilizacion"]
        res[f"Lq_{r}"] = rec.loc[r, "Lq_media"]
        if rec.loc[r, "Tipo"] != "Servidor":
            res[f"Bloqueo_d_anio_{r}"] = rec.loc[r, "Bloqueo_d"] / ((t1 - t0) / 365)
    res.update(eco)
    rc = ind_fin["Perfil"].isin(PERFILES_SLA_LATENCIA)
    res["Latencia_EVI_media_RC_d"] = ind_fin.loc[rc, "Latencia_EVI"].mean()
    res["Mort_Sin_UCI_rescates"] = ind_fin.loc[ind_fin["Perfil"].isin(PERFILES_RESCATE), "Sin_UCI"].mean()
    for k, pref in CLAVES_SLA.items():
        fila = sla.loc[sla.index.str.startswith(pref)].iloc[0]
        res[f"SLA_valor_{k}"] = fila["Valor"]
        res[f"Cumple_{k}"] = fila["Cumple"] == "Sí"
    res["Ocup_media_UCI"] = res["SLA_valor_SLA3"]
    rec_uci = [r for r in capacidades(cfg) if r.startswith("UCI")]
    res["Pct_meses_UCI_90"] = (ocupacion_periodica(seg, rec_uci, cfg.UCI_TOTAL_CAPACIDAD, t0, t1) > param["SLA_OCUP_UCI"]).mean()
    res["Pct_RC_con_demora"] = ind_fin.loc[rc, "Demora_critica"].mean() if rc.any() else 0.0
    # Acceso a la UCI por perfil (espera de plaza y fallos sobre las peticiones)
    uci = d[d["Station_in"].eq(UCI) & (d["T_in"] >= t0) & (d["T_in"] < t1)]
    for perfil in TIPOS:
        u = uci[uci["Perfil"] == perfil]
        res[f"Peticiones_UCI_anio_{perfil}"] = len(u) / dias_v * 365
        # si en la réplica no hay peticiones de ese perfil, no hay espera ni fallos: se registra 0
        res[f"Espera_UCI_media_{perfil}"] = u["T_wait"].mean() if len(u) else 0.0
        res[f"Espera_UCI_P95_{perfil}"] = u["T_wait"].quantile(0.95) if len(u) else 0.0
        res[f"Pct_peticiones_UCI_fallidas_{perfil}"] = u["Sin_UCI"].mean() if len(u) else 0.0
    # Uso del secuenciador genético (recurso escaso)
    pg = d[d["Station_in"].eq("Pruebas_Geneticas_Avanzadas") & (d["T_in"] >= t0) & (d["T_in"] < t1) & ~d["Fin"]]
    res["Espera_media_Pruebas_Geneticas"] = pg["T_wait"].mean() if len(pg) else 0.0

    return {"traza": d, "individuos": ind, "individuos_fin": ind_fin, "operativos": op,
            "estaciones": est, "recursos": rec, "segmentos": seg, "biologicos": bio,
            "economicos": eco_tab, "economicos_glob": eco, "slas": sla, "resumen": res,
            "dias_ventana": dias_v, "t0": t0, "t1": t1, "t_fin": t_fin}


TABLAS_MC = ["operativos", "biologicos", "estaciones", "recursos", "economicos"]


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
    if tabla in ("operativos", "biologicos", "economicos"):
        media = media.reindex(TIPOS + ["Global"])
    return media


def slas_mc(mc):
    """Para cada SLA: valor medio de la métrica, IC95 y % de réplicas que lo cumplen."""
    r = mc["resumen"]
    p = PARAM
    nombres = {"SLA1": f"Latencia EVI ≤ {p['SLA_LATENCIA_D']:.2f} d (RescateCritico): % que cumplen (≥{p['SLA_LATENCIA_CUMPL']:.0%})",
               "SLA2_Fundador": f"Ratio espera Fundador (<{p['SLA_RATIO_ESPERA']:.0%})",
               "SLA2_Cautivo": f"Ratio espera Cautivo (<{p['SLA_RATIO_ESPERA']:.0%})",
               "SLA3": f"Ocupación media UCI Veterinaria (≤{p['SLA_OCUP_UCI']:.0%})",
               "SLA4": f"Mortalidad Sin_UCI en rescates (<{p['SLA_SIN_UCI']:.0%})",
               "SLA5": f"Tasa de liberación global (≥{p['SLA_LIBERACION']:.0%})",
               "SLA6": f"Margen neto ≥ margen exigido ({p['MARGEN_EXIGIDO']:.0%})"}
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
        # sin variación entre réplicas: diferencia nula (p = 1) o constante (p = 0)
        pval[k] = (stats.ttest_1samp(x, 0).pvalue if len(x) > 1 and x.std() > 0
                   else (1.0 if (len(x) == 0 or abs(x.mean()) < 1e-12) else 0.0))
    out = pd.DataFrame({"Base": b.mean(), "Escenario": e.mean(), "Δ": dif.mean(),
                        "Δ%": _div(dif.mean(), b.mean().abs()),
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
        "Cobertura del coste anual (Δ/coste)": (m / anual) if anual else "no aplica (sin coste anual)",
        "Periodo de amortización del coste único (años)": ((unico / m) if m > 0 else "no se amortiza") if unico
        else "no aplica (sin coste único)",
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
    + ["Latencia_EVI_media_RC_d", "SLA_valor_SLA1", "Pct_RC_con_demora"]
    + [f"Tasa_liberacion_{t}" for t in TIPOS + ["Global"]]
    + [f"Tasa_mortalidad_{t}" for t in TIPOS + ["Global"]]
    + [f"Tasa_cautividad_{t}" for t in TIPOS + ["Global"]]
    + ["Liberados_anio", "Tasa_mort_Sin_UCI_Global", "Mort_Sin_UCI_rescates", "Tasa_Modulo_Cria_Global",
       "Tasa_UCI_Global", "Tasa_UCI_desde_Ingreso", "Tasa_UCI_desde_Cirugia", "Tasa_UCI_desde_Modulo_Cria"]
    + ["Util_Equipo_Veterinario", "Util_Pruebas_Geneticas_Avanzadas", "Espera_media_Pruebas_Geneticas",
       "Util_Recinto_Recuperacion", "Util_Cuarentena", "Util_Cirugia_Veterinaria", "Util_Modulo_Cria",
       "Util_Recuperacion_PostCritica", "Ocup_media_UCI", "Pct_meses_UCI_90",
       "Bloqueo_d_anio_Recinto_Recuperacion", "Bloqueo_d_anio_Cirugia_Veterinaria", "Bloqueo_d_anio_Cuarentena"]
    + ["Coste_medio_individuo", "Subvencion_anio", "Beneficio_operativo_anio", "Margen_operativo",
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
        rec_uci = [r for r in capacidades(cfg) if r.startswith("UCI")]
        detalle = {
            "individuos_fin": k["individuos_fin"][["Perfil", "T_sys", "T_ser", "T_q", "Ratio_espera", "Desenlace",
                                                   "Latencia_EVI"]].copy(),
            "ocupacion_uci": ocupacion_periodica(k["segmentos"], rec_uci, cfg.UCI_TOTAL_CAPACIDAD, k["t0"], k["t1"]),
            "ocupacion_uci_bolsas": {r: ocupacion_periodica(k["segmentos"], [r], capacidades(cfg)[r], k["t0"], k["t1"])
                                     for r in rec_uci},
        }
    return dict(k["resumen"]), tablas, detalle


def _replica_worker(args):
    cfg, dias, semilla, param, i = args
    res, tablas, det = ejecutar_replica(cfg, dias, semilla, param, guardar_detalle=(i == 0))
    return i, res, tablas, det


def montecarlo(cfg, n_sim=30, dias=3650, semilla_base=1000, param=None, progreso=None,
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
    if k.startswith(("Tasa_liberacion", "SLA_valor_SLA1", "Margen", "Beneficio", "Subvencion", "Liberados_anio")):
        return +1
    if k.startswith(("T_sys", "T_q", "Ratio_espera", "Latencia", "Tasa_mortalidad", "Tasa_mort_Sin_UCI",
                     "Mort_Sin_UCI", "Tasa_cautividad", "Coste", "Bloqueo", "Espera_", "Pct_peticiones_UCI_fallidas",
                     "Pct_meses_UCI_90", "Pct_RC_con_demora", "Lq_")):
        return -1
    return 0


ETIQUETAS_KPI = {
    "T_sys_media_Fundador": "T. medio en el sistema · Fundador (días)",
    "T_sys_media_Cautivo": "T. medio en el sistema · Cautivo (días)",
    "T_sys_media_RescateLeve": "T. medio en el sistema · RescateLeve (días)",
    "T_sys_media_RescateCritico": "T. medio en el sistema · RescateCritico (días)",
    "T_sys_media_Global": "T. medio en el sistema · Global (días)",
    "T_q_media_Global": "T. medio en cola · Global (días)",
    "Latencia_EVI_media_RC_d": "Latencia hasta Eval. Vet. Inicial · RescateCritico (días)",
    "Pct_RC_con_demora": "% RescateCritico con demora crítica (> latencia SLA1)",
    "SLA_valor_SLA1": "% RescateCritico atendidos en ≤ 0,25 días (6 h)",
    "Tasa_liberacion_Fundador": "Liberación · Fundador",
    "Tasa_liberacion_Cautivo": "Liberación · Cautivo",
    "Tasa_liberacion_RescateLeve": "Liberación · RescateLeve",
    "Tasa_liberacion_RescateCritico": "Liberación · RescateCritico",
    "Tasa_liberacion_Global": "Tasa de liberación global",
    "Tasa_cautividad_Global": "Tasa de cautividad permanente",
    "Tasa_mortalidad_Fundador": "Mortalidad · Fundador",
    "Tasa_mortalidad_Cautivo": "Mortalidad · Cautivo",
    "Tasa_mortalidad_RescateLeve": "Mortalidad · RescateLeve",
    "Tasa_mortalidad_RescateCritico": "Mortalidad · RescateCritico",
    "Tasa_mortalidad_Global": "Mortalidad · Global",
    "Tasa_mort_Sin_UCI_Global": "Mortalidad Sin_UCI (s/ total individuos)",
    "Mort_Sin_UCI_rescates": "Mortalidad Sin_UCI (s/ rescates)",
    "Liberados_anio": "Individuos liberados al año",
    "Ocup_media_UCI": "Ocupación media UCI Veterinaria",
    "Util_Modulo_Cria": "Utilización Módulo de Cría",
    "Util_Pruebas_Geneticas_Avanzadas": "Utilización Pruebas Genéticas",
    "Util_Recinto_Recuperacion": "Utilización Recinto de Recuperación",
    "Util_Equipo_Veterinario": "Utilización Equipo Veterinario",
    "Coste_medio_individuo": "Coste medio por individuo (€)",
    "Subvencion_anio": "Subvención recibida (€/año)",
    "Beneficio_operativo_anio": "Beneficio operativo (€/año)",
    "Margen_operativo": "Margen operativo bruto",
    "Margen_neto": "Margen operativo neto (tras coste fijo)",
    "Margen_exigido": "Margen exigido por el patronato",
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
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, d=1):
    if isinstance(x, str):
        return x
    return "—" if x is None or pd.isna(x) else num(100 * x, d) + " %"


def eur(x, d=0):
    if isinstance(x, str):
        return x
    return "—" if x is None or pd.isna(x) else num(x, d) + " €"


def fmt_kpi(k, v):
    """Formatea un valor según el tipo de KPI."""
    if isinstance(v, str):
        return v
    if pd.isna(v):
        return "—"
    if k.startswith(("Tasa_", "Util_", "Ratio_", "Margen", "Ocup", "Pct_", "Mort_", "SLA_valor_SLA1", "Cumple", "%")):
        return pct(v, 2 if k.startswith(("Tasa_mort_Sin", "Mort_Sin")) else 1)
    if k in ("Ingresos_anio", "Liberados_anio") or k.startswith("Peticiones_"):
        return num(v, 1)
    if any(t in k for t in ("Coste", "Subvencion", "Beneficio", "€", "Resultado")):
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
                "Liberación global": m["resumen"]["Tasa_liberacion_Global"].mean(),
                "Mortalidad global": m["resumen"]["Tasa_mortalidad_Global"].mean(),
                "Mortalidad Sin_UCI rescates": m["resumen"]["Mort_Sin_UCI_rescates"].mean(),
                "T_sys global (días)": m["resumen"]["T_sys_media_Global"].mean(),
                "Margen neto": m["resumen"]["Margen_neto"].mean()}
        if et == ref:
            fila.update({"Resultado neto anual (€)": 0.0, "Resultado neto 1er año (€)": 0.0, "Descartado": False})
        else:
            e = resultado_economico(base, m)
            c = comparar_escenarios(base, m, ["Tasa_mortalidad_Global", "Mort_Sin_UCI_rescates"])
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
        f"Diagnóstico de partida ({ref}). La simulación estima una tasa de liberación del "
        f"{pct(base['resumen']['Tasa_liberacion_Global'].mean())}, una mortalidad global del "
        f"{pct(base['resumen']['Tasa_mortalidad_Global'].mean())}, una tasa de cautividad permanente del "
        f"{pct(base['resumen']['Tasa_cautividad_Global'].mean())} y un margen operativo del "
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
