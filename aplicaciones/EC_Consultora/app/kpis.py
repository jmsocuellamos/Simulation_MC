# -*- coding: utf-8 -*-
"""KPIs, SLAs, análisis Monte Carlo y comparación de escenarios de la consultora de reestructuración de deuda.

Funciones del cuaderno '04 Reestructuración de deuda' (Tareas 1 y 2), con las herramientas de síntesis y
redacción automática de la aplicación. Todos los tiempos en días."""
import math
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

from motor import TIPOS, simular

DESENLACES = ["Regularizado", "Liquidacion_Ordenada", "Insolvencia_Total"]


PARAM = {
    # Honorarios por desenlace (€) según perfil (Sección A.05)
    "HONORARIOS": {
        "Regularizado":         {"Preventivo": 840, "Impago_Leve": 2625, "Impago_Grave": 8600, "Riesgo_Concursal": 16200},
        "Liquidacion_Ordenada": {"Preventivo": 150, "Impago_Leve": 350, "Impago_Grave": 1100, "Riesgo_Concursal": 1400},
    },
    # Coste de oportunidad por insolvencia total (€) según perfil
    "PEN_INSOLVENCIA": {"Preventivo": 450, "Impago_Leve": 1100, "Impago_Grave": 6600, "Riesgo_Concursal": 16500},
    # Penalización por demora: expediente Riesgo_Concursal con latencia hasta el Diagnóstico Financiero > SLA_LATENCIA_D
    "PEN_DEMORA": 500,
    # Coste por procedimiento realizado (€)
    "COSTE_SERVICIO": {"Peritaje_Independiente": 900, "Consulta_Juridica": 400, "Auditoria_Contable": 150,
                       "Simulacion_Estres_Financiero": 300, "Comite_Evaluacion": 80, "Diagnostico_Financiero": 60,
                       "Diagnostico_Financiero_IT": 60, "Apertura_Expediente": 20, "Homologacion_Final": 120},
    # Coste por día de estancia (€/día)
    "COSTE_DIA": {"Recopilacion_Documental": 15, "Mesa_Negociacion": 40, "Mesa_Negociacion_PostIntervencion": 40,
                  "Reestructuracion_Profunda": 60, "Intervencion_Judicial": 150},
    # Margen operativo objetivo de la dirección (15-20%): se exige el límite inferior
    "MARGEN_EXIGIDO": 0.15,
    "MARGEN_OBJETIVO_SUP": 0.20,
    # SLAs
    "SLA_LATENCIA_D": 1.0,              # 1 día hasta Diagnóstico Financiero (Riesgo_Concursal)
    "SLA_LATENCIA_CUMPL": 0.95,         # se exige que el 95% de los expedientes Riesgo_Concursal lo cumpla
    "SLA_RATIO_ESPERA": 0.08,           # Preventivo/Impago_Leve
    "SLA_OCUP_INT": 0.90,
    "SLA_SIN_INT": 0.02,                # sobre Impago_Grave/Riesgo_Concursal
    "SLA_REGULARIZACION": 0.77,         # tasa de regularización global mínima
    # Ventanas de análisis (días)
    "T_WARMUP": 180,
    "T_COOLDOWN": 365,
}


PERFILES_SLA_LATENCIA = ["Riesgo_Concursal"]


PERFILES_SLA_ESPERA = ["Preventivo", "Impago_Leve"]


PERFILES_GRAVES = ["Impago_Grave", "Riesgo_Concursal"]


ESTACION_PRIMER_CONTACTO = ("Diagnostico_Financiero", "Diagnostico_Financiero_IT")


INTERV = "Intervencion_Judicial"


MESA_PRIO = "Mesa_Prioritaria"


ESTACIONES_DEL_ANIO = [("Q1", 0, 90), ("Q2", 90, 181), ("Q3", 181, 273), ("Q4", 273, 365)]


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
    """Etiqueta de la periodo del año a la que pertenece el día del año `dia` (0-364)."""
    for nombre, ini, fin in ESTACIONES_DEL_ANIO:
        if (ini > fin and (dia >= ini or dia < fin)) or (ini < fin and ini <= dia < fin):
            return nombre


def preparar_traza(df):
    """Ordena la traza por expediente y tiempo y añade columnas auxiliares:
    * `T_end_real`: instante real de salida de la estación (= T_in de la fila siguiente).
    * `T_hueco`: espera no registrada en T_wait (espera de plaza en la Mesa Prioritaria).
    * `Mesa_prev`: mesa física que ocupaba el expediente en la fila anterior.
    * `Fin`: True si la fila está cortada por el final de la simulación."""
    d = df.sort_values(["Id_expediente", "T_in", "T_out"], kind="mergesort").reset_index(drop=True).copy()
    g = d.groupby("Id_expediente", sort=False)
    sig = g["T_in"].shift(-1)
    d["T_end_real"] = np.maximum(sig.fillna(d["T_out"]), d["T_out"])
    d["T_hueco"] = (d["T_end_real"] - d["T_out"]).clip(lower=0)
    d.loc[d["T_hueco"] < 1e-3, "T_hueco"] = 0.0
    d["Mesa_prev"] = g["Mesa"].shift(1).fillna("Cartera_Espera")
    d["Fin"] = d["Station_out"].eq("Fin_Simulacion")
    d["Sin_Int"] = d["ID_server"].eq("Sin_Intervencion_Disponible")
    return d


def horizonte(df):
    """Duración total (días) de la traza, redondeada al año."""
    return math.ceil(df["T_out"].max() / 365) * 365


def tasa_ingresos_estacion(df, dias=None):
    """Tasa de ingresos (expedientes/día) y reparto por perfil en cada periodo del año."""
    adm = df.loc[df["Station_in"] == "Apertura_Expediente", ["Id_expediente", "Perfil", "T_in"]].drop_duplicates("Id_expediente")
    if dias is None:
        dias = horizonte(df)
    anios = dias / 365
    adm["Estacion_anio"] = (adm["T_in"].astype(int) % 365).map(estacion_del_anio)
    dias_est = {n: ((fin - ini) % 365) * anios for n, ini, fin in ESTACIONES_DEL_ANIO}
    tab = adm.groupby("Estacion_anio")["Id_expediente"].count().rename("Ingresos").to_frame()
    tab["Días"] = tab.index.map(dias_est)
    tab["Tasa_obs (exp/día)"] = tab["Ingresos"] / tab["Días"]
    tab["Tasa_teórica"] = tab.index.map(TASAS_TEORICAS)
    mix = pd.crosstab(adm["Estacion_anio"], adm["Perfil"], normalize="index").reindex(columns=TIPOS)
    tab = tab.join(mix.add_prefix("%")).reindex([n for n, _, _ in ESTACIONES_DEL_ANIO])
    total = pd.Series({"Ingresos": len(adm), "Días": dias, "Tasa_obs (exp/día)": len(adm) / dias,
                       "Tasa_teórica": sum(TASAS_TEORICAS[n] * ((fin - ini) % 365) for n, ini, fin in ESTACIONES_DEL_ANIO) / 365})
    total = pd.concat([total, adm["Perfil"].value_counts(normalize=True).reindex(TIPOS).add_prefix("%")])
    out = pd.concat([tab, pd.DataFrame([total], index=["Total"])])
    out["Ingresos/año"] = out["Ingresos"] / anios
    return out


def tiempos_servicio(df, cfg=None):
    """Estadísticos del tiempo de servicio (días) por estación y perfil.
    Si se pasa `cfg`, añade el valor nominal de la configuración para comparar."""
    d = df.loc[~df["Station_out"].eq("Fin_Simulacion") & ~df["ID_server"].eq("Sin_Intervencion_Disponible")]
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


def probabilidades_derivacion(df, separar_sin_int=True):
    """Probabilidades de derivación estimadas (Station_in -> Station_out) por perfil.
    Excluye las filas cortadas por el final de la simulación. Si `separar_sin_int`, los
    insolvencias totales por falta de plaza de Intervención Judicial se muestran como destino 'Insolvencia_Sin_Int'."""
    d = df.loc[~df["Station_out"].eq("Fin_Simulacion"), ["Perfil", "Station_in", "Station_out", "ID_server"]].copy()
    if separar_sin_int:
        d.loc[d["ID_server"].eq("Sin_Intervencion_Disponible"), "Station_out"] = "Insolvencia_Sin_Int"
    cnt = d.groupby(["Perfil", "Station_in", "Station_out"]).size().rename("n").reset_index()
    cnt["p"] = cnt["n"] / cnt.groupby(["Perfil", "Station_in"])["n"].transform("sum")
    return cnt


def matriz_derivacion(df, tipo):
    """Matriz de transición (filas: origen, columnas: destino) para un perfil."""
    p = probabilidades_derivacion(df)
    p = p[p["Perfil"] == tipo]
    return p.pivot(index="Station_in", columns="Station_out", values="p").fillna(0).round(3)


def tabla_expedientes(d, param=PARAM):
    """Una fila por expediente con sus tiempos, desenlace, ruta y resultado económico.
    `d` debe venir de `preparar_traza`."""
    g = d.groupby("Id_expediente", sort=False)
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

    visita = lambda est: d.loc[d["Station_in"].eq(est)].groupby("Id_expediente").size().reindex(ind.index).fillna(0) > 0
    ind["Mesa_Negociacion"] = visita("Mesa_Negociacion")
    ind["Estres"] = visita("Simulacion_Estres_Financiero")
    ind["Recopilacion_Documental"] = visita("Recopilacion_Documental")
    ind["Intervencion"] = visita(INTERV)
    ind["Sin_Int"] = d.loc[d["Sin_Int"]].groupby("Id_expediente").size().reindex(ind.index).fillna(0) > 0
    # Origen de la (primera) derivación a Intervención Judicial
    d_int = d.assign(Prev=g["Station_in"].shift(1)).loc[d["Station_in"].eq(INTERV)]
    prev = d_int.groupby("Id_expediente")["Prev"].first().reindex(ind.index)
    ind["Int_desde"] = np.select(
        [prev.isin(["Mesa_Negociacion", "Mesa_Negociacion_PostIntervencion"]), prev.eq("Simulacion_Estres_Financiero"), prev.notna()],
        ["Mesa_Negociacion", "Estres", "Diagnostico"], default="")

    # Latencia Apertura de Expediente -> primer contacto en Diagnóstico Financiero
    it = d.loc[d["Station_in"].isin(ESTACION_PRIMER_CONTACTO) & ~d["Fin"]].groupby("Id_expediente").first()
    contacto = it["T_end_real"] - it["T_server"]
    ind["Latencia_DF"] = contacto.reindex(ind.index) - ind["T_llegada"]

    # Economía por expediente
    ok = d.loc[~d["Fin"]]
    coste = ok["Station_in"].map(param["COSTE_SERVICIO"]).fillna(0) \
        + ok["Station_in"].map(param["COSTE_DIA"]).fillna(0) * ok["T_server"]
    ind["Coste_var"] = coste.groupby(ok["Id_expediente"]).sum().reindex(ind.index).fillna(0)
    # Demora en atención del riesgo concursal: expediente Riesgo_Concursal con latencia hasta el DF superior al SLA
    ind["Demora_critica"] = ind["Perfil"].isin(PERFILES_SLA_LATENCIA) & (ind["Latencia_DF"].fillna(0) > param["SLA_LATENCIA_D"])
    ind["Penalizacion"] = np.where(ind["Desenlace"].eq("Insolvencia_Total"), ind["Perfil"].map(param["PEN_INSOLVENCIA"]), 0) \
        + np.where(ind["Demora_critica"], param["PEN_DEMORA"], 0)
    ind["Ingreso"] = 0.0
    for des, tabla in param["HONORARIOS"].items():
        m = ind["Desenlace"].eq(des)
        ind.loc[m, "Ingreso"] = ind.loc[m, "Perfil"].map(tabla).astype(float)
    return ind


def ventana_expedientes(ind, t_fin, param=PARAM):
    """Selecciona los expedientes ingresadas en [warm-up, fin - cool-down]; devuelve las finalizadas,
    todas las de la ventana y la duración de la ventana (días)."""
    t0, t1 = param["T_WARMUP"], t_fin - param["T_COOLDOWN"]
    sel = ind[(ind["T_llegada"] >= t0) & (ind["T_llegada"] < t1)]
    return sel[sel["Finalizado"]], sel, (t1 - t0)


def kpis_operativos(exp_fin):
    """KPIs temporales (días) globales y por perfil sobre expedientes finalizados."""
    def resumen(x):
        return pd.Series({
            "Expedientes": len(x),
            "T_sys_media": x["T_sys"].mean(), "T_sys_mediana": x["T_sys"].median(),
            "T_sys_P90": x["T_sys"].quantile(0.9),
            "T_ser_media": x["T_ser"].mean(), "T_q_media": x["T_q"].mean(),
            "T_q_P90": x["T_q"].quantile(0.9),
            "Ratio_espera_medio": x["Ratio_espera"].mean(),
            "Ratio_espera_agregado": x["T_q"].sum() / x["T_sys"].sum() if x["T_sys"].sum() > 0 else np.nan,
        })
    tab = exp_fin.groupby("Perfil").apply(resumen, include_groups=False).reindex(TIPOS)
    tab.loc["Global"] = resumen(exp_fin)
    return tab


def kpis_estaciones(d, t0, t1):
    """Carga y espera por estación (visitas con llegada en [t0, t1))."""
    x = d[(d["T_in"] >= t0) & (d["T_in"] < t1) & ~d["Fin"]].copy()
    x["Espera"] = x["T_wait"] + x["T_hueco"]
    tab = x.groupby("Station_in").agg(
        Visitas=("Id_expediente", "count"),
        Espera_media_d=("Espera", "mean"),
        Espera_P90_d=("Espera", lambda s: s.quantile(0.9)),
        Servicio_medio_d=("T_server", "mean"),
        Carga_total_d=("T_server", "sum"),
        Espera_total_d=("Espera", "sum"),
        Cola_media_llegada=("Exp_en_cola", "mean"),
        Cola_max_llegada=("Exp_en_cola", "max"),
    )
    tab["%Espera_total"] = tab["Espera_total_d"] / tab["Espera_total_d"].sum()
    return tab.sort_values("Espera_total_d", ascending=False)


def _clip_dur(ini, fin, t0, t1):
    """Duración del solape de los intervalos [ini, fin] con la ventana [t0, t1]."""
    return (np.minimum(fin, t1) - np.maximum(ini, t0)).clip(lower=0)


def capacidades(cfg):
    """Capacidad de cada recurso (servidores, mesas, Mesa Prioritaria y bolsas de Intervención Judicial)."""
    cap = dict(cfg.SERVIDORES_CAPACIDAD)
    cap.update(cfg.MESAS_CAPACIDAD)
    cap[MESA_PRIO] = cfg.MESA_PRIORITARIA_CAPACIDAD
    if cfg.INTERVENCION_MODE == "pooled":
        cap[INTERV] = cfg.INTERVENCION_TOTAL_CAPACIDAD
    else:
        cap["Int_Concursal"] = cfg.INTERVENCION_CONCURSAL_CAPACIDAD
        cap["Int_Grave"] = cfg.INTERVENCION_GRAVE_CAPACIDAD
        cap["Int_Base"] = cfg.INTERVENCION_BASE_CAPACIDAD
    return cap


def estaciones_mesa(cfg):
    """Estaciones que retienen una mesa/plaza: mesas físicas, sus alias y la Intervención Judicial."""
    return set(cfg.MESAS_CAPACIDAD) | set(getattr(cfg, "MESA_ALIAS", {})) | {INTERV}


def _nombre_recurso(sala, tipo, cfg):
    """Traduce la mesa física de la traza al nombre del recurso (bolsas de Intervención Judicial)."""
    if cfg.INTERVENCION_MODE == "pooled":
        return np.asarray(sala)
    bolsa = np.select([tipo.eq("Riesgo_Concursal"), tipo.eq("Impago_Grave")], ["Int_Concursal", "Int_Grave"], "Int_Base")
    return np.where(sala.eq(INTERV), bolsa, sala)


def segmentos_ocupacion(d, cfg):
    """Segmentos de ocupación de recursos físicos (Mesa Prioritaria, mesas, Intervención Judicial).
    Tipo 'Estancia': el expediente ocupa la mesa en su estación.
    Tipo 'Bloqueo'  : el expediente ocupa la mesa anterior esperando plaza en otra unidad."""
    es_mesa = d["Station_in"].isin(estaciones_mesa(cfg)) & ~d["Fin"]
    # En la fila de captura de la Mesa Prioritaria, el hueco es la espera de plaza (al principio de la fila)
    ini = np.where(d["Fin"], d["T_in"], d["T_in"] + np.where(es_mesa, d["T_wait"], d["T_hueco"]))
    est = pd.DataFrame({"Recurso": _nombre_recurso(d["Mesa"], d["Perfil"], cfg),
                        "Ini": ini, "Fin": d["T_end_real"].values, "Tipo": "Estancia",
                        "Perfil": d["Perfil"].values, "Id_expediente": d["Id_expediente"].values})
    b = d[es_mesa & (d["T_wait"] > 0)]
    blo = pd.DataFrame({"Recurso": _nombre_recurso(b["Mesa_prev"], b["Perfil"], cfg),
                        "Ini": b["T_in"].values, "Fin": (b["T_in"] + b["T_wait"]).values,
                        "Tipo": "Bloqueo", "Perfil": b["Perfil"].values, "Id_expediente": b["Id_expediente"].values})
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
    ec = d[d["Station_in"].isin(estaciones_mesa(cfg))]
    rec_ec = pd.Series(_nombre_recurso(ec["Station_in"].replace(getattr(cfg, "MESA_ALIAS", {})), ec["Perfil"], cfg), index=ec.index)
    cola_c = _clip_dur(ec["T_in"], ec["T_in"] + ec["T_wait"], t0, t1).groupby(rec_ec).sum()
    hb = d[d["T_hueco"] > 0]
    cola_c.loc[MESA_PRIO] = _clip_dur(hb["T_in"], hb["T_in"] + hb["T_hueco"], t0, t1).sum()
    for r in cap:
        if r in cfg.SERVIDORES_CAPACIDAD:
            continue
        filas[r] = {"Tipo": "Mesa", "Capacidad": cap[r], "Ocupacion_d": ocup_c.get(r, 0.0),
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


def kpis_recuperacion(exp_fin, dias_ventana):
    """Regularización, liquidación ordenada, insolvencia total, derivación a Intervención Judicial y al Simulación de Estrés (global y por perfil)."""
    def resumen(x):
        return pd.Series({
            "Expedientes": len(x),
            "Tasa_regularizacion": (x["Desenlace"] == "Regularizado").mean(),
            "Tasa_insolvencia": (x["Desenlace"] == "Insolvencia_Total").mean(),
            "Tasa_liquidacion": (x["Desenlace"] == "Liquidacion_Ordenada").mean(),
            "Tasa_insolv_Sin_Int": x["Sin_Int"].mean(),
            "Regularizados_anio": (x["Desenlace"] == "Regularizado").sum() / dias_ventana * 365,
            "Tasa_Negociacion": x["Mesa_Negociacion"].mean(),
            "Tasa_Int": x["Intervencion"].mean(),
            "Tasa_Int_desde_Diagnostico": (x["Int_desde"] == "Diagnostico").mean(),
            "Tasa_Int_desde_Estres": (x["Int_desde"] == "Estres").mean(),
            "Tasa_Int_desde_Negociacion": (x["Int_desde"] == "Mesa_Negociacion").mean(),
            "Tasa_Estres": x["Estres"].mean(),
        })
    tab = exp_fin.groupby("Perfil").apply(resumen, include_groups=False).reindex(TIPOS)
    tab.loc["Global"] = resumen(exp_fin)
    return tab


def coste_fijo_periodo(cfg, dias_ventana):
    """Coste fijo imputable a la ventana: los anuales se prorratean (días/365) y los únicos se
    imputan íntegramente (criterio prudente)."""
    return sum(v * dias_ventana / 365 if "anual" in k else v for k, v in getattr(cfg, "COSTES_FIJOS", {}).items())


def kpis_economicos(exp_fin, cfg, dias_ventana, param=PARAM):
    """Facturación, costes, penalizaciones, margen operativo bruto y neto (tras coste fijo)."""
    def resumen(x):
        ing, cv, pen = x["Ingreso"].sum(), x["Coste_var"].sum(), x["Penalizacion"].sum()
        sub = {f"Facturacion_{des}": x.loc[x["Desenlace"] == des, "Ingreso"].sum() for des in param["HONORARIOS"]}
        return pd.Series({"Expedientes": len(x), "Facturacion_total": ing, **sub, "Coste_variable": cv,
                          "Penalizaciones": pen, "Coste_total": cv + pen,
                          "Coste_medio_expediente": (cv + pen) / max(len(x), 1),
                          "Facturacion_media_expediente": ing / max(len(x), 1),
                          "Beneficio": ing - cv - pen,
                          "Margen": (ing - cv - pen) / ing if ing > 0 else np.nan})
    tab = exp_fin.groupby("Perfil").apply(resumen, include_groups=False).reindex(TIPOS)
    tab.loc["Global"] = resumen(exp_fin)
    g = tab.loc["Global"]
    cf = coste_fijo_periodo(cfg, dias_ventana)
    anios = dias_ventana / 365
    glob = {
        "Facturacion_anio": g["Facturacion_total"] / anios,
        "Coste_total_anio": g["Coste_total"] / anios,
        "Coste_medio_expediente": g["Coste_medio_expediente"],
        "Beneficio_operativo_anio": g["Beneficio"] / anios,
        "Margen_operativo": g["Margen"],
        "Coste_fijo_periodo": cf,
        "Margen_neto": (g["Beneficio"] - cf) / g["Facturacion_total"] if g["Facturacion_total"] > 0 else np.nan,
        "Beneficio_neto_anio": (g["Beneficio"] - cf) / anios,
        "Margen_exigido": param["MARGEN_EXIGIDO"],
    }
    return tab, glob


def evaluar_slas(exp_fin, seg, cfg, t0, t1, econ_glob, param=PARAM):
    """Evalúa los seis SLAs de la Sección A.08 (el SLA2 se evalúa por separado para Preventivo y Impago_Leve).
    Devuelve un DataFrame con la métrica, su valor, el umbral, si se cumple («Sí»/«No») y un detalle."""
    filas = []
    lim_d = param["SLA_LATENCIA_D"]
    # 1) Latencia Apertura de Expediente -> Diagnóstico Financiero (Riesgo_Concursal)
    rc = exp_fin[exp_fin["Perfil"].isin(PERFILES_SLA_LATENCIA)]
    lat = rc["Latencia_DF"].dropna()
    pct_ok = (lat <= param["SLA_LATENCIA_D"]).mean() if len(lat) else 0.0
    filas.append((f"SLA1 Latencia DF ≤ {lim_d:g} d (Riesgo_Concursal)", f"% expedientes ≤ {lim_d:g} d", pct_ok,
                  f"≥ {param['SLA_LATENCIA_CUMPL']:.0%}", pct_ok >= param["SLA_LATENCIA_CUMPL"],
                  f"latencia media {lat.mean() if len(lat) else 0:.2f} d · P95 {lat.quantile(0.95) if len(lat) else 0:.2f} d"))
    # 2) Ratio de espera (Preventivo y Impago_Leve)
    for t in PERFILES_SLA_ESPERA:
        x = exp_fin[exp_fin["Perfil"] == t]
        r = x["Ratio_espera"].mean() if len(x) else 0.0
        filas.append((f"SLA2 Ratio espera ({t})", "T_q/T_sys medio", r,
                      f"< {param['SLA_RATIO_ESPERA']:.0%}", r < param["SLA_RATIO_ESPERA"],
                      f"T_q medio {x['T_q'].mean() if len(x) else 0:.2f} d de {x['T_sys'].mean() if len(x) else 0:.2f} d en el sistema"))
    # 3) Ocupación estructural de Intervención Judicial
    rec_int = [r for r in capacidades(cfg) if r.startswith("Int")]
    occ = ocupacion_periodica(seg, rec_int, cfg.INTERVENCION_TOTAL_CAPACIDAD, t0, t1)
    filas.append((f"SLA3 Ocupación media Intervención Judicial ≤ {param['SLA_OCUP_INT']:.0%}", "ocupación media", occ.mean(),
                  f"≤ {param['SLA_OCUP_INT']:.0%}", occ.mean() <= param["SLA_OCUP_INT"],
                  f"{(occ > param['SLA_OCUP_INT']).mean():.1%} de los meses por encima del {param['SLA_OCUP_INT']:.0%}"))
    # 4) Insolvencia evitable Sin_Int en Impago_Grave/Riesgo_Concursal
    gc = exp_fin[exp_fin["Perfil"].isin(PERFILES_GRAVES)]
    m = gc["Sin_Int"].mean() if len(gc) else 0.0
    filas.append((f"SLA4 Insolvencias Sin_Int < {param['SLA_SIN_INT']:.0%} (Impago_Grave/Riesgo_Concursal)", "tasa", m,
                  f"< {param['SLA_SIN_INT']:.0%}", m < param["SLA_SIN_INT"],
                  f"{int(gc['Sin_Int'].sum())} insolvencias totales por falta de plaza de Intervención Judicial"))
    # 5) Tasa de regularización global
    lib = (exp_fin["Desenlace"] == "Regularizado").mean() if len(exp_fin) else 0.0
    filas.append((f"SLA5 Regularización ≥ {param['SLA_REGULARIZACION']:.0%}", "tasa", lib,
                  f"≥ {param['SLA_REGULARIZACION']:.0%}", lib >= param["SLA_REGULARIZACION"],
                  f"{int((exp_fin['Desenlace'] == 'Regularizado').sum())} regularizados de {len(exp_fin)} expedientes finalizados"))
    # 6) Margen operativo neto ≥ margen exigido
    filas.append(("SLA6 Margen neto ≥ margen exigido", "margen neto", econ_glob["Margen_neto"],
                  f"≥ {econ_glob['Margen_exigido']:.0%}", econ_glob["Margen_neto"] >= econ_glob["Margen_exigido"],
                  f"margen operativo antes de coste fijo {econ_glob['Margen_operativo']:.1%}"))
    out = pd.DataFrame(filas, columns=["SLA", "Métrica", "Valor", "Umbral", "Cumple", "Detalle"]).set_index("SLA")
    out["Cumple"] = out["Cumple"].map({True: "Sí", False: "No"})
    return out


CLAVES_SLA = {"SLA1": "SLA1", "SLA2_Preventivo": "SLA2 Ratio espera (Preventivo)",
              "SLA2_Impago_Leve": "SLA2 Ratio espera (Impago_Leve)", "SLA3": "SLA3", "SLA4": "SLA4",
              "SLA5": "SLA5", "SLA6": "SLA6"}


def calcular_kpis(df, cfg, dias=None, param=PARAM):
    """Calcula TODOS los KPIs y SLAs de una traza (histórica o simulada).
    Devuelve un diccionario con las tablas y un resumen plano `resumen` (una cifra por KPI),
    pensado para acumular réplicas en el análisis Monte Carlo."""
    d = preparar_traza(df)
    t_fin = dias if dias is not None else horizonte(df)
    t0, t1 = param["T_WARMUP"], t_fin - param["T_COOLDOWN"]
    ind = tabla_expedientes(d, param)
    exp_fin, exp_vent, dias_v = ventana_expedientes(ind, t_fin, param)
    op = kpis_operativos(exp_fin)
    est = kpis_estaciones(d, t0, t1)
    rec = kpis_recursos(d, cfg, t0, t1)
    seg = segmentos_ocupacion(d, cfg)
    bio = kpis_recuperacion(exp_fin, dias_v)
    eco_tab, eco = kpis_economicos(exp_fin, cfg, dias_v, param)
    sla = evaluar_slas(exp_fin, seg, cfg, t0, t1, eco, param)

    res = {"Expedientes_ventana": len(exp_vent), "Expedientes_finalizados": len(exp_fin),
           "Pct_finalizados": len(exp_fin) / max(len(exp_vent), 1), "Ingresos_anio": len(exp_vent) / dias_v * 365}
    for perfil in TIPOS + ["Global"]:
        for k in ["T_sys_media", "T_ser_media", "T_q_media", "Ratio_espera_medio"]:
            res[f"{k}_{perfil}"] = op.loc[perfil, k]
        for k in ["Tasa_regularizacion", "Tasa_insolvencia", "Tasa_liquidacion", "Tasa_insolv_Sin_Int", "Tasa_Negociacion", "Tasa_Estres", "Tasa_Int"]:
            res[f"{k}_{perfil}"] = bio.loc[perfil, k]
    for k in ["Regularizados_anio", "Tasa_Int_desde_Diagnostico", "Tasa_Int_desde_Estres", "Tasa_Int_desde_Negociacion", "Tasa_Estres"]:
        res[k] = bio.loc["Global", k]
    for r in rec.index:
        res[f"Util_{r}"] = rec.loc[r, "Utilizacion"]
        res[f"Lq_{r}"] = rec.loc[r, "Lq_media"]
        if rec.loc[r, "Tipo"] != "Servidor":
            res[f"Bloqueo_d_anio_{r}"] = rec.loc[r, "Bloqueo_d"] / ((t1 - t0) / 365)
    res.update(eco)
    rc = exp_fin["Perfil"].isin(PERFILES_SLA_LATENCIA)
    res["Latencia_DF_media_RC_d"] = exp_fin.loc[rc, "Latencia_DF"].mean()
    res["Insolv_Sin_Int_GC"] = exp_fin.loc[exp_fin["Perfil"].isin(PERFILES_GRAVES), "Sin_Int"].mean()
    for k, pref in CLAVES_SLA.items():
        fila = sla.loc[sla.index.str.startswith(pref)].iloc[0]
        res[f"SLA_valor_{k}"] = fila["Valor"]
        res[f"Cumple_{k}"] = fila["Cumple"] == "Sí"
    res["Ocup_media_Int"] = res["SLA_valor_SLA3"]
    rec_int = [r for r in capacidades(cfg) if r.startswith("Int")]
    res["Pct_meses_Int_90"] = (ocupacion_periodica(seg, rec_int, cfg.INTERVENCION_TOTAL_CAPACIDAD, t0, t1) > param["SLA_OCUP_INT"]).mean()
    res["Pct_RC_con_demora"] = exp_fin.loc[rc, "Demora_critica"].mean() if rc.any() else 0.0
    # Acceso a Intervención Judicial por perfil (espera de plaza y fallos sobre las peticiones)
    interv = d[d["Station_in"].eq(INTERV) & (d["T_in"] >= t0) & (d["T_in"] < t1)]
    for perfil in TIPOS:
        u = interv[interv["Perfil"] == perfil]
        res[f"Peticiones_Int_anio_{perfil}"] = len(u) / dias_v * 365
        # si en la réplica no hay peticiones de ese perfil, no hay espera ni fallos: se registra 0
        res[f"Espera_Int_media_{perfil}"] = u["T_wait"].mean() if len(u) else 0.0
        res[f"Espera_Int_P95_{perfil}"] = u["T_wait"].quantile(0.95) if len(u) else 0.0
        res[f"Pct_peticiones_Int_fallidas_{perfil}"] = u["Sin_Int"].mean() if len(u) else 0.0
    # Uso del Peritaje Independiente (recurso escaso)
    pg = d[d["Station_in"].eq("Peritaje_Independiente") & (d["T_in"] >= t0) & (d["T_in"] < t1) & ~d["Fin"]]
    res["Espera_media_Peritaje"] = pg["T_wait"].mean() if len(pg) else 0.0

    return {"traza": d, "expedientes": ind, "expedientes_fin": exp_fin, "operativos": op,
            "estaciones": est, "recursos": rec, "segmentos": seg, "recuperacion": bio,
            "economicos": eco_tab, "economicos_glob": eco, "slas": sla, "resumen": res,
            "dias_ventana": dias_v, "t0": t0, "t1": t1, "t_fin": t_fin}


TABLAS_MC = ["operativos", "recuperacion", "estaciones", "recursos", "economicos"]


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
    if tabla in ("operativos", "recuperacion", "economicos"):
        media = media.reindex(TIPOS + ["Global"])
    return media


def slas_mc(mc):
    """Para cada SLA: valor medio de la métrica, IC95 y % de réplicas que lo cumplen."""
    r = mc["resumen"]
    p = PARAM
    nombres = {"SLA1": f"Latencia DF ≤ {p['SLA_LATENCIA_D']:g} d (Riesgo_Concursal): % que cumplen (≥{p['SLA_LATENCIA_CUMPL']:.0%})",
               "SLA2_Preventivo": f"Ratio espera Preventivo (<{p['SLA_RATIO_ESPERA']:.0%})",
               "SLA2_Impago_Leve": f"Ratio espera Impago_Leve (<{p['SLA_RATIO_ESPERA']:.0%})",
               "SLA3": f"Ocupación media Intervención Judicial (≤{p['SLA_OCUP_INT']:.0%})",
               "SLA4": f"Insolvencia total Sin_Int en Impago_Grave/Riesgo_Concursal (<{p['SLA_SIN_INT']:.0%})",
               "SLA5": f"Tasa de regularización global (≥{p['SLA_REGULARIZACION']:.0%})",
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
    + ["Latencia_DF_media_RC_d", "SLA_valor_SLA1", "Pct_RC_con_demora"]
    + [f"Tasa_regularizacion_{t}" for t in TIPOS + ["Global"]]
    + [f"Tasa_insolvencia_{t}" for t in TIPOS + ["Global"]]
    + [f"Tasa_liquidacion_{t}" for t in TIPOS + ["Global"]]
    + ["Regularizados_anio", "Tasa_insolv_Sin_Int_Global", "Insolv_Sin_Int_GC", "Tasa_Negociacion_Global",
       "Tasa_Estres_Global", "Tasa_Int_Global", "Tasa_Int_desde_Diagnostico", "Tasa_Int_desde_Estres", "Tasa_Int_desde_Negociacion"]
    + ["Util_Equipo_Analistas", "Util_Peritaje_Independiente", "Espera_media_Peritaje",
       "Util_Mesa_Prioritaria", "Util_Recopilacion_Documental", "Util_Simulacion_Estres_Financiero", "Util_Mesa_Negociacion",
       "Util_Reestructuracion_Profunda", "Ocup_media_Int", "Pct_meses_Int_90",
       "Bloqueo_d_anio_Mesa_Prioritaria", "Bloqueo_d_anio_Simulacion_Estres_Financiero", "Bloqueo_d_anio_Recopilacion_Documental",
       "Bloqueo_d_anio_Mesa_Negociacion"]
    + ["Coste_medio_expediente", "Facturacion_anio", "Beneficio_operativo_anio", "Margen_operativo",
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
        rec_int = [r for r in capacidades(cfg) if r.startswith("Int")]
        detalle = {
            "expedientes_fin": k["expedientes_fin"][["Perfil", "T_sys", "T_ser", "T_q", "Ratio_espera", "Desenlace",
                                                   "Latencia_DF"]].copy(),
            "ocupacion_cont": ocupacion_periodica(k["segmentos"], rec_int, cfg.INTERVENCION_TOTAL_CAPACIDAD, k["t0"], k["t1"]),
            "ocupacion_cont_bolsas": {r: ocupacion_periodica(k["segmentos"], [r], capacidades(cfg)[r], k["t0"], k["t1"])
                                     for r in rec_int},
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
    if k.startswith(("Tasa_regularizacion", "SLA_valor_SLA1", "Margen", "Beneficio", "Facturacion", "Regularizados_anio")):
        return +1
    if k.startswith(("T_sys", "T_q", "Ratio_espera", "Latencia", "Tasa_insolvencia", "Insolvencia_Sin_Int", "Tasa_liquidacion",
                     "Coste", "Bloqueo", "Espera_", "Pct_peticiones_Int_fallidas", "Pct_meses_Int_90", "Pct_RC_con_demora", "Lq_")):
        return -1
    return 0


ETIQUETAS_KPI = {
    "T_sys_media_Preventivo": "T. medio en el sistema · Preventivo (días)",
    "T_sys_media_Impago_Leve": "T. medio en el sistema · Impago_Leve (días)",
    "T_sys_media_Impago_Grave": "T. medio en el sistema · Impago_Grave (días)",
    "T_sys_media_Riesgo_Concursal": "T. medio en el sistema · Riesgo_Concursal (días)",
    "T_sys_media_Global": "T. medio en el sistema · Global (días)",
    "T_q_media_Global": "T. medio en cola · Global (días)",
    "Latencia_DF_media_RC_d": "Latencia hasta Diagnóstico Financiero · Riesgo_Concursal (días)",
    "SLA_valor_SLA1": "% Riesgo_Concursal diagnosticados en ≤ 1 día",
    "Pct_RC_con_demora": "% Riesgo_Concursal con demora (> 1 día)",
    "Tasa_regularizacion_Preventivo": "Regularización · Preventivo",
    "Tasa_regularizacion_Impago_Leve": "Regularización · Impago_Leve",
    "Tasa_regularizacion_Impago_Grave": "Regularización · Impago_Grave",
    "Tasa_regularizacion_Riesgo_Concursal": "Regularización · Riesgo_Concursal",
    "Tasa_regularizacion_Global": "Tasa de regularización global",
    "Tasa_liquidacion_Global": "Tasa de liquidación ordenada",
    "Tasa_insolvencia_Preventivo": "Insolvencia total · Preventivo",
    "Tasa_insolvencia_Impago_Leve": "Insolvencia total · Impago_Leve",
    "Tasa_insolvencia_Impago_Grave": "Insolvencia total · Impago_Grave",
    "Tasa_insolvencia_Riesgo_Concursal": "Insolvencia total · Riesgo_Concursal",
    "Tasa_insolvencia_Global": "Insolvencia total · Global",
    "Tasa_insolv_Sin_Int_Global": "Insolvencia Sin_Int (s/ total expedientes)",
    "Insolv_Sin_Int_GC": "Insolvencia Sin_Int (s/ Impago_Grave+Riesgo_Concursal)",
    "Regularizados_anio": "Órdenes devueltas a servicio al año",
    "Ocup_media_Int": "Ocupación media Intervención Judicial",
    "Util_Mesa_Negociacion": "Utilización Mesa de Negociación",
    "Util_Recopilacion_Documental": "Utilización Recopilación Documental",
    "Util_Peritaje_Independiente": "Utilización Peritaje Independiente",
    "Util_Mesa_Prioritaria": "Utilización Mesa Prioritaria",
    "Util_Equipo_Analistas": "Utilización Equipo Técnico",
    "Coste_medio_expediente": "Coste medio por expediente (€)",
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
    if k.startswith(("Tasa_", "Util_", "Ratio_", "Margen", "Ocup", "Pct_", "Perdida_", "SLA_valor_SLA1", "Cumple", "%")):
        return pct(v, 2 if k.startswith(("Tasa_insolvencia_Sin", "Perdida_Sin")) else 1)
    if k in ("Ingresos_anio", "Regularizados_anio") or k.startswith("Peticiones_"):
        return num(v, 1)
    if any(t in k for t in ("Coste", "Facturacion", "Beneficio", "€", "Resultado")):
        return eur(v)
    return num(v, 2)


def ranking_protocolos(protocolos, ref):
    """Ordena los protocolos: primero los que no empeoran la insolvencia total, luego por nº de SLAs
    cumplidos y, a igualdad, por resultado económico neto anual frente a la referencia."""
    base = protocolos[ref]
    filas = []
    for et, m in protocolos.items():
        s = slas_mc(m)
        fila = {"Protocolo": et, "SLAs cumplidos": int((s["Veredicto"] == "Cumple").sum()),
                "SLAs cumplidos (lista)": ", ".join(s.index[s["Veredicto"] == "Cumple"]),
                "Regularización": m["resumen"]["Tasa_regularizacion_Global"].mean(),
                "Insolvencia total global": m["resumen"]["Tasa_insolvencia_Global"].mean(),
                "Insolvencia Sin_Int G/C": m["resumen"]["Insolv_Sin_Int_GC"].mean(),
                "T_sys global (días)": m["resumen"]["T_sys_media_Global"].mean(),
                "Margen neto": m["resumen"]["Margen_neto"].mean()}
        if et == ref:
            fila.update({"Resultado neto anual (€)": 0.0, "Resultado neto 1er año (€)": 0.0, "Descartado": False})
        else:
            e = resultado_economico(base, m)
            c = comparar_escenarios(base, m, ["Tasa_insolvencia_Global", "Insolv_Sin_Int_GC"])
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
        f"Diagnóstico de partida ({ref}). La simulación estima una tasa de regularización del "
        f"{pct(base['resumen']['Tasa_regularizacion_Global'].mean())}, una tasa de insolvencia total del "
        f"{pct(base['resumen']['Tasa_insolvencia_Global'].mean())}, una tasa de liquidación ordenada del "
        f"{pct(base['resumen']['Tasa_liquidacion_Global'].mean())} y un margen operativo del "
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
                f"Recomendación. Se recomienda implantar {mejor}. Es la alternativa que, sin empeorar la insolvencia total, "
                f"cumple más SLAs ({int(m['SLAs cumplidos'])}: {m['SLAs cumplidos (lista)'] or '—'}) con el mejor resultado "
                f"económico neto ({eur(m['Resultado neto anual (€)'])}/año frente a {ref}; "
                f"{eur(m['Resultado neto 1er año (€)'])} el primer año) y un margen neto del {pct(m['Margen neto'])}.")
        else:
            mejor = ref
            par.append(f"Recomendación. Ninguna alternativa mejora a {ref} en SLAs y resultado económico a la vez; "
                       f"se recomienda mantener {ref} o revisar el diseño de los escenarios.")
    else:
        mejor = ref
        par.append(f"Recomendación. Todas las alternativas empeoran la insolvencia total frente a {ref}; se recomienda mantenerlo.")
    pend = slas_mc(protocolos[mejor])
    pend = pend.index[pend["Veredicto"] != "Cumple"]
    if len(pend):
        cuellos = tabla_media_mc(protocolos[mejor], "recursos")
        cuellos = cuellos[cuellos["Utilizacion"] >= 0.85].index.tolist()
        par.append(f"Aun con la opción recomendada siguen sin cumplirse {', '.join(pend)}."
                   + (f" Los recursos con utilización ≥ 0,85 ({', '.join(cuellos)}) son la siguiente palanca a estudiar."
                      if cuellos else ""))
    return par, rk
