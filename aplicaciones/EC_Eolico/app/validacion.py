# -*- coding: utf-8 -*-
"""Comprobaciones de coherencia de las configuraciones del taller eólico antes de simular."""
from motor import TIPOS, ESTADOS_FINALES, es_observacion


def estaciones_alcanzables(cfg, tipo):
    """Estaciones que puede visitar una orden de `tipo` partiendo de Recepción."""
    vistos, pila = set(), ["Recepcion"]
    while pila:
        e = pila.pop()
        if e in vistos or e in ESTADOS_FINALES:
            continue
        vistos.add(e)
        for dest, p in cfg.PROB_DERIVACION.get(e, {}).get(tipo, {}).items():
            if p > 0:
                pila.append(dest)
    return vistos


def validar_config(cfg):
    """Devuelve (errores, avisos). Con errores la simulación fallaría o daría resultados sin sentido."""
    errores, avisos = [], []
    pools = set(cfg.SERVIDORES_CAPACIDAD)
    camas = set(cfg.BAHIAS_CAPACIDAD) | set(cfg.BAHIA_ALIAS)
    for alias, destino in cfg.BAHIA_ALIAS.items():
        if destino not in cfg.BAHIAS_CAPACIDAD:
            errores.append(f"El alias de bahía '{alias}' apunta a '{destino}', que no es una bahía definida.")
    for nombre, cap in {**cfg.SERVIDORES_CAPACIDAD, **cfg.BAHIAS_CAPACIDAD}.items():
        if cap is None or cap < 1:
            errores.append(f"La capacidad de '{nombre}' debe ser ≥ 1.")
    if cfg.BAHIA_URGENTE_CAPACIDAD < 1:
        errores.append("La capacidad de la Bahía Urgente debe ser ≥ 1.")
    if cfg.CONTENCION_MODE == "pooled":
        if cfg.CONTENCION_TOTAL_CAPACIDAD < 1:
            errores.append("La capacidad total de la Contención de Emergencia debe ser ≥ 1.")
    else:
        for k in ("CONTENCION_CRITICO_CAPACIDAD", "CONTENCION_GRAVE_CAPACIDAD", "CONTENCION_BASE_CAPACIDAD"):
            if getattr(cfg, k) < 1:
                errores.append(f"{k} debe ser ≥ 1 (el motor necesita las tres bolsas).")
    if cfg.LIMITE_ESPERA_CONTENCION <= 0:
        errores.append("El límite de espera de Contención debe ser positivo.")

    for tipo in TIPOS:
        for est in sorted(estaciones_alcanzables(cfg, tipo)):
            fila = cfg.PROB_DERIVACION.get(est, {}).get(tipo)
            if not fila:
                errores.append(f"{tipo}: la estación '{est}' es alcanzable pero no tiene probabilidades de derivación.")
                continue
            if any(p < 0 for p in fila.values()):
                errores.append(f"{tipo}: '{est}' tiene probabilidades negativas.")
            s = sum(fila.values())
            if abs(s - 1) > 1e-6:
                errores.append(f"{tipo}: las probabilidades de salida de '{est}' suman {s:.4f} (deben sumar 1).")
            if cfg.TIEMPOS_SERVICIO.get(est, {}).get(tipo) is None:
                errores.append(f"{tipo}: falta el tiempo de servicio de '{est}'.")
            elif cfg.TIEMPOS_SERVICIO[est][tipo] <= 0:
                errores.append(f"{tipo}: el tiempo de servicio de '{est}' debe ser positivo.")
            es_servidor = not (es_observacion(est) or est in camas or est == "Contencion_Emergencia")
            if es_servidor and cfg.POOL_ESTACION.get(est, est) not in pools:
                errores.append(f"La estación '{est}' no tiene servidor asociado (añádela a servidores o a la tabla de asignación).")
    for tipo, est in cfg.BAHIA_URGENTE_CAPTURA.items():
        if est and est not in estaciones_alcanzables(cfg, tipo):
            avisos.append(f"{tipo} captura la Bahía Urgente en '{est}', pero ese perfil nunca pasa por esa estación.")
    for k, v in getattr(cfg, "COSTES_FIJOS", {}).items():
        if v < 0:
            errores.append(f"El coste fijo '{k}' no puede ser negativo.")
    return errores, avisos


def validar_llegadas(lista):
    """Valida la tabla de estaciones del año (lista de dicts Inicio/Fin/Tasa/perfiles; días del año 0-365)."""
    errores = []
    dias = [0] * 365
    for r in lista:
        ini, fin = int(r["Inicio"]), int(r["Fin"])
        if not (0 <= ini <= 364 and 0 <= fin <= 365) or ini == fin:
            errores.append(f"Periodo {ini}-{fin}: días del año no válidos.")
            continue
        h = ini
        while h % 365 != fin % 365:
            dias[h % 365] += 1
            h += 1
        if float(r["Tasa"]) <= 0:
            errores.append(f"Periodo {ini}-{fin}: la tasa debe ser positiva.")
        s = sum(float(r[t]) for t in TIPOS)
        if abs(s - 1) > 1e-6:
            errores.append(f"Periodo {ini}-{fin}: el reparto por perfil suma {s:.3f} (debe sumar 1).")
    sin = [h for h, c in enumerate(dias) if c == 0]
    dup = [h for h, c in enumerate(dias) if c > 1]
    if sin:
        errores.append(f"Días del año sin periodo asignado: {_rangos(sin)}.")
    if dup:
        errores.append(f"Días del año asignados a más de un periodo: {_rangos(dup)}.")
    return errores


def _rangos(lista):
    """[1,2,3,7,8] -> '1–3, 7–8'."""
    out, ini, prev = [], None, None
    for x in lista:
        if ini is None:
            ini = prev = x
        elif x == prev + 1:
            prev = x
        else:
            out.append(f"{ini}–{prev}" if ini != prev else f"{ini}"); ini = prev = x
    if ini is not None:
        out.append(f"{ini}–{prev}" if ini != prev else f"{ini}")
    return ", ".join(out)
