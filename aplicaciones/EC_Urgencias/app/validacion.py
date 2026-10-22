# -*- coding: utf-8 -*-
"""Comprobaciones de coherencia de las configuraciones antes de simular."""
from motor import TIPOS, ESTADOS_FINALES, es_observacion


def estaciones_alcanzables(cfg, tipo):
    """Estaciones que puede visitar un paciente de `tipo` partiendo de Admisión."""
    vistos, pila = set(), ["Admision"]
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
    camas = set(cfg.CAMAS_CAPACIDAD) | set(cfg.CAMA_ALIAS)
    for alias, destino in cfg.CAMA_ALIAS.items():
        if destino not in cfg.CAMAS_CAPACIDAD:
            errores.append(f"El alias de cama '{alias}' apunta a '{destino}', que no es una cama definida.")
    for nombre, cap in {**cfg.SERVIDORES_CAPACIDAD, **cfg.CAMAS_CAPACIDAD}.items():
        if cap is None or cap < 1:
            errores.append(f"La capacidad de '{nombre}' debe ser ≥ 1.")
    if cfg.BOX_CAPACIDAD < 1:
        errores.append("La capacidad del Box debe ser ≥ 1.")
    if cfg.UCI_MODE == "pooled":
        if cfg.UCI_TOTAL_CAPACIDAD < 1:
            errores.append("La capacidad total de la UCI debe ser ≥ 1.")
    else:
        for k in ("UCI_ROJO_CAPACIDAD", "UCI_NARANJA_CAPACIDAD", "UCI_BASE_CAPACIDAD"):
            if getattr(cfg, k) < 1:
                errores.append(f"{k} debe ser ≥ 1 (el motor necesita las tres bolsas).")
    if cfg.LIMITE_ESPERA_UCI <= 0:
        errores.append("El límite de espera de UCI debe ser positivo.")

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
            es_servidor = not (es_observacion(est) or est in camas or est == "UCI")
            if es_servidor and cfg.POOL_ESTACION.get(est, est) not in pools:
                errores.append(f"La estación '{est}' no tiene servidor asociado (añádela a servidores o a la tabla de asignación).")
            if est == "UCI" and tipo == "Verde":
                avisos.append("Los pacientes Verde pueden llegar a UCI: se les asigna la prioridad/bolsa de Amarillo.")
    for tipo, est in cfg.BOX_CAPTURA.items():
        if est and est not in estaciones_alcanzables(cfg, tipo):
            avisos.append(f"{tipo} captura el Box en '{est}', pero ese perfil nunca pasa por esa estación.")
    for k, v in getattr(cfg, "COSTES_FIJOS", {}).items():
        if v < 0:
            errores.append(f"El coste fijo '{k}' no puede ser negativo.")
    return errores, avisos


def validar_llegadas(lista):
    """Valida la tabla de franjas horarias (lista de dicts Inicio/Fin/Tasa/perfiles)."""
    errores = []
    horas = [0] * 24
    for r in lista:
        ini, fin = int(r["Inicio"]), int(r["Fin"])
        if not (0 <= ini <= 23 and 0 <= fin <= 24) or ini == fin:
            errores.append(f"Franja {ini}-{fin}: horas no válidas.")
            continue
        h = ini
        while h % 24 != fin % 24:
            horas[h % 24] += 1
            h += 1
        if float(r["Tasa"]) <= 0:
            errores.append(f"Franja {ini}-{fin}: la tasa debe ser positiva.")
        s = sum(float(r[t]) for t in TIPOS)
        if abs(s - 1) > 1e-6:
            errores.append(f"Franja {ini}-{fin}: el reparto por perfil suma {s:.3f} (debe sumar 1).")
    sin = [h for h, c in enumerate(horas) if c == 0]
    dup = [h for h, c in enumerate(horas) if c > 1]
    if sin:
        errores.append(f"Horas del día sin franja asignada: {sin}.")
    if dup:
        errores.append(f"Horas asignadas a más de una franja: {dup}.")
    return errores
