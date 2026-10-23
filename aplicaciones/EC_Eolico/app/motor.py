# -*- coding: utf-8 -*-
"""Motor de simulación (SimPy) del taller de mantenimiento y reacondicionamiento eólico y
configuraciones de los escenarios. Tiempo en días.

Es el mismo gemelo digital del cuaderno '03 Taller Eólico', con dos cambios menores:
* los ingresos por estación del año pueden ir dentro de la configuración (cfg.ESTACIONES_ANIO), para
  que la aplicación los modifique como parámetro común a todos los escenarios;
* cualquier estación cuyo nombre empiece por 'Monitorizacion_' se trata como estación de monitorización.
"""
import copy
import random

import pandas as pd
import simpy

class Config:
    """Contenedor simple de los parámetros de un escenario (capacidades,
    tiempos de servicio, probabilidades de derivación, costes...)."""

    def to_dict(self):
        """Representación serializable (JSON) de la configuración."""
        d = copy.deepcopy(vars(self))
        if "ESTACIONES_ANIO" in d:
            d["ESTACIONES_ANIO"] = llegadas_a_lista(d["ESTACIONES_ANIO"])
        return d

    @classmethod
    def from_dict(cls, d):
        cfg = cls()
        for k, v in copy.deepcopy(d).items():
            setattr(cfg, k, v)
        if isinstance(getattr(cfg, "ESTACIONES_ANIO", None), list):
            cfg.ESTACIONES_ANIO = lista_a_llegadas(cfg.ESTACIONES_ANIO)
        return cfg


# --- Parámetros compartidos por TODOS los escenarios (no cambian) ---------
TIPOS = ["Programado", "Leve", "Grave", "Critico"]

# Estacionalidad meteorológica: (tasa_ordenes_dia, [pProgramado,pLeve,pGrave,pCritico])
ESTACIONES_ANIO = {
    (335, 59):  (1.50, [0.30, 0.25, 0.30, 0.15]),   # Invierno: temporales -> más averías graves/críticas
    (59, 151):  (1.05, [0.55, 0.25, 0.15, 0.05]),   # Primavera: revisiones programadas tras el invierno
    (151, 243): (0.825, [0.35, 0.35, 0.20, 0.10]),  # Verano: menor actividad, sobrecalentamientos eléctricos
    (243, 335): (1.125, [0.40, 0.30, 0.20, 0.10]),  # Otoño: mantenimiento preventivo previo al invierno
}


def llegadas_a_lista(llegadas):
    """{(ini, fin): (tasa, probs)} -> lista de dicts (formato editable/JSON)."""
    return [{"Inicio": i, "Fin": f, "Tasa": t, **{tp: p for tp, p in zip(TIPOS, probs)}}
            for (i, f), (t, probs) in llegadas.items()]


def lista_a_llegadas(lista):
    """Lista de dicts -> {(ini, fin): (tasa, probs)} (formato del motor)."""
    return {(int(r["Inicio"]), int(r["Fin"])): (float(r["Tasa"]), [float(r[tp]) for tp in TIPOS]) for r in lista}


def obtener_perfil_temporada(reloj_dias, llegadas=None):
    llegadas = ESTACIONES_ANIO if llegadas is None else llegadas
    dia_anno = int(reloj_dias) % 365
    for (inicio, fin), (tasa, probs) in llegadas.items():
        if inicio > fin:
            if dia_anno >= inicio or dia_anno < fin:
                return tasa, probs
        else:
            if inicio <= dia_anno < fin:
                return tasa, probs
    raise RuntimeError("Día del año no cubierto")


def generar_tiempo_servicio(cfg, estacion, tipo):
    base = cfg.TIEMPOS_SERVICIO.get(estacion, {}).get(tipo)
    if base is None:
        raise KeyError(f"Sin tiempo de servicio definido para {estacion}/{tipo}")
    return base * random.uniform(0.75, 1.25)


def determinar_destino(cfg, estacion, tipo):
    opciones = cfg.PROB_DERIVACION[estacion][tipo]
    destinos = list(opciones.keys())
    pesos = list(opciones.values())
    return random.choices(destinos, weights=pesos, k=1)[0]


def _verificar_filas(prob):
    """Comprueba que todas las filas de PROB_DERIVACION suman 1.0."""
    errores = []
    for estacion, por_tipo in prob.items():
        for tipo, destinos in por_tipo.items():
            s = sum(destinos.values())
            if abs(s - 1.0) > 1e-9:
                errores.append((estacion, tipo, s))
    return errores


ESTADOS_FINALES = {"Operativo", "Baja_Definitiva", "Perdida_Total"}
OBS_ESTACIONES = {"Monitorizacion_Leve", "Monitorizacion_Grave", "Monitorizacion_Critico"}


def es_observacion(estacion):
    """Las estaciones de monitorización (sin recurso propio) son las que empiezan por 'Monitorizacion_'."""
    return estacion in OBS_ESTACIONES or estacion.startswith("Monitorizacion_")


class Taller:
    def __init__(self, env, cfg, simpy_module):
        self.env = env
        self.cfg = cfg
        self.simpy = simpy_module
        self.log_datos = []
        self.activos = {}

        self.servidores = {
            nombre: simpy_module.Resource(env, capacity=cap)
            for nombre, cap in cfg.SERVIDORES_CAPACIDAD.items()
        }
        self.bahias = {
            nombre: simpy_module.Resource(env, capacity=cap)
            for nombre, cap in cfg.BAHIAS_CAPACIDAD.items()
        }
        self.bahia_urgente = simpy_module.Resource(env, capacity=cfg.BAHIA_URGENTE_CAPACIDAD)

        if cfg.CONTENCION_MODE == "pooled":
            self.contencion = {"POOL": simpy_module.PriorityResource(env, capacity=cfg.CONTENCION_TOTAL_CAPACIDAD)}
        else:
            self.contencion = {
                "Critico": simpy_module.Resource(env, capacity=cfg.CONTENCION_CRITICO_CAPACIDAD),
                "Grave": simpy_module.Resource(env, capacity=cfg.CONTENCION_GRAVE_CAPACIDAD),
                "Base": simpy_module.Resource(env, capacity=cfg.CONTENCION_BASE_CAPACIDAD),
            }

    def info_cola(self, resource):
        q = len([r for r in resource.queue if not r.triggered and not getattr(r, "_cancelled", False)])
        return q, q + resource.count

    def recurso_de_bahia(self, nombre_logico):
        alias = self.cfg.BAHIA_ALIAS.get(nombre_logico, nombre_logico)
        return self.bahias[alias]

    def pool_de_estacion(self, estacion):
        return self.cfg.POOL_ESTACION.get(estacion, estacion)

    def registrar(self, id_ord, tipo, t_in, estacion, t_wait, t_server, id_recurso,
                  bahia, q_len, sys_len, destino):
        self.log_datos.append({
            "Id_orden": id_ord, "Perfil": tipo, "T_in": round(t_in, 4),
            "Station_in": estacion, "T_wait": round(t_wait, 4),
            "T_server": round(t_server, 4), "ID_server": id_recurso,
            "Bahia": bahia, "Ord_en_cola": q_len, "Ord_en_sistema": sys_len,
            "T_out": round(t_in + t_wait + t_server, 4), "Station_out": destino,
            "T_system": round(t_wait + t_server, 4),
        })

    def finalizar_simulacion(self):
        end_time = self.env.now
        for ord_id, data in list(self.activos.items()):
            t_transcurrido = end_time - data["t_llegada_estacion"]
            self.log_datos.append({
                "Id_orden": ord_id, "Perfil": data["tipo"], "T_in": round(data["t_llegada_estacion"], 4),
                "Station_in": data["estacion"], "T_wait": round(t_transcurrido, 4),
                "T_server": 0.0, "ID_server": data.get("id_recurso", "N/A"),
                "Bahia": data.get("bahia", "N/A"), "Ord_en_cola": data.get("q_len", 0),
                "Ord_en_sistema": data.get("sys_len", 0), "T_out": round(end_time, 4),
                "Station_out": "Fin_Simulacion", "T_system": round(t_transcurrido, 4),
            })


def liberar_bahia(taller, bahia_propia):
    """Libera una bahía propia: un recurso físico normal (registrado en
    taller.bahias) o una plaza de Contención de Emergencia (clave especial
    'CONT::<bolsa>')."""
    if bahia_propia is None:
        return
    nombre, req = bahia_propia
    if nombre.startswith("CONT::"):
        bolsa = nombre.split("::", 1)[1]
        taller.contencion[bolsa].release(req)
    else:
        taller.bahias[nombre].release(req)


def comportamiento_orden(env, taller, simpy, id_ord, tipo):
    cfg = taller.cfg
    estacion_actual = "Recepcion"
    mi_bahia = "Taller_Espera"
    urgente_propia = None     # request concedida de la Bahía Urgente, si la orden ya la ocupa
    bahia_propia = None        # (nombre_logico, request) de la bahía o plaza de contención que ocupa
    t_llegada = env.now

    taller.activos[id_ord] = {
        "tipo": tipo, "t_llegada_estacion": t_llegada, "estacion": estacion_actual, "bahia": mi_bahia,
    }

    # Captura inmediata de la Bahía Urgente para Critico desde Recepción
    if cfg.BAHIA_URGENTE_CAPTURA.get(tipo) == "Recepcion":
        req_urgente = taller.bahia_urgente.request()
        yield req_urgente
        urgente_propia = req_urgente
        mi_bahia = "Bahia_Urgente"

    while estacion_actual not in ESTADOS_FINALES:
        taller.activos[id_ord].update({
            "t_llegada_estacion": t_llegada, "estacion": estacion_actual, "bahia": mi_bahia,
        })

        # Captura de la Bahía Urgente en el momento indicado por la configuración
        if urgente_propia is None and cfg.BAHIA_URGENTE_CAPTURA.get(tipo) == estacion_actual:
            req_urgente = taller.bahia_urgente.request()
            yield req_urgente
            urgente_propia = req_urgente
            mi_bahia = "Bahia_Urgente"
            taller.activos[id_ord]["bahia"] = mi_bahia

        t_wait = 0.0
        id_recurso = "N/A"
        q_len, sys_len = 0, 0

        if es_observacion(estacion_actual):
            # No hay recurso propio: se retiene la bahía ya ocupada.
            id_recurso = mi_bahia
            t_wait = 0.0

        elif estacion_actual in cfg.BAHIAS_CAPACIDAD or estacion_actual in cfg.BAHIA_ALIAS:
            recurso = taller.recurso_de_bahia(estacion_actual)
            alias = cfg.BAHIA_ALIAS.get(estacion_actual, estacion_actual)
            if bahia_propia is not None and bahia_propia[0] == alias:
                t_wait = 0.0  # ya se encontraba en esa misma bahía
            else:
                q_len, sys_len = taller.info_cola(recurso)
                t0 = env.now
                req_bahia = recurso.request()
                yield req_bahia
                t_wait = env.now - t0
                # libera la bahía anterior (si tenía una física distinta)
                if bahia_propia is not None:
                    liberar_bahia(taller, bahia_propia)
                    bahia_propia = None
                elif urgente_propia is not None:
                    taller.bahia_urgente.release(urgente_propia)
                    urgente_propia = None
                bahia_propia = (alias, req_bahia)
            id_recurso = alias
            mi_bahia = alias
            taller.activos[id_ord]["bahia"] = mi_bahia

        elif estacion_actual == "Contencion_Emergencia":
            prioridad = {"Critico": 0, "Grave": 1, "Leve": 2, "Programado": 2}[tipo]
            if cfg.CONTENCION_MODE == "pooled":
                pool = taller.contencion["POOL"]
                q_len, sys_len = taller.info_cola(pool)
                t0 = env.now
                req_cont = pool.request(priority=prioridad)
            else:
                bolsa = {"Critico": "Critico", "Grave": "Grave"}.get(tipo, "Base")
                pool = taller.contencion[bolsa]
                q_len, sys_len = taller.info_cola(pool)
                t0 = env.now
                req_cont = pool.request()

            # Carrera entre conseguir plaza y agotar el límite de espera
            resultado = yield req_cont | env.timeout(cfg.LIMITE_ESPERA_CONTENCION)
            if req_cont in resultado:
                t_wait = env.now - t0
                if bahia_propia is not None:
                    liberar_bahia(taller, bahia_propia)
                    bahia_propia = None
                elif urgente_propia is not None:
                    taller.bahia_urgente.release(urgente_propia)
                    urgente_propia = None
                bahia_propia = ("CONT::" + (bolsa if cfg.CONTENCION_MODE != "pooled" else "POOL"), req_cont)
                id_recurso = "Contencion_Emergencia"
                mi_bahia = "Contencion_Emergencia"
            else:
                req_cont.cancel()
                t_wait = env.now - t0
                # sin plaza de contención disponible dentro del límite: pérdida total
                taller.registrar(id_ord, tipo, t_llegada, estacion_actual,
                                  t_wait, 0.0, "Sin_Contencion_Disponible", mi_bahia, q_len, sys_len,
                                  "Perdida_Total")
                estacion_actual = "Perdida_Total"
                break
            taller.activos[id_ord]["bahia"] = mi_bahia

        else:
            # Estación "servidor": el recurso se libera justo tras el servicio
            pool_nombre = taller.pool_de_estacion(estacion_actual)
            recurso = taller.servidores[pool_nombre]
            q_len, sys_len = taller.info_cola(recurso)
            t0 = env.now
            req = recurso.request()
            yield req
            t_wait = env.now - t0
            id_recurso = pool_nombre

        t_servicio = generar_tiempo_servicio(cfg, estacion_actual, tipo)
        yield env.timeout(t_servicio)

        if (not es_observacion(estacion_actual)
                and estacion_actual not in cfg.BAHIAS_CAPACIDAD
                and estacion_actual not in cfg.BAHIA_ALIAS
                and estacion_actual != "Contencion_Emergencia"):
            taller.servidores[taller.pool_de_estacion(estacion_actual)].release(req)

        destino = determinar_destino(cfg, estacion_actual, tipo)

        if destino in ESTADOS_FINALES:
            if urgente_propia is not None:
                taller.bahia_urgente.release(urgente_propia)
                urgente_propia = None
            if bahia_propia is not None:
                liberar_bahia(taller, bahia_propia)
                bahia_propia = None

        taller.registrar(id_ord, tipo, t_llegada, estacion_actual,
                          t_wait, t_servicio, id_recurso, mi_bahia, q_len, sys_len, destino)

        estacion_actual = destino
        t_llegada = env.now

    if id_ord in taller.activos:
        del taller.activos[id_ord]
    if urgente_propia is not None:
        taller.bahia_urgente.release(urgente_propia)
    if bahia_propia is not None:
        liberar_bahia(taller, bahia_propia)


def generador_ordenes(env, taller, simpy, cfg):
    id_ord = 1
    llegadas = getattr(cfg, "ESTACIONES_ANIO", None)
    while True:
        tasa, probs = obtener_perfil_temporada(env.now, llegadas)
        yield env.timeout(random.expovariate(tasa))
        tipo = random.choices(TIPOS, weights=probs)[0]
        env.process(comportamiento_orden(env, taller, simpy, f"OT_{id_ord}", tipo))
        id_ord += 1


def simular(cfg, dias, semilla, simpy_module=None):
    """Ejecuta la simulación de `cfg` durante `dias` días con la semilla dada.
    Devuelve un DataFrame de pandas con la traza completa (mismo formato que
    `traza_taller_eolico.csv.gz`). `dias` es directamente la unidad de tiempo
    de la simulación: no se aplica ningún factor de conversión adicional."""
    if simpy_module is None:
        simpy_module = simpy
    random.seed(semilla)
    env = simpy_module.Environment()
    taller = Taller(env, cfg, simpy_module)
    env.process(generador_ordenes(env, taller, simpy_module, cfg))
    env.run(until=dias)
    taller.finalizar_simulacion()
    df_sim = pd.DataFrame(taller.log_datos)
    if len(df_sim):
        df_sim = df_sim.sort_values(by=["T_in"]).reset_index(drop=True)
    return df_sim


def construir_config_base():
    cfg = Config()
    cfg.nombre = "Línea Base (protocolo actual)"

    cfg.SERVIDORES_CAPACIDAD = {
        "Recepcion": 1,
        "Equipo_Tecnico": 1,                    # atiende Inspeccion_Tecnica y Valoracion_Preliminar
        "Diagnostico_Electrico": 1,
        "Diagnostico_END": 1,                   # único banco de Ensayos No Destructivos -> cuello de botella deliberado
        "Consulta_Ingeniero_Fiabilidad": 1,
        "Puesta_En_Marcha": 1,
    }

    # Capacidades "bahía" calibradas por Ley de Little: se midió la ocupación
    # real (L = lambda * W) simulando con capacidad artificialmente ilimitada
    # durante 40 años, y se dimensionó cada recurso por encima del percentil
    # 90-95 de ocupación observada (no sólo de la media: los tiempos de
    # reparación tienen varianza alta, y dimensionar sólo por la media
    # produce inestabilidad -colas que crecen sin límite- a largo plazo).
    cfg.BAHIA_URGENTE_CAPACIDAD = 12
    cfg.BAHIAS_CAPACIDAD = {
        "Espera_Repuestos": 20,
        "Banco_Ensayo_Sobrecarga": 1,
        "Bahia_Reparacion": 20,
        "Bahia_Reconstruccion_PostIncidente": 3,
    }
    cfg.BAHIA_ALIAS = {"Bahia_Reparacion_PostIncidente": "Bahia_Reparacion"}
    cfg.POOL_ESTACION = {
        "Inspeccion_Tecnica": "Equipo_Tecnico",
        "Valoracion_Preliminar": "Equipo_Tecnico",
    }
    cfg.BAHIA_URGENTE_CAPTURA = {"Critico": "Recepcion", "Grave": "Inspeccion_Tecnica"}

    cfg.CONTENCION_MODE = "partitioned"     # "partitioned" (línea base) o "pooled" (Escenario C)
    cfg.CONTENCION_CRITICO_CAPACIDAD = 2
    cfg.CONTENCION_GRAVE_CAPACIDAD = 1
    cfg.CONTENCION_BASE_CAPACIDAD = 1
    cfg.CONTENCION_TOTAL_CAPACIDAD = (cfg.CONTENCION_CRITICO_CAPACIDAD
                                       + cfg.CONTENCION_GRAVE_CAPACIDAD
                                       + cfg.CONTENCION_BASE_CAPACIDAD)  # 4
    cfg.LIMITE_ESPERA_CONTENCION = 1.0   # días

    cfg.TIEMPOS_SERVICIO = {
        "Recepcion":                     {t: 0.02 for t in TIPOS},
        "Inspeccion_Tecnica":            {t: 0.25 for t in TIPOS},
        "Espera_Repuestos": {
            "Programado": 5.0, "Leve": 7.0, "Grave": 10.0, "Critico": 4.0,
        },
        "Diagnostico_Electrico":         {t: 0.15 for t in TIPOS},
        "Diagnostico_END":               {t: 2.5 for t in TIPOS},
        "Consulta_Ingeniero_Fiabilidad": {t: 0.25 for t in TIPOS},
        "Valoracion_Preliminar":         {"Programado": 0.25, "Leve": 0.25, "Grave": 0.30, "Critico": 0.30},
        "Banco_Ensayo_Sobrecarga":       {"Grave": 0.30, "Critico": 0.30},
        "Bahia_Reparacion": {
            "Programado": 3.0, "Leve": 6.0, "Grave": 15.0, "Critico": 20.0,
        },
        "Bahia_Reparacion_PostIncidente": {
            "Programado": 2.0, "Leve": 3.0, "Grave": 5.0, "Critico": 7.0,
        },
        "Contencion_Emergencia": {
            "Programado": 2.0, "Leve": 2.0, "Grave": 4.0, "Critico": 5.0,
        },
        "Bahia_Reconstruccion_PostIncidente": {t: 10.0 for t in TIPOS},
        "Puesta_En_Marcha":              {t: 0.5 for t in TIPOS},
        "Monitorizacion_Leve":           {"Leve": 2.0},
        "Monitorizacion_Grave":          {"Grave": 3.0},
        "Monitorizacion_Critico":        {"Critico": 5.0},
    }

    cfg.PROB_DERIVACION = {
        "Recepcion": {t: {"Inspeccion_Tecnica": 1.0} for t in TIPOS},

        "Inspeccion_Tecnica": {
            "Programado": {"Espera_Repuestos": 1.0},
            "Leve":       {"Espera_Repuestos": 1.0},
            "Grave":      {"Diagnostico_Electrico": 0.40, "Diagnostico_END": 0.30,
                            "Consulta_Ingeniero_Fiabilidad": 0.10, "Banco_Ensayo_Sobrecarga": 0.10,
                            "Contencion_Emergencia": 0.10},
            "Critico":    {"Diagnostico_Electrico": 0.25, "Consulta_Ingeniero_Fiabilidad": 0.10,
                            "Banco_Ensayo_Sobrecarga": 0.25, "Contencion_Emergencia": 0.40},
        },

        # Espera_Repuestos cumple DOS papeles: espera de repuestos DE ENTRADA
        # (Programado/Leve, antes de iniciar el diagnóstico) y espera de
        # repuestos DE CIERRE (Grave/Critico, ya avanzada la reparación,
        # antes de la puesta en marcha final o de una nueva intervención).
        "Espera_Repuestos": {
            "Programado": {"Diagnostico_Electrico": 1.0},
            "Leve":       {"Diagnostico_Electrico": 1.0},
            "Grave":      {"Puesta_En_Marcha": 0.55, "Bahia_Reparacion": 0.30, "Baja_Definitiva": 0.15},
            "Critico":    {"Puesta_En_Marcha": 0.45, "Bahia_Reparacion": 0.25, "Baja_Definitiva": 0.30},
        },

        "Diagnostico_Electrico": {
            "Programado": {"Valoracion_Preliminar": 0.85, "Diagnostico_END": 0.15},
            "Leve":       {"Valoracion_Preliminar": 0.70, "Diagnostico_END": 0.30},
            "Grave":      {"Diagnostico_END": 0.35, "Consulta_Ingeniero_Fiabilidad": 0.25,
                            "Valoracion_Preliminar": 0.40},
            "Critico":    {"Diagnostico_END": 0.20, "Consulta_Ingeniero_Fiabilidad": 0.15,
                            "Banco_Ensayo_Sobrecarga": 0.25, "Valoracion_Preliminar": 0.40},
        },

        "Diagnostico_END": {t: {"Valoracion_Preliminar": 1.0} for t in TIPOS},

        "Consulta_Ingeniero_Fiabilidad": {
            "Grave":   {"Valoracion_Preliminar": 0.60, "Banco_Ensayo_Sobrecarga": 0.40},
            "Critico": {"Banco_Ensayo_Sobrecarga": 0.50, "Contencion_Emergencia": 0.30,
                        "Valoracion_Preliminar": 0.20},
        },

        "Valoracion_Preliminar": {
            "Programado": {"Bahia_Reparacion": 0.55, "Puesta_En_Marcha": 0.45},
            "Leve":       {"Bahia_Reparacion": 0.40, "Puesta_En_Marcha": 0.40, "Monitorizacion_Leve": 0.20},
            "Grave":      {"Bahia_Reparacion": 0.35, "Espera_Repuestos": 0.15, "Monitorizacion_Grave": 0.20,
                            "Baja_Definitiva": 0.20, "Perdida_Total": 0.05, "Consulta_Ingeniero_Fiabilidad": 0.05},
            "Critico":    {"Contencion_Emergencia": 0.25, "Bahia_Reparacion": 0.20, "Espera_Repuestos": 0.10,
                            "Monitorizacion_Critico": 0.15, "Baja_Definitiva": 0.20, "Perdida_Total": 0.05,
                            "Banco_Ensayo_Sobrecarga": 0.05},
        },

        "Banco_Ensayo_Sobrecarga": {
            "Grave":   {"Contencion_Emergencia": 0.15, "Espera_Repuestos": 0.59,
                        "Baja_Definitiva": 0.20, "Perdida_Total": 0.06},
            "Critico": {"Contencion_Emergencia": 0.45, "Espera_Repuestos": 0.27,
                        "Baja_Definitiva": 0.20, "Perdida_Total": 0.08},
        },

        "Monitorizacion_Leve":    {"Leve": {"Puesta_En_Marcha": 0.65, "Monitorizacion_Leve": 0.20,
                                             "Bahia_Reparacion": 0.15}},
        "Monitorizacion_Grave":   {"Grave": {"Puesta_En_Marcha": 0.30, "Monitorizacion_Grave": 0.25,
                                              "Bahia_Reparacion": 0.25, "Baja_Definitiva": 0.17,
                                              "Perdida_Total": 0.03}},
        "Monitorizacion_Critico": {"Critico": {"Monitorizacion_Critico": 0.10, "Bahia_Reparacion": 0.40,
                                                "Contencion_Emergencia": 0.25, "Baja_Definitiva": 0.20,
                                                "Perdida_Total": 0.05}},

        "Bahia_Reparacion": {
            "Programado": {"Puesta_En_Marcha": 0.55, "Bahia_Reparacion": 0.30,
                            "Espera_Repuestos": 0.10, "Baja_Definitiva": 0.05},
            "Leve":       {"Puesta_En_Marcha": 0.50, "Bahia_Reparacion": 0.26, "Espera_Repuestos": 0.10,
                            "Contencion_Emergencia": 0.05, "Baja_Definitiva": 0.08, "Perdida_Total": 0.01},
            "Grave":      {"Puesta_En_Marcha": 0.35, "Bahia_Reparacion": 0.26, "Espera_Repuestos": 0.12,
                            "Contencion_Emergencia": 0.10, "Baja_Definitiva": 0.14, "Perdida_Total": 0.03},
            "Critico":    {"Puesta_En_Marcha": 0.25, "Bahia_Reparacion": 0.20, "Espera_Repuestos": 0.10,
                            "Contencion_Emergencia": 0.15, "Baja_Definitiva": 0.24, "Perdida_Total": 0.06},
        },

        "Contencion_Emergencia": {
            "Programado": {"Bahia_Reparacion_PostIncidente": 0.90, "Baja_Definitiva": 0.05, "Perdida_Total": 0.05},
            "Leve":       {"Bahia_Reparacion_PostIncidente": 0.90, "Baja_Definitiva": 0.05, "Perdida_Total": 0.05},
            "Grave":      {"Bahia_Reparacion_PostIncidente": 0.73, "Baja_Definitiva": 0.15, "Perdida_Total": 0.12},
            "Critico":    {"Bahia_Reparacion_PostIncidente": 0.62, "Baja_Definitiva": 0.20, "Perdida_Total": 0.18},
        },

        "Bahia_Reparacion_PostIncidente": {
            "Programado": {"Puesta_En_Marcha": 0.40, "Bahia_Reconstruccion_PostIncidente": 0.60},
            "Leve":       {"Puesta_En_Marcha": 0.55, "Bahia_Reconstruccion_PostIncidente": 0.45},
            "Grave":      {"Puesta_En_Marcha": 0.45, "Bahia_Reconstruccion_PostIncidente": 0.55},
            "Critico":    {"Puesta_En_Marcha": 0.35, "Bahia_Reconstruccion_PostIncidente": 0.65},
        },

        "Bahia_Reconstruccion_PostIncidente": {t: {"Puesta_En_Marcha": 0.80, "Baja_Definitiva": 0.20} for t in TIPOS},

        "Puesta_En_Marcha": {t: {"Operativo": 1.0} for t in TIPOS},
    }

    cfg.COSTES_FIJOS = {}   # el protocolo actual no tiene coste fijo adicional
    return cfg


def construir_config_ia_predictiva():
    cfg = construir_config_base()
    cfg.nombre = "Escenario A - Mantenimiento Predictivo basado en IA"

    # 1) Tiempos de servicio: el gemelo digital acelera drásticamente los
    #    ensayos no destructivos y agiliza la lectura del ingeniero de fiabilidad.
    cfg.TIEMPOS_SERVICIO = copy.deepcopy(cfg.TIEMPOS_SERVICIO)
    for tipo in cfg.TIEMPOS_SERVICIO["Diagnostico_END"]:
        cfg.TIEMPOS_SERVICIO["Diagnostico_END"][tipo] *= 0.55
    for tipo in cfg.TIEMPOS_SERVICIO["Consulta_Ingeniero_Fiabilidad"]:
        cfg.TIEMPOS_SERVICIO["Consulta_Ingeniero_Fiabilidad"][tipo] *= 0.65

    # 2) Derivaciones: el diagnóstico predictivo detecta antes el alcance real
    #    de la avería -> menos Perdida_Total y menos Baja_Definitiva para
    #    Grave/Critico.
    cfg.PROB_DERIVACION = copy.deepcopy(cfg.PROB_DERIVACION)
    cfg.PROB_DERIVACION["Valoracion_Preliminar"]["Grave"] = {
        "Bahia_Reparacion": 0.50, "Espera_Repuestos": 0.15, "Monitorizacion_Grave": 0.20,
        "Baja_Definitiva": 0.10, "Perdida_Total": 0.02, "Consulta_Ingeniero_Fiabilidad": 0.03,
    }
    cfg.PROB_DERIVACION["Valoracion_Preliminar"]["Critico"] = {
        "Contencion_Emergencia": 0.25, "Bahia_Reparacion": 0.33, "Espera_Repuestos": 0.10,
        "Monitorizacion_Critico": 0.15, "Baja_Definitiva": 0.10, "Perdida_Total": 0.02,
        "Banco_Ensayo_Sobrecarga": 0.05,
    }

    # 3) Coste del escenario
    cfg.COSTES_FIJOS = {"licencia_gemelo_digital_anual": 54_000}
    return cfg



def construir_config_respuesta_rapida():
    cfg = construir_config_base()
    cfg.nombre = "Escenario B - Equipo de Respuesta Rápida"

    # 1) Nuevo recurso: segundo equipo técnico, dedicado en exclusiva a las
    #    averías críticas.
    cfg.SERVIDORES_CAPACIDAD = copy.deepcopy(cfg.SERVIDORES_CAPACIDAD)
    cfg.SERVIDORES_CAPACIDAD["Equipo_Respuesta_Rapida"] = 1

    # 2) Nueva estación de inspección inicial para Critico, atendida por este
    #    equipo dedicado, independiente de "Equipo_Tecnico".
    cfg.POOL_ESTACION = copy.deepcopy(cfg.POOL_ESTACION)
    cfg.POOL_ESTACION["Inspeccion_Tecnica_RR"] = "Equipo_Respuesta_Rapida"

    cfg.TIEMPOS_SERVICIO = copy.deepcopy(cfg.TIEMPOS_SERVICIO)
    cfg.TIEMPOS_SERVICIO["Inspeccion_Tecnica_RR"] = {"Critico": 0.08}

    # 3) Derivaciones: Critico entra ahora por la vía rápida en lugar de
    #    compartir "Equipo_Tecnico" con el resto de perfiles.
    cfg.PROB_DERIVACION = copy.deepcopy(cfg.PROB_DERIVACION)
    cfg.PROB_DERIVACION["Recepcion"]["Critico"] = {"Inspeccion_Tecnica_RR": 1.0}
    cfg.PROB_DERIVACION["Inspeccion_Tecnica_RR"] = {
        "Critico": {"Diagnostico_Electrico": 0.20, "Consulta_Ingeniero_Fiabilidad": 0.10,
                    "Banco_Ensayo_Sobrecarga": 0.25, "Contencion_Emergencia": 0.45},
    }

    # 4) Coste del escenario
    cfg.COSTES_FIJOS = {"personal_respuesta_rapida_anual": 46_000}
    return cfg



def construir_config_contencion_dinamica():
    cfg = construir_config_base()
    cfg.nombre = "Escenario C - Gestión Dinámica de Contención de Emergencia"
    cfg.CONTENCION_MODE = "pooled"   # único cambio real: pool único con prioridad por gravedad
    # cfg.CONTENCION_TOTAL_CAPACIDAD se mantiene IGUAL que en la línea base (4
    # plazas): no se añade ninguna plaza nueva, sólo se reorganiza su gestión.
    cfg.COSTES_FIJOS = {"reorganizacion_unica": 5_000}   # coste único, no anual
    return cfg

ESCENARIOS_POR_DEFECTO = {
    "Línea base": construir_config_base,
    "A · IA predictiva": construir_config_ia_predictiva,
    "B · Respuesta rápida": construir_config_respuesta_rapida,
    "C · Contención compartida": construir_config_contencion_dinamica,
}


def combinar_configs(*cfgs, nombre=None, base=None):
    """Configuración que aplica sobre `base` (línea base por defecto) todos los cambios que cada
    escenario de `cfgs` introduce respecto a esa base (a nivel de fila de las tablas)."""
    base = base if base is not None else construir_config_base()
    out = Config.from_dict(base.to_dict())
    for c in cfgs:
        for attr, val in vars(c).items():
            if attr in ("nombre", "descripcion"):
                continue
            if attr == "COSTES_FIJOS":
                out.COSTES_FIJOS = {**out.COSTES_FIJOS, **val}
                continue
            bval = getattr(base, attr, None)
            if isinstance(val, dict) and isinstance(bval, dict):
                actual = getattr(out, attr)
                for k, v in val.items():
                    if k not in bval:
                        actual[k] = copy.deepcopy(v)
                    elif v != bval[k]:
                        if isinstance(v, dict) and isinstance(bval[k], dict):
                            for k2, v2 in v.items():
                                if bval[k].get(k2) != v2:
                                    actual[k][k2] = copy.deepcopy(v2)
                        else:
                            actual[k] = copy.deepcopy(v)
            elif val != bval:
                setattr(out, attr, copy.deepcopy(val))
    out.nombre = nombre or " + ".join(getattr(c, "nombre", "?") for c in cfgs)
    return out
