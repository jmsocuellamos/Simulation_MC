# -*- coding: utf-8 -*-
"""Motor de simulación (SimPy) de la red de colas hospitalaria y configuraciones de los escenarios.

Es el mismo gemelo digital del cuaderno '01 Red de Colas Hospitalaria', con dos cambios menores:
* las llegadas horarias pueden ir dentro de la configuración (cfg.LLEGADAS_HORARIAS), para que se
  puedan modificar desde la aplicación como parámetro común a todos los escenarios;
* cualquier estación cuyo nombre empiece por 'Obs_' se trata como estación de observación.
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
        if "LLEGADAS_HORARIAS" in d:
            d["LLEGADAS_HORARIAS"] = llegadas_a_lista(d["LLEGADAS_HORARIAS"])
        return d

    @classmethod
    def from_dict(cls, d):
        cfg = cls()
        for k, v in copy.deepcopy(d).items():
            setattr(cfg, k, v)
        if isinstance(getattr(cfg, "LLEGADAS_HORARIAS", None), list):
            cfg.LLEGADAS_HORARIAS = lista_a_llegadas(cfg.LLEGADAS_HORARIAS)
        return cfg


# --- Parámetros compartidos por TODOS los escenarios (no cambian) ---------
TIPOS = ["Verde", "Amarillo", "Naranja", "Rojo"]

LLEGADAS_HORARIAS = {
    (22, 7):  (3.0, [0.15, 0.35, 0.35, 0.15]),
    (7, 10):  (4.5, [0.30, 0.30, 0.25, 0.15]),
    (10, 14): (6.0, [0.30, 0.35, 0.30, 0.05]),
    (14, 17): (4.5, [0.40, 0.30, 0.25, 0.05]),
    (17, 22): (7.0, [0.25, 0.30, 0.30, 0.15]),
}


def llegadas_a_lista(llegadas):
    """{(ini, fin): (tasa, probs)} -> lista de dicts (formato editable/JSON)."""
    return [{"Inicio": i, "Fin": f, "Tasa": t, **{tp: p for tp, p in zip(TIPOS, probs)}}
            for (i, f), (t, probs) in llegadas.items()]


def lista_a_llegadas(lista):
    """Lista de dicts -> {(ini, fin): (tasa, probs)} (formato del motor)."""
    return {(int(r["Inicio"]), int(r["Fin"])): (float(r["Tasa"]), [float(r[tp]) for tp in TIPOS]) for r in lista}


def obtener_perfil_horario(reloj_horas, llegadas=None):
    llegadas = LLEGADAS_HORARIAS if llegadas is None else llegadas
    hora_dia = int(reloj_horas) % 24
    for (inicio, fin), (tasa, probs) in llegadas.items():
        if inicio > fin:
            if hora_dia >= inicio or hora_dia < fin:
                return tasa, probs
        else:
            if inicio <= hora_dia < fin:
                return tasa, probs
    raise RuntimeError("Franja horaria no cubierta")


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


ESTADOS_FINALES = {"Alta", "Fallece"}
OBS_ESTACIONES = {"Obs_Verde", "Obs_Amarillo", "Obs_Naranja", "Obs_Rojo"}


def es_observacion(estacion):
    """Las estaciones de observación (sin recurso propio) son las que empiezan por 'Obs_'."""
    return estacion in OBS_ESTACIONES or estacion.startswith("Obs_")


class Hospital:
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
        self.camas = {
            nombre: simpy_module.Resource(env, capacity=cap)
            for nombre, cap in cfg.CAMAS_CAPACIDAD.items()
        }
        self.box = simpy_module.Resource(env, capacity=cfg.BOX_CAPACIDAD)

        if cfg.UCI_MODE == "pooled":
            self.uci = {"POOL": simpy_module.PriorityResource(env, capacity=cfg.UCI_TOTAL_CAPACIDAD)}
        else:
            self.uci = {
                "Rojo": simpy_module.Resource(env, capacity=cfg.UCI_ROJO_CAPACIDAD),
                "Naranja": simpy_module.Resource(env, capacity=cfg.UCI_NARANJA_CAPACIDAD),
                "Base": simpy_module.Resource(env, capacity=cfg.UCI_BASE_CAPACIDAD),
            }

    def info_cola(self, resource):
        q = len([r for r in resource.queue if not r.triggered and not getattr(r, "_cancelled", False)])
        return q, q + resource.count

    def resource_de_cama(self, nombre_logico):
        alias = self.cfg.CAMA_ALIAS.get(nombre_logico, nombre_logico)
        return self.camas[alias]

    def pool_de_estacion(self, estacion):
        return self.cfg.POOL_ESTACION.get(estacion, estacion)

    def registrar(self, id_pac, tipo, t_in, estacion, t_wait, t_server, id_recurso,
                  sala, q_len, sys_len, destino):
        self.log_datos.append({
            "Id_pat": id_pac, "Pat_type": tipo, "T_in": round(t_in, 4),
            "Station_in": estacion, "T_wait": round(t_wait, 4),
            "T_server": round(t_server, 4), "ID_server": id_recurso,
            "Wait_room": sala, "Pat_in_queue": q_len, "Pat_in_system": sys_len,
            "T_out": round(t_in + t_wait + t_server, 4), "Station_out": destino,
            "T_system": round(t_wait + t_server, 4),
        })

    def finalizar_simulacion(self):
        end_time = self.env.now
        for pac_id, data in list(self.activos.items()):
            t_transcurrido = end_time - data["t_llegada_estacion"]
            self.log_datos.append({
                "Id_pat": pac_id, "Pat_type": data["tipo"], "T_in": round(data["t_llegada_estacion"], 4),
                "Station_in": data["estacion"], "T_wait": round(t_transcurrido, 4),
                "T_server": 0.0, "ID_server": data.get("id_recurso", "N/A"),
                "Wait_room": data.get("sala", "N/A"), "Pat_in_queue": data.get("q_len", 0),
                "Pat_in_system": data.get("sys_len", 0), "T_out": round(end_time, 4),
                "Station_out": "Fin_Simulacion", "T_system": round(t_transcurrido, 4),
            })


def liberar_cama(hosp, cama_propia):
    """Libera una 'cama' propia: un recurso físico normal (registrado en
    hosp.camas) o una cama de UCI (clave especial 'UCI::<bolsa>')."""
    if cama_propia is None:
        return
    nombre, req = cama_propia
    if nombre.startswith("UCI::"):
        bolsa = nombre.split("::", 1)[1]
        hosp.uci[bolsa].release(req)
    else:
        hosp.camas[nombre].release(req)


def comportamiento_paciente(env, hosp, simpy, id_paciente, tipo_paciente):
    cfg = hosp.cfg
    estacion_actual = "Admision"
    mi_sala = "Sala_Espera_Urgencias"
    box_propio = None       # request concedida del Box, si el paciente ya lo ocupa
    cama_propia = None      # (nombre_logico, request) de la cama que ocupa
    t_llegada = env.now

    hosp.activos[id_paciente] = {
        "tipo": tipo_paciente, "t_llegada_estacion": t_llegada, "estacion": estacion_actual,
        "sala": mi_sala,
    }

    # Captura inmediata de Box para Rojo desde Admisión (paciente crítico)
    if cfg.BOX_CAPTURA.get(tipo_paciente) == "Admision":
        req_box = hosp.box.request()
        yield req_box
        box_propio = req_box
        mi_sala = "Box_Urgencias"

    while estacion_actual not in ESTADOS_FINALES:
        hosp.activos[id_paciente].update({
            "t_llegada_estacion": t_llegada, "estacion": estacion_actual, "sala": mi_sala,
        })

        # Captura de Box en el momento indicado por la configuración
        if box_propio is None and cfg.BOX_CAPTURA.get(tipo_paciente) == estacion_actual:
            req_box = hosp.box.request()
            yield req_box
            box_propio = req_box
            mi_sala = "Box_Urgencias"
            hosp.activos[id_paciente]["sala"] = mi_sala

        t_wait = 0.0
        id_recurso = "N/A"
        q_len, sys_len = 0, 0

        if es_observacion(estacion_actual):
            # No hay recurso propio: se retiene la sala/box ya ocupado.
            id_recurso = mi_sala
            t_wait = 0.0

        elif estacion_actual in cfg.CAMAS_CAPACIDAD or estacion_actual in cfg.CAMA_ALIAS:
            recurso = hosp.resource_de_cama(estacion_actual)
            alias = cfg.CAMA_ALIAS.get(estacion_actual, estacion_actual)
            if cama_propia is not None and cama_propia[0] == alias:
                t_wait = 0.0  # ya se encontraba en esa misma cama
            else:
                q_len, sys_len = hosp.info_cola(recurso)
                t0 = env.now
                req_cama = recurso.request()
                yield req_cama
                t_wait = env.now - t0
                # libera la cama/box anterior (si tenía una física distinta)
                if cama_propia is not None:
                    liberar_cama(hosp, cama_propia)
                    cama_propia = None
                elif box_propio is not None:
                    hosp.box.release(box_propio)
                    box_propio = None
                cama_propia = (alias, req_cama)
            id_recurso = alias
            mi_sala = alias
            hosp.activos[id_paciente]["sala"] = mi_sala

        elif estacion_actual == "UCI":
            prioridad = {"Rojo": 0, "Naranja": 1, "Amarillo": 2, "Verde": 2}[tipo_paciente]
            if cfg.UCI_MODE == "pooled":
                pool = hosp.uci["POOL"]
                q_len, sys_len = hosp.info_cola(pool)
                t0 = env.now
                req_uci = pool.request(priority=prioridad)
            else:
                bolsa = {"Rojo": "Rojo", "Naranja": "Naranja"}.get(tipo_paciente, "Base")
                pool = hosp.uci[bolsa]
                q_len, sys_len = hosp.info_cola(pool)
                t0 = env.now
                req_uci = pool.request()

            # Carrera entre conseguir cama y agotar el límite de espera
            resultado = yield req_uci | env.timeout(cfg.LIMITE_ESPERA_UCI)
            if req_uci in resultado:
                t_wait = env.now - t0
                if cama_propia is not None:
                    liberar_cama(hosp, cama_propia)
                    cama_propia = None
                elif box_propio is not None:
                    hosp.box.release(box_propio)
                    box_propio = None
                cama_propia = ("UCI::" + (bolsa if cfg.UCI_MODE != "pooled" else "POOL"), req_uci)
                id_recurso = "UCI"
                mi_sala = "UCI"
            else:
                req_uci.cancel()
                t_wait = env.now - t0
                # sin cama de UCI disponible dentro del límite: desenlace fatal
                hosp.registrar(id_paciente, tipo_paciente, t_llegada, estacion_actual,
                                t_wait, 0.0, "Sin_UCI_Disponible", mi_sala, q_len, sys_len, "Fallece")
                estacion_actual = "Fallece"
                break
            hosp.activos[id_paciente]["sala"] = mi_sala

        else:
            # Estación "servidor": el recurso se libera justo tras el servicio
            pool_nombre = hosp.pool_de_estacion(estacion_actual)
            recurso = hosp.servidores[pool_nombre]
            q_len, sys_len = hosp.info_cola(recurso)
            t0 = env.now
            req = recurso.request()
            yield req
            t_wait = env.now - t0
            id_recurso = pool_nombre

        t_servicio = generar_tiempo_servicio(cfg, estacion_actual, tipo_paciente)
        yield env.timeout(t_servicio)

        if (not es_observacion(estacion_actual)
                and estacion_actual not in cfg.CAMAS_CAPACIDAD
                and estacion_actual not in cfg.CAMA_ALIAS
                and estacion_actual != "UCI"):
            hosp.servidores[hosp.pool_de_estacion(estacion_actual)].release(req)

        destino = determinar_destino(cfg, estacion_actual, tipo_paciente)

        if destino in ESTADOS_FINALES:
            if box_propio is not None:
                hosp.box.release(box_propio)
                box_propio = None
            if cama_propia is not None:
                liberar_cama(hosp, cama_propia)
                cama_propia = None

        hosp.registrar(id_paciente, tipo_paciente, t_llegada, estacion_actual,
                        t_wait, t_servicio, id_recurso, mi_sala, q_len, sys_len, destino)

        estacion_actual = destino
        t_llegada = env.now

    if id_paciente in hosp.activos:
        del hosp.activos[id_paciente]
    if box_propio is not None:
        hosp.box.release(box_propio)
    if cama_propia is not None:
        liberar_cama(hosp, cama_propia)


def generador_llegadas(env, hosp, simpy, cfg):
    id_paciente = 1
    llegadas = getattr(cfg, "LLEGADAS_HORARIAS", None)
    while True:
        tasa, probs = obtener_perfil_horario(env.now, llegadas)
        yield env.timeout(random.expovariate(tasa))
        tipo = random.choices(TIPOS, weights=probs)[0]
        env.process(comportamiento_paciente(env, hosp, simpy, f"PAC_{id_paciente}", tipo))
        id_paciente += 1


def simular(cfg, dias, semilla, simpy_module=None):
    """Ejecuta la simulación de `cfg` durante `dias` días con la semilla dada.
    Devuelve un DataFrame de pandas con la traza completa (mismo formato que
    `traza_urgencias_red.csv`)."""
    if simpy_module is None:
        simpy_module = simpy
    random.seed(semilla)
    env = simpy_module.Environment()
    hosp = Hospital(env, cfg, simpy_module)
    env.process(generador_llegadas(env, hosp, simpy_module, cfg))
    env.run(until=dias * 24)
    hosp.finalizar_simulacion()
    df_sim = pd.DataFrame(hosp.log_datos)
    if len(df_sim):
        df_sim = df_sim.sort_values(by=["T_in"]).reset_index(drop=True)
    return df_sim


def construir_config_base():
    cfg = Config()
    cfg.nombre = "Línea Base (protocolo actual)"

    cfg.SERVIDORES_CAPACIDAD = {
        "Admision": 3, "Triaje": 4, "Aislamiento": 6, "Medicos": 7,
        "Pruebas_Basicas": 4, "Pruebas_Avanzadas": 4,
        "Consulta_Especialista": 3, "Farmacia": 3,
    }

    cfg.BOX_CAPACIDAD = 55
    cfg.CAMAS_CAPACIDAD = {
        "Planta": 120,
        "Quirofano": 3,
        "Rehabilitacion": 43,
    }
    cfg.CAMA_ALIAS = {"Planta_PostUCI": "Planta"}
    cfg.POOL_ESTACION = {"Diag_Inicial": "Medicos", "Diag_Preliminar": "Medicos"}
    cfg.BOX_CAPTURA = {"Rojo": "Admision", "Naranja": "Diag_Inicial"}

    cfg.UCI_MODE = "partitioned"       # "partitioned" (línea base) o "pooled" (Escenario C)
    cfg.UCI_ROJO_CAPACIDAD = 45
    cfg.UCI_NARANJA_CAPACIDAD = 30
    cfg.UCI_BASE_CAPACIDAD = 4
    cfg.UCI_TOTAL_CAPACIDAD = cfg.UCI_ROJO_CAPACIDAD + cfg.UCI_NARANJA_CAPACIDAD + cfg.UCI_BASE_CAPACIDAD  # 79
    cfg.LIMITE_ESPERA_UCI = 12.0        # horas

    cfg.TIEMPOS_SERVICIO = {
        "Admision":              {"Verde": 5/60,  "Amarillo": 5/60,  "Naranja": 4/60,  "Rojo": 2/60},
        "Triaje":                {"Verde": 10/60, "Amarillo": 8/60,  "Naranja": 6/60,  "Rojo": 4/60},
        "Aislamiento":           {"Verde": 20/60, "Amarillo": 20/60, "Naranja": 20/60, "Rojo": 20/60},
        "Diag_Inicial":          {"Verde": 10/60, "Amarillo": 10/60, "Naranja": 10/60, "Rojo": 10/60},
        "Pruebas_Basicas":       {"Verde": 15/60, "Amarillo": 15/60, "Naranja": 15/60, "Rojo": 15/60},
        "Pruebas_Avanzadas":     {"Verde": 45/60, "Amarillo": 45/60, "Naranja": 60/60, "Rojo": 60/60},
        "Consulta_Especialista": {"Verde": 30/60, "Amarillo": 30/60, "Naranja": 30/60, "Rojo": 30/60},
        "Diag_Preliminar":       {"Verde": 10/60, "Amarillo": 10/60, "Naranja": 15/60, "Rojo": 20/60},
        "Farmacia":              {"Verde": 8/60,  "Amarillo": 8/60,  "Naranja": 8/60,  "Rojo": 8/60},
        "Quirofano":             {"Naranja": 2.5, "Rojo": 3.5},
        "Planta":                {"Amarillo": 24.0, "Naranja": 48.0, "Rojo": 72.0},
        "UCI":                   {"Amarillo": 24.0, "Naranja": 72.0, "Rojo": 96.0},
        "Planta_PostUCI":        {"Amarillo": 24.0, "Naranja": 36.0, "Rojo": 48.0},
        "Rehabilitacion":        {"Amarillo": 48.0, "Naranja": 72.0, "Rojo": 96.0},
        "Obs_Verde":             {"Verde": 6.0},
        "Obs_Amarillo":          {"Amarillo": 6.0},
        "Obs_Naranja":           {"Naranja": 6.0},
        "Obs_Rojo":              {"Rojo": 6.0},
    }

    cfg.PROB_DERIVACION = {
        "Admision": {t: {"Triaje": 1.0} for t in TIPOS},

        "Triaje": {
            "Verde":    {"Aislamiento": 0.05, "Diag_Inicial": 0.95},
            "Amarillo": {"Aislamiento": 0.10, "Diag_Inicial": 0.90},
            "Naranja":  {"Aislamiento": 0.12, "Diag_Inicial": 0.88},
            "Rojo":     {"Diag_Inicial": 1.0},
        },

        "Aislamiento": {t: {"Diag_Inicial": 1.0} for t in ("Verde", "Amarillo", "Naranja")},

        "Diag_Inicial": {
            "Verde":    {"Farmacia": 0.15, "Pruebas_Basicas": 0.65, "Pruebas_Avanzadas": 0.20},
            "Amarillo": {"Farmacia": 0.05, "Pruebas_Basicas": 0.55, "Pruebas_Avanzadas": 0.40},
            "Naranja":  {"Pruebas_Basicas": 0.40, "Pruebas_Avanzadas": 0.42, "Consulta_Especialista": 0.18},
            "Rojo":     {"Pruebas_Basicas": 0.30, "Pruebas_Avanzadas": 0.30, "Consulta_Especialista": 0.15,
                         "Planta": 0.10, "Quirofano": 0.15},
        },

        "Pruebas_Basicas": {
            "Verde":    {"Diag_Preliminar": 0.85, "Pruebas_Avanzadas": 0.15},
            "Amarillo": {"Diag_Preliminar": 0.45, "Pruebas_Avanzadas": 0.55},
            "Naranja":  {"Pruebas_Avanzadas": 0.60, "Diag_Preliminar": 0.40},
            "Rojo":     {"Pruebas_Avanzadas": 0.40, "Diag_Preliminar": 0.60},
        },

        "Pruebas_Avanzadas": {t: {"Diag_Preliminar": 1.0} for t in TIPOS},

        "Consulta_Especialista": {
            "Amarillo": {"Planta": 0.60, "Farmacia": 0.40},
            "Naranja":  {"Diag_Preliminar": 0.55, "Quirofano": 0.45},
            "Rojo":     {"Quirofano": 0.55, "UCI": 0.25, "Planta": 0.20},
        },

        "Diag_Preliminar": {
            "Verde":    {"Farmacia": 0.85, "Obs_Verde": 0.15},
            "Amarillo": {"Farmacia": 0.35, "Obs_Amarillo": 0.55, "Consulta_Especialista": 0.10},
            "Naranja":  {"Planta": 0.30, "Obs_Naranja": 0.35, "Consulta_Especialista": 0.15, "Quirofano": 0.20},
            "Rojo":     {"UCI": 0.25, "Planta": 0.30, "Quirofano": 0.20, "Obs_Rojo": 0.15, "Consulta_Especialista": 0.10},
        },

        "Quirofano": {
            "Naranja": {"UCI": 0.35, "Planta": 0.55, "Fallece": 0.10},
            "Rojo":    {"UCI": 0.65, "Planta": 0.25, "Fallece": 0.10},
        },

        "Obs_Verde":    {"Verde": {"Farmacia": 1.0}},
        "Obs_Amarillo": {"Amarillo": {"Farmacia": 0.60, "Obs_Amarillo": 0.25, "Planta": 0.15}},
        "Obs_Naranja":  {"Naranja": {"Farmacia": 0.29, "Obs_Naranja": 0.30, "Planta": 0.35, "UCI": 0.06}},
        "Obs_Rojo":     {"Rojo": {"Obs_Rojo": 0.10, "Planta": 0.45, "UCI": 0.35, "Farmacia": 0.10}},

        "Planta": {
            "Amarillo": {"Farmacia": 0.65, "Planta": 0.10, "UCI": 0.12, "Rehabilitacion": 0.08, "Fallece": 0.05},
            "Naranja":  {"Farmacia": 0.45, "Planta": 0.30, "UCI": 0.08, "Fallece": 0.07, "Rehabilitacion": 0.10},
            "Rojo":     {"Planta": 0.20, "UCI": 0.35, "Farmacia": 0.20, "Fallece": 0.15, "Rehabilitacion": 0.10},
        },

        "UCI": {
            "Amarillo": {"Planta_PostUCI": 0.90, "Fallece": 0.10},
            "Naranja":  {"Planta_PostUCI": 0.75, "Fallece": 0.25},
            "Rojo":     {"Planta_PostUCI": 0.65, "Fallece": 0.35},
        },

        "Planta_PostUCI": {
            "Amarillo": {"Farmacia": 0.80, "Rehabilitacion": 0.20},
            "Naranja":  {"Farmacia": 0.55, "Rehabilitacion": 0.45},
            "Rojo":     {"Farmacia": 0.45, "Rehabilitacion": 0.55},
        },

        "Rehabilitacion": {t: {"Farmacia": 1.0} for t in ("Amarillo", "Naranja", "Rojo")},

        "Farmacia": {t: {"Alta": 1.0} for t in TIPOS},
    }

    cfg.COSTES_FIJOS = {}   # el protocolo actual no tiene coste fijo adicional
    return cfg


def construir_config_ia():
    cfg = construir_config_base()
    cfg.nombre = "Escenario A - IA Diagnóstica"

    # 1) Tiempos de servicio: la IA acelera el diagnóstico inicial y preliminar
    cfg.TIEMPOS_SERVICIO = copy.deepcopy(cfg.TIEMPOS_SERVICIO)
    for tipo in cfg.TIEMPOS_SERVICIO["Diag_Inicial"]:
        cfg.TIEMPOS_SERVICIO["Diag_Inicial"][tipo] *= 0.70
    for tipo in cfg.TIEMPOS_SERVICIO["Diag_Preliminar"]:
        cfg.TIEMPOS_SERVICIO["Diag_Preliminar"][tipo] *= 0.75

    # 2) Derivaciones: menos necesidad de Pruebas Avanzadas (mejor precisión inicial).
    #    "Rojo" no se modifica: protocolo diagnóstico completo siempre en emergencia vital.
    cfg.PROB_DERIVACION = copy.deepcopy(cfg.PROB_DERIVACION)
    cfg.PROB_DERIVACION["Diag_Inicial"]["Verde"] = {
        "Farmacia": 0.35, "Pruebas_Basicas": 0.55, "Pruebas_Avanzadas": 0.10,
    }
    cfg.PROB_DERIVACION["Diag_Inicial"]["Amarillo"] = {
        "Farmacia": 0.15, "Pruebas_Basicas": 0.60, "Pruebas_Avanzadas": 0.25,
    }
    cfg.PROB_DERIVACION["Diag_Inicial"]["Naranja"] = {
        "Pruebas_Basicas": 0.45, "Pruebas_Avanzadas": 0.32, "Consulta_Especialista": 0.23,
    }

    # 3) Coste del escenario
    cfg.COSTES_FIJOS = {"licencia_ia_anual": 480_000}
    return cfg



def construir_config_fasttrack():
    cfg = construir_config_base()
    cfg.nombre = "Escenario B - Circuito Rápido (Fast-Track)"

    # 1) Nuevo recurso: equipo médico del circuito rápido
    cfg.SERVIDORES_CAPACIDAD = copy.deepcopy(cfg.SERVIDORES_CAPACIDAD)
    cfg.SERVIDORES_CAPACIDAD["Medicos_Rapidos"] = 3

    # 2) Nueva estación Diag_Rapido y su tiempo de servicio
    cfg.TIEMPOS_SERVICIO = copy.deepcopy(cfg.TIEMPOS_SERVICIO)
    cfg.TIEMPOS_SERVICIO["Diag_Rapido"] = {"Verde": 6/60, "Amarillo": 8/60}

    # 3) Diag_Rapido usa su propio equipo (Medicos_Rapidos), independiente
    #    de la bolsa "Medicos" que atiende Diag_Inicial/Diag_Preliminar.
    cfg.POOL_ESTACION = copy.deepcopy(cfg.POOL_ESTACION)
    cfg.POOL_ESTACION["Diag_Rapido"] = "Medicos_Rapidos"

    # 4) Derivaciones: Triaje deriva una fracción de Verde/Amarillo al circuito
    #    rápido; Naranja/Rojo no cambian.
    cfg.PROB_DERIVACION = copy.deepcopy(cfg.PROB_DERIVACION)
    cfg.PROB_DERIVACION["Triaje"] = {
        "Verde":    {"Aislamiento": 0.05, "Diag_Rapido": 0.70, "Diag_Inicial": 0.25},
        "Amarillo": {"Aislamiento": 0.10, "Diag_Rapido": 0.30, "Diag_Inicial": 0.60},
        "Naranja":  {"Aislamiento": 0.12, "Diag_Inicial": 0.88},
        "Rojo":     {"Diag_Inicial": 1.0},
    }
    cfg.PROB_DERIVACION["Diag_Rapido"] = {
        "Verde":    {"Farmacia": 0.75, "Pruebas_Basicas": 0.25},
        "Amarillo": {"Farmacia": 0.45, "Pruebas_Basicas": 0.50, "Diag_Inicial": 0.05},
    }

    # 5) Coste del escenario
    cfg.COSTES_FIJOS = {"personal_fasttrack_anual": 310_000}
    return cfg



def construir_config_bedpool():
    cfg = construir_config_base()
    cfg.nombre = "Escenario C - Gestión Dinámica de Camas UCI"
    cfg.UCI_MODE = "pooled"   # único cambio real: pool único con prioridad por gravedad
    # cfg.UCI_TOTAL_CAPACIDAD se mantiene IGUAL que en la línea base (79 camas):
    # no se añade ninguna cama nueva, sólo se reorganiza su gestión.
    cfg.COSTES_FIJOS = {"reorganizacion_unica": 45_000}   # coste único, no anual
    return cfg

ESCENARIOS_POR_DEFECTO = {
    "Línea base": construir_config_base,
    "A · IA": construir_config_ia,
    "B · Fast-Track": construir_config_fasttrack,
    "C · UCI compartida": construir_config_bedpool,
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
