# -*- coding: utf-8 -*-
"""Motor de simulación (SimPy) de la consultora de reestructuración de deuda y
configuraciones de los escenarios. Tiempo en días.

Es el mismo gemelo digital del cuaderno '04 Reestructuración de deuda', con dos cambios menores:
* los ingresos por periodo del año pueden ir dentro de la configuración (cfg.ESTACIONES_ANIO), para
  que la aplicación los modifique como parámetro común a todos los escenarios;
* cualquier estación cuyo nombre empiece por 'Seguimiento_' se trata como estación de monitorización.
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
TIPOS = ["Preventivo", "Impago_Leve", "Impago_Grave", "Riesgo_Concursal"]

# Estacionalidad meteorológica: (tasa_ordenes_dia, [pPreventivo,pImpago_Leve,pImpago_Grave,pRiesgo_Concursal])
ESTACIONES_ANIO = {
    (0, 90):    (0.54, [0.45, 0.30, 0.18, 0.07]),   # Q1: cierre del ejercicio anterior, campaña de la renta
    (90, 181):  (0.42, [0.30, 0.35, 0.25, 0.10]),   # Q2: resultados 1T, vencimiento de líneas de crédito
    (181, 273): (0.30, [0.20, 0.25, 0.35, 0.20]),   # Q3: verano, menor actividad salvo casos ya graves
    (273, 365): (0.48, [0.25, 0.25, 0.30, 0.20]),   # Q4: cierre de ejercicio, renovación de covenants
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


ESTADOS_FINALES = {"Regularizado", "Liquidacion_Ordenada", "Insolvencia_Total"}
OBS_ESTACIONES = {"Seguimiento_Leve", "Seguimiento_Grave", "Seguimiento_Concursal"}


def es_observacion(estacion):
    """Las estaciones de monitorización (sin recurso propio) son las que empiezan por 'Seguimiento_'."""
    return estacion in OBS_ESTACIONES or estacion.startswith("Seguimiento_")


class Consultora:
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
            for nombre, cap in cfg.MESAS_CAPACIDAD.items()
        }
        self.bahia_urgente = simpy_module.Resource(env, capacity=cfg.MESA_PRIORITARIA_CAPACIDAD)

        if cfg.INTERVENCION_MODE == "pooled":
            self.contencion = {"POOL": simpy_module.PriorityResource(env, capacity=cfg.INTERVENCION_TOTAL_CAPACIDAD)}
        else:
            self.contencion = {
                "Riesgo_Concursal": simpy_module.Resource(env, capacity=cfg.INTERVENCION_CONCURSAL_CAPACIDAD),
                "Impago_Grave": simpy_module.Resource(env, capacity=cfg.INTERVENCION_GRAVE_CAPACIDAD),
                "Base": simpy_module.Resource(env, capacity=cfg.INTERVENCION_BASE_CAPACIDAD),
            }

    def info_cola(self, resource):
        q = len([r for r in resource.queue if not r.triggered and not getattr(r, "_cancelled", False)])
        return q, q + resource.count

    def recurso_de_bahia(self, nombre_logico):
        alias = self.cfg.MESA_ALIAS.get(nombre_logico, nombre_logico)
        return self.bahias[alias]

    def pool_de_estacion(self, estacion):
        return self.cfg.POOL_ESTACION.get(estacion, estacion)

    def registrar(self, id_ord, tipo, t_in, estacion, t_wait, t_server, id_recurso,
                  bahia, q_len, sys_len, destino):
        self.log_datos.append({
            "Id_expediente": id_ord, "Perfil": tipo, "T_in": round(t_in, 4),
            "Station_in": estacion, "T_wait": round(t_wait, 4),
            "T_server": round(t_server, 4), "ID_server": id_recurso,
            "Mesa": bahia, "Exp_en_cola": q_len, "Exp_en_sistema": sys_len,
            "T_out": round(t_in + t_wait + t_server, 4), "Station_out": destino,
            "T_system": round(t_wait + t_server, 4),
        })

    def finalizar_simulacion(self):
        end_time = self.env.now
        for ord_id, data in list(self.activos.items()):
            t_transcurrido = end_time - data["t_llegada_estacion"]
            self.log_datos.append({
                "Id_expediente": ord_id, "Perfil": data["tipo"], "T_in": round(data["t_llegada_estacion"], 4),
                "Station_in": data["estacion"], "T_wait": round(t_transcurrido, 4),
                "T_server": 0.0, "ID_server": data.get("id_recurso", "N/A"),
                "Mesa": data.get("bahia", "N/A"), "Exp_en_cola": data.get("q_len", 0),
                "Exp_en_sistema": data.get("sys_len", 0), "T_out": round(end_time, 4),
                "Station_out": "Fin_Simulacion", "T_system": round(t_transcurrido, 4),
            })


def liberar_bahia(consultora, bahia_propia):
    """Libera una mesa propia: un recurso físico normal (registrado en
    consultora.bahias) o una plaza de Intervención Judicial (clave especial
    'INTERV::<bolsa>')."""
    if bahia_propia is None:
        return
    nombre, req = bahia_propia
    if nombre.startswith("INTERV::"):
        bolsa = nombre.split("::", 1)[1]
        consultora.contencion[bolsa].release(req)
    else:
        consultora.bahias[nombre].release(req)


def comportamiento_orden(env, consultora, simpy, id_ord, tipo):
    cfg = consultora.cfg
    estacion_actual = "Apertura_Expediente"
    mi_bahia = "Cartera_Espera"
    urgente_propia = None     # request concedida de la Mesa Prioritaria, si el expediente ya la ocupa
    bahia_propia = None        # (nombre_logico, request) de la mesa o plaza de intervención que ocupa
    t_llegada = env.now

    consultora.activos[id_ord] = {
        "tipo": tipo, "t_llegada_estacion": t_llegada, "estacion": estacion_actual, "bahia": mi_bahia,
    }

    # Captura inmediata de la Mesa Prioritaria para Riesgo_Concursal desde Apertura de Expediente
    if cfg.MESA_PRIORITARIA_CAPTURA.get(tipo) == "Apertura_Expediente":
        req_urgente = consultora.bahia_urgente.request()
        yield req_urgente
        urgente_propia = req_urgente
        mi_bahia = "Mesa_Prioritaria"

    while estacion_actual not in ESTADOS_FINALES:
        consultora.activos[id_ord].update({
            "t_llegada_estacion": t_llegada, "estacion": estacion_actual, "bahia": mi_bahia,
        })

        # Captura de la Mesa Prioritaria en el momento indicado por la configuración
        if urgente_propia is None and cfg.MESA_PRIORITARIA_CAPTURA.get(tipo) == estacion_actual:
            req_urgente = consultora.bahia_urgente.request()
            yield req_urgente
            urgente_propia = req_urgente
            mi_bahia = "Mesa_Prioritaria"
            consultora.activos[id_ord]["bahia"] = mi_bahia

        t_wait = 0.0
        id_recurso = "N/A"
        q_len, sys_len = 0, 0

        if es_observacion(estacion_actual):
            # No hay recurso propio: se retiene la mesa ya ocupada.
            id_recurso = mi_bahia
            t_wait = 0.0

        elif estacion_actual in cfg.MESAS_CAPACIDAD or estacion_actual in cfg.MESA_ALIAS:
            recurso = consultora.recurso_de_bahia(estacion_actual)
            alias = cfg.MESA_ALIAS.get(estacion_actual, estacion_actual)
            if bahia_propia is not None and bahia_propia[0] == alias:
                t_wait = 0.0  # ya se encontraba en esa misma mesa
            else:
                q_len, sys_len = consultora.info_cola(recurso)
                t0 = env.now
                req_bahia = recurso.request()
                yield req_bahia
                t_wait = env.now - t0
                # libera la mesa anterior (si tenía una física distinta)
                if bahia_propia is not None:
                    liberar_bahia(consultora, bahia_propia)
                    bahia_propia = None
                elif urgente_propia is not None:
                    consultora.bahia_urgente.release(urgente_propia)
                    urgente_propia = None
                bahia_propia = (alias, req_bahia)
            id_recurso = alias
            mi_bahia = alias
            consultora.activos[id_ord]["bahia"] = mi_bahia

        elif estacion_actual == "Intervencion_Judicial":
            prioridad = {"Riesgo_Concursal": 0, "Impago_Grave": 1, "Impago_Leve": 2, "Preventivo": 2}[tipo]
            if cfg.INTERVENCION_MODE == "pooled":
                pool = consultora.contencion["POOL"]
                q_len, sys_len = consultora.info_cola(pool)
                t0 = env.now
                req_cont = pool.request(priority=prioridad)
            else:
                bolsa = {"Riesgo_Concursal": "Riesgo_Concursal", "Impago_Grave": "Impago_Grave"}.get(tipo, "Base")
                pool = consultora.contencion[bolsa]
                q_len, sys_len = consultora.info_cola(pool)
                t0 = env.now
                req_cont = pool.request()

            # Carrera entre conseguir plaza y agotar el límite de espera
            resultado = yield req_cont | env.timeout(cfg.LIMITE_ESPERA_INTERVENCION)
            if req_cont in resultado:
                t_wait = env.now - t0
                if bahia_propia is not None:
                    liberar_bahia(consultora, bahia_propia)
                    bahia_propia = None
                elif urgente_propia is not None:
                    consultora.bahia_urgente.release(urgente_propia)
                    urgente_propia = None
                bahia_propia = ("INTERV::" + (bolsa if cfg.INTERVENCION_MODE != "pooled" else "POOL"), req_cont)
                id_recurso = "Intervencion_Judicial"
                mi_bahia = "Intervencion_Judicial"
            else:
                req_cont.cancel()
                t_wait = env.now - t0
                # sin plaza de contención disponible dentro del límite: insolvencia total
                consultora.registrar(id_ord, tipo, t_llegada, estacion_actual,
                                  t_wait, 0.0, "Sin_Intervencion_Disponible", mi_bahia, q_len, sys_len,
                                  "Insolvencia_Total")
                estacion_actual = "Insolvencia_Total"
                break
            consultora.activos[id_ord]["bahia"] = mi_bahia

        else:
            # Estación "servidor": el recurso se libera justo tras el servicio
            pool_nombre = consultora.pool_de_estacion(estacion_actual)
            recurso = consultora.servidores[pool_nombre]
            q_len, sys_len = consultora.info_cola(recurso)
            t0 = env.now
            req = recurso.request()
            yield req
            t_wait = env.now - t0
            id_recurso = pool_nombre

        t_servicio = generar_tiempo_servicio(cfg, estacion_actual, tipo)
        yield env.timeout(t_servicio)

        if (not es_observacion(estacion_actual)
                and estacion_actual not in cfg.MESAS_CAPACIDAD
                and estacion_actual not in cfg.MESA_ALIAS
                and estacion_actual != "Intervencion_Judicial"):
            consultora.servidores[consultora.pool_de_estacion(estacion_actual)].release(req)

        destino = determinar_destino(cfg, estacion_actual, tipo)

        if destino in ESTADOS_FINALES:
            if urgente_propia is not None:
                consultora.bahia_urgente.release(urgente_propia)
                urgente_propia = None
            if bahia_propia is not None:
                liberar_bahia(consultora, bahia_propia)
                bahia_propia = None

        consultora.registrar(id_ord, tipo, t_llegada, estacion_actual,
                          t_wait, t_servicio, id_recurso, mi_bahia, q_len, sys_len, destino)

        estacion_actual = destino
        t_llegada = env.now

    if id_ord in consultora.activos:
        del consultora.activos[id_ord]
    if urgente_propia is not None:
        consultora.bahia_urgente.release(urgente_propia)
    if bahia_propia is not None:
        liberar_bahia(consultora, bahia_propia)


def generador_ordenes(env, consultora, simpy, cfg):
    id_ord = 1
    llegadas = getattr(cfg, "ESTACIONES_ANIO", None)
    while True:
        tasa, probs = obtener_perfil_temporada(env.now, llegadas)
        yield env.timeout(random.expovariate(tasa))
        tipo = random.choices(TIPOS, weights=probs)[0]
        env.process(comportamiento_orden(env, consultora, simpy, f"EXP_{id_ord}", tipo))
        id_ord += 1


def simular(cfg, dias, semilla, simpy_module=None):
    """Ejecuta la simulación de `cfg` durante `dias` días con la semilla dada.
    Devuelve un DataFrame de pandas con la traza completa (mismo formato que
    `traza_consultora_eolico.csv.gz`). `dias` es directamente la unidad de tiempo
    de la simulación: no se aplica ningún factor de conversión adicional."""
    if simpy_module is None:
        simpy_module = simpy
    random.seed(semilla)
    env = simpy_module.Environment()
    consultora = Consultora(env, cfg, simpy_module)
    env.process(generador_ordenes(env, consultora, simpy_module, cfg))
    env.run(until=dias)
    consultora.finalizar_simulacion()
    df_sim = pd.DataFrame(consultora.log_datos)
    if len(df_sim):
        df_sim = df_sim.sort_values(by=["T_in"]).reset_index(drop=True)
    return df_sim


def construir_config_base():
    cfg = Config()
    cfg.nombre = "Línea Base (protocolo actual)"

    cfg.SERVIDORES_CAPACIDAD = {
        "Apertura_Expediente": 1,
        "Equipo_Analistas": 1,                  # atiende Diagnostico_Financiero y Comite_Evaluacion
        "Auditoria_Contable": 1,
        "Peritaje_Independiente": 1,            # único perito independiente -> cuello de botella deliberado
        "Consulta_Juridica": 1,
        "Homologacion_Final": 2,
    }

    # Capacidades "mesa" calibradas por Ley de Little: se midió la ocupación
    # real (L = lambda * W) simulando con capacidad artificialmente ilimitada
    # durante 40 años, y se dimensionó cada recurso por encima del percentil
    # 90-95 de ocupación observada (no solo de la media: los tiempos de
    # negociación tienen varianza alta, y dimensionar solo por la media
    # produce inestabilidad -colas que crecen sin límite- a largo plazo).
    cfg.MESA_PRIORITARIA_CAPACIDAD = 20
    cfg.MESAS_CAPACIDAD = {
        "Recopilacion_Documental": 25,
        "Simulacion_Estres_Financiero": 2,
        "Mesa_Negociacion": 35,
        "Reestructuracion_Profunda": 6,
    }
    cfg.MESA_ALIAS = {"Mesa_Negociacion_PostIntervencion": "Mesa_Negociacion"}
    cfg.POOL_ESTACION = {
        "Diagnostico_Financiero": "Equipo_Analistas",
        "Comite_Evaluacion": "Equipo_Analistas",
    }
    cfg.MESA_PRIORITARIA_CAPTURA = {"Riesgo_Concursal": "Apertura_Expediente", "Impago_Grave": "Diagnostico_Financiero"}

    cfg.INTERVENCION_MODE = "partitioned"     # "partitioned" (línea base) o "pooled" (Escenario C)
    cfg.INTERVENCION_CONCURSAL_CAPACIDAD = 2
    cfg.INTERVENCION_GRAVE_CAPACIDAD = 2
    cfg.INTERVENCION_BASE_CAPACIDAD = 1
    cfg.INTERVENCION_TOTAL_CAPACIDAD = (cfg.INTERVENCION_CONCURSAL_CAPACIDAD
                                         + cfg.INTERVENCION_GRAVE_CAPACIDAD
                                         + cfg.INTERVENCION_BASE_CAPACIDAD)  # 5
    cfg.LIMITE_ESPERA_INTERVENCION = 4.0    # días

    cfg.TIEMPOS_SERVICIO = {
        "Apertura_Expediente":           {t: 0.05 for t in TIPOS},
        "Diagnostico_Financiero":        {"Preventivo": 0.40, "Impago_Leve": 0.40, "Impago_Grave": 1.00, "Riesgo_Concursal": 1.00},
        "Recopilacion_Documental": {
            "Preventivo": 7.0, "Impago_Leve": 10.0, "Impago_Grave": 15.0, "Riesgo_Concursal": 5.0,
        },
        "Auditoria_Contable":            {t: 1.0 for t in TIPOS},
        "Peritaje_Independiente":        {t: 7.0 for t in TIPOS},
        "Consulta_Juridica":             {t: 2.0 for t in TIPOS},
        "Comite_Evaluacion":             {"Preventivo": 0.40, "Impago_Leve": 0.40, "Impago_Grave": 1.00, "Riesgo_Concursal": 1.00},
        "Simulacion_Estres_Financiero":  {"Impago_Grave": 1.0, "Riesgo_Concursal": 1.0},
        "Mesa_Negociacion": {
            "Preventivo": 15.0, "Impago_Leve": 30.0, "Impago_Grave": 60.0, "Riesgo_Concursal": 90.0,
        },
        "Mesa_Negociacion_PostIntervencion": {
            "Preventivo": 10.0, "Impago_Leve": 15.0, "Impago_Grave": 25.0, "Riesgo_Concursal": 35.0,
        },
        "Intervencion_Judicial": {
            "Preventivo": 10.0, "Impago_Leve": 10.0, "Impago_Grave": 15.0, "Riesgo_Concursal": 20.0,
        },
        "Reestructuracion_Profunda":     {t: 45.0 for t in TIPOS},
        "Homologacion_Final":            {t: 2.0 for t in TIPOS},
        "Seguimiento_Leve":              {"Impago_Leve": 10.0},
        "Seguimiento_Grave":             {"Impago_Grave": 15.0},
        "Seguimiento_Concursal":         {"Riesgo_Concursal": 20.0},
    }

    cfg.PROB_DERIVACION = {
        "Apertura_Expediente": {t: {"Diagnostico_Financiero": 1.0} for t in TIPOS},

        "Diagnostico_Financiero": {
            "Preventivo":       {"Recopilacion_Documental": 1.0},
            "Impago_Leve":      {"Recopilacion_Documental": 1.0},
            "Impago_Grave":     {"Auditoria_Contable": 0.40, "Peritaje_Independiente": 0.30,
                                  "Consulta_Juridica": 0.10, "Simulacion_Estres_Financiero": 0.10,
                                  "Intervencion_Judicial": 0.10},
            "Riesgo_Concursal": {"Auditoria_Contable": 0.25, "Consulta_Juridica": 0.10,
                                  "Simulacion_Estres_Financiero": 0.25, "Intervencion_Judicial": 0.40},
        },

        # Recopilacion_Documental cumple DOS papeles: recopilación documental
        # DE ENTRADA (Preventivo/Impago_Leve, antes del análisis detallado) y
        # recopilación documental DE CIERRE (Impago_Grave/Riesgo_Concursal, ya
        # avanzado el expediente, ante una petición de documentación adicional).
        "Recopilacion_Documental": {
            "Preventivo":       {"Auditoria_Contable": 1.0},
            "Impago_Leve":      {"Auditoria_Contable": 1.0},
            "Impago_Grave":     {"Homologacion_Final": 0.55, "Mesa_Negociacion": 0.30, "Liquidacion_Ordenada": 0.15},
            "Riesgo_Concursal": {"Homologacion_Final": 0.45, "Mesa_Negociacion": 0.25, "Liquidacion_Ordenada": 0.30},
        },

        "Auditoria_Contable": {
            "Preventivo":       {"Comite_Evaluacion": 0.85, "Peritaje_Independiente": 0.15},
            "Impago_Leve":      {"Comite_Evaluacion": 0.70, "Peritaje_Independiente": 0.30},
            "Impago_Grave":     {"Peritaje_Independiente": 0.35, "Consulta_Juridica": 0.25, "Comite_Evaluacion": 0.40},
            "Riesgo_Concursal": {"Peritaje_Independiente": 0.20, "Consulta_Juridica": 0.15,
                                  "Simulacion_Estres_Financiero": 0.25, "Comite_Evaluacion": 0.40},
        },

        "Peritaje_Independiente": {t: {"Comite_Evaluacion": 1.0} for t in TIPOS},

        "Consulta_Juridica": {
            "Impago_Grave":     {"Comite_Evaluacion": 0.60, "Simulacion_Estres_Financiero": 0.40},
            "Riesgo_Concursal": {"Simulacion_Estres_Financiero": 0.50, "Intervencion_Judicial": 0.30,
                                  "Comite_Evaluacion": 0.20},
        },

        "Comite_Evaluacion": {
            "Preventivo":       {"Mesa_Negociacion": 0.55, "Homologacion_Final": 0.45},
            "Impago_Leve":      {"Mesa_Negociacion": 0.40, "Homologacion_Final": 0.40, "Seguimiento_Leve": 0.20},
            "Impago_Grave":     {"Mesa_Negociacion": 0.35, "Recopilacion_Documental": 0.15, "Seguimiento_Grave": 0.20,
                                  "Liquidacion_Ordenada": 0.20, "Insolvencia_Total": 0.05, "Consulta_Juridica": 0.05},
            "Riesgo_Concursal": {"Intervencion_Judicial": 0.25, "Mesa_Negociacion": 0.20, "Recopilacion_Documental": 0.10,
                                  "Seguimiento_Concursal": 0.15, "Liquidacion_Ordenada": 0.20, "Insolvencia_Total": 0.05,
                                  "Simulacion_Estres_Financiero": 0.05},
        },

        "Simulacion_Estres_Financiero": {
            "Impago_Grave":     {"Intervencion_Judicial": 0.15, "Recopilacion_Documental": 0.59,
                                  "Liquidacion_Ordenada": 0.20, "Insolvencia_Total": 0.06},
            "Riesgo_Concursal": {"Intervencion_Judicial": 0.45, "Recopilacion_Documental": 0.27,
                                  "Liquidacion_Ordenada": 0.20, "Insolvencia_Total": 0.08},
        },

        "Seguimiento_Leve":     {"Impago_Leve": {"Homologacion_Final": 0.65, "Seguimiento_Leve": 0.20,
                                                  "Mesa_Negociacion": 0.15}},
        "Seguimiento_Grave":    {"Impago_Grave": {"Homologacion_Final": 0.30, "Seguimiento_Grave": 0.25,
                                                   "Mesa_Negociacion": 0.25, "Liquidacion_Ordenada": 0.17,
                                                   "Insolvencia_Total": 0.03}},
        "Seguimiento_Concursal": {"Riesgo_Concursal": {"Seguimiento_Concursal": 0.10, "Mesa_Negociacion": 0.40,
                                                        "Intervencion_Judicial": 0.25, "Liquidacion_Ordenada": 0.20,
                                                        "Insolvencia_Total": 0.05}},

        "Mesa_Negociacion": {
            "Preventivo":       {"Homologacion_Final": 0.55, "Mesa_Negociacion": 0.30,
                                  "Recopilacion_Documental": 0.10, "Liquidacion_Ordenada": 0.05},
            "Impago_Leve":      {"Homologacion_Final": 0.50, "Mesa_Negociacion": 0.26, "Recopilacion_Documental": 0.10,
                                  "Intervencion_Judicial": 0.05, "Liquidacion_Ordenada": 0.08, "Insolvencia_Total": 0.01},
            "Impago_Grave":     {"Homologacion_Final": 0.35, "Mesa_Negociacion": 0.26, "Recopilacion_Documental": 0.12,
                                  "Intervencion_Judicial": 0.10, "Liquidacion_Ordenada": 0.14, "Insolvencia_Total": 0.03},
            "Riesgo_Concursal": {"Homologacion_Final": 0.25, "Mesa_Negociacion": 0.20, "Recopilacion_Documental": 0.10,
                                  "Intervencion_Judicial": 0.15, "Liquidacion_Ordenada": 0.24, "Insolvencia_Total": 0.06},
        },

        "Intervencion_Judicial": {
            "Preventivo":       {"Mesa_Negociacion_PostIntervencion": 0.90, "Liquidacion_Ordenada": 0.05, "Insolvencia_Total": 0.05},
            "Impago_Leve":      {"Mesa_Negociacion_PostIntervencion": 0.90, "Liquidacion_Ordenada": 0.05, "Insolvencia_Total": 0.05},
            "Impago_Grave":     {"Mesa_Negociacion_PostIntervencion": 0.73, "Liquidacion_Ordenada": 0.15, "Insolvencia_Total": 0.12},
            "Riesgo_Concursal": {"Mesa_Negociacion_PostIntervencion": 0.62, "Liquidacion_Ordenada": 0.20, "Insolvencia_Total": 0.18},
        },

        "Mesa_Negociacion_PostIntervencion": {
            "Preventivo":       {"Homologacion_Final": 0.40, "Reestructuracion_Profunda": 0.60},
            "Impago_Leve":      {"Homologacion_Final": 0.55, "Reestructuracion_Profunda": 0.45},
            "Impago_Grave":     {"Homologacion_Final": 0.45, "Reestructuracion_Profunda": 0.55},
            "Riesgo_Concursal": {"Homologacion_Final": 0.35, "Reestructuracion_Profunda": 0.65},
        },

        "Reestructuracion_Profunda": {t: {"Homologacion_Final": 0.80, "Liquidacion_Ordenada": 0.20} for t in TIPOS},

        "Homologacion_Final": {t: {"Regularizado": 1.0} for t in TIPOS},
    }

    cfg.COSTES_FIJOS = {}   # el protocolo actual no tiene coste fijo adicional
    return cfg


def construir_config_analitica_predictiva():
    cfg = construir_config_base()
    cfg.nombre = "Escenario A - Analítica Predictiva de Riesgo de Impago"

    # 1) Tiempos de servicio: el modelo predictivo acelera drásticamente el
    #    peritaje independiente y agiliza la lectura del abogado.
    cfg.TIEMPOS_SERVICIO = copy.deepcopy(cfg.TIEMPOS_SERVICIO)
    for tipo in cfg.TIEMPOS_SERVICIO["Peritaje_Independiente"]:
        cfg.TIEMPOS_SERVICIO["Peritaje_Independiente"][tipo] *= 0.55
    for tipo in cfg.TIEMPOS_SERVICIO["Consulta_Juridica"]:
        cfg.TIEMPOS_SERVICIO["Consulta_Juridica"][tipo] *= 0.65

    # 2) Derivaciones: el scoring predictivo anticipa mejor el alcance real
    #    del deterioro financiero -> menos Insolvencia_Total y menos
    #    Liquidacion_Ordenada para Impago_Grave/Riesgo_Concursal.
    cfg.PROB_DERIVACION = copy.deepcopy(cfg.PROB_DERIVACION)
    cfg.PROB_DERIVACION["Comite_Evaluacion"]["Impago_Grave"] = {
        "Mesa_Negociacion": 0.50, "Recopilacion_Documental": 0.15, "Seguimiento_Grave": 0.20,
        "Liquidacion_Ordenada": 0.10, "Insolvencia_Total": 0.02, "Consulta_Juridica": 0.03,
    }
    cfg.PROB_DERIVACION["Comite_Evaluacion"]["Riesgo_Concursal"] = {
        "Intervencion_Judicial": 0.25, "Mesa_Negociacion": 0.33, "Recopilacion_Documental": 0.10,
        "Seguimiento_Concursal": 0.15, "Liquidacion_Ordenada": 0.10, "Insolvencia_Total": 0.02,
        "Simulacion_Estres_Financiero": 0.05,
    }

    # 3) Coste del escenario
    cfg.COSTES_FIJOS = {"licencia_analitica_predictiva_anual": 8_000}
    return cfg



def construir_config_intervencion_temprana():
    cfg = construir_config_base()
    cfg.nombre = "Escenario B - Equipo de Intervención Temprana"

    # 1) Nuevo recurso: segundo equipo de analistas, dedicado en exclusiva a
    #    los expedientes en riesgo concursal.
    cfg.SERVIDORES_CAPACIDAD = copy.deepcopy(cfg.SERVIDORES_CAPACIDAD)
    cfg.SERVIDORES_CAPACIDAD["Equipo_Intervencion_Temprana"] = 1

    # 2) Nueva estación de diagnóstico inicial para Riesgo_Concursal, atendida
    #    por este equipo dedicado, independiente de "Equipo_Analistas".
    cfg.POOL_ESTACION = copy.deepcopy(cfg.POOL_ESTACION)
    cfg.POOL_ESTACION["Diagnostico_Financiero_IT"] = "Equipo_Intervencion_Temprana"

    cfg.TIEMPOS_SERVICIO = copy.deepcopy(cfg.TIEMPOS_SERVICIO)
    cfg.TIEMPOS_SERVICIO["Diagnostico_Financiero_IT"] = {"Riesgo_Concursal": 0.35}

    # 3) Derivaciones: Riesgo_Concursal entra ahora por la vía dedicada en
    #    lugar de compartir "Equipo_Analistas" con el resto de perfiles.
    cfg.PROB_DERIVACION = copy.deepcopy(cfg.PROB_DERIVACION)
    cfg.PROB_DERIVACION["Apertura_Expediente"]["Riesgo_Concursal"] = {"Diagnostico_Financiero_IT": 1.0}
    cfg.PROB_DERIVACION["Diagnostico_Financiero_IT"] = {
        "Riesgo_Concursal": {"Auditoria_Contable": 0.20, "Consulta_Juridica": 0.10,
                              "Simulacion_Estres_Financiero": 0.25, "Intervencion_Judicial": 0.45},
    }

    # 4) Coste del escenario
    cfg.COSTES_FIJOS = {"personal_intervencion_temprana_anual": 15_000}
    return cfg



def construir_config_intervencion_dinamica():
    cfg = construir_config_base()
    cfg.nombre = "Escenario C - Gestión Dinámica de Turnos de Intervención Judicial"
    cfg.INTERVENCION_MODE = "pooled"   # único cambio real: pool único con prioridad por gravedad
    # cfg.INTERVENCION_TOTAL_CAPACIDAD se mantiene IGUAL que en la línea base
    # (5 plazas): no se añade ninguna plaza nueva, sólo se reorganiza su gestión.
    cfg.COSTES_FIJOS = {"reorganizacion_unica": 4_000}   # coste único, no anual
    return cfg

ESCENARIOS_POR_DEFECTO = {
    "Línea base": construir_config_base,
    "A · Analítica predictiva": construir_config_analitica_predictiva,
    "B · Intervención temprana": construir_config_intervencion_temprana,
    "C · Turnos compartidos": construir_config_intervencion_dinamica,
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
