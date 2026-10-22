# -*- coding: utf-8 -*-
"""Motor de simulación (SimPy) del centro de cría en cautividad y reintroducción y configuraciones
de los escenarios. Tiempo en días.

Es el mismo gemelo digital del cuaderno '02 Centro de Cría en Cautividad y Reintroducción', con dos
cambios menores:
* los ingresos por estación del año pueden ir dentro de la configuración (cfg.ESTACIONES_ANIO), para
  que la aplicación los modifique como parámetro común a todos los escenarios;
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
TIPOS = ["Fundador", "Cautivo", "RescateLeve", "RescateCritico"]

ESTACIONES_ANIO = {
    (335, 59):  (0.08, [0.20, 0.10, 0.40, 0.30]),   # Invierno: baja actividad, traslados planificados
    (59, 151):  (0.20, [0.05, 0.55, 0.30, 0.10]),   # Primavera: temporada de partos en cautividad
    (151, 243): (0.22, [0.05, 0.10, 0.35, 0.50]),   # Verano: incendios, sequía -> más rescates críticos
    (243, 335): (0.18, [0.10, 0.10, 0.40, 0.40]),   # Otoño: caza, atropellos -> más rescates
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


ESTADOS_FINALES = {"Liberado", "Fallece", "Cautividad_Permanente"}
OBS_ESTACIONES = {"Obs_Cautivo", "Obs_RescateLeve", "Obs_RescateCritico"}


def es_observacion(estacion):
    """Las estaciones de observación (sin recurso propio) son las que empiezan por 'Obs_'."""
    return estacion in OBS_ESTACIONES or estacion.startswith("Obs_")


class Centro:
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
        self.jaulas = {
            nombre: simpy_module.Resource(env, capacity=cap)
            for nombre, cap in cfg.JAULAS_CAPACIDAD.items()
        }
        self.recinto = simpy_module.Resource(env, capacity=cfg.RECINTO_CAPACIDAD)

        if cfg.UCI_MODE == "pooled":
            self.uci = {"POOL": simpy_module.PriorityResource(env, capacity=cfg.UCI_TOTAL_CAPACIDAD)}
        else:
            self.uci = {
                "Critico": simpy_module.Resource(env, capacity=cfg.UCI_CRITICO_CAPACIDAD),
                "Leve": simpy_module.Resource(env, capacity=cfg.UCI_LEVE_CAPACIDAD),
                "Base": simpy_module.Resource(env, capacity=cfg.UCI_BASE_CAPACIDAD),
            }

    def info_cola(self, resource):
        q = len([r for r in resource.queue if not r.triggered and not getattr(r, "_cancelled", False)])
        return q, q + resource.count

    def recurso_de_jaula(self, nombre_logico):
        alias = self.cfg.JAULA_ALIAS.get(nombre_logico, nombre_logico)
        return self.jaulas[alias]

    def pool_de_estacion(self, estacion):
        return self.cfg.POOL_ESTACION.get(estacion, estacion)

    def registrar(self, id_ind, tipo, t_in, estacion, t_wait, t_server, id_recurso,
                  sala, q_len, sys_len, destino):
        self.log_datos.append({
            "Id_ind": id_ind, "Perfil": tipo, "T_in": round(t_in, 4),
            "Station_in": estacion, "T_wait": round(t_wait, 4),
            "T_server": round(t_server, 4), "ID_server": id_recurso,
            "Recinto": sala, "Ind_en_cola": q_len, "Ind_en_sistema": sys_len,
            "T_out": round(t_in + t_wait + t_server, 4), "Station_out": destino,
            "T_system": round(t_wait + t_server, 4),
        })

    def finalizar_simulacion(self):
        end_time = self.env.now
        for ind_id, data in list(self.activos.items()):
            t_transcurrido = end_time - data["t_llegada_estacion"]
            self.log_datos.append({
                "Id_ind": ind_id, "Perfil": data["tipo"], "T_in": round(data["t_llegada_estacion"], 4),
                "Station_in": data["estacion"], "T_wait": round(t_transcurrido, 4),
                "T_server": 0.0, "ID_server": data.get("id_recurso", "N/A"),
                "Recinto": data.get("sala", "N/A"), "Ind_en_cola": data.get("q_len", 0),
                "Ind_en_sistema": data.get("sys_len", 0), "T_out": round(end_time, 4),
                "Station_out": "Fin_Simulacion", "T_system": round(t_transcurrido, 4),
            })


def liberar_jaula(centro, jaula_propia):
    """Libera una 'jaula' propia: un recurso físico normal (registrado en
    centro.jaulas) o una plaza de UCI (clave especial 'UCI::<bolsa>')."""
    if jaula_propia is None:
        return
    nombre, req = jaula_propia
    if nombre.startswith("UCI::"):
        bolsa = nombre.split("::", 1)[1]
        centro.uci[bolsa].release(req)
    else:
        centro.jaulas[nombre].release(req)


def comportamiento_individuo(env, centro, simpy, id_ind, tipo):
    cfg = centro.cfg
    estacion_actual = "Admision"
    mi_sala = "Recinto_Espera"
    recinto_propio = None     # request concedida del Recinto de Recuperación, si el individuo ya lo ocupa
    jaula_propia = None        # (nombre_logico, request) de la jaula o plaza de UCI que ocupa
    t_llegada = env.now

    centro.activos[id_ind] = {
        "tipo": tipo, "t_llegada_estacion": t_llegada, "estacion": estacion_actual, "sala": mi_sala,
    }

    # Captura inmediata del Recinto de Recuperación para RescateCritico desde Admisión
    if cfg.RECINTO_CAPTURA.get(tipo) == "Admision":
        req_recinto = centro.recinto.request()
        yield req_recinto
        recinto_propio = req_recinto
        mi_sala = "Recinto_Recuperacion"

    while estacion_actual not in ESTADOS_FINALES:
        centro.activos[id_ind].update({
            "t_llegada_estacion": t_llegada, "estacion": estacion_actual, "sala": mi_sala,
        })

        # Captura del Recinto de Recuperación en el momento indicado por la configuración
        if recinto_propio is None and cfg.RECINTO_CAPTURA.get(tipo) == estacion_actual:
            req_recinto = centro.recinto.request()
            yield req_recinto
            recinto_propio = req_recinto
            mi_sala = "Recinto_Recuperacion"
            centro.activos[id_ind]["sala"] = mi_sala

        t_wait = 0.0
        id_recurso = "N/A"
        q_len, sys_len = 0, 0

        if es_observacion(estacion_actual):
            # No hay recurso propio: se retiene la sala/recinto ya ocupado.
            id_recurso = mi_sala
            t_wait = 0.0

        elif estacion_actual in cfg.JAULAS_CAPACIDAD or estacion_actual in cfg.JAULA_ALIAS:
            recurso = centro.recurso_de_jaula(estacion_actual)
            alias = cfg.JAULA_ALIAS.get(estacion_actual, estacion_actual)
            if jaula_propia is not None and jaula_propia[0] == alias:
                t_wait = 0.0  # ya se encontraba en esa misma jaula
            else:
                q_len, sys_len = centro.info_cola(recurso)
                t0 = env.now
                req_jaula = recurso.request()
                yield req_jaula
                t_wait = env.now - t0
                # libera la jaula anterior (si tenía una física distinta)
                if jaula_propia is not None:
                    liberar_jaula(centro, jaula_propia)
                    jaula_propia = None
                elif recinto_propio is not None:
                    centro.recinto.release(recinto_propio)
                    recinto_propio = None
                jaula_propia = (alias, req_jaula)
            id_recurso = alias
            mi_sala = alias
            centro.activos[id_ind]["sala"] = mi_sala

        elif estacion_actual == "UCI_Veterinaria":
            prioridad = {"RescateCritico": 0, "RescateLeve": 1, "Cautivo": 2, "Fundador": 2}[tipo]
            if cfg.UCI_MODE == "pooled":
                pool = centro.uci["POOL"]
                q_len, sys_len = centro.info_cola(pool)
                t0 = env.now
                req_uci = pool.request(priority=prioridad)
            else:
                bolsa = {"RescateCritico": "Critico", "RescateLeve": "Leve"}.get(tipo, "Base")
                pool = centro.uci[bolsa]
                q_len, sys_len = centro.info_cola(pool)
                t0 = env.now
                req_uci = pool.request()

            # Carrera entre conseguir plaza y agotar el límite de espera
            resultado = yield req_uci | env.timeout(cfg.LIMITE_ESPERA_UCI)
            if req_uci in resultado:
                t_wait = env.now - t0
                if jaula_propia is not None:
                    liberar_jaula(centro, jaula_propia)
                    jaula_propia = None
                elif recinto_propio is not None:
                    centro.recinto.release(recinto_propio)
                    recinto_propio = None
                jaula_propia = ("UCI::" + (bolsa if cfg.UCI_MODE != "pooled" else "POOL"), req_uci)
                id_recurso = "UCI_Veterinaria"
                mi_sala = "UCI_Veterinaria"
            else:
                req_uci.cancel()
                t_wait = env.now - t0
                # sin plaza de UCI disponible dentro del límite: desenlace fatal
                centro.registrar(id_ind, tipo, t_llegada, estacion_actual,
                                  t_wait, 0.0, "Sin_UCI_Disponible", mi_sala, q_len, sys_len, "Fallece")
                estacion_actual = "Fallece"
                break
            centro.activos[id_ind]["sala"] = mi_sala

        else:
            # Estación "servidor": el recurso se libera justo tras el servicio
            pool_nombre = centro.pool_de_estacion(estacion_actual)
            recurso = centro.servidores[pool_nombre]
            q_len, sys_len = centro.info_cola(recurso)
            t0 = env.now
            req = recurso.request()
            yield req
            t_wait = env.now - t0
            id_recurso = pool_nombre

        t_servicio = generar_tiempo_servicio(cfg, estacion_actual, tipo)
        yield env.timeout(t_servicio)

        if (not es_observacion(estacion_actual)
                and estacion_actual not in cfg.JAULAS_CAPACIDAD
                and estacion_actual not in cfg.JAULA_ALIAS
                and estacion_actual != "UCI_Veterinaria"):
            centro.servidores[centro.pool_de_estacion(estacion_actual)].release(req)

        destino = determinar_destino(cfg, estacion_actual, tipo)

        if destino in ESTADOS_FINALES:
            if recinto_propio is not None:
                centro.recinto.release(recinto_propio)
                recinto_propio = None
            if jaula_propia is not None:
                liberar_jaula(centro, jaula_propia)
                jaula_propia = None

        centro.registrar(id_ind, tipo, t_llegada, estacion_actual,
                          t_wait, t_servicio, id_recurso, mi_sala, q_len, sys_len, destino)

        estacion_actual = destino
        t_llegada = env.now

    if id_ind in centro.activos:
        del centro.activos[id_ind]
    if recinto_propio is not None:
        centro.recinto.release(recinto_propio)
    if jaula_propia is not None:
        liberar_jaula(centro, jaula_propia)


def generador_entradas(env, centro, simpy, cfg):
    id_ind = 1
    llegadas = getattr(cfg, "ESTACIONES_ANIO", None)
    while True:
        tasa, probs = obtener_perfil_temporada(env.now, llegadas)
        yield env.timeout(random.expovariate(tasa))
        tipo = random.choices(TIPOS, weights=probs)[0]
        env.process(comportamiento_individuo(env, centro, simpy, f"IND_{id_ind}", tipo))
        id_ind += 1


def simular(cfg, dias, semilla, simpy_module=None):
    """Ejecuta la simulación de `cfg` durante `dias` días con la semilla dada.
    Devuelve un DataFrame de pandas con la traza completa (mismo formato que
    `traza_centro_conservacion.csv.gz`). `dias` es directamente la unidad de
    tiempo de la simulación: no se aplica ningún factor de conversión
    adicional."""
    if simpy_module is None:
        simpy_module = simpy
    random.seed(semilla)
    env = simpy_module.Environment()
    centro = Centro(env, cfg, simpy_module)
    env.process(generador_entradas(env, centro, simpy_module, cfg))
    env.run(until=dias)
    centro.finalizar_simulacion()
    df_sim = pd.DataFrame(centro.log_datos)
    if len(df_sim):
        df_sim = df_sim.sort_values(by=["T_in"]).reset_index(drop=True)
    return df_sim


def construir_config_base():
    cfg = Config()
    cfg.nombre = "Línea Base (protocolo actual)"

    cfg.SERVIDORES_CAPACIDAD = {
        "Admision": 1,
        "Revision_Ingreso": 1,
        "Equipo_Veterinario": 2,           # atiende Evaluacion_Veterinaria_Inicial y Evaluacion_Preliminar
        "Analitica_Basica": 1,
        "Pruebas_Geneticas_Avanzadas": 1,  # único secuenciador/analista genético -> cuello de botella deliberado
        "Consulta_Genetista": 1,
        "Preparacion_Suelta": 1,
    }

    # Capacidades "jaula/recinto" calibradas por Ley de Little: se midió la
    # ocupación media real (L = lambda * W) simulando con capacidad
    # artificialmente ilimitada durante 100 años, y se dimensionó cada
    # recurso por encima del percentil 90-95 de ocupación observada (no solo
    # de la media: los tiempos de estancia en el Módulo de Cría tienen
    # varianza muy alta, y dimensionar solo por la media produce
    # inestabilidad -colas que crecen sin límite- a largo plazo).
    cfg.RECINTO_CAPACIDAD = 3
    cfg.JAULAS_CAPACIDAD = {
        "Cuarentena": 5,
        "Cirugia_Veterinaria": 1,
        "Modulo_Cria": 22,
        "Recuperacion_PostCritica": 3,
    }
    cfg.JAULA_ALIAS = {"Modulo_Cria_PostUCI": "Modulo_Cria"}
    cfg.POOL_ESTACION = {
        "Evaluacion_Veterinaria_Inicial": "Equipo_Veterinario",
        "Evaluacion_Preliminar": "Equipo_Veterinario",
    }
    cfg.RECINTO_CAPTURA = {"RescateCritico": "Admision", "RescateLeve": "Evaluacion_Veterinaria_Inicial"}

    cfg.UCI_MODE = "partitioned"       # "partitioned" (línea base) o "pooled" (Escenario C)
    cfg.UCI_CRITICO_CAPACIDAD = 3
    cfg.UCI_LEVE_CAPACIDAD = 1
    cfg.UCI_BASE_CAPACIDAD = 1
    cfg.UCI_TOTAL_CAPACIDAD = cfg.UCI_CRITICO_CAPACIDAD + cfg.UCI_LEVE_CAPACIDAD + cfg.UCI_BASE_CAPACIDAD  # 5
    cfg.LIMITE_ESPERA_UCI = 2.0         # días

    cfg.TIEMPOS_SERVICIO = {
        "Admision":                     {t: 0.02 for t in TIPOS},
        "Revision_Ingreso":             {t: 0.05 for t in TIPOS},
        "Cuarentena": {
            "Fundador": 21.0, "Cautivo": 14.0, "RescateLeve": 10.0, "RescateCritico": 10.0,
        },
        "Evaluacion_Veterinaria_Inicial": {t: 0.15 for t in TIPOS},
        "Analitica_Basica":              {t: 0.10 for t in TIPOS},
        "Pruebas_Geneticas_Avanzadas":   {t: 3.0 for t in TIPOS},
        "Consulta_Genetista":            {t: 0.30 for t in TIPOS},
        "Evaluacion_Preliminar":         {t: 0.20 for t in TIPOS},
        "Cirugia_Veterinaria":           {"RescateLeve": 0.15, "RescateCritico": 0.25},
        "Modulo_Cria": {
            "Fundador": 400.0, "Cautivo": 300.0, "RescateLeve": 250.0, "RescateCritico": 200.0,
        },
        "Modulo_Cria_PostUCI": {
            "Fundador": 20.0, "Cautivo": 15.0, "RescateLeve": 12.0, "RescateCritico": 18.0,
        },
        "UCI_Veterinaria": {
            "Fundador": 10.0, "Cautivo": 10.0, "RescateLeve": 12.0, "RescateCritico": 15.0,
        },
        "Recuperacion_PostCritica":      {t: 30.0 for t in TIPOS},
        "Preparacion_Suelta":            {t: 0.5 for t in TIPOS},
        "Obs_Cautivo":                   {"Cautivo": 5.0},
        "Obs_RescateLeve":               {"RescateLeve": 7.0},
        "Obs_RescateCritico":            {"RescateCritico": 10.0},
    }

    cfg.PROB_DERIVACION = {
        "Admision": {t: {"Revision_Ingreso": 1.0} for t in TIPOS},

        "Revision_Ingreso": {
            "Fundador":     {"Cuarentena": 1.0},
            "Cautivo":      {"Cuarentena": 1.0},
            "RescateLeve":  {"Evaluacion_Veterinaria_Inicial": 1.0},
            "RescateCritico": {"Evaluacion_Veterinaria_Inicial": 1.0},
        },

        # Cuarentena cumple DOS papeles: cuarentena de INGRESO (Fundador/
        # Cautivo, recién llegados) y cuarentena de REINTEGRACIÓN
        # (RescateLeve/Critico, una vez estabilizados, antes de unirse a la
        # población general o liberarse).
        "Cuarentena": {
            "Fundador":       {"Evaluacion_Veterinaria_Inicial": 1.0},
            "Cautivo":        {"Evaluacion_Veterinaria_Inicial": 1.0},
            "RescateLeve":    {"Preparacion_Suelta": 0.75, "Modulo_Cria": 0.25},
            "RescateCritico": {"Preparacion_Suelta": 0.60, "Modulo_Cria": 0.15, "Cautividad_Permanente": 0.25},
        },

        "Evaluacion_Veterinaria_Inicial": {
            "Fundador":       {"Analitica_Basica": 0.60, "Pruebas_Geneticas_Avanzadas": 0.40},
            "Cautivo":        {"Analitica_Basica": 0.70, "Pruebas_Geneticas_Avanzadas": 0.25, "Preparacion_Suelta": 0.05},
            "RescateLeve":    {"Analitica_Basica": 0.45, "Pruebas_Geneticas_Avanzadas": 0.15,
                               "Consulta_Genetista": 0.15, "Cirugia_Veterinaria": 0.10, "Evaluacion_Preliminar": 0.15},
            "RescateCritico": {"Analitica_Basica": 0.30, "Consulta_Genetista": 0.10,
                               "Cirugia_Veterinaria": 0.30, "UCI_Veterinaria": 0.30},
        },

        "Analitica_Basica": {
            "Fundador":       {"Pruebas_Geneticas_Avanzadas": 0.90, "Evaluacion_Preliminar": 0.10},
            "Cautivo":        {"Evaluacion_Preliminar": 0.75, "Pruebas_Geneticas_Avanzadas": 0.25},
            "RescateLeve":    {"Pruebas_Geneticas_Avanzadas": 0.20, "Evaluacion_Preliminar": 0.80},
            "RescateCritico": {"Pruebas_Geneticas_Avanzadas": 0.10, "Evaluacion_Preliminar": 0.90},
        },

        "Pruebas_Geneticas_Avanzadas": {t: {"Evaluacion_Preliminar": 1.0} for t in TIPOS},

        "Consulta_Genetista": {
            "Fundador":       {"Modulo_Cria": 0.70, "Preparacion_Suelta": 0.30},
            "RescateLeve":    {"Evaluacion_Preliminar": 0.60, "Cirugia_Veterinaria": 0.40},
            "RescateCritico": {"Cirugia_Veterinaria": 0.50, "UCI_Veterinaria": 0.30, "Evaluacion_Preliminar": 0.20},
        },

        "Evaluacion_Preliminar": {
            "Fundador":       {"Modulo_Cria": 0.55, "Consulta_Genetista": 0.30, "Preparacion_Suelta": 0.15},
            "Cautivo":        {"Modulo_Cria": 0.35, "Preparacion_Suelta": 0.45, "Obs_Cautivo": 0.20},
            "RescateLeve":    {"Cuarentena": 0.20, "Obs_RescateLeve": 0.35, "Preparacion_Suelta": 0.15,
                               "Consulta_Genetista": 0.10, "Cautividad_Permanente": 0.15, "Fallece": 0.05},
            "RescateCritico": {"UCI_Veterinaria": 0.30, "Cirugia_Veterinaria": 0.20, "Cuarentena": 0.10,
                               "Obs_RescateCritico": 0.15, "Cautividad_Permanente": 0.15, "Fallece": 0.10},
        },

        "Cirugia_Veterinaria": {
            "RescateLeve":    {"UCI_Veterinaria": 0.15, "Cuarentena": 0.55, "Cautividad_Permanente": 0.20, "Fallece": 0.10},
            "RescateCritico": {"UCI_Veterinaria": 0.45, "Cuarentena": 0.20, "Cautividad_Permanente": 0.20, "Fallece": 0.15},
        },

        "Obs_Cautivo":        {"Cautivo": {"Preparacion_Suelta": 0.65, "Obs_Cautivo": 0.20, "Modulo_Cria": 0.15}},
        "Obs_RescateLeve":    {"RescateLeve": {"Preparacion_Suelta": 0.30, "Obs_RescateLeve": 0.25,
                                               "Cuarentena": 0.25, "Cautividad_Permanente": 0.15, "Fallece": 0.05}},
        "Obs_RescateCritico": {"RescateCritico": {"Obs_RescateCritico": 0.10, "Cuarentena": 0.35,
                                                  "UCI_Veterinaria": 0.25, "Cautividad_Permanente": 0.20, "Fallece": 0.10}},

        "Modulo_Cria": {
            "Fundador":       {"Preparacion_Suelta": 0.35, "Modulo_Cria": 0.35, "UCI_Veterinaria": 0.08,
                               "Cautividad_Permanente": 0.17, "Fallece": 0.05},
            "Cautivo":        {"Preparacion_Suelta": 0.55, "Modulo_Cria": 0.20, "UCI_Veterinaria": 0.10,
                               "Cautividad_Permanente": 0.10, "Fallece": 0.05},
            "RescateLeve":    {"Preparacion_Suelta": 0.40, "Modulo_Cria": 0.25, "UCI_Veterinaria": 0.10,
                               "Cautividad_Permanente": 0.18, "Fallece": 0.07},
            "RescateCritico": {"Preparacion_Suelta": 0.25, "Modulo_Cria": 0.20, "UCI_Veterinaria": 0.15,
                               "Cautividad_Permanente": 0.25, "Fallece": 0.15},
        },

        "UCI_Veterinaria": {
            "Fundador":       {"Modulo_Cria_PostUCI": 0.85, "Cautividad_Permanente": 0.05, "Fallece": 0.10},
            "Cautivo":        {"Modulo_Cria_PostUCI": 0.85, "Cautividad_Permanente": 0.05, "Fallece": 0.10},
            "RescateLeve":    {"Modulo_Cria_PostUCI": 0.65, "Cautividad_Permanente": 0.15, "Fallece": 0.20},
            "RescateCritico": {"Modulo_Cria_PostUCI": 0.45, "Cautividad_Permanente": 0.20, "Fallece": 0.35},
        },

        "Modulo_Cria_PostUCI": {
            "Fundador":       {"Preparacion_Suelta": 0.40, "Recuperacion_PostCritica": 0.60},
            "Cautivo":        {"Preparacion_Suelta": 0.55, "Recuperacion_PostCritica": 0.45},
            "RescateLeve":    {"Preparacion_Suelta": 0.45, "Recuperacion_PostCritica": 0.55},
            "RescateCritico": {"Preparacion_Suelta": 0.35, "Recuperacion_PostCritica": 0.65},
        },

        "Recuperacion_PostCritica": {t: {"Preparacion_Suelta": 0.80, "Cautividad_Permanente": 0.20} for t in TIPOS},

        "Preparacion_Suelta": {t: {"Liberado": 1.0} for t in TIPOS},
    }

    cfg.COSTES_FIJOS = {}   # el protocolo actual no tiene coste fijo adicional
    return cfg


def construir_config_ia_genetica():
    cfg = construir_config_base()
    cfg.nombre = "Escenario A - Genotipado Asistido por IA"

    # 1) Tiempos de servicio: la IA de bioinformática acelera drásticamente el
    #    análisis genético y agiliza la lectura del genetista.
    cfg.TIEMPOS_SERVICIO = copy.deepcopy(cfg.TIEMPOS_SERVICIO)
    for tipo in cfg.TIEMPOS_SERVICIO["Pruebas_Geneticas_Avanzadas"]:
        cfg.TIEMPOS_SERVICIO["Pruebas_Geneticas_Avanzadas"][tipo] *= 0.35   # 3.0d -> ~1.05d
    for tipo in cfg.TIEMPOS_SERVICIO["Consulta_Genetista"]:
        cfg.TIEMPOS_SERVICIO["Consulta_Genetista"][tipo] *= 0.60

    # 2) Derivaciones: mejores emparejamientos -> menos Cautividad_Permanente
    #    por bajo valor genético, más Preparacion_Suelta, para Fundador/Cautivo.
    cfg.PROB_DERIVACION = copy.deepcopy(cfg.PROB_DERIVACION)
    cfg.PROB_DERIVACION["Consulta_Genetista"]["Fundador"] = {
        "Modulo_Cria": 0.80, "Preparacion_Suelta": 0.20,
    }
    cfg.PROB_DERIVACION["Modulo_Cria"]["Fundador"] = {
        "Preparacion_Suelta": 0.40, "Modulo_Cria": 0.35, "UCI_Veterinaria": 0.08,
        "Cautividad_Permanente": 0.12, "Fallece": 0.05,
    }
    cfg.PROB_DERIVACION["Modulo_Cria"]["Cautivo"] = {
        "Preparacion_Suelta": 0.59, "Modulo_Cria": 0.20, "UCI_Veterinaria": 0.10,
        "Cautividad_Permanente": 0.06, "Fallece": 0.05,
    }

    # 3) Coste del escenario
    cfg.COSTES_FIJOS = {"licencia_ia_genomica_anual": 62_000}
    return cfg



def construir_config_rescate_rapido():
    cfg = construir_config_base()
    cfg.nombre = "Escenario B - Equipo de Rescate Rápido"

    # 1) Nuevo recurso: segundo equipo veterinario, dedicado en exclusiva a
    #    los rescates leves.
    cfg.SERVIDORES_CAPACIDAD = copy.deepcopy(cfg.SERVIDORES_CAPACIDAD)
    cfg.SERVIDORES_CAPACIDAD["Equipo_Rescate_Rapido"] = 1

    # 2) Nuevas estaciones (valoración inicial y preliminar del RescateLeve)
    #    atendidas por este equipo dedicado, independiente de "Equipo_Veterinario".
    cfg.POOL_ESTACION = copy.deepcopy(cfg.POOL_ESTACION)
    cfg.POOL_ESTACION["Evaluacion_Veterinaria_Inicial_RL"] = "Equipo_Rescate_Rapido"
    cfg.POOL_ESTACION["Evaluacion_Preliminar_RL"] = "Equipo_Rescate_Rapido"

    cfg.TIEMPOS_SERVICIO = copy.deepcopy(cfg.TIEMPOS_SERVICIO)
    cfg.TIEMPOS_SERVICIO["Evaluacion_Veterinaria_Inicial_RL"] = {"RescateLeve": 0.12}
    cfg.TIEMPOS_SERVICIO["Evaluacion_Preliminar_RL"] = {"RescateLeve": 0.15}

    # 3) El RescateLeve captura el Recinto de Recuperación en su nueva estación de entrada
    cfg.RECINTO_CAPTURA = dict(cfg.RECINTO_CAPTURA)
    cfg.RECINTO_CAPTURA["RescateLeve"] = "Evaluacion_Veterinaria_Inicial_RL"

    # 4) Derivaciones: RescateLeve entra ahora por la vía rápida en lugar de
    #    compartir "Equipo_Veterinario" con el resto de perfiles.
    cfg.PROB_DERIVACION = copy.deepcopy(cfg.PROB_DERIVACION)
    cfg.PROB_DERIVACION["Revision_Ingreso"]["RescateLeve"] = {"Evaluacion_Veterinaria_Inicial_RL": 1.0}
    cfg.PROB_DERIVACION["Evaluacion_Veterinaria_Inicial_RL"] = {
        "RescateLeve": {"Analitica_Basica": 0.45, "Pruebas_Geneticas_Avanzadas": 0.15,
                        "Consulta_Genetista": 0.15, "Cirugia_Veterinaria": 0.10,
                        "Evaluacion_Preliminar_RL": 0.15},
    }
    cfg.PROB_DERIVACION["Evaluacion_Preliminar_RL"] = {
        "RescateLeve": {"Cuarentena": 0.20, "Obs_RescateLeve": 0.35, "Preparacion_Suelta": 0.15,
                        "Consulta_Genetista": 0.10, "Cautividad_Permanente": 0.15, "Fallece": 0.05},
    }
    # Al recibir antes atención veterinaria dedicada, mejora ligeramente la
    # fracción de rescates leves que se estabiliza sin más incidencias.
    cfg.PROB_DERIVACION["Analitica_Basica"]["RescateLeve"] = {
        "Pruebas_Geneticas_Avanzadas": 0.15, "Evaluacion_Preliminar_RL": 0.85,
    }

    # 5) Coste del escenario
    cfg.COSTES_FIJOS = {"personal_rescate_rapido_anual": 38_000}
    return cfg



def construir_config_uci_dinamica():
    cfg = construir_config_base()
    cfg.nombre = "Escenario C - Gestión Dinámica de UCI Veterinaria"
    cfg.UCI_MODE = "pooled"   # único cambio real: pool único con prioridad por criticidad
    # cfg.UCI_TOTAL_CAPACIDAD se mantiene IGUAL que en la línea base (5 plazas):
    # no se añade ninguna plaza nueva, sólo se reorganiza su gestión.
    cfg.COSTES_FIJOS = {"reorganizacion_unica": 4_500}   # coste único, no anual
    return cfg

ESCENARIOS_POR_DEFECTO = {
    "Línea base": construir_config_base,
    "A · IA genómica": construir_config_ia_genetica,
    "B · Rescate rápido": construir_config_rescate_rapido,
    "C · UCI compartida": construir_config_uci_dinamica,
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
