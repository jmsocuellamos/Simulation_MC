# -*- coding: utf-8 -*-
"""Modelo de colas M/M/s con capacidad infinita: teoría (Erlang C), simulación de eventos discretos,
costes, dimensionamiento y comparación de escenarios.

Basado en «D-190. Sistemas de Colas», Javier Morales y Asun Martínez Mayoral, equipo IA4LEGOS (CC BY-SA 4.0)."""
import math
import random

import numpy as np
import pandas as pd

try:
    from scipy import stats as _stats
except Exception:          # pragma: no cover - sin scipy se usa la aproximación normal
    _stats = None

UNIDADES = {"s": ("segundos", "segundo"), "min": ("minutos", "minuto"), "h": ("horas", "hora"),
            "días": ("días", "día"), "sem": ("semanas", "semana")}


def unidad_larga(u):
    return UNIDADES.get(u, (u, u))[0]


def unidad_sing(u):
    return UNIDADES.get(u, (u, u))[1]


# ==============================================================================================
# Teoría: M/M/s con capacidad infinita (fórmulas de Erlang C)
# ==============================================================================================
def teoria(lam, mu, s):
    """Distribución límite y KPIs del sistema M/M/s.

    a = λ/μ (carga ofrecida, Erlangs) y ρ = a/s (carga por servidor). Solo hay régimen estacionario si
    ρ < 1; si no, la cola crece sin límite: L, Lq, W y Wq son infinitos y la distribución límite no existe."""
    lam, mu, s = float(lam), float(mu), int(s)
    a = lam / mu
    rho = a / s
    estable = rho < 1
    suma = sum(a ** n / math.factorial(n) for n in range(s))
    term_s = a ** s / math.factorial(s)
    if not estable:
        return {"lambda": lam, "mu": mu, "s": s, "a": a, "rho": rho, "estable": False, "P0": 0.0, "Pespera": 1.0,
                "L": math.inf, "Lq": math.inf, "W": math.inf, "Wq": math.inf, "Ocupados": float(s), "pn": None}
    P0 = 1.0 / (suma + term_s / (1 - rho))
    Pespera = term_s / (1 - rho) * P0                  # C(s, a) de Erlang
    Lq = Pespera * rho / (1 - rho)
    L = Lq + a
    # La distribución es infinita en teoría: se trunca donde la cola geométrica ya es despreciable
    cola = max(20, math.ceil(math.log(1e-4) / math.log(rho))) if rho > 0 else 20
    n_max = min(400, s + cola)
    pn = np.array([P0 * a ** n / math.factorial(n) if n < s else P0 * term_s * rho ** (n - s)
                   for n in range(n_max + 1)])
    return {"lambda": lam, "mu": mu, "s": s, "a": a, "rho": rho, "estable": True, "P0": P0, "Pespera": Pespera,
            "L": L, "Lq": Lq, "W": L / lam, "Wq": Lq / lam, "Ocupados": a, "pn": pn}


def costes(sis, cq, cs):
    """Costes por unidad de tiempo.
    CTS: servidores ocupados, a c_t = c_q + c_s (el cliente en servicio también consume su tiempo);
    CTSn: servidores desocupados, a c_s; CTQ: espera de los clientes en cola, λ·c_q·Wq; CT = CTS + CTSn + CTQ."""
    ct = cq + cs
    CTS = ct * sis["Ocupados"]
    CTSn = cs * (sis["s"] - sis["Ocupados"])
    CTQ = sis["lambda"] * cq * sis["Wq"] if sis["estable"] else math.inf
    CT = CTS + CTSn + CTQ if sis["estable"] else math.inf
    return {"c_t": ct, "CTS": CTS, "CTSn": CTSn, "CTQ": CTQ, "CT": CT}


def barrido(lam, mu, cq, cs, s_max):
    """Coste total y KPIs para s = 1 … s_max (los valores con ρ ≥ 1 se marcan como inestables)."""
    filas = []
    for s in range(1, int(s_max) + 1):
        t = teoria(lam, mu, s)
        c = costes(t, cq, cs)
        filas.append({"s": s, "ρ": t["rho"], "Estable": t["estable"], "P(espera)": t["Pespera"], "L": t["L"],
                      "Lq": t["Lq"], "W": t["W"], "Wq": t["Wq"], "CTS": c["CTS"], "CTSn": c["CTSn"],
                      "CTQ": c["CTQ"], "CT": c["CT"]})
    df = pd.DataFrame(filas).set_index("s")
    est = df[df["Estable"]]
    s_opt = int(est["CT"].idxmin()) if len(est) else None
    return df, s_opt


# ==============================================================================================
# Simulación de eventos discretos (relojes exponenciales exactos, sin discretizar el tiempo)
# ==============================================================================================
N_DIAGRAMA = 60          # clientes que se muestran en el diagrama de funcionamiento
CALENTAMIENTO = 0.05      # fracción inicial de cada réplica que se descarta (el sistema arranca vacío)


def simular_replica(lam, mu, s, t_max, semilla, n_traza=0, calentamiento=CALENTAMIENTO):
    """Una réplica del proceso de nacimiento-muerte durante `t_max` unidades de tiempo (las del modelo).
    Las estadísticas se acumulan tras descartar el periodo inicial de calentamiento (`calentamiento`·t_max), para
    evitar el sesgo de arrancar con el sistema vacío. Devuelve L, Lq, W, Wq (Little), P(espera) observada, servidores
    ocupados medios, la distribución p_n ponderada por tiempo y, si `n_traza` > 0, los primeros `n_traza` puntos de la
    trayectoria N(t) (desde el instante 0)."""
    rng = random.Random(semilla)
    expo = rng.expovariate
    t0 = calentamiento * t_max
    t = 0.0
    n = 0
    llegadas = 0
    esperan = 0
    area_n = area_q = 0.0
    t_estado = {}
    traza = [(0.0, 0)] if n_traza else None
    prox_lleg = expo(lam)
    prox_sal = math.inf
    while True:
        t_sig = prox_lleg if prox_lleg <= prox_sal else prox_sal
        fin = t_sig >= t_max
        if fin:
            t_sig = t_max
        if t_sig > t0:                       # solo cuenta el tramo posterior al calentamiento
            dt = t_sig - max(t, t0)
            area_n += n * dt
            if n > s:
                area_q += (n - s) * dt
            t_estado[n] = t_estado.get(n, 0.0) + dt
        t = t_sig
        if fin:
            break
        if prox_lleg <= prox_sal:
            if t >= t0:
                llegadas += 1
                if n >= s:
                    esperan += 1
            n += 1
            prox_sal = t + expo(min(n, s) * mu)
            prox_lleg = t + expo(lam)
        else:
            n -= 1
            prox_sal = t + expo(min(n, s) * mu) if n > 0 else math.inf
        if traza is not None and len(traza) < n_traza:
            traza.append((t, n))
    T = t_max - t0
    L = area_n / T
    Lq = area_q / T
    n_max = max(t_estado) if t_estado else 0
    pn = np.array([t_estado.get(i, 0.0) / T for i in range(n_max + 1)])
    return {"L": L, "Lq": Lq, "W": L / lam, "Wq": L / lam - 1 / mu, "Pespera": esperan / llegadas if llegadas else 0.0,
            "Llegadas": llegadas, "Ocupados": L - Lq, "T": T, "pn": pn, "traza": traza}


def simular_clientes(lam, mu, s, n_clientes, semilla):
    """Simulación cliente a cliente (cola FIFO, s servidores) de los primeros `n_clientes` clientes, para el diagrama
    de funcionamiento: llegada, inicio y fin de servicio y servidor de cada cliente."""
    rng = random.Random(semilla)
    libre = [0.0] * s                  # instante en que queda libre cada servidor
    clientes = []
    t = rng.expovariate(lam)
    for i in range(1, int(n_clientes) + 1):
        k = min(range(s), key=lambda j: libre[j])
        inicio = max(t, libre[k])
        fin = inicio + rng.expovariate(mu)
        libre[k] = fin
        clientes.append({"Cliente": i, "Llegada": t, "Inicio": inicio, "Fin": fin, "Servidor": k + 1,
                         "Espera": inicio - t, "Servicio": fin - inicio})
        t += rng.expovariate(lam)
    return pd.DataFrame(clientes, columns=["Cliente", "Llegada", "Inicio", "Fin", "Servidor", "Espera", "Servicio"])


def _tq(n, alpha=0.05):
    if n < 2:
        return float("nan")
    if _stats is not None:
        return float(_stats.t.ppf(1 - alpha / 2, n - 1))
    return 1.96


def montecarlo(lam, mu, s, t_max=3000.0, replicas=8, semilla=1000, progreso=None):
    """`replicas` réplicas independientes (réplica i -> semilla + i) y combinación de resultados."""
    filas, pn_list = [], []
    traza = None
    for r in range(replicas):
        res = simular_replica(lam, mu, s, t_max, semilla + r, n_traza=400 if r == 0 else 0)
        filas.append({"Réplica": r + 1, "Semilla": semilla + r, "L": res["L"], "Lq": res["Lq"], "W": res["W"],
                      "Wq": res["Wq"], "P(espera)": res["Pespera"], "Servidores ocupados": res["Ocupados"],
                      "Llegadas": res["Llegadas"], "Tiempo simulado": res["T"]})
        pn_list.append((res["pn"], res["T"]))
        if r == 0:
            traza = res["traza"]
        if progreso:
            progreso(r + 1, replicas)
    rep = pd.DataFrame(filas).set_index("Réplica")
    n_max = max(len(p) for p, _ in pn_list)
    acum = np.zeros(n_max)
    for p, T in pn_list:
        acum[:len(p)] += p * T
    pn = acum / sum(T for _, T in pn_list)
    return {"replicas": rep, "pn": pn, "traza": traza}


def resumen_mc(rep, sis):
    """Media Monte Carlo, IC95% (variabilidad entre réplicas) y valor teórico de cada KPI."""
    n = len(rep)
    tq = _tq(n)
    filas = []
    for k, teo in [("L", sis["L"]), ("Lq", sis["Lq"]), ("W", sis["W"]), ("Wq", sis["Wq"]),
                   ("P(espera)", sis["Pespera"]), ("Servidores ocupados", sis["Ocupados"])]:
        v = rep[k].astype(float)
        m = v.mean()
        se = v.std(ddof=1) / math.sqrt(n) if n > 1 else 0.0
        lo, hi = m - tq * se, m + tq * se
        filas.append({"KPI": k, "Teórico (fórmula)": teo, "Simulado · media": m, "Simulado · IC95 inf": lo,
                      "Simulado · IC95 sup": hi,
                      "Error relativo": (m - teo) / teo if (sis["estable"] and teo not in (0, math.inf)) else None,
                      "Teórico dentro del IC95": (lo <= teo <= hi) if sis["estable"] else None})
    return pd.DataFrame(filas).set_index("KPI")


def p_mayor(sis, k):
    """P(N > k) teórica exacta (incluye toda la cola geométrica, sin truncar)."""
    if not sis["estable"]:
        return 1.0
    s, a, rho, P0 = sis["s"], sis["a"], sis["rho"], sis["P0"]
    cola = P0 * a ** s / math.factorial(s) / (1 - rho)           # P(N ≥ s)
    if k >= s - 1:
        return cola * rho ** (k + 1 - s)
    return sum(P0 * a ** n / math.factorial(n) for n in range(k + 1, s)) + cola


def tabla_probabilidades(sis, pn_mc=None, umbral=0.9999, n_min=None):
    """Probabilidad de n clientes en el sistema: p_n = P(N = n), acumulada P(N ≤ n) y cola P(N > n),
    teóricas y (si se indican) simuladas. Se trunca donde la probabilidad acumulada teórica supera `umbral`."""
    s = sis["s"]
    if sis["estable"]:
        teo = np.asarray(sis["pn"])
        corte = int(np.searchsorted(np.cumsum(teo), umbral)) + 1
    else:
        teo = None
        corte = int(np.searchsorted(np.cumsum(pn_mc), umbral)) + 1 if pn_mc is not None else 0
    corte = max(corte, s + 1, n_min or 0)
    idx = pd.Index(range(corte), name="n")
    df = pd.DataFrame(index=idx)
    df["Región"] = ["servidores (n < s)" if n < s else "cola (n ≥ s)" for n in idx]
    df["Clientes en cola"] = [max(0, n - s) for n in idx]
    df["Servidores ocupados"] = [min(n, s) for n in idx]
    if teo is not None:
        v = np.zeros(corte); m = min(corte, len(teo)); v[:m] = teo[:m]
        df["Teórica · P(N = n)"] = v
        df["Teórica · P(N > n)"] = [p_mayor(sis, n) for n in idx]
        df["Teórica · P(N ≤ n)"] = 1 - df["Teórica · P(N > n)"]
        df = df[[c for c in df.columns if not c.startswith("Teórica")] +
                ["Teórica · P(N = n)", "Teórica · P(N ≤ n)", "Teórica · P(N > n)"]]
    if pn_mc is not None:
        w = np.zeros(corte); m = min(corte, len(pn_mc)); w[:m] = pn_mc[:m]
        df["Simulada · P(N = n)"] = w
        acum = np.cumsum(pn_mc)
        df["Simulada · P(N ≤ n)"] = [acum[min(n, len(acum) - 1)] for n in idx]
        df["Simulada · P(N > n)"] = 1 - df["Simulada · P(N ≤ n)"]
    return df


def prob_evento(sis, pn_mc, tipo, k):
    """P(N = k), P(N ≤ k), P(N ≥ k) o P(N > k), teórica y simulada."""
    def calc(p):
        if p is None:
            return None
        p = np.asarray(p)
        tot = lambda a, b: float(p[a:b].sum())
        if tipo == "=":
            return float(p[k]) if k < len(p) else 0.0
        if tipo == "≤":
            return min(1.0, tot(0, k + 1))
        if tipo == "≥":
            return max(0.0, 1 - tot(0, k))
        return max(0.0, 1 - tot(0, k + 1))
    teo = None
    if sis["estable"]:          # exacta: la cola geométrica se suma en forma cerrada
        teo = {"=": lambda: (p_mayor(sis, k - 1) if k > 0 else 1.0) - p_mayor(sis, k),
               "≤": lambda: 1 - p_mayor(sis, k),
               "≥": lambda: p_mayor(sis, k - 1) if k > 0 else 1.0,
               ">": lambda: p_mayor(sis, k)}[tipo]()
    return teo, calc(pn_mc)


# ==============================================================================================
# Objetivos de servicio (equivalentes a los SLAs de las aplicaciones de los casos de estudio)
# ==============================================================================================
OBJETIVOS_DEF = {"RHO_MAX": 0.85, "WQ_MAX": 0.25, "PESPERA_MAX": 0.50, "LQ_MAX": 2.0}


def evaluar_objetivos(sis, obj):
    """Cumplimiento de los objetivos de servicio (sobre los valores teóricos). Un objetivo vacío (None) no se evalúa."""
    filas = []
    def fila(nombre, valor, umbral, cumple, fmt):
        filas.append({"Objetivo": nombre, "Valor": valor, "Umbral": fmt(umbral), "Cumple": "Sí" if cumple else "No"})
    fila("Sistema estable (ρ < 100 %)", sis["rho"], 1.0, sis["estable"], lambda u: "< 100 %")
    if obj.get("RHO_MAX") is not None:
        fila("Ocupación por servidor ρ", sis["rho"], obj["RHO_MAX"], sis["rho"] <= obj["RHO_MAX"], lambda u: f"≤ {pct(u, 0)}")
    if obj.get("WQ_MAX") is not None:
        fila("Tiempo medio en cola Wq", sis["Wq"], obj["WQ_MAX"], sis["Wq"] <= obj["WQ_MAX"], lambda u: f"≤ {num(u, 3)}")
    if obj.get("PESPERA_MAX") is not None:
        fila("Probabilidad de esperar P(espera)", sis["Pespera"], obj["PESPERA_MAX"], sis["Pespera"] <= obj["PESPERA_MAX"],
             lambda u: f"≤ {pct(u, 0)}")
    if obj.get("LQ_MAX") is not None:
        fila("Clientes medios en cola Lq", sis["Lq"], obj["LQ_MAX"], sis["Lq"] <= obj["LQ_MAX"], lambda u: f"≤ {num(u, 2)}")
    return pd.DataFrame(filas).set_index("Objetivo")


# ==============================================================================================
# Análisis completo de un escenario y comparación
# ==============================================================================================
def analizar_escenario(esc, comunes, ajustes, progreso=None):
    lam, mu, s = float(esc["lambda"]), float(esc["mu"]), int(esc["s"])
    cq = float(comunes["cq"])
    cs = float(esc["cs"]) if esc.get("cs") not in (None, "") else float(comunes["cs"])
    sis = teoria(lam, mu, s)
    cst = costes(sis, cq, cs)
    mc = montecarlo(lam, mu, s, float(ajustes["duracion"]), int(ajustes["replicas"]), int(ajustes["semilla"]), progreso)
    sw, s_opt = barrido(lam, mu, cq, cs, max(int(comunes["s_max"]), s))
    clientes = simular_clientes(lam, mu, s, N_DIAGRAMA, int(ajustes["semilla"]))
    return {"esc": dict(esc, cs=cs, cq=cq), "teoria": sis, "costes": cst, "mc": mc,
            "resumen_mc": resumen_mc(mc["replicas"], sis), "barrido": sw, "s_opt": s_opt,
            "objetivos": evaluar_objetivos(sis, comunes["objetivos"]), "clientes": clientes}


def tabla_comparativa(res):
    """KPIs de todos los escenarios (en columnas): parámetros, resultados TEÓRICOS (fórmulas), resultados SIMULADOS
    (media de las réplicas Monte Carlo) y costes (teóricos)."""
    cols = {}
    for nombre, r in res.items():
        t, c, m = r["teoria"], r["costes"], r["mc"]["replicas"]
        cols[nombre] = {
            "Parámetro · λ": t["lambda"], "Parámetro · μ": t["mu"], "Parámetro · s": t["s"],
            "Parámetro · c_s (€/servidor)": r["esc"]["cs"],
            "Teórico · a (Erlangs)": t["a"], "Teórico · ρ": t["rho"], "Teórico · estable": "Sí" if t["estable"] else "No",
            "Teórico · P0": t["P0"], "Teórico · P(espera)": t["Pespera"], "Teórico · L": t["L"], "Teórico · Lq": t["Lq"],
            "Teórico · W": t["W"], "Teórico · Wq": t["Wq"], "Teórico · servidores ocupados": t["Ocupados"],
            "Simulado · P(espera)": m["P(espera)"].mean(), "Simulado · L": m["L"].mean(), "Simulado · Lq": m["Lq"].mean(),
            "Simulado · W": m["W"].mean(), "Simulado · Wq": m["Wq"].mean(),
            "Simulado · servidores ocupados": m["Servidores ocupados"].mean(),
            "Coste teórico · CTS": c["CTS"], "Coste teórico · CTSn": c["CTSn"], "Coste teórico · CTQ": c["CTQ"],
            "Coste teórico · CT": c["CT"],
            "Dimensionamiento · s óptimo (mín. CT)": r["s_opt"] if r["s_opt"] is not None else "—",
            "Objetivos cumplidos (teóricos)": f"{(r['objetivos']['Cumple'] == 'Sí').sum()} de {len(r['objetivos'])}",
        }
    return pd.DataFrame(cols)


def tabla_objetivos(res):
    return pd.DataFrame({n: r["objetivos"]["Cumple"] for n, r in res.items()})


# ==============================================================================================
# Formato y textos automáticos
# ==============================================================================================
def num(x, d=2):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    s = f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, d=1):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    return num(100 * x, d) + " %"


def eur(x, d=2):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    return num(x, d) + " €"


def fmt_kpi(k, c, v):
    """Formato (fila, columna, valor) según la fila de la tabla comparativa."""
    if isinstance(v, str) or v is None:
        return "—" if v is None else v
    base = k.split(" · ")[-1]
    if base in ("ρ", "P0", "P(espera)") or k.startswith("Error"):
        return pct(v, 1)
    if base in ("CTS", "CTSn", "CTQ", "CT") or "€" in k:
        return eur(v)
    if base == "s" or base.startswith("s óptimo"):
        return num(v, 0)
    return num(v, 3)


def texto_escenario(nombre, r, unidad):
    t, c = r["teoria"], r["costes"]
    u = unidad
    if not t["estable"]:
        return (f"{nombre}: con λ = {num(t['lambda'], 3)} y s·μ = {num(t['s'] * t['mu'], 3)} clientes/{unidad_sing(u)}, "
                f"la carga por servidor es ρ = {pct(t['rho'])} ≥ 100 %: el sistema es inestable, la cola crece sin límite "
                f"y no existe régimen estacionario. Hacen falta al menos {math.floor(t['a']) + 1} servidores.")
    ok = r["resumen_mc"]["Teórico dentro del IC95"].iloc[:4]
    val = ("la simulación reproduce los valores teóricos (todos dentro del IC95 %)" if ok.all() else
           f"la simulación reproduce los valores teóricos salvo en {', '.join(ok[~ok].index)} (fuera del IC95 %, "
           "lo que puede deberse al azar o a una duración de réplica insuficiente)")
    txt = (f"{nombre}: ρ = {pct(t['rho'])} por servidor, P(espera) = {pct(t['Pespera'])}, L = {num(t['L'], 3)} y "
           f"Lq = {num(t['Lq'], 3)} clientes, W = {num(t['W'], 4)} y Wq = {num(t['Wq'], 4)} {unidad_larga(u)} (valores teóricos); {val}. "
           f"Coste total CT = {eur(c['CT'])}/{unidad_sing(u)} (CTS {eur(c['CTS'])}, CTSn {eur(c['CTSn'])}, "
           f"CTQ {eur(c['CTQ'])}).")
    if r["s_opt"] is not None:
        txt += (" Su número de servidores ya es el de menor coste total." if r["s_opt"] == t["s"] else
                f" El número de servidores de menor coste total sería s = {r['s_opt']} "
                f"(CT = {eur(r['barrido'].loc[r['s_opt'], 'CT'])}/{unidad_sing(u)}).")
    n_ok = (r["objetivos"]["Cumple"] == "Sí").sum()
    txt += f" Cumple {n_ok} de los {len(r['objetivos'])} objetivos de servicio."
    return txt


def texto_recomendacion(res, unidad):
    filas = []
    for n, r in res.items():
        filas.append((n, (r["objetivos"]["Cumple"] == "Sí").sum(), r["costes"]["CT"], r["teoria"]["estable"]))
    est = [f for f in filas if f[3]]
    if not est:
        return "Ningún escenario es estable: hay que aumentar el número de servidores o la tasa de servicio."
    max_ok = max(f[1] for f in est)
    cand = sorted([f for f in est if f[1] == max_ok], key=lambda f: f[2])
    mejor = cand[0]
    barato = min(est, key=lambda f: f[2])
    txt = (f"Se recomienda «{mejor[0]}»: es el escenario que cumple más objetivos de servicio ({mejor[1]} de "
           f"{len(res[mejor[0]]['objetivos'])}) y, entre ellos, el de menor coste total ({eur(mejor[2])}/{unidad_sing(unidad)}).")
    if barato[0] != mejor[0]:
        txt += (f" El escenario más barato es «{barato[0]}» ({eur(barato[2])}/{unidad_sing(unidad)}), pero cumple menos "
                f"objetivos ({barato[1]}).")
    inest = [f[0] for f in filas if not f[3]]
    if inest:
        txt += f" Escenarios inestables (descartados): {', '.join(inest)}."
    return txt
