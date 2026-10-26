# -*- coding: utf-8 -*-
"""Modelo de colas M/M/s/K (capacidad finita K): teoría, simulación de eventos discretos con bloqueo,
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
# Teoría: M/M/s/K (capacidad finita K = clientes en servicio + en cola)
# ==============================================================================================
def teoria(lam, mu, s, K):
    """Distribución límite y KPIs del sistema M/M/s/K (proceso de nacimiento-muerte truncado en K).

    a = λ/μ (carga ofrecida, Erlangs), ρ = a/s (carga nominal por servidor). Con capacidad finita SIEMPRE existe
    régimen estacionario: si ρ ≥ 1 la cola no crece sin límite, sino que el sistema se llena y rechaza clientes.
    Las llegadas que encuentran N = K se pierden: λ̄ = λ(1 − p_K) es la tasa efectiva de entrada."""
    lam, mu, s, K = float(lam), float(mu), int(s), int(K)
    K = max(K, s)
    a = lam / mu
    rho = a / s
    suma = sum(a ** n / math.factorial(n) for n in range(s))
    term_s = a ** s / math.factorial(s)
    geo = (K - s + 1) if abs(rho - 1) < 1e-9 else (1 - rho ** (K - s + 1)) / (1 - rho)
    P0 = 1.0 / (suma + term_s * geo)
    pn = np.array([P0 * a ** n / math.factorial(n) if n < s else P0 * term_s * rho ** (n - s)
                   for n in range(K + 1)])
    L = float(np.dot(np.arange(K + 1), pn))
    Pbloq = float(pn[K])
    lam_bar = lam * (1 - Pbloq)
    Lq = max(0.0, L - lam_bar / mu)
    W = L / lam_bar
    Wq = max(0.0, W - 1 / mu)
    # Probabilidad de que un cliente ACEPTADO tenga que esperar (PASTA: ve N = n con probabilidad p_n, n < K)
    Pespera = float(pn[s:K].sum() / (1 - Pbloq)) if K > s else 0.0
    return {"lambda": lam, "mu": mu, "s": s, "K": K, "a": a, "rho": rho, "rho_eff": lam_bar / (s * mu),
            "estable": True, "estado": "Subcrítico (ρ < 1)" if rho < 1 else "Saturado (ρ ≥ 1, lo contiene el bloqueo)",
            "P0": P0, "Pbloqueo": Pbloq, "lambda_bar": lam_bar, "Pespera": Pespera,
            "L": L, "Lq": Lq, "W": W, "Wq": Wq, "Ocupados": L - Lq, "pn": pn}


def costes(sis, cq, cs):
    """Costes por unidad de tiempo (como en la clase M_M_s_K del cuaderno).
    CTS: servidores ocupados (L − Lq), a c_t = c_q + c_s; CTSn: servidores desocupados, a c_s;
    CTQ: espera en cola, λ·c_q·Wq (con λ nominal); CT = CTS + CTSn + CTQ."""
    ct = cq + cs
    CTS = ct * sis["Ocupados"]
    CTSn = cs * (sis["s"] - sis["Ocupados"])
    CTQ = sis["lambda"] * cq * sis["Wq"]
    return {"c_t": ct, "CTS": CTS, "CTSn": CTSn, "CTQ": CTQ, "CT": CTS + CTSn + CTQ}


def barrido(lam, mu, cq, cs, s_max, K):
    """Coste total y KPIs para s = 1 … min(s_max, K), manteniendo la capacidad K."""
    filas = []
    for s in range(1, min(int(s_max), int(K)) + 1):
        t = teoria(lam, mu, s, K)
        c = costes(t, cq, cs)
        filas.append({"s": s, "ρ": t["rho"], "ρ̄": t["rho_eff"], "P(bloqueo)": t["Pbloqueo"], "λ̄": t["lambda_bar"],
                      "L": t["L"], "Lq": t["Lq"], "W": t["W"], "Wq": t["Wq"], "CTS": c["CTS"], "CTSn": c["CTSn"],
                      "CTQ": c["CTQ"], "CT": c["CT"]})
    df = pd.DataFrame(filas).set_index("s")
    s_opt = int(df["CT"].idxmin()) if len(df) else None
    return df, s_opt


# ==============================================================================================
# Simulación de eventos discretos (relojes exponenciales exactos, sin discretizar el tiempo)
# ==============================================================================================
N_DIAGRAMA = 60          # clientes (llegadas) que se muestran en el diagrama de funcionamiento
CALENTAMIENTO = 0.05      # fracción inicial de cada réplica que se descarta (el sistema arranca vacío)


def simular_replica(lam, mu, s, K, t_max, semilla, n_traza=0, calentamiento=CALENTAMIENTO):
    """Una réplica del proceso M/M/s/K durante `t_max` unidades de tiempo (las del modelo). Una llegada que encuentra
    N = K clientes se rechaza (bloqueo). Las estadísticas se acumulan tras descartar el calentamiento inicial.
    Devuelve L, Lq (áreas bajo N(t) y la cola), λ̄ observada, W = L/λ̄ y Wq = Lq/λ̄ (Little), P(bloqueo) y P(espera)
    observadas, servidores ocupados medios, la distribución p_n ponderada por tiempo y la traza N(t)."""
    rng = random.Random(semilla)
    expo = rng.expovariate
    K = max(int(K), int(s))
    t0 = calentamiento * t_max
    t = 0.0
    n = 0
    llegadas = rechazos = esperan = 0
    area_n = area_q = 0.0
    t_estado = np.zeros(K + 1)
    traza = [(0.0, 0)] if n_traza else None
    prox_lleg = expo(lam)
    prox_sal = math.inf
    while True:
        t_sig = prox_lleg if prox_lleg <= prox_sal else prox_sal
        fin = t_sig >= t_max
        if fin:
            t_sig = t_max
        if t_sig > t0:
            dt = t_sig - max(t, t0)
            area_n += n * dt
            if n > s:
                area_q += (n - s) * dt
            t_estado[n] += dt
        t = t_sig
        if fin:
            break
        if prox_lleg <= prox_sal:
            medir = t >= t0
            if medir:
                llegadas += 1
            if n >= K:                       # sistema lleno: el cliente se pierde
                if medir:
                    rechazos += 1
            else:
                if medir and n >= s:
                    esperan += 1
                n += 1
                prox_sal = t + expo(min(n, s) * mu)      # reloj exponencial: se puede volver a muestrear
            prox_lleg = t + expo(lam)
        else:
            n -= 1
            prox_sal = t + expo(min(n, s) * mu) if n > 0 else math.inf
        if traza is not None and len(traza) < n_traza:
            traza.append((t, n))
    T = t_max - t0
    L, Lq = area_n / T, area_q / T
    aceptadas = llegadas - rechazos
    lam_bar = aceptadas / T
    W = L / lam_bar if lam_bar > 0 else float("nan")
    Wq = Lq / lam_bar if lam_bar > 0 else float("nan")
    return {"L": L, "Lq": Lq, "W": W, "Wq": Wq, "Pbloqueo": rechazos / llegadas if llegadas else 0.0,
            "Pespera": esperan / aceptadas if aceptadas else 0.0, "lambda_bar": lam_bar, "Llegadas": llegadas,
            "Rechazos": rechazos, "Ocupados": L - Lq, "T": T, "pn": t_estado / T, "traza": traza}


def simular_clientes(lam, mu, s, K, n_clientes, semilla):
    """Simulación cliente a cliente (cola FIFO, s servidores, capacidad K) de las primeras `n_clientes` llegadas,
    para el diagrama de funcionamiento. Una llegada que encuentra K clientes en el sistema se rechaza."""
    rng = random.Random(semilla)
    K = max(int(K), int(s))
    libre = [0.0] * s                  # instante en que queda libre cada servidor
    salidas = []                       # instantes de salida de los clientes aceptados
    clientes = []
    t = rng.expovariate(lam)
    for i in range(1, int(n_clientes) + 1):
        en_sistema = sum(1 for f in salidas if f > t)
        if en_sistema >= K:
            clientes.append({"Cliente": i, "Llegada": t, "Inicio": float("nan"), "Fin": float("nan"), "Servidor": 0,
                             "Espera": float("nan"), "Servicio": float("nan"), "Rechazado": True})
        else:
            k = min(range(s), key=lambda j: libre[j])
            inicio = max(t, libre[k])
            fin = inicio + rng.expovariate(mu)
            libre[k] = fin
            salidas.append(fin)
            clientes.append({"Cliente": i, "Llegada": t, "Inicio": inicio, "Fin": fin, "Servidor": k + 1,
                             "Espera": inicio - t, "Servicio": fin - inicio, "Rechazado": False})
        t += rng.expovariate(lam)
    return pd.DataFrame(clientes, columns=["Cliente", "Llegada", "Inicio", "Fin", "Servidor", "Espera", "Servicio",
                                           "Rechazado"])


def _tq(n, alpha=0.05):
    if n < 2:
        return float("nan")
    if _stats is not None:
        return float(_stats.t.ppf(1 - alpha / 2, n - 1))
    return 1.96


def montecarlo(lam, mu, s, K, t_max=5000.0, replicas=8, semilla=1000, progreso=None):
    """`replicas` réplicas independientes (réplica i -> semilla + i) y combinación de resultados."""
    filas, pn_list = [], []
    traza = None
    for r in range(replicas):
        res = simular_replica(lam, mu, s, K, t_max, semilla + r, n_traza=400 if r == 0 else 0)
        filas.append({"Réplica": r + 1, "Semilla": semilla + r, "L": res["L"], "Lq": res["Lq"], "W": res["W"],
                      "Wq": res["Wq"], "P(bloqueo)": res["Pbloqueo"], "P(espera)": res["Pespera"],
                      "λ̄": res["lambda_bar"], "Servidores ocupados": res["Ocupados"], "Llegadas": res["Llegadas"],
                      "Rechazos": res["Rechazos"], "Tiempo simulado": res["T"]})
        pn_list.append((res["pn"], res["T"]))
        if r == 0:
            traza = res["traza"]
        if progreso:
            progreso(r + 1, replicas)
    rep = pd.DataFrame(filas).set_index("Réplica")
    pn = sum(p * T for p, T in pn_list) / sum(T for _, T in pn_list)
    return {"replicas": rep, "pn": pn, "traza": traza}


KPIS_MC = ["L", "Lq", "W", "Wq", "P(bloqueo)", "λ̄", "P(espera)", "Servidores ocupados"]


def resumen_mc(rep, sis):
    """Media Monte Carlo, IC95% (variabilidad entre réplicas) y valor teórico de cada KPI."""
    n = len(rep)
    tq = _tq(n)
    teo_d = {"L": sis["L"], "Lq": sis["Lq"], "W": sis["W"], "Wq": sis["Wq"], "P(bloqueo)": sis["Pbloqueo"],
             "λ̄": sis["lambda_bar"], "P(espera)": sis["Pespera"], "Servidores ocupados": sis["Ocupados"]}
    filas = []
    for k in KPIS_MC:
        teo = teo_d[k]
        v = rep[k].astype(float)
        m = v.mean()
        se = v.std(ddof=1) / math.sqrt(n) if n > 1 else 0.0
        lo, hi = m - tq * se, m + tq * se
        filas.append({"KPI": k, "Teórico (fórmula)": teo, "Simulado · media": m, "Simulado · IC95 inf": lo,
                      "Simulado · IC95 sup": hi,
                      "Error relativo": (m - teo) / teo if teo > 1e-12 else None,
                      "Teórico dentro del IC95": bool(lo - 1e-12 <= teo <= hi + 1e-12)})
    return pd.DataFrame(filas).set_index("KPI")


def p_mayor(sis, k):
    """P(N > k) teórica exacta (distribución finita n = 0 … K)."""
    pn = np.asarray(sis["pn"])
    if k < 0:
        return 1.0
    return float(pn[k + 1:].sum()) if k + 1 < len(pn) else 0.0


def region(n, s, K):
    if n == K:
        return "lleno · bloqueo (n = K)"
    return "servidores (n < s)" if n < s else "cola (s ≤ n < K)"


def tabla_probabilidades(sis, pn_mc=None, umbral=None, n_min=None):
    """Probabilidad de n clientes en el sistema, n = 0 … K: p_n = P(N = n), acumulada P(N ≤ n) y cola P(N > n),
    teóricas y (si se indican) simuladas."""
    s, K = sis["s"], sis["K"]
    idx = pd.Index(range(K + 1), name="n")
    df = pd.DataFrame(index=idx)
    df["Región"] = [region(n, s, K) for n in idx]
    df["Clientes en cola"] = [max(0, n - s) for n in idx]
    df["Servidores ocupados"] = [min(n, s) for n in idx]
    teo = np.asarray(sis["pn"])
    df["Teórica · P(N = n)"] = teo
    df["Teórica · P(N ≤ n)"] = np.minimum(1.0, np.cumsum(teo))
    df["Teórica · P(N > n)"] = [p_mayor(sis, n) for n in idx]
    if pn_mc is not None:
        w = np.zeros(K + 1); m = min(K + 1, len(pn_mc)); w[:m] = pn_mc[:m]
        df["Simulada · P(N = n)"] = w
        df["Simulada · P(N ≤ n)"] = np.minimum(1.0, np.cumsum(w))
        df["Simulada · P(N > n)"] = np.maximum(0.0, 1 - df["Simulada · P(N ≤ n)"])
    return df


def prob_evento(sis, pn_mc, tipo, k):
    """P(N = k), P(N ≤ k), P(N ≥ k) o P(N > k), teórica y simulada (sumas finitas exactas)."""
    def calc(p):
        if p is None:
            return None
        p = np.asarray(p)
        tot = lambda a, b: float(p[max(a, 0):max(b, 0)].sum())
        if tipo == "=":
            return float(p[k]) if 0 <= k < len(p) else 0.0
        if tipo == "≤":
            return min(1.0, tot(0, k + 1))
        if tipo == "≥":
            return max(0.0, min(1.0, tot(k, len(p))))
        return max(0.0, min(1.0, tot(k + 1, len(p))))
    return calc(sis["pn"]), calc(pn_mc)


# ==============================================================================================
# Objetivos de servicio (equivalentes a los SLAs de las aplicaciones de los casos de estudio)
# ==============================================================================================
OBJETIVOS_DEF = {"RHO_MAX": 0.85, "WQ_MAX": 0.25, "PBLOQ_MAX": 0.05, "LQ_MAX": 1.0}


def evaluar_objetivos(sis, obj):
    """Cumplimiento de los objetivos de servicio (sobre los valores teóricos). Un objetivo vacío (None) no se evalúa."""
    filas = []
    def fila(nombre, valor, umbral, cumple, fmt):
        filas.append({"Objetivo": nombre, "Valor": valor, "Umbral": fmt(umbral), "Cumple": "Sí" if cumple else "No"})
    if obj.get("RHO_MAX") is not None:
        fila("Ocupación efectiva por servidor ρ̄", sis["rho_eff"], obj["RHO_MAX"], sis["rho_eff"] <= obj["RHO_MAX"],
             lambda u: f"≤ {pct(u, 0)}")
    if obj.get("WQ_MAX") is not None:
        fila("Tiempo medio en cola Wq", sis["Wq"], obj["WQ_MAX"], sis["Wq"] <= obj["WQ_MAX"], lambda u: f"≤ {num(u, 3)}")
    if obj.get("PBLOQ_MAX") is not None:
        fila("Probabilidad de bloqueo P(bloqueo)", sis["Pbloqueo"], obj["PBLOQ_MAX"], sis["Pbloqueo"] <= obj["PBLOQ_MAX"],
             lambda u: f"≤ {pct(u, 1)}")
    if obj.get("LQ_MAX") is not None:
        fila("Clientes medios en cola Lq", sis["Lq"], obj["LQ_MAX"], sis["Lq"] <= obj["LQ_MAX"], lambda u: f"≤ {num(u, 2)}")
    return pd.DataFrame(filas, columns=["Objetivo", "Valor", "Umbral", "Cumple"]).set_index("Objetivo")


# ==============================================================================================
# Análisis completo de un escenario y comparación
# ==============================================================================================
def analizar_escenario(esc, comunes, ajustes, progreso=None):
    lam, mu, s = float(esc["lambda"]), float(esc["mu"]), int(esc["s"])
    K = max(int(esc["K"]), s)
    cq = float(comunes["cq"])
    cs = float(esc["cs"]) if esc.get("cs") not in (None, "") else float(comunes["cs"])
    sis = teoria(lam, mu, s, K)
    cst = costes(sis, cq, cs)
    mc = montecarlo(lam, mu, s, K, float(ajustes["duracion"]), int(ajustes["replicas"]), int(ajustes["semilla"]),
                    progreso)
    sw, s_opt = barrido(lam, mu, cq, cs, max(int(comunes["s_max"]), s), K)
    clientes = simular_clientes(lam, mu, s, K, N_DIAGRAMA, int(ajustes["semilla"]))
    return {"esc": dict(esc, cs=cs, cq=cq, K=K), "teoria": sis, "costes": cst, "mc": mc,
            "resumen_mc": resumen_mc(mc["replicas"], sis), "barrido": sw, "s_opt": s_opt,
            "objetivos": evaluar_objetivos(sis, comunes["objetivos"]), "clientes": clientes}


def tabla_comparativa(res):
    """KPIs de todos los escenarios (en columnas): parámetros, resultados TEÓRICOS (fórmulas), resultados SIMULADOS
    (media de las réplicas Monte Carlo) y costes (teóricos)."""
    cols = {}
    for nombre, r in res.items():
        t, c, m = r["teoria"], r["costes"], r["mc"]["replicas"]
        cols[nombre] = {
            "Parámetro · λ": t["lambda"], "Parámetro · μ": t["mu"], "Parámetro · s": t["s"], "Parámetro · K": t["K"],
            "Parámetro · c_s (€/servidor)": r["esc"]["cs"],
            "Teórico · a (Erlangs)": t["a"], "Teórico · ρ (nominal)": t["rho"], "Teórico · ρ̄ (efectiva)": t["rho_eff"],
            "Teórico · estado": t["estado"].split(" (")[0],
            "Teórico · P0": t["P0"], "Teórico · P(bloqueo)": t["Pbloqueo"], "Teórico · λ̄": t["lambda_bar"],
            "Teórico · P(espera)": t["Pespera"], "Teórico · L": t["L"], "Teórico · Lq": t["Lq"],
            "Teórico · W": t["W"], "Teórico · Wq": t["Wq"], "Teórico · servidores ocupados": t["Ocupados"],
            "Simulado · P(bloqueo)": m["P(bloqueo)"].mean(), "Simulado · λ̄": m["λ̄"].mean(),
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
    if base.startswith(("ρ", "P0", "P(espera)", "P(bloqueo)")) or k.startswith("Error"):
        return pct(v, 1)
    if base in ("CTS", "CTSn", "CTQ", "CT") or "€" in k:
        return eur(v)
    if base in ("s", "K") or base.startswith("s óptimo"):
        return num(v, 0)
    return num(v, 3)


def texto_escenario(nombre, r, unidad):
    t, c = r["teoria"], r["costes"]
    u = unidad
    ok = r["resumen_mc"]["Teórico dentro del IC95"].loc[["L", "Lq", "W", "Wq", "P(bloqueo)"]]
    val = ("la simulación reproduce los valores teóricos (todos dentro del IC95 %)" if ok.all() else
           f"la simulación reproduce los valores teóricos salvo en {', '.join(ok[~ok].index)} (fuera del IC95 %, "
           "lo que puede deberse al azar o a una duración de réplica insuficiente)")
    sat = ("" if t["rho"] < 1 else
           f" La carga nominal ρ = {pct(t['rho'])} ≥ 100 % sería inestable con capacidad infinita; aquí la capacidad "
           f"K = {t['K']} la contiene a costa de rechazar clientes.")
    txt = (f"{nombre}: M/M/{t['s']}/{t['K']} con ρ = {pct(t['rho'])} nominal y ρ̄ = {pct(t['rho_eff'])} efectiva por "
           f"servidor; se rechaza el {pct(t['Pbloqueo'], 2)} de las llegadas (λ̄ = {num(t['lambda_bar'], 3)} clientes/"
           f"{unidad_sing(u)} aceptados). L = {num(t['L'], 3)} y Lq = {num(t['Lq'], 3)} clientes, W = {num(t['W'], 4)} y "
           f"Wq = {num(t['Wq'], 4)} {unidad_larga(u)} (valores teóricos); {val}.{sat} "
           f"Coste total CT = {eur(c['CT'])}/{unidad_sing(u)} (CTS {eur(c['CTS'])}, CTSn {eur(c['CTSn'])}, "
           f"CTQ {eur(c['CTQ'])}).")
    if r["s_opt"] is not None:
        txt += (" Su número de servidores ya es el de menor coste total para esa capacidad." if r["s_opt"] == t["s"] else
                f" Con la misma capacidad, el número de servidores de menor coste total sería s = {r['s_opt']} "
                f"(CT = {eur(r['barrido'].loc[r['s_opt'], 'CT'])}/{unidad_sing(u)}).")
    n_ok = (r["objetivos"]["Cumple"] == "Sí").sum()
    txt += f" Cumple {n_ok} de los {len(r['objetivos'])} objetivos de servicio."
    return txt


def texto_recomendacion(res, unidad):
    filas = [(n, (r["objetivos"]["Cumple"] == "Sí").sum(), r["costes"]["CT"], r["teoria"]["Pbloqueo"])
             for n, r in res.items()]
    max_ok = max(f[1] for f in filas)
    mejor = sorted([f for f in filas if f[1] == max_ok], key=lambda f: f[2])[0]
    barato = min(filas, key=lambda f: f[2])
    txt = (f"Se recomienda «{mejor[0]}»: es el escenario que cumple más objetivos de servicio ({mejor[1]} de "
           f"{len(res[mejor[0]]['objetivos'])}) y, entre ellos, el de menor coste total ({eur(mejor[2])}/"
           f"{unidad_sing(unidad)}), con un bloqueo del {pct(mejor[3], 2)}.")
    if barato[0] != mejor[0]:
        txt += (f" El escenario más barato es «{barato[0]}» ({eur(barato[2])}/{unidad_sing(unidad)}), pero cumple menos "
                f"objetivos ({barato[1]}) y rechaza el {pct(barato[3], 2)} de las llegadas. Recuerde que el coste total "
                "no incluye el coste de los clientes perdidos.")
    return txt
