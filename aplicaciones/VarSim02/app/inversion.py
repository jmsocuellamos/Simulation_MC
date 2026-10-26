# -*- coding: utf-8 -*-
"""Riesgo y retorno de planes de inversión: el retorno R de cada plan es una variable discreta (valores vᵢ con
probabilidades pᵢ) que se genera por el método de la transformada inversa.

Teoría exacta (media, riesgo σ, probabilidad de pérdida, percentiles), simulación Monte Carlo con réplicas,
comparación de planes (escenarios) con objetivos y sensibilidad del estimador frente a nsim.

Basado en el Ejemplo 2 de «B-70. Distribuciones derivadas», Javier Morales y Asun Martínez Mayoral, equipo IA4LEGOS
(CC BY-SA 4.0)."""
import math

import numpy as np
import pandas as pd

try:
    from scipy import stats as _stats
except Exception:          # pragma: no cover
    _stats = None

CUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
N_DIAGRAMA = 60                 # simulaciones que se muestran en el diagrama de la transformada inversa
MAX_TRAMOS = 8
Z95 = 1.959963984540054


def normalizar(p):
    p = np.clip(np.asarray(p, dtype=float), 0, None)
    s = p.sum()
    return p / s if s > 0 else np.full(len(p), 1 / len(p))


def limpiar_plan(valores, probs):
    """Quita los tramos vacíos y devuelve (valores, probabilidades normalizadas)."""
    pares = [(float(v), float(p)) for v, p in zip(valores, probs)
             if v is not None and p is not None and not (isinstance(v, float) and math.isnan(v))
             and not (isinstance(p, float) and math.isnan(p))]
    if not pares:
        pares = [(0.0, 1.0)]
    v = np.array([a for a, _ in pares]); p = normalizar([b for _, b in pares])
    return v, p


# ==============================================================================================
# Teoría
# ==============================================================================================
def _cuantil(v, p, q):
    """Cuantil de la distribución discreta: menor valor x con P(R ≤ x) ≥ q."""
    orden = np.argsort(v, kind="stable")
    vs, F = v[orden], np.cumsum(p[orden])
    return float(vs[min(len(vs) - 1, int(np.searchsorted(F, q - 1e-12)))])


def teoria(valores, probs, I0):
    """KPIs teóricos de un plan: E(R) = Σ pᵢ·vᵢ, σ(R) = √(Σ pᵢ·(vᵢ − E(R))²), P(pérdida) = P(R < I₀),
    resultado neto esperado E(R) − I₀ y rentabilidad esperada (E(R) − I₀)/I₀."""
    v, p = limpiar_plan(valores, probs)
    I0 = float(I0)
    m = float(p @ v)
    sd = math.sqrt(max(0.0, float(p @ (v - m) ** 2)))
    F = np.cumsum(p); F[-1] = 1.0
    sis = {"v": tuple(float(x) for x in v), "p": tuple(float(x) for x in p), "F": tuple(float(x) for x in F),
           "I0": I0, "tramos": len(v), "E": m, "sd": sd, "CV": sd / m if m else float("nan"),
           "Ploss": float(p[v < I0].sum()), "neto": m - I0, "rent": (m - I0) / I0 if I0 else float("nan"),
           "perdida_media": float((p * np.maximum(I0 - v, 0)).sum()),
           "min": float(v[p > 0].min()), "max": float(v[p > 0].max())}
    sis["cuantiles"] = {q: _cuantil(v, p, q) for q in CUANTILES}
    sis["P5"] = sis["cuantiles"][0.05]
    sis["P5_neto"] = sis["P5"] - I0
    return sis


def cdf(sis, x):
    v, p = np.array(sis["v"]), np.array(sis["p"])
    return float(p[v <= x].sum())


# ==============================================================================================
# Simulación: transformada inversa
# ==============================================================================================
def transformada_inversa(u, F):
    """Índice del tramo: menor i tal que u ≤ F(i) (vectorizado)."""
    return np.minimum(np.searchsorted(np.asarray(F), u, side="left"), len(F) - 1)


def simular_plan(sis, nsim, semilla, n_traza=0):
    """nsim retornos del plan por transformada inversa. Devuelve los retornos, el tramo de cada simulación y, si se
    pide, las primeras simulaciones (U, tramo, retorno, resultado neto)."""
    rng = np.random.default_rng(int(semilla))
    U = rng.random(int(nsim))
    idx = transformada_inversa(U, sis["F"])
    R = np.asarray(sis["v"])[idx]
    traza = None
    if n_traza:
        k = min(n_traza, len(U))
        traza = pd.DataFrame({"Simulación": np.arange(1, k + 1), "U": U[:k], "Tramo": idx[:k] + 1, "Retorno": R[:k],
                              "Neto": R[:k] - sis["I0"]})
    return {"R": R, "idx": idx, "traza": traza}


def _tq(n, alpha=0.05):
    if n < 2:
        return float("nan")
    return float(_stats.t.ppf(1 - alpha / 2, n - 1)) if _stats is not None else 1.96


def montecarlo(sis, nsim, replicas, semilla, progreso=None):
    """`replicas` réplicas de `nsim` simulaciones (réplica i -> semilla + i). Las mismas semillas en todos los
    planes (números aleatorios comunes)."""
    filas, Rs, Is = [], [], []
    traza = None
    I0 = sis["I0"]
    for r in range(int(replicas)):
        s = simular_plan(sis, nsim, int(semilla) + r, n_traza=N_DIAGRAMA if r == 0 else 0)
        R = s["R"]
        if r == 0:
            traza = s["traza"]
        m = float(R.mean()); sd = float(R.std(ddof=1)) if len(R) > 1 else 0.0
        fila = {"Réplica": r + 1, "Semilla": int(semilla) + r, "Simulaciones": len(R), "Media R": m, "σ(R)": sd,
                "P(pérdida)": float((R < I0).mean()), "Neto medio": m - I0,
                "Rentabilidad": (m - I0) / I0 if I0 else float("nan"), "P5": float(np.quantile(R, 0.05, method="inverted_cdf"))}
        fr = np.bincount(s["idx"], minlength=sis["tramos"]) / len(R)
        for i in range(sis["tramos"]):
            fila[f"Frec. tramo {i + 1}"] = fr[i]
        filas.append(fila)
        Rs.append(R); Is.append(s["idx"])
        if progreso:
            progreso(r + 1, replicas)
    return {"replicas": pd.DataFrame(filas).set_index("Réplica"), "R": np.concatenate(Rs),
            "idx": np.concatenate(Is), "traza": traza}


def _kpis(sis):
    k = [("Retorno medio E(R)", "Media R", sis["E"]), ("Riesgo σ(R)", "σ(R)", sis["sd"]),
         ("Probabilidad de pérdida P(R < I₀)", "P(pérdida)", sis["Ploss"]),
         ("Resultado neto medio E(R) − I₀", "Neto medio", sis["neto"]),
         ("Rentabilidad esperada", "Rentabilidad", sis["rent"]), ("Percentil 5 de R", "P5", sis["P5"])]
    for i in range(sis["tramos"]):
        k.append((f"Proporción del tramo {i + 1} (R = {eur(sis['v'][i])})", f"Frec. tramo {i + 1}", sis["p"][i]))
    return k


def resumen_mc(mc, sis):
    """Media de las réplicas, IC95 % (t de Student con la variabilidad entre réplicas) y valor teórico."""
    rep = mc["replicas"]
    n = len(rep)
    tq = _tq(n)
    filas = []
    for nombre, col, teo in _kpis(sis):
        v = rep[col].astype(float)
        m = float(v.mean())
        se = float(v.std(ddof=1)) / math.sqrt(n) if n > 1 else float("nan")
        lo, hi = (m - tq * se, m + tq * se) if n > 1 else (float("nan"), float("nan"))
        tol = 1e-9 * max(1.0, abs(teo))
        filas.append({"KPI": nombre, "Teórico": teo, "Simulado · media": m, "IC95 inf": lo, "IC95 sup": hi,
                      "Error relativo": (m - teo) / abs(teo) if abs(teo) > 1e-12 else None,
                      "Teórico en IC95": bool(lo - tol <= teo <= hi + tol) if n > 1 else None})
    return pd.DataFrame(filas).set_index("KPI")


# ==============================================================================================
# Distribución del retorno
# ==============================================================================================
def tabla_tramos(sis, mc=None):
    """Tramos del plan: retorno, resultado neto, probabilidad, acumulada F e intervalo de U que lo produce en la
    transformada inversa; con la frecuencia simulada si se indica."""
    filas = []
    for i in range(sis["tramos"]):
        lo = 0.0 if i == 0 else sis["F"][i - 1]
        filas.append({"Tramo": i + 1, "Retorno R (€)": sis["v"][i], "Neto R − I₀ (€)": sis["v"][i] - sis["I0"],
                      "Intervalo de U": f"({num(lo, 2)}, {num(sis['F'][i], 2)}]",
                      "Teórica · P(R = v)": sis["p"][i], "Teórica · F": sis["F"][i]})
    df = pd.DataFrame(filas).set_index("Tramo")
    if mc is not None:
        fr = np.bincount(mc["idx"], minlength=sis["tramos"]) / len(mc["idx"])
        df["Simulada · P(R = v)"] = fr
        df["Simulada · F"] = np.cumsum(fr)
    return df


def tabla_cuantiles(sis, mc=None):
    df = pd.DataFrame(index=pd.Index([f"P{int(round(q * 100))}" for q in CUANTILES], name="Percentil"))
    df["Teórico (€)"] = [sis["cuantiles"][q] for q in CUANTILES]
    if mc is not None:
        df["Simulado (€)"] = [float(np.quantile(mc["R"], q, method="inverted_cdf")) for q in CUANTILES]
    df["Teórico neto (€)"] = df["Teórico (€)"] - sis["I0"]
    return df


def prob_retorno(sis, R, tipo, x):
    """P(R ≤ x), P(R < x), P(R ≥ x) o P(R > x), teórica y simulada."""
    v, p = np.array(sis["v"]), np.array(sis["p"])
    R = np.asarray(R) if R is not None else None
    op = {"≤": np.less_equal, "<": np.less, "≥": np.greater_equal, ">": np.greater}[tipo]
    return float(p[op(v, x)].sum()), (float(op(R, x).mean()) if R is not None else None)


# ==============================================================================================
# Objetivos (equivalentes a los SLAs de los casos de estudio)
# ==============================================================================================
OBJETIVOS_DEF = {"RENT_MIN": 0.102, "PLOSS_MAX": 0.05, "SIGMA_MAX": 1000.0, "P5NETO_MIN": 0.0}


def evaluar_objetivos(sis, obj):
    filas = []
    def fila(nombre, valor, txt, cumple):
        filas.append({"Objetivo": nombre, "Valor": valor, "Umbral": txt, "Cumple": "Sí" if cumple else "No"})
    if obj.get("RENT_MIN") is not None:
        fila("Rentabilidad esperada", sis["rent"], f"≥ {pct(obj['RENT_MIN'], 1)}", sis["rent"] >= obj["RENT_MIN"] - 1e-12)
    if obj.get("PLOSS_MAX") is not None:
        fila("Probabilidad de pérdida", sis["Ploss"], f"≤ {pct(obj['PLOSS_MAX'], 1)}", sis["Ploss"] <= obj["PLOSS_MAX"] + 1e-12)
    if obj.get("SIGMA_MAX") is not None:
        fila("Riesgo σ(R)", sis["sd"], f"≤ {eur(obj['SIGMA_MAX'])}", sis["sd"] <= obj["SIGMA_MAX"] + 1e-9)
    if obj.get("P5NETO_MIN") is not None:
        fila("Resultado neto en el peor 5 % (P5 − I₀)", sis["P5_neto"], f"≥ {eur(obj['P5NETO_MIN'])}",
             sis["P5_neto"] >= obj["P5NETO_MIN"] - 1e-9)
    return pd.DataFrame(filas, columns=["Objetivo", "Valor", "Umbral", "Cumple"]).set_index("Objetivo")


# ==============================================================================================
# Análisis de un escenario (plan) y comparación
# ==============================================================================================
def analizar_escenario(esc, comunes, ajustes, progreso=None):
    sis = teoria(esc["valores"], esc["probs"], comunes["I0"])
    mc = montecarlo(sis, ajustes["nsim"], ajustes["replicas"], ajustes["semilla"], progreso)
    return {"esc": dict(esc), "teoria": sis, "mc": mc, "resumen_mc": resumen_mc(mc, sis),
            "objetivos": evaluar_objetivos(sis, comunes["objetivos"])}


def tabla_comparativa(res):
    cols = {}
    for n, r in res.items():
        t, rm = r["teoria"], r["resumen_mc"]
        cols[n] = {
            "Parámetro · tramos": t["tramos"],
            "Parámetro · retornos posibles": " / ".join(num(x, 0) for x in t["v"]),
            "Parámetro · probabilidades": " / ".join(pct(x, 0) for x in t["p"]),
            "Teórico · E(R)": t["E"], "Teórico · σ(R)": t["sd"], "Teórico · P(pérdida)": t["Ploss"],
            "Teórico · neto E(R) − I₀": t["neto"], "Teórico · rentabilidad": t["rent"],
            "Teórico · percentil 5 de R": t["P5"], "Teórico · pérdida esperada": t["perdida_media"],
            "Simulado · E(R)": rm.loc["Retorno medio E(R)", "Simulado · media"],
            "Simulado · σ(R)": rm.loc["Riesgo σ(R)", "Simulado · media"],
            "Simulado · P(pérdida)": rm.loc["Probabilidad de pérdida P(R < I₀)", "Simulado · media"],
            "Simulado · rentabilidad": rm.loc["Rentabilidad esperada", "Simulado · media"],
            "Objetivos cumplidos": f"{(r['objetivos']['Cumple'] == 'Sí').sum()} de {len(r['objetivos'])}",
        }
    return pd.DataFrame(cols)


def tabla_objetivos(res):
    return pd.DataFrame({n: r["objetivos"]["Cumple"] for n, r in res.items()})


# ==============================================================================================
# Sensibilidad del estimador Monte Carlo frente a nsim
# ==============================================================================================
def valores_barrido(vmin, vmax, puntos):
    v = np.geomspace(max(1, vmin), max(vmin + 1, vmax), max(2, int(puntos)))
    return sorted(set(int(round(x)) for x in v))


def sensibilidad(sis, valores, replicas, semilla, progreso=None):
    filas = []
    tot = len(valores) * int(replicas)
    k = 0
    for nsim in valores:
        medias, s1, s2, nt = [], 0.0, 0.0, 0
        for r in range(int(replicas)):
            R = simular_plan(sis, nsim, int(semilla) + r)["R"]
            medias.append(float(R.mean())); s1 += float(R.sum()); s2 += float((R ** 2).sum()); nt += len(R)
            k += 1
            if progreso:
                progreso(k, tot)
        m = s1 / nt
        var = max(0.0, (s2 - nt * m * m) / max(1, nt - 1))
        se = math.sqrt(var / nt)
        filas.append({"nsim": nsim, "Simulaciones totales": nt, "Media combinada": m, "IC95 inf": m - Z95 * se,
                      "IC95 sup": m + Z95 * se,
                      "σ entre réplicas": float(np.std(medias, ddof=1)) if len(medias) > 1 else 0.0,
                      "σ teórica de la media": sis["sd"] / math.sqrt(nsim),
                      "Teórico": sis["E"], "Error relativo": (m - sis["E"]) / abs(sis["E"]) if sis["E"] else None,
                      "Medias por réplica": medias})
    return pd.DataFrame(filas).set_index("nsim")


def texto_sensibilidad(df, plan, sis):
    if sis["sd"] == 0:
        return (f"El plan «{plan}» es determinista (un único retorno con probabilidad 1): no hay variabilidad que "
                "reducir y la σ entre réplicas es 0 para cualquier nsim."), None
    a, z = df["σ entre réplicas"].iloc[0], df["σ entre réplicas"].iloc[-1]
    red = a / z if a > 0 and z > 0 else None
    dentro = ((df["IC95 inf"] <= df["Teórico"]) & (df["Teórico"] <= df["IC95 sup"])).sum()
    txt = ("A medida que crece nsim, la media combinada se acerca al retorno teórico y la dispersión entre semillas "
           "se reduce" + (f" ({num(red, 1)} veces menor entre el primer y el último punto; la teoría, σ/√nsim, "
                          f"predice {num(math.sqrt(df.index[-1] / df.index[0]), 1)} veces)." if red else "."))
    txt += f" El valor teórico queda dentro del IC95 % en {dentro} de los {len(df)} puntos."
    return txt, red


# ==============================================================================================
# Formato y textos
# ==============================================================================================
def num(x, d=2):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    if round(float(x), d) == 0:
        x = 0.0                      # evita «-0»
    s = f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, d=1):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return num(100 * x, d) + " %"


def eur(x, d=0):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return num(x, d) + " €"


def fmt_kpi(k, c, v):
    if isinstance(v, str) or v is None:
        return "—" if v is None else v
    if "P(" in k or "rentabilidad" in k:
        return pct(v, 2)
    if "tramos" in k:
        return num(v, 0)
    return eur(v, 0)


def fmt_mc(fila, col, v):
    if col == "Teórico en IC95":
        return "—" if v is None else ("Sí" if v else "No")
    if col == "Error relativo":
        return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else pct(v, 2)
    if "Proporción" in fila or "Probabilidad" in fila or "Rentabilidad" in fila:
        return pct(v, 2)
    return eur(v, 0)


def fmt_obj(fila, col, v):
    if col == "Valor":
        return pct(v, 2) if ("Rentabilidad" in fila or "Probabilidad" in fila) else eur(v, 0)
    return str(v)


def texto_escenario(nombre, r):
    t, rm = r["teoria"], r["resumen_mc"]
    ok = rm["Teórico en IC95"].dropna()
    val = ("la simulación reproduce los valores teóricos (todos dentro del IC95 %)" if ok.all() else
           f"la simulación reproduce los valores teóricos salvo en {', '.join(ok[~ok].index)} (fuera del IC95 %, "
           "lo que puede deberse al azar o a un número de simulaciones insuficiente)")
    tramos = ", ".join(f"{eur(v)} ({pct(p, 0)})" for v, p in zip(t["v"], t["p"]))
    txt = (f"{nombre}: retornos {tramos}. Retorno esperado E(R) = {eur(t['E'])} (resultado neto {eur(t['neto'])}, "
           f"rentabilidad {pct(t['rent'], 2)}), riesgo σ(R) = {eur(t['sd'])} y probabilidad de pérdida "
           f"P(R < I₀) = {pct(t['Ploss'], 1)} (valores teóricos); {val}.")
    if t["sd"] == 0:
        txt += " Es un plan determinista: el retorno es siempre el mismo."
    n_ok = (r["objetivos"]["Cumple"] == "Sí").sum()
    txt += f" Cumple {n_ok} de los {len(r['objetivos'])} objetivos."
    return txt


def texto_recomendacion(res):
    filas = [(n, (r["objetivos"]["Cumple"] == "Sí").sum(), r["teoria"]) for n, r in res.items()]
    max_ok = max(f[1] for f in filas)
    mejor = sorted([f for f in filas if f[1] == max_ok], key=lambda f: (-f[2]["E"], f[2]["sd"]))[0]
    rentable = max(filas, key=lambda f: f[2]["E"])
    seguro = min(filas, key=lambda f: (f[2]["sd"], -f[2]["E"]))
    txt = (f"Se recomienda «{mejor[0]}»: es el plan que cumple más objetivos ({mejor[1]} de "
           f"{len(res[mejor[0]]['objetivos'])}) y, entre ellos, el de mayor retorno esperado ({eur(mejor[2]['E'])}, "
           f"rentabilidad {pct(mejor[2]['rent'], 2)}, σ = {eur(mejor[2]['sd'])}).")
    if rentable[0] != mejor[0]:
        txt += (f" El plan de mayor retorno medio es «{rentable[0]}» ({eur(rentable[2]['E'])}), pero con un riesgo "
                f"σ = {eur(rentable[2]['sd'])} y una probabilidad de pérdida del {pct(rentable[2]['Ploss'], 1)}.")
    if seguro[0] != mejor[0]:
        txt += f" El más seguro es «{seguro[0]}» (σ = {eur(seguro[2]['sd'])}), con menor retorno ({eur(seguro[2]['E'])})."
    txt += " La elección final depende del perfil de riesgo de la empresa, no solo de maximizar la media."
    return txt
