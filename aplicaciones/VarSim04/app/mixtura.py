# -*- coding: utf-8 -*-
"""Beneficio esperado por mixtura: cada cliente que entra en la tienda compra (T = 1) con probabilidad Pr(compra) y,
si compra, elige uno de varios modelos de horno (transformada inversa sobre sus probabilidades condicionadas). El
beneficio es B = 0 si no compra y B = ganancia del modelo si compra.

Teoría exacta de la mixtura, simulación Monte Carlo en dos pasos con réplicas, comparación de estrategias
comerciales (escenarios) con objetivos y sensibilidad del estimador frente a nsim.

Basado en el Ejemplo 4 de «B-70. Distribuciones derivadas», Javier Morales y Asun Martínez Mayoral, equipo IA4LEGOS
(CC BY-SA 4.0)."""
import math

import numpy as np
import pandas as pd

try:
    from scipy import stats as _stats
except Exception:          # pragma: no cover
    _stats = None

N_DIAGRAMA = 60                 # clientes que se muestran en el diagrama de funcionamiento
MAX_MODELOS = 8
Z95 = 1.959963984540054
NO_COMPRA = "No compra"


def normalizar(p):
    p = np.clip(np.asarray(p, dtype=float), 0, None)
    s = p.sum()
    return p / s if s > 0 else np.full(len(p), 1 / len(p))


def limpiar_modelos(nombres, valores, probs):
    filas = [(str(n), float(v), float(p)) for n, v, p in zip(nombres, valores, probs)
             if v is not None and p is not None and not (isinstance(v, float) and math.isnan(v))
             and not (isinstance(p, float) and math.isnan(p))]
    if not filas:
        filas = [("Modelo", 0.0, 1.0)]
    return [f[0] or f"Modelo {i + 1}" for i, f in enumerate(filas)], np.array([f[1] for f in filas]), \
        normalizar([f[2] for f in filas])


# ==============================================================================================
# Teoría de la mixtura
# ==============================================================================================
def teoria(pcompra, nombres, valores, probs, coste=0.0, visitantes=200, gama_alta=None):
    """E(B|compra) = Σ pᵢ·bᵢ ; E(B) = Pr(compra)·E(B|compra) ; Var(B) = Pr(compra)·Σ pᵢ·bᵢ² − E(B)².
    `coste` es el coste comercial por cliente que entra; `gama_alta` el número de modelos «de gama alta» (los
    últimos de la lista) para el objetivo correspondiente."""
    nombres, b, p = limpiar_modelos(nombres, valores, probs)
    pc = float(min(max(pcompra, 0.0), 1.0))
    F = np.cumsum(p); F[-1] = 1.0
    ec = float(p @ b)
    e2c = float(p @ b ** 2)
    E = pc * ec
    var = max(0.0, pc * e2c - E ** 2)
    sdc = math.sqrt(max(0.0, e2c - ec ** 2))
    k_alta = len(b) - 1 if gama_alta is None else int(gama_alta)
    alta = float(p[len(b) - k_alta:].sum()) if k_alta > 0 else 0.0
    sis = {"pc": pc, "modelos": tuple(nombres), "b": tuple(float(x) for x in b), "p": tuple(float(x) for x in p),
           "F": tuple(float(x) for x in F), "coste": float(coste), "visitantes": int(visitantes),
           "E_compra": ec, "sd_compra": sdc, "E": E, "sd": math.sqrt(var), "E_neto": E - float(coste),
           "dia_neto": (E - float(coste)) * int(visitantes), "gama_alta": alta,
           "contrib": tuple(pc * float(pi) * float(bi) for pi, bi in zip(p, b)),
           "coste_no_compra": ec - E}
    sis["CV"] = sis["sd"] / E if E else float("nan")
    return sis


def resultados_posibles(sis):
    """Desenlaces de la mixtura: no compra (B = 0) o compra del modelo i (B = bᵢ), con su probabilidad
    incondicional y su contribución a E(B)."""
    filas = [{"Desenlace": NO_COMPRA, "Beneficio B (€)": 0.0, "P(modelo | compra)": None,
              "Teórica · P(desenlace)": 1 - sis["pc"], "Contribución a E(B) (€)": 0.0}]
    for m, b, p, c in zip(sis["modelos"], sis["b"], sis["p"], sis["contrib"]):
        filas.append({"Desenlace": f"Compra · {m}", "Beneficio B (€)": b, "P(modelo | compra)": p,
                      "Teórica · P(desenlace)": sis["pc"] * p, "Contribución a E(B) (€)": c})
    return pd.DataFrame(filas).set_index("Desenlace")


# ==============================================================================================
# Simulación en dos pasos
# ==============================================================================================
def transformada_inversa(u, F):
    return np.minimum(np.searchsorted(np.asarray(F), u, side="left"), len(F) - 1)


def un_cliente(sis, u1, u2):
    """Camino de un cliente con U₁ y U₂ dados: (compra, índice del modelo o −1, beneficio)."""
    if u1 < sis["pc"]:
        k = int(transformada_inversa(np.array([u2]), sis["F"])[0])
        return True, k, sis["b"][k]
    return False, -1, 0.0


def simular_clientes(sis, nsim, semilla, n_traza=0):
    """Paso 1: U₁ ~ Unif(0,1); compra si U₁ < Pr(compra). Paso 2 (solo si compra): U₂ ~ Unif(0,1) y transformada
    inversa sobre las probabilidades de los modelos. Devuelve el beneficio, el indicador de compra y el modelo de
    cada cliente (−1 si no compra)."""
    rng = np.random.default_rng(int(semilla))
    n = int(nsim)
    U1 = rng.random(n)
    U2 = rng.random(n)                     # se genera para todos, pero solo se usa si el cliente compra
    T = U1 < sis["pc"]
    idx = np.where(T, transformada_inversa(U2, sis["F"]), -1)
    B = np.where(T, np.asarray(sis["b"])[np.maximum(idx, 0)], 0.0)
    traza = None
    if n_traza:
        k = min(n_traza, n)
        traza = pd.DataFrame({"Cliente": np.arange(1, k + 1), "U1": U1[:k], "Compra": T[:k],
                              "U2": np.where(T[:k], U2[:k], np.nan),
                              "Modelo": [sis["modelos"][i] if i >= 0 else "—" for i in idx[:k]], "Beneficio": B[:k]})
    return {"B": B, "T": T, "idx": idx, "traza": traza}


def _tq(n, alpha=0.05):
    if n < 2:
        return float("nan")
    return float(_stats.t.ppf(1 - alpha / 2, n - 1)) if _stats is not None else 1.96


def montecarlo(sis, nsim, replicas, semilla, progreso=None):
    """`replicas` réplicas de `nsim` clientes (réplica i -> semilla + i); mismas semillas en todas las estrategias."""
    filas, Bs, Ts, Is = [], [], [], []
    traza = None
    for r in range(int(replicas)):
        s = simular_clientes(sis, nsim, int(semilla) + r, n_traza=N_DIAGRAMA if r == 0 else 0)
        B, T, idx = s["B"], s["T"], s["idx"]
        if r == 0:
            traza = s["traza"]
        nc = int(T.sum())
        m = float(B.mean())
        fila = {"Réplica": r + 1, "Semilla": int(semilla) + r, "Clientes": len(B), "Compradores": nc,
                "E(B)": m, "E(B | compra)": float(B[T].mean()) if nc else float("nan"),
                "P(compra)": nc / len(B), "E(B) neto": m - sis["coste"],
                "σ(B)": float(B.std(ddof=1)) if len(B) > 1 else 0.0}
        fr = np.bincount(idx[T], minlength=len(sis["b"])) / max(1, nc)
        for i, mod in enumerate(sis["modelos"]):
            fila[f"P({mod} | compra)"] = fr[i]
        filas.append(fila)
        Bs.append(B); Ts.append(T); Is.append(idx)
        if progreso:
            progreso(r + 1, replicas)
    return {"replicas": pd.DataFrame(filas).set_index("Réplica"), "B": np.concatenate(Bs), "T": np.concatenate(Ts),
            "idx": np.concatenate(Is), "traza": traza}


def _kpis(sis):
    k = [("Beneficio medio por cliente E(B)", "E(B)", sis["E"]),
         ("Beneficio medio por comprador E(B | compra)", "E(B | compra)", sis["E_compra"]),
         ("Proporción que compra P(compra)", "P(compra)", sis["pc"]),
         ("Beneficio neto por cliente E(B) − coste", "E(B) neto", sis["E_neto"]),
         ("Desviación típica σ(B)", "σ(B)", sis["sd"])]
    for m, p in zip(sis["modelos"], sis["p"]):
        k.append((f"Proporción de «{m}» entre compradores", f"P({m} | compra)", p))
    return k


def resumen_mc(mc, sis):
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


def tabla_desenlaces(sis, mc=None):
    df = resultados_posibles(sis)
    if mc is not None:
        n = len(mc["B"])
        sim = [float((~mc["T"]).sum()) / n] + [float(((mc["idx"] == i) & mc["T"]).sum()) / n for i in range(len(sis["b"]))]
        df["Simulada · P(desenlace)"] = sim
        df["Simulada · contribución (€)"] = [0.0] + [s * b for s, b in zip(sim[1:], sis["b"])]
    return df


def tabla_modelos(sis, mc=None):
    """Distribución condicionada a la compra: probabilidad, acumulada e intervalo de U₂ de cada modelo."""
    filas = []
    for i, (m, b, p) in enumerate(zip(sis["modelos"], sis["b"], sis["p"])):
        lo = 0.0 if i == 0 else sis["F"][i - 1]
        filas.append({"Modelo": m, "Ganancia (€)": b, "Intervalo de U₂": f"({num(lo, 2)}, {num(sis['F'][i], 2)}]",
                      "Teórica · P(modelo | compra)": p, "Teórica · F": sis["F"][i]})
    df = pd.DataFrame(filas).set_index("Modelo")
    if mc is not None:
        T = mc["T"]
        fr = np.bincount(mc["idx"][T], minlength=len(sis["b"])) / max(1, int(T.sum()))
        df["Simulada · P(modelo | compra)"] = fr
        df["Simulada · F"] = np.cumsum(fr)
    return df


# ==============================================================================================
# Objetivos (equivalentes a los SLAs de los casos de estudio)
# ==============================================================================================
OBJETIVOS_DEF = {"BNETO_MIN": 29.0, "PCOMPRA_MIN": 0.55, "BCOMPRA_MIN": 58.0, "ALTA_MIN": 0.80}


def evaluar_objetivos(sis, obj):
    filas = []
    def fila(nombre, valor, txt, cumple):
        filas.append({"Objetivo": nombre, "Valor": valor, "Umbral": txt, "Cumple": "Sí" if cumple else "No"})
    if obj.get("BNETO_MIN") is not None:
        fila("Beneficio neto por cliente", sis["E_neto"], f"≥ {eur(obj['BNETO_MIN'], 2)}", sis["E_neto"] >= obj["BNETO_MIN"] - 1e-9)
    if obj.get("PCOMPRA_MIN") is not None:
        fila("Proporción que compra", sis["pc"], f"≥ {pct(obj['PCOMPRA_MIN'], 1)}", sis["pc"] >= obj["PCOMPRA_MIN"] - 1e-12)
    if obj.get("BCOMPRA_MIN") is not None:
        fila("Beneficio medio por comprador", sis["E_compra"], f"≥ {eur(obj['BCOMPRA_MIN'], 2)}",
             sis["E_compra"] >= obj["BCOMPRA_MIN"] - 1e-9)
    if obj.get("ALTA_MIN") is not None:
        fila("Compras de gama alta (todos salvo el 1.er modelo)", sis["gama_alta"], f"≥ {pct(obj['ALTA_MIN'], 1)}",
             sis["gama_alta"] >= obj["ALTA_MIN"] - 1e-12)
    return pd.DataFrame(filas, columns=["Objetivo", "Valor", "Umbral", "Cumple"]).set_index("Objetivo")


# ==============================================================================================
# Análisis de un escenario (estrategia) y comparación
# ==============================================================================================
def teoria_escenario(esc, comunes):
    return teoria(esc["pcompra"] / 100, esc["modelos"], esc["valores"], esc["probs"], esc.get("coste") or 0.0,
                  comunes["visitantes"])


def analizar_escenario(esc, comunes, ajustes, progreso=None):
    sis = teoria_escenario(esc, comunes)
    mc = montecarlo(sis, ajustes["nsim"], ajustes["replicas"], ajustes["semilla"], progreso)
    return {"esc": dict(esc), "teoria": sis, "mc": mc, "resumen_mc": resumen_mc(mc, sis),
            "objetivos": evaluar_objetivos(sis, comunes["objetivos"])}


def tabla_comparativa(res):
    cols = {}
    for n, r in res.items():
        t, rm = r["teoria"], r["resumen_mc"]
        cols[n] = {
            "Parámetro · P(compra)": t["pc"],
            "Parámetro · modelos": " / ".join(t["modelos"]),
            "Parámetro · ganancias (€)": " / ".join(num(b, 0) for b in t["b"]),
            "Parámetro · P(modelo | compra)": " / ".join(pct(p, 0) for p in t["p"]),
            "Parámetro · coste por cliente (€)": t["coste"],
            "Teórico · E(B | compra)": t["E_compra"], "Teórico · E(B) por cliente": t["E"],
            "Teórico · σ(B)": t["sd"], "Teórico · beneficio neto por cliente": t["E_neto"],
            "Teórico · compras de gama alta": t["gama_alta"],
            "Teórico · beneficio neto diario": t["dia_neto"],
            "Simulado · P(compra)": rm.loc["Proporción que compra P(compra)", "Simulado · media"],
            "Simulado · E(B | compra)": rm.loc["Beneficio medio por comprador E(B | compra)", "Simulado · media"],
            "Simulado · E(B) por cliente": rm.loc["Beneficio medio por cliente E(B)", "Simulado · media"],
            "Simulado · beneficio neto por cliente": rm.loc["Beneficio neto por cliente E(B) − coste", "Simulado · media"],
            "Objetivos cumplidos": f"{(r['objetivos']['Cumple'] == 'Sí').sum()} de {len(r['objetivos'])}",
        }
    return pd.DataFrame(cols)


def tabla_objetivos(res):
    return pd.DataFrame({n: r["objetivos"]["Cumple"] for n, r in res.items()})


# ==============================================================================================
# Sensibilidad del estimador Monte Carlo frente a nsim
# ==============================================================================================
METRICAS = {"total": "Beneficio medio · todos los clientes", "compradores": "Beneficio medio · solo compradores",
            "proporcion": "Proporción que compra"}


def valores_barrido(vmin, vmax, puntos):
    v = np.geomspace(max(1, vmin), max(vmin + 1, vmax), max(2, int(puntos)))
    return sorted(set(int(round(x)) for x in v))


def _teo_metrica(sis, metrica):
    return {"total": sis["E"], "compradores": sis["E_compra"], "proporcion": sis["pc"]}[metrica]


def _sd_teo(sis, metrica, nsim):
    """Desviación típica teórica del estimador con nsim clientes."""
    if metrica == "total":
        return sis["sd"] / math.sqrt(nsim)
    if metrica == "proporcion":
        return math.sqrt(sis["pc"] * (1 - sis["pc"]) / nsim)
    return sis["sd_compra"] / math.sqrt(max(1e-9, nsim * sis["pc"]))      # nº de compradores ≈ nsim·Pr(compra)


def sensibilidad(sis, metrica, valores, replicas, semilla, progreso=None):
    """Para cada nsim, `replicas` simulaciones con semillas distintas; media de las réplicas, IC95 % con la
    variabilidad entre réplicas (t de Student) y σ entre réplicas frente a la σ teórica del estimador."""
    filas = []
    tot = len(valores) * int(replicas)
    k = 0
    tq = _tq(int(replicas))
    teo = _teo_metrica(sis, metrica)
    for nsim in valores:
        medias = []
        for r in range(int(replicas)):
            s = simular_clientes(sis, nsim, int(semilla) + r)
            B, T = s["B"], s["T"]
            if metrica == "total":
                medias.append(float(B.mean()))
            elif metrica == "compradores":
                medias.append(float(B[T].mean()) if T.any() else 0.0)
            else:
                medias.append(float(T.mean()))
            k += 1
            if progreso:
                progreso(k, tot)
        m = float(np.mean(medias))
        sdr = float(np.std(medias, ddof=1)) if len(medias) > 1 else 0.0
        se = sdr / math.sqrt(len(medias))
        filas.append({"nsim": nsim, "Media de las réplicas": m, "IC95 inf": m - tq * se, "IC95 sup": m + tq * se,
                      "σ entre réplicas": sdr, "σ teórica del estimador": _sd_teo(sis, metrica, nsim),
                      "Teórico": teo, "Error relativo": (m - teo) / abs(teo) if teo else None,
                      "Medias por réplica": medias})
    return pd.DataFrame(filas).set_index("nsim")


def texto_sensibilidad(df, metrica, estrategia):
    a, z = df["σ entre réplicas"].iloc[0], df["σ entre réplicas"].iloc[-1]
    red = a / z if a > 0 and z > 0 else None
    dentro = ((df["IC95 inf"] <= df["Teórico"]) & (df["Teórico"] <= df["IC95 sup"])).sum()
    txt = (f"Estrategia «{estrategia}», métrica «{METRICAS[metrica]}». A medida que crece nsim, la media de las réplicas "
           "se acerca al valor teórico y la dispersión entre semillas se reduce")
    txt += (f" ({num(red, 1)} veces menor entre el primer y el último punto; la teoría, 1/√nsim, predice "
            f"{num(math.sqrt(df.index[-1] / df.index[0]), 1)} veces)." if red else ".")
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
        x = 0.0
    s = f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, d=1):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return num(100 * x, d) + " %"


def eur(x, d=2):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return num(x, d) + " €"


def fmt_kpi(k, c, v):
    if isinstance(v, str) or v is None:
        return "—" if v is None else v
    if "P(" in k or "gama alta" in k:
        return pct(v, 2)
    if "diario" in k:
        return eur(v, 0)
    return eur(v, 2)


def fmt_mc(fila, col, v):
    if col == "Teórico en IC95":
        return "—" if v is None else ("Sí" if v else "No")
    if col == "Error relativo":
        return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else pct(v, 2)
    if "Proporción" in fila:
        return pct(v, 2)
    return eur(v, 2)


def fmt_obj(fila, col, v):
    if col == "Valor":
        return pct(v, 2) if ("Proporción" in fila or "gama" in fila) else eur(v, 2)
    return str(v)


def fmt_metrica(metrica):
    return (lambda v: pct(v, 2)) if metrica == "proporcion" else (lambda v: eur(v, 2))


def texto_escenario(nombre, r):
    t, rm = r["teoria"], r["resumen_mc"]
    ok = rm["Teórico en IC95"].dropna()
    val = ("la simulación reproduce los valores teóricos (todos dentro del IC95 %)" if ok.all() else
           f"la simulación reproduce los valores teóricos salvo en {', '.join(ok[~ok].index)} (fuera del IC95 %, "
           "lo que puede deberse al azar o a un número de clientes insuficiente)")
    txt = (f"{nombre}: compra el {pct(t['pc'], 0)} de los clientes; entre los compradores, "
           + ", ".join(f"{m} {pct(p, 0)}" for m, p in zip(t["modelos"], t["p"])) +
           f". Beneficio medio por comprador E(B|compra) = {eur(t['E_compra'])} y por cliente que entra "
           f"E(B) = {eur(t['E'])} (valores teóricos); {val}.")
    if t["coste"]:
        txt += f" Descontado el coste comercial ({eur(t['coste'])} por cliente), el beneficio neto es {eur(t['E_neto'])} por cliente"
    else:
        txt += f" Beneficio neto por cliente: {eur(t['E_neto'])}"
    txt += f" y {eur(t['dia_neto'], 0)} al día ({num(t['visitantes'], 0)} clientes)."
    n_ok = (r["objetivos"]["Cumple"] == "Sí").sum()
    txt += f" Cumple {n_ok} de los {len(r['objetivos'])} objetivos."
    return txt


def texto_recomendacion(res):
    filas = [(n, (r["objetivos"]["Cumple"] == "Sí").sum(), r["teoria"]) for n, r in res.items()]
    max_ok = max(f[1] for f in filas)
    mejor = sorted([f for f in filas if f[1] == max_ok], key=lambda f: -f[2]["E_neto"])[0]
    rentable = max(filas, key=lambda f: f[2]["E_neto"])
    peor = min(filas, key=lambda f: f[2]["E_neto"])
    txt = (f"Se recomienda «{mejor[0]}»: es la estrategia que cumple más objetivos ({mejor[1]} de "
           f"{len(res[mejor[0]]['objetivos'])}) y, entre ellas, la de mayor beneficio neto por cliente "
           f"({eur(mejor[2]['E_neto'])}, {eur(mejor[2]['dia_neto'], 0)} al día).")
    if rentable[0] != mejor[0]:
        txt += (f" La de mayor beneficio neto es «{rentable[0]}» ({eur(rentable[2]['E_neto'])} por cliente), pero "
                f"cumple menos objetivos ({rentable[1]}).")
    if peor[0] != mejor[0]:
        g = mejor[2]["E_neto"] - peor[2]["E_neto"]
        if g > 0:
            txt += (f" Frente a «{peor[0]}», aporta {eur(g)} más por cliente ({eur(g * mejor[2]['visitantes'], 0)} al día).")
    t = mejor[2]
    txt += (f" En la estrategia recomendada, que no todos los clientes compren reduce el beneficio medio de "
            f"{eur(t['E_compra'])} (compradores) a {eur(t['E'])} (todos los clientes).")
    return txt
