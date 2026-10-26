# -*- coding: utf-8 -*-
"""EC1 · Optimización de costes en «Fly-Next»: coste de una entrega por el método de composición (ruta T → D, G, M)
y coste diario de la operación con tres franjas horarias de tráfico.

Bloque A · una entrega:  T ∈ {1, 2, 3} (Urbana, Regional, Premium); D | T, G | T y M | T independientes;
    Y = D·G·P_fuel + M·C_labor + peaje(D), con peaje(D) = K·I{D > umbral} + t·(D − umbral)⁺.
Bloque B · un día:  N_h ~ Poisson(λ_franja(h)) pedidos por hora; el tráfico multiplica G y M según (franja, ruta);
    Y_con e Y_sin se calculan con las mismas variables base (simulación pareada) y Δ = Y_con − Y_sin.

Teoría exacta: momentos y descomposición de la varianza por términos, función de distribución de Y por integración
numérica (percentiles y probabilidades de cola), coste diario como suma compuesta de Poisson (aproximación normal para
sus percentiles). Simulación: transformada inversa con números aleatorios comunes entre escenarios y macro-réplicas.

Material de Javier Morales, equipo IA4LEGOS (CC BY-SA 4.0)."""
import copy
import math

import numpy as np
import pandas as pd

try:
    from scipy import stats as _stats
    from scipy.special import ndtri as _ndtri
except Exception:          # pragma: no cover
    _stats = None
    _ndtri = None

RUTAS = (1, 2, 3)
NOMBRE_RUTA = {1: "Urbana", 2: "Regional", 3: "Premium"}
FRANJAS = ("Valle", "Llano", "Punta")
HORAS_FRANJA = {h: "Valle" for h in list(range(0, 7)) + [22, 23]}
HORAS_FRANJA.update({h: "Llano" for h in [7, 8] + list(range(12, 17)) + [20, 21]})
HORAS_FRANJA.update({h: "Punta" for h in list(range(9, 12)) + list(range(17, 20))})
TERMINOS = ("Combustible D·G·P_fuel", "Mano de obra M·C_labor", "Peaje", "Externalización")
Z95 = 1.959963984540054
N_DIAGRAMA = 40

# ==============================================================================================
# Parámetros por defecto (cuaderno 01_FlyNext)
# ==============================================================================================
COMUNES_DEF = {
    "D": {1: {"tipo": "poisson", "mu": 15.0, "sd": None},
          2: {"tipo": "norm", "mu": 80.0, "sd": 10.0},
          3: {"tipo": "norm", "mu": 300.0, "sd": 50.0}},
    "G": {1: (0.10, 0.15), 2: (0.08, 0.12), 3: (0.05, 0.07)},
    "M": {1: 0.5, 2: 1.2, 3: 3.0},
    "lambda": {"Valle": 8.0, "Llano": 25.0, "Punta": 45.0},
    "f_retraso": {"Valle": {1: 1.00, 2: 1.00, 3: 1.00}, "Llano": {1: 1.40, 2: 1.20, 3: 1.05},
                  "Punta": {1: 2.20, 2: 1.60, 3: 1.10}},
    "f_consumo": {"Valle": {1: 1.00, 2: 1.00, 3: 1.00}, "Llano": {1: 1.08, 2: 1.04, 3: 1.00},
                  "Punta": {1: 1.20, 2: 1.10, 3: 1.02}},
    "umbral_Y": 150.0, "entregas_ano": 50000, "presupuesto": 30000.0, "capacidad": 0,
    "objetivos": {"EY_MAX": 36.0, "PCOLA_MAX": 0.04, "DIA_MAX": 26300.0, "PPRES_MAX": 0.02},
}
ESC_BASE = {"p": [0.60, 0.30, 0.10], "P_fuel": 1.5, "C_labor": 25.0, "K": 50.0, "umbral_D": 100.0, "tarifa_km": 0.0,
            "factor_G": 1.0, "desplazar": 0.0, "externalizar": False, "coste_ext": 120.0, "reduccion_punta": 0.0,
            "coste_impl": 0.0, "descripcion": ""}


def escenario(**cambios):
    e = copy.deepcopy(ESC_BASE)
    e.update(cambios)
    return e


# ==============================================================================================
# Teoría: tipos de pedido y sus momentos
# ==============================================================================================
def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _poisson_pmf(mu, kmax=None):
    kmax = kmax or int(mu + 12 * math.sqrt(mu) + 25)
    k = np.arange(kmax + 1)
    if _stats is not None:
        return k, _stats.poisson.pmf(k, mu)
    lp = k * math.log(mu) - mu - np.array([math.lgamma(x + 1) for x in k])     # pragma: no cover
    return k, np.exp(lp)


def momentos_D(dD, c):
    """E(D), E(D²), P(D > c), E[(D−c)⁺], E[((D−c)⁺)²], E[D·I(D>c)], E[D·(D−c)⁺]."""
    if dD["tipo"] == "poisson":
        k, p = _poisson_pmf(dD["mu"])
        exc = np.maximum(k - c, 0.0)
        ind = (k > c).astype(float)
        return {"ED": float(p @ k), "ED2": float(p @ k ** 2), "PI": float(p @ ind), "E1": float(p @ exc),
                "E2": float(p @ exc ** 2), "EDI": float(p @ (k * ind)), "EDE": float(p @ (k * exc))}
    mu, sd = dD["mu"], dD["sd"]
    z = (c - mu) / sd
    Q, f = 1 - Phi(z), phi(z)
    E1 = (mu - c) * Q + sd * f
    E2 = ((mu - c) ** 2 + sd ** 2) * Q + (mu - c) * sd * f
    return {"ED": mu, "ED2": mu ** 2 + sd ** 2, "PI": Q, "E1": E1, "E2": E2, "EDI": mu * Q + sd * f, "EDE": E2 + c * E1}


def tipo_pedido(com, esc, ruta, franja=None, diferencia=False):
    """Parámetros de un tipo de pedido (ruta y, opcionalmente, franja de entrega). Y = a·D·G + b·M + peaje(D) (+ cte).
    Con `diferencia=True` describe Δ = Y_con − Y_sin = P·D·G·(f_G − 1) + C·M·(f_M − 1)."""
    g0, g1 = com["G"][ruta]
    fprov = esc["factor_G"]
    fG = com["f_consumo"][franja][ruta] if franja else 1.0
    fM = com["f_retraso"][franja][ruta] if franja else 1.0
    if franja == "Punta" and esc.get("reduccion_punta"):
        fM = fM * (1 - esc["reduccion_punta"])
    tp = {"ruta": ruta, "D": com["D"][ruta], "g0": g0 * fprov, "g1": g1 * fprov, "m": com["M"][ruta],
          "a": esc["P_fuel"] * fG, "b": esc["C_labor"] * fM, "K": esc["K"], "t": esc["tarifa_km"], "c": esc["umbral_D"],
          "const": None}
    if diferencia:
        tp.update(a=esc["P_fuel"] * (fG - 1), b=esc["C_labor"] * (fM - 1), K=0.0, t=0.0)
    if ruta == 3 and esc.get("externalizar"):
        tp["const"] = 0.0 if diferencia else float(esc["coste_ext"])
    return tp


def momentos_tipo(tp):
    """Media de cada término (combustible, mano de obra, peaje, externalización) y matriz E[Xi·Xj]."""
    E = np.zeros(4); M2 = np.zeros((4, 4))
    if tp["const"] is not None:
        E[3] = tp["const"]; M2[3, 3] = tp["const"] ** 2
        return E, M2
    md = momentos_D(tp["D"], tp["c"])
    g0, g1 = tp["g0"], tp["g1"]
    EG, EG2 = (g0 + g1) / 2, (g0 * g0 + g0 * g1 + g1 * g1) / 3
    a, b, m, K, t = tp["a"], tp["b"], tp["m"], tp["K"], tp["t"]
    Ep = K * md["PI"] + t * md["E1"]
    Ep2 = K * K * md["PI"] + 2 * K * t * md["E1"] + t * t * md["E2"]
    EDp = K * md["EDI"] + t * md["EDE"]
    E[:3] = [a * md["ED"] * EG, b * m, Ep]
    M2[0, 0] = a * a * md["ED2"] * EG2
    M2[1, 1] = b * b * 2 * m * m
    M2[2, 2] = Ep2
    M2[0, 1] = M2[1, 0] = a * md["ED"] * EG * b * m
    M2[0, 2] = M2[2, 0] = a * EG * EDp
    M2[1, 2] = M2[2, 1] = b * m * Ep
    return E, M2


def _rejilla_D(dD, c, n=700):
    """Valores de D y sus probabilidades (Poisson exacta; normal por celdas con un borde en el umbral c)."""
    if dD["tipo"] == "poisson":
        k, p = _poisson_pmf(dD["mu"])
        return k.astype(float), p
    mu, sd = dD["mu"], dD["sd"]
    lo, hi = mu - 8.5 * sd, mu + 8.5 * sd
    if lo < c < hi:
        f = (c - lo) / (hi - lo)
        n1 = max(10, int(n * f)); n2 = max(10, n - n1)
        bordes = np.concatenate([np.linspace(lo, c, n1 + 1), np.linspace(c, hi, n2 + 1)[1:]])
    else:
        bordes = np.linspace(lo, hi, n + 1)
    F = np.array([Phi((x - mu) / sd) for x in bordes])
    F[0], F[-1] = 0.0, 1.0
    return np.maximum((bordes[:-1] + bordes[1:]) / 2, 0.0), np.diff(F)


def cdf_tipo(tp, y):
    """P(Y ≤ y) de un tipo de pedido, integrando de forma exacta en G (uniforme) y M (exponencial) y numéricamente
    en D."""
    y = np.atleast_1d(np.asarray(y, dtype=float))
    if tp["const"] is not None:
        return (y >= tp["const"]).astype(float)
    d, pd_ = _rejilla_D(tp["D"], tp["c"])
    toll = tp["K"] * (d > tp["c"]) + tp["t"] * np.maximum(d - tp["c"], 0.0)
    w = y[None, :] - toll[:, None]                       # (nd, ny)
    cc = (tp["a"] * d)[:, None] * np.ones_like(w)
    s = tp["b"] * tp["m"]
    g0, g1 = tp["g0"], tp["g1"]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if s <= 0:
            P = np.where(cc > 0, np.clip((w / np.where(cc > 0, cc, 1) - g0) / (g1 - g0), 0, 1), (w >= 0).astype(float))
        else:
            up = np.clip(np.where(cc > 0, w / np.where(cc > 0, cc, 1), g1), g0, g1)
            ok = (w > cc * g0)
            ccs = np.where(cc > 0, cc, 1.0)
            Fup = up - (s / ccs) * np.exp(-np.maximum(w - cc * up, 0) / s)
            F0 = g0 - (s / ccs) * np.exp(-np.maximum(w - cc * g0, 0) / s)
            P_int = np.where(ok, (Fup - F0) / (g1 - g0), 0.0)
            P_c0 = np.where(w > 0, 1 - np.exp(-np.maximum(w, 0) / s), 0.0)
            P = np.where(cc > 0, P_int, P_c0)
    P = np.clip(np.nan_to_num(P), 0, 1)
    return pd_ @ P


def mezcla(componentes):
    """componentes: lista de (peso, tipo). Devuelve E (4), Cov (4x4), E(Y), Var(Y)."""
    E = np.zeros(4); M2 = np.zeros((4, 4))
    for w, tp in componentes:
        e, m2 = momentos_tipo(tp)
        E += w * e; M2 += w * m2
    cov = M2 - np.outer(E, E)
    return E, cov, float(E.sum()), float(cov.sum())


# ==============================================================================================
# Teoría · Bloque A (una entrega, sin franjas)
# ==============================================================================================
def _rejilla_y(tipos):
    hi = 0.0
    saltos = []
    for tp in tipos:
        if tp["const"] is not None:
            hi = max(hi, tp["const"] * 1.2)
            saltos += [tp["const"] * (1 - 1e-9), tp["const"]]       # salto de la distribución en el coste fijo
            continue
        E, M2 = momentos_tipo(tp)
        sd = math.sqrt(max(M2.sum() - E.sum() ** 2, 0))
        hi = max(hi, E.sum() + 12 * sd)
    return np.unique(np.concatenate([np.linspace(0, hi, 2000), saltos]))


def teoria_A(com, esc):
    p = normalizar(esc["p"])
    tipos = {r: tipo_pedido(com, esc, r) for r in RUTAS}
    E, cov, EY, VY = mezcla([(p[r - 1], tipos[r]) for r in RUTAS])
    yg = _rejilla_y(tipos.values())
    cdfs = {r: cdf_tipo(tipos[r], yg) for r in RUTAS}
    F = sum(p[r - 1] * cdfs[r] for r in RUTAS)
    u = float(com["umbral_Y"])
    cola = {r: 1 - float(cdf_tipo(tipos[r], [u])[0]) for r in RUTAS}
    Pcola = float(sum(p[r - 1] * cola[r] for r in RUTAS))
    por_ruta = {}
    for r in RUTAS:
        e, c2, ey, vy = mezcla([(1.0, tipos[r])])
        por_ruta[r] = {"p": p[r - 1], "EY": ey, "sd": math.sqrt(max(vy, 0)), "CV": math.sqrt(max(vy, 0)) / ey if ey else 0.0,
                       "Pcola": cola[r], "peso_cola": p[r - 1] * cola[r] / Pcola if Pcola > 0 else 0.0,
                       "E_terminos": e}
    comp = cov.sum(axis=1) / VY if VY > 0 else np.zeros(4)        # contribución de cada término a Var(Y)
    return {"p": p, "EY": EY, "VY": VY, "sdY": math.sqrt(VY), "CV": math.sqrt(VY) / EY, "Pcola": Pcola,
            "P90": cuantil_rejilla(yg, F, 0.90), "P50": cuantil_rejilla(yg, F, 0.50), "P99": cuantil_rejilla(yg, F, 0.99),
            "E_terminos": E, "cov": cov, "contrib": comp, "por_ruta": por_ruta, "y": yg, "F": F, "tipos": tipos,
            "anual": EY * com["entregas_ano"]}


def cuantil_rejilla(y, F, q):
    i = int(np.searchsorted(F, q))
    if i <= 0:
        return float(y[0])
    if i >= len(y):
        return float(y[-1])
    f0, f1 = F[i - 1], F[i]
    return float(y[i - 1] + (q - f0) / (f1 - f0) * (y[i] - y[i - 1])) if f1 > f0 else float(y[i])


def normalizar(p):
    p = np.clip(np.asarray(p, dtype=float), 0, None)
    return p / p.sum() if p.sum() > 0 else np.full(len(p), 1 / len(p))


# ==============================================================================================
# Teoría · Bloque B (un día con franjas)
# ==============================================================================================
def _cuenta_hora(lam, cap):
    """Media y varianza de los pedidos atendidos en una hora (Poisson, con capacidad máxima opcional) y media de los
    rechazados."""
    if not cap:
        return lam, lam, 0.0
    k, p = _poisson_pmf(lam)
    at = np.minimum(k, cap)
    m = float(p @ at)
    return m, float(p @ at ** 2) - m * m, float(p @ np.maximum(k - cap, 0))


def componentes_hora(com, esc, franja):
    """Mezcla de los pedidos que llegan en una hora de la franja: (peso, tipo_con, tipo_sin, tipo_delta, franja de
    entrega). El desplazamiento lleva una fracción de los urbanos de Punta a Valle."""
    p = normalizar(esc["p"])
    out = []
    for r in RUTAS:
        partes = [(1.0, franja)]
        if franja == "Punta" and r == 1 and esc.get("desplazar"):
            s = float(esc["desplazar"])
            partes = [(1 - s, "Punta"), (s, "Valle")]
        for w, fe in partes:
            if w <= 0:
                continue
            out.append((p[r - 1] * w, tipo_pedido(com, esc, r, fe), tipo_pedido(com, esc, r, None),
                        tipo_pedido(com, esc, r, fe, diferencia=True), fe, r))
    return out


def teoria_B(com, esc, alpha=0.05):
    cap = int(com.get("capacidad") or 0)
    tot = {"con": [0.0, 0.0], "sin": [0.0, 0.0], "delta": [0.0, 0.0]}
    fr = {f: {"n": 0.0, "con": 0.0, "sin": 0.0, "delta": 0.0} for f in FRANJAS}
    pedidos_m = pedidos_v = rechazados = 0.0
    orden = {"con": [0.0, 0.0], "sin": [0.0, 0.0], "delta": [0.0, 0.0]}      # momentos por pedido (todo el día)
    for h in range(24):
        f = HORAS_FRANJA[h]
        lam = float(com["lambda"][f])
        m, v, rej = _cuenta_hora(lam, cap)
        pedidos_m += m; pedidos_v += v; rechazados += rej
        comps = componentes_hora(com, esc, f)
        for clave, idx in (("con", 1), ("sin", 2), ("delta", 3)):
            E1 = E2 = 0.0
            for c_ in comps:
                e, m2 = momentos_tipo(c_[idx])
                E1 += c_[0] * e.sum(); E2 += c_[0] * m2.sum()
                if clave == "con":
                    fr[c_[4]]["n"] += m * c_[0]
                fr[c_[4]][clave] += m * c_[0] * e.sum()
            tot[clave][0] += m * E1
            tot[clave][1] += m * (E2 - E1 * E1) + v * E1 * E1
            orden[clave][0] += m * E1; orden[clave][1] += m * E2
    out = {"pedidos": pedidos_m, "var_pedidos": pedidos_v, "rechazados": rechazados}
    for clave in tot:
        mu, var = tot[clave]
        out[f"dia_{clave}"] = mu
        out[f"sd_dia_{clave}"] = math.sqrt(max(var, 0))
        out[f"pedido_{clave}"] = orden[clave][0] / pedidos_m
        out[f"var_pedido_{clave}"] = orden[clave][1] / pedidos_m - (orden[clave][0] / pedidos_m) ** 2
    mu, sd = out["dia_con"], out["sd_dia_con"]
    z95 = 1.6448536269514722
    out["P95"] = mu + z95 * sd
    out["CVaR"] = mu + sd * phi(z95) / alpha
    out["Ppres"] = 1 - Phi((com["presupuesto"] - mu) / sd) if sd > 0 else float(mu > com["presupuesto"])
    out["pct_sobre"] = out["dia_delta"] / mu if mu else 0.0
    out["franjas"] = {f: {"n": d["n"], "Ycon": d["con"] / d["n"] if d["n"] else 0.0,
                          "Ysin": d["sin"] / d["n"] if d["n"] else 0.0, "delta": d["delta"] / d["n"] if d["n"] else 0.0,
                          "dia_con": d["con"], "dia_delta": d["delta"],
                          "cuota_sobre": d["delta"] / out["dia_delta"] if out["dia_delta"] else 0.0} for f, d in fr.items()}
    # independencia frente a simulación pareada (O8.4): Var(Δ) frente a Var(Y_con) + Var(Y_sin)
    out["var_pareada"] = out["var_pedido_delta"]
    out["var_independiente"] = out["var_pedido_con"] + out["var_pedido_sin"]
    # capacidad horaria para atender la demanda de Punta (O6.1)
    lamP = float(com["lambda"]["Punta"])
    if _stats is not None:
        out["cap95_hora"] = int(_stats.poisson.ppf(0.95, lamP))
        n_p = sum(1 for h in range(24) if HORAS_FRANJA[h] == "Punta")
        c = int(lamP)
        while _stats.poisson.cdf(c, lamP) ** n_p < 0.95:
            c += 1
        out["cap95_dia"] = c
    out["pedidos_poisson"] = sum(float(com["lambda"][HORAS_FRANJA[h]]) for h in range(24))
    return out


def reduccion_equilibrio(com, esc):
    """Reducción porcentual r del factor de retraso en Punta (f → f·(1 − r)) que iguala el sobrecoste de Punta y de
    Llano (O5.3). El sobrecoste de Punta es lineal en r."""
    e0 = dict(esc, reduccion_punta=0.0)
    b0 = teoria_B(com, e0)["franjas"]
    e1 = dict(esc, reduccion_punta=1.0)
    b1 = teoria_B(com, e1)["franjas"]
    sP0, sP1, sL = b0["Punta"]["dia_delta"], b1["Punta"]["dia_delta"], b0["Llano"]["dia_delta"]
    if sP0 == sP1:
        return None
    return (sP0 - sL) / (sP0 - sP1)


def teoria(com, esc):
    return {"A": teoria_A(com, esc), "B": teoria_B(com, esc), "r_equilibrio": reduccion_equilibrio(com, esc)}


# ==============================================================================================
# Simulación (transformada inversa con números aleatorios comunes)
# ==============================================================================================
def _ppf_norm(u):
    return _ndtri(u) if _ndtri is not None else _stats.norm.ppf(u)


def generar_pedidos(com, esc, U, franja=None):
    """Variables base y costes de los pedidos a partir de uniformes U (5 × n): U_T, U_D, U_G, U_M, U_desplazamiento.
    `franja` (array de nombres o None) determina los factores de tráfico; devuelve también la franja de entrega."""
    p = normalizar(esc["p"]); Fp = np.cumsum(p)
    n = U.shape[1]
    T = 1 + (U[0] > Fp[0]).astype(int) + (U[0] > Fp[1]).astype(int)
    D = np.zeros(n); G = np.zeros(n); M = np.zeros(n)
    for r in RUTAS:
        mk = T == r
        if not mk.any():
            continue
        dD = com["D"][r]
        if dD["tipo"] == "poisson":         # inversa de la Poisson con su función de distribución tabulada
            k, pk = _poisson_pmf(dD["mu"])
            Fk = np.cumsum(pk); Fk[-1] = 1.0
            D[mk] = np.minimum(np.searchsorted(Fk, U[1, mk], side="left"), len(Fk) - 1).astype(float)
        else:
            D[mk] = dD["mu"] + dD["sd"] * _ppf_norm(U[1, mk])
        g0, g1 = com["G"][r]
        G[mk] = (g0 + (g1 - g0) * U[2, mk]) * esc["factor_G"]
        M[mk] = -com["M"][r] * np.log1p(-U[3, mk])
    peaje = esc["K"] * (D > esc["umbral_D"]) + esc["tarifa_km"] * np.maximum(D - esc["umbral_D"], 0.0)
    x_fuel, x_lab = D * G * esc["P_fuel"], M * esc["C_labor"]
    out = {"T": T, "D": D, "G": G, "M": M, "peaje": peaje, "fuel": x_fuel, "labor": x_lab, "ext": np.zeros(n)}
    Ysin = x_fuel + x_lab + peaje
    if franja is not None:
        fe = np.array(franja, dtype=object)
        if esc.get("desplazar"):
            fe = np.where((fe == "Punta") & (T == 1) & (U[4] < esc["desplazar"]), "Valle", fe)
        fG = np.ones(n); fM = np.ones(n)
        for f in FRANJAS:
            for r in RUTAS:
                mk = (fe == f) & (T == r)
                fG[mk] = com["f_consumo"][f][r]
                fM[mk] = com["f_retraso"][f][r] * ((1 - esc["reduccion_punta"]) if (f == "Punta" and esc.get("reduccion_punta")) else 1.0)
        Ycon = x_fuel * fG + x_lab * fM + peaje
        out["franja"] = fe
    else:
        Ycon = Ysin
    if esc.get("externalizar"):
        mk = T == 3
        for k_ in ("fuel", "labor", "peaje"):
            out[k_] = np.where(mk, 0.0, out[k_])
        out["ext"] = np.where(mk, float(esc["coste_ext"]), 0.0)
        Ysin = np.where(mk, float(esc["coste_ext"]), Ysin)
        Ycon = np.where(mk, float(esc["coste_ext"]), Ycon)
    out["Ysin"], out["Ycon"], out["delta"] = Ysin, Ycon, Ycon - Ysin
    return out


def _tq(n, alpha=0.05):
    if n < 2:
        return float("nan")
    return float(_stats.t.ppf(1 - alpha / 2, n - 1)) if _stats is not None else 1.96


def simular_A(com, esc, nsim, replicas, semilla, progreso=None):
    """Bloque A: `replicas` macro-réplicas de `nsim` entregas (réplica i → semilla + i)."""
    u = float(com["umbral_Y"])
    filas, Ys, Ts, traza = [], [], [], None
    terms = []
    for r in range(int(replicas)):
        rng = np.random.default_rng(int(semilla) + r)
        U = rng.random((5, int(nsim)))
        o = generar_pedidos(com, esc, U)
        Y, T = o["Ysin"], o["T"]
        X = np.vstack([o["fuel"], o["labor"], o["peaje"], o["ext"]])
        cov = np.cov(X)
        vy = float(Y.var(ddof=1))
        fila = {"Réplica": r + 1, "Semilla": int(semilla) + r, "Entregas": len(Y), "E(Y)": float(Y.mean()),
                "σ(Y)": math.sqrt(vy), "Var(Y)": vy, "P(Y > umbral)": float((Y > u).mean()),
                "P90": float(np.quantile(Y, 0.90))}
        for k_, nom in enumerate(TERMINOS):
            fila[f"Contrib. {nom}"] = float(cov[k_].sum() / vy) if vy > 0 else 0.0
        for ruta in RUTAS:
            mk = T == ruta
            yr = Y[mk]
            fila[f"E(Y | {NOMBRE_RUTA[ruta]})"] = float(yr.mean()) if mk.any() else float("nan")
            fila[f"P(Y > umbral | {NOMBRE_RUTA[ruta]})"] = float((yr > u).mean()) if mk.any() else float("nan")
            fila[f"CV({NOMBRE_RUTA[ruta]})"] = float(yr.std(ddof=1) / yr.mean()) if mk.sum() > 1 else float("nan")
            fila[f"Peso en la cola · {NOMBRE_RUTA[ruta]}"] = float((mk & (Y > u)).sum() / max(1, (Y > u).sum()))
        filas.append(fila)
        Ys.append(Y); Ts.append(T)
        if r == 0:
            traza = pd.DataFrame({"Entrega": np.arange(1, N_DIAGRAMA + 1), "U_T": U[0, :N_DIAGRAMA],
                                  "Ruta": [NOMBRE_RUTA[t] for t in T[:N_DIAGRAMA]], "D (km)": o["D"][:N_DIAGRAMA],
                                  "G (L/km)": o["G"][:N_DIAGRAMA], "M (h)": o["M"][:N_DIAGRAMA],
                                  "Peaje (€)": o["peaje"][:N_DIAGRAMA] + o["ext"][:N_DIAGRAMA], "Y (€)": Y[:N_DIAGRAMA]})
        if progreso:
            progreso()
    return {"replicas": pd.DataFrame(filas).set_index("Réplica"), "Y": np.concatenate(Ys), "T": np.concatenate(Ts),
            "traza": traza}


def simular_B(com, esc, ndias, replicas, semilla, progreso=None):
    """Bloque B: `replicas` macro-réplicas de `ndias` días (réplica i → semilla + 1000 + i)."""
    cap = int(com.get("capacidad") or 0)
    franja_h = np.array([HORAS_FRANJA[h] for h in range(24)], dtype=object)
    lam_h = np.array([float(com["lambda"][f]) for f in franja_h])
    filas, dias_all = [], []
    acc = np.zeros(6)            # sumas pedido a pedido: Ycon, Ycon², Ysin, Ysin², Δ, Δ²
    n_ped = 0
    for r in range(int(replicas)):
        rng = np.random.default_rng(int(semilla) + 1000 + r)
        N = rng.poisson(np.tile(lam_h, int(ndias)))                       # pedidos por bloque día-hora
        dia_b = np.repeat(np.arange(int(ndias)), 24)
        fr_b = np.tile(franja_h, int(ndias))
        n_tot = int(N.sum())
        U = rng.random((5, n_tot))
        bloque = np.repeat(np.arange(len(N)), N)
        inicio = np.repeat(np.cumsum(N) - N, N)
        rank = np.arange(n_tot) - inicio
        atendido = rank < cap if cap else np.ones(n_tot, dtype=bool)
        o = generar_pedidos(com, esc, U, fr_b[bloque])
        dia = dia_b[bloque]
        w = atendido.astype(float)
        dc = np.bincount(dia, weights=o["Ycon"] * w, minlength=int(ndias))
        ds = np.bincount(dia, weights=o["Ysin"] * w, minlength=int(ndias))
        dd = np.bincount(dia, weights=o["delta"] * w, minlength=int(ndias))
        dn = np.bincount(dia, weights=w, minlength=int(ndias))
        drej = np.bincount(dia, weights=1 - w, minlength=int(ndias))
        q = float(np.quantile(dc, 0.95))
        cola = np.sort(dc)[-max(1, int(math.ceil(0.05 * len(dc)))):]
        fila = {"Réplica": r + 1, "Semilla": int(semilla) + 1000 + r, "Días": int(ndias),
                "Coste diario con tráfico": float(dc.mean()), "Coste diario sin tráfico": float(ds.mean()),
                "Sobrecoste diario": float(dd.mean()), "% sobrecoste": float(dd.mean() / dc.mean()),
                "P95 coste diario": q, "CVaR 5 % coste diario": float(cola.mean()),
                "P(coste diario > presupuesto)": float((dc > com["presupuesto"]).mean()),
                "Pedidos por día": float(dn.mean() + drej.mean()), "Var. pedidos por día": float((dn + drej).var(ddof=1)),
                "Rechazados por día": float(drej.mean())}
        fe = o["franja"]
        tot_d = float((o["delta"] * w).sum())
        for f in FRANJAS:
            mk = (fe == f) & atendido
            nn = max(1, int(mk.sum()))
            fila[f"E(Y_con) · {f}"] = float(o["Ycon"][mk].sum() / nn)
            fila[f"E(Y_sin) · {f}"] = float(o["Ysin"][mk].sum() / nn)
            fila[f"E(Δ) · {f}"] = float(o["delta"][mk].sum() / nn)
            fila[f"Cuota del sobrecoste · {f}"] = float(o["delta"][mk].sum() / tot_d) if tot_d else 0.0
        filas.append(fila)
        dias_all.append(pd.DataFrame({"Réplica": r + 1, "Día": np.arange(1, int(ndias) + 1), "Coste con tráfico": dc,
                                      "Coste sin tráfico": ds, "Sobrecoste": dd, "Pedidos": dn + drej, "Rechazados": drej}))
        a_ = atendido
        acc += [o["Ycon"][a_].sum(), (o["Ycon"][a_] ** 2).sum(), o["Ysin"][a_].sum(), (o["Ysin"][a_] ** 2).sum(),
                o["delta"][a_].sum(), (o["delta"][a_] ** 2).sum()]
        n_ped += int(a_.sum())
        if progreso:
            progreso()
    var = lambda s1, s2: s2 / n_ped - (s1 / n_ped) ** 2
    return {"replicas": pd.DataFrame(filas).set_index("Réplica"), "dias": pd.concat(dias_all, ignore_index=True),
            "var_pareada": var(acc[4], acc[5]), "var_independiente": var(acc[0], acc[1]) + var(acc[2], acc[3]),
            "n_pedidos": n_ped}


def resumen(rep, teo_por_col):
    """Media de las réplicas, IC95 % (t de Student) y valor teórico para cada columna indicada."""
    n = len(rep)
    tq = _tq(n)
    filas = []
    for nombre, (col, teo) in teo_por_col.items():
        v = rep[col].astype(float).dropna()
        m = float(v.mean())
        se = float(v.std(ddof=1)) / math.sqrt(len(v)) if len(v) > 1 else float("nan")
        lo, hi = m - tq * se, m + tq * se
        ok = None
        if teo is not None and lo == lo:
            tol = 1e-9 * max(1.0, abs(teo))
            ok = bool(lo - tol <= teo <= hi + tol)
        filas.append({"KPI": nombre, "Teórico": teo, "Simulado · media": m, "IC95 inf": lo, "IC95 sup": hi,
                      "Error relativo": (m - teo) / abs(teo) if (teo is not None and abs(teo) > 1e-12) else None,
                      "Teórico en IC95": ok})
    return pd.DataFrame(filas).set_index("KPI")


def resumen_A(mc, tA, umbral):
    u = num(umbral, 0)
    d = {"Coste esperado E(Y) (€)": ("E(Y)", tA["EY"]),
         "Desviación típica σ(Y) (€)": ("σ(Y)", tA["sdY"]),
         f"P(Y > {u} €)": ("P(Y > umbral)", tA["Pcola"]),
         "Percentil 90 de Y (€)": ("P90", tA["P90"])}
    for r in RUTAS:
        nr = NOMBRE_RUTA[r]
        d[f"E(Y | {nr}) (€)"] = (f"E(Y | {nr})", tA["por_ruta"][r]["EY"])
        d[f"P(Y > {u} € | {nr})"] = (f"P(Y > umbral | {nr})", tA["por_ruta"][r]["Pcola"])
        d[f"CV(Y | {nr})"] = (f"CV({nr})", tA["por_ruta"][r]["CV"])
    for r in RUTAS:
        d[f"Peso de {NOMBRE_RUTA[r]} en la cola Y > {u} €"] = (f"Peso en la cola · {NOMBRE_RUTA[r]}", tA["por_ruta"][r]["peso_cola"])
    for k_, nom in enumerate(TERMINOS):
        d[f"Contribución a Var(Y) · {nom}"] = (f"Contrib. {nom}", float(tA["contrib"][k_]))
    return resumen(mc["replicas"], d)


def resumen_B(mc, tB):
    d = {"Coste diario esperado con tráfico (€)": ("Coste diario con tráfico", tB["dia_con"]),
         "Coste diario esperado sin tráfico (€)": ("Coste diario sin tráfico", tB["dia_sin"]),
         "Sobrecoste diario por tráfico (€)": ("Sobrecoste diario", tB["dia_delta"]),
         "Sobrecoste en % del coste diario": ("% sobrecoste", tB["pct_sobre"]),
         "Percentil 95 del coste diario (€) · aprox. normal": ("P95 coste diario", tB["P95"]),
         "CVaR 5 % del coste diario (€) · aprox. normal": ("CVaR 5 % coste diario", tB["CVaR"]),
         "P(coste diario > presupuesto) · aprox. normal": ("P(coste diario > presupuesto)", tB["Ppres"]),
         "Pedidos por día": ("Pedidos por día", tB["pedidos_poisson"]),
         "Varianza de los pedidos por día (Poisson)": ("Var. pedidos por día", tB["pedidos_poisson"]),
         "Pedidos rechazados por día": ("Rechazados por día", tB["rechazados"])}
    for f in FRANJAS:
        d[f"E(Y_con) por pedido · {f} (€)"] = (f"E(Y_con) · {f}", tB["franjas"][f]["Ycon"])
        d[f"E(Y_sin) por pedido · {f} (€)"] = (f"E(Y_sin) · {f}", tB["franjas"][f]["Ysin"])
        d[f"E(Δ) sobrecoste por pedido · {f} (€)"] = (f"E(Δ) · {f}", tB["franjas"][f]["delta"])
    for f in FRANJAS:
        d[f"Cuota del sobrecoste diario · {f}"] = (f"Cuota del sobrecoste · {f}", tB["franjas"][f]["cuota_sobre"])
    return resumen(mc["replicas"], d)


# ==============================================================================================
# Objetivos
# ==============================================================================================
OBJ_NOMBRES = {"EY_MAX": "Coste esperado por entrega E(Y)", "PCOLA_MAX": "Probabilidad de entrega cara P(Y > umbral)",
               "DIA_MAX": "Coste diario esperado con tráfico", "PPRES_MAX": "P(coste diario > presupuesto)"}


def evaluar_objetivos(t, obj, com):
    A, B = t["A"], t["B"]
    filas = []
    def fila(clave, valor, txt, cumple):
        filas.append({"Objetivo": OBJ_NOMBRES[clave], "Valor": valor, "Umbral": txt, "Cumple": "Sí" if cumple else "No"})
    if obj.get("EY_MAX") is not None:
        fila("EY_MAX", A["EY"], f"≤ {eur(obj['EY_MAX'], 2)}", A["EY"] <= obj["EY_MAX"] + 1e-9)
    if obj.get("PCOLA_MAX") is not None:
        fila("PCOLA_MAX", A["Pcola"], f"≤ {pct(obj['PCOLA_MAX'], 1)}", A["Pcola"] <= obj["PCOLA_MAX"] + 1e-12)
    if obj.get("DIA_MAX") is not None:
        fila("DIA_MAX", B["dia_con"], f"≤ {eur(obj['DIA_MAX'])}", B["dia_con"] <= obj["DIA_MAX"] + 1e-9)
    if obj.get("PPRES_MAX") is not None:
        fila("PPRES_MAX", B["Ppres"], f"≤ {pct(obj['PPRES_MAX'], 1)}", B["Ppres"] <= obj["PPRES_MAX"] + 1e-12)
    return pd.DataFrame(filas, columns=["Objetivo", "Valor", "Umbral", "Cumple"]).set_index("Objetivo")


# ==============================================================================================
# Análisis de un escenario, comparación y diferencias pareadas frente al escenario base
# ==============================================================================================
def analizar_escenario(esc, com, aj, progreso=None):
    t = teoria(com, esc)
    mA = simular_A(com, esc, aj["nsim"], aj["replicas"], aj["semilla"], progreso)
    mB = simular_B(com, esc, aj["ndias"], aj["replicas_B"], aj["semilla"], progreso)
    return {"esc": dict(esc), "teoria": t, "mcA": mA, "mcB": mB, "resA": resumen_A(mA, t["A"], com["umbral_Y"]),
            "resB": resumen_B(mB, t["B"]), "objetivos": evaluar_objetivos(t, com["objetivos"], com)}


def diferencias(res, base, com):
    """Ahorro de cada escenario frente al base: teórico y simulado (diferencias pareadas por réplica, mismas semillas)."""
    if base not in res:
        return None
    rb = res[base]
    filas = []
    for n, r in res.items():
        if n == base:
            continue
        dA = rb["mcA"]["replicas"]["E(Y)"] - r["mcA"]["replicas"]["E(Y)"]
        dB = rb["mcB"]["replicas"]["Coste diario con tráfico"] - r["mcB"]["replicas"]["Coste diario con tráfico"]
        dS = rb["mcB"]["replicas"]["Sobrecoste diario"] - r["mcB"]["replicas"]["Sobrecoste diario"]
        tq = _tq(len(dA)); tqb = _tq(len(dB))
        ic = lambda v, t_: (float(v.mean() - t_ * v.std(ddof=1) / math.sqrt(len(v))),
                            float(v.mean() + t_ * v.std(ddof=1) / math.sqrt(len(v))))
        ta, tb = rb["teoria"], r["teoria"]
        impl = float(r["esc"].get("coste_impl") or 0.0)
        filas.append({"Escenario": n,
                      "Ahorro por entrega · teórico (€)": ta["A"]["EY"] - tb["A"]["EY"],
                      "Ahorro por entrega · simulado (€)": float(dA.mean()),
                      "IC95 ahorro por entrega (€)": ic(dA, tq),
                      "Ahorro anual · teórico (€)": (ta["A"]["EY"] - tb["A"]["EY"]) * com["entregas_ano"],
                      "Ahorro anual · simulado (€)": float(dA.mean()) * com["entregas_ano"],
                      "Ahorro diario · teórico (€)": ta["B"]["dia_con"] - tb["B"]["dia_con"],
                      "Ahorro diario · simulado (€)": float(dB.mean()),
                      "IC95 ahorro diario (€)": ic(dB, tqb),
                      "Ahorro en sobrecoste diario · teórico (€)": ta["B"]["dia_delta"] - tb["B"]["dia_delta"],
                      "Ahorro en sobrecoste diario · simulado (€)": float(dS.mean()),
                      "Coste de implantación (€/día)": impl,
                      "Ahorro diario neto · teórico (€)": ta["B"]["dia_con"] - tb["B"]["dia_con"] - impl,
                      "Cambio en σ(Y) · teórico (€)": tb["A"]["sdY"] - ta["A"]["sdY"],
                      "Cambio en P(Y > umbral) · teórico": tb["A"]["Pcola"] - ta["A"]["Pcola"]})
    return pd.DataFrame(filas).set_index("Escenario") if filas else None


def tabla_comparativa(res, com):
    cols = {}
    u = num(com["umbral_Y"], 0)
    for n, r in res.items():
        A, B, e = r["teoria"]["A"], r["teoria"]["B"], r["esc"]
        ra, rb = r["resA"], r["resB"]
        cols[n] = {
            "Parámetro · p(Urbana / Regional / Premium)": " / ".join(num(x, 2) for x in A["p"]),
            "Parámetro · P_fuel / C_labor": f"{num(e['P_fuel'], 2)} / {num(e['C_labor'], 1)}",
            "Parámetro · peaje (K fijo / €/km sobre umbral)": f"{num(e['K'], 0)} / {num(e['tarifa_km'], 2)}",
            "Parámetro · consumo del proveedor (×G)": num(e["factor_G"], 2),
            "Parámetro · urbanos de Punta a Valle": pct(e["desplazar"], 0),
            "Parámetro · Premium externalizado": f"sí ({eur(e['coste_ext'])})" if e["externalizar"] else "no",
            "Teórico · E(Y) por entrega (€)": A["EY"], "Teórico · σ(Y) (€)": A["sdY"],
            f"Teórico · P(Y > {u} €)": A["Pcola"], "Teórico · P90 de Y (€)": A["P90"],
            "Teórico · coste anual (€)": A["anual"],
            "Teórico · coste diario con tráfico (€)": B["dia_con"], "Teórico · sobrecoste diario (€)": B["dia_delta"],
            "Teórico · % sobrecoste": B["pct_sobre"], "Teórico · P95 coste diario (€)": B["P95"],
            "Teórico · P(coste diario > presupuesto)": B["Ppres"],
            "Simulado · E(Y) por entrega (€)": ra.loc["Coste esperado E(Y) (€)", "Simulado · media"],
            f"Simulado · P(Y > {u} €)": ra.loc[f"P(Y > {u} €)", "Simulado · media"],
            "Simulado · coste diario con tráfico (€)": rb.loc["Coste diario esperado con tráfico (€)", "Simulado · media"],
            "Simulado · sobrecoste diario (€)": rb.loc["Sobrecoste diario por tráfico (€)", "Simulado · media"],
            "Simulado · P95 coste diario (€)": rb.loc["Percentil 95 del coste diario (€) · aprox. normal", "Simulado · media"],
            "Objetivos cumplidos": f"{(r['objetivos']['Cumple'] == 'Sí').sum()} de {len(r['objetivos'])}",
        }
    return pd.DataFrame(cols)


def tabla_objetivos(res):
    return pd.DataFrame({n: r["objetivos"]["Cumple"] for n, r in res.items()})


# ==============================================================================================
# O3.1 · Optimización del vector de probabilidades de las rutas
# ==============================================================================================
def optimizar_rutas(mu, p0, reduccion):
    """min ‖p − p₀‖²  s.a.  Σ pⱼ·μⱼ ≤ (1 − r)·Σ p₀ⱼ·μⱼ,  Σ pⱼ = 1,  pⱼ ≥ 0  (μⱼ = E(Y | T = j)).
    Se resuelve sobre una rejilla fina del símplex (paso 0,001) y se devuelve la solución más cercana a p₀."""
    mu = np.asarray(mu, dtype=float); p0 = normalizar(p0)
    objetivo = (1 - reduccion) * float(p0 @ mu)
    paso = 0.001
    g = np.arange(0, 1 + 1e-9, paso)
    p1, p2 = np.meshgrid(g, g, indexing="ij")
    p3 = 1 - p1 - p2
    ok = p3 >= -1e-12
    E = p1 * mu[0] + p2 * mu[1] + np.clip(p3, 0, 1) * mu[2]
    fact = ok & (E <= objetivo + 1e-9)
    if not fact.any():
        return None
    dist = (p1 - p0[0]) ** 2 + (p2 - p0[1]) ** 2 + (p3 - p0[2]) ** 2
    dist = np.where(fact, dist, np.inf)
    i, j = np.unravel_index(int(np.argmin(dist)), dist.shape)
    p = np.array([g[i], g[j], max(0.0, 1 - g[i] - g[j])])
    return {"p": p, "E": float(p @ mu), "objetivo": objetivo, "E0": float(p0 @ mu), "distancia": float(math.sqrt(dist[i, j]))}


def analizar_optimizacion(com, esc, reduccion, mcA=None, nsim=10000, replicas=5, semilla=1000):
    """Resuelve O3.1 con las medias teóricas por ruta y con las estimadas por simulación, y verifica la solución
    re-simulando con las mismas semillas."""
    tA = teoria_A(com, esc)
    mu_teo = [tA["por_ruta"][r]["EY"] for r in RUTAS]
    sol_teo = optimizar_rutas(mu_teo, esc["p"], reduccion)
    if mcA is None:
        mcA = simular_A(com, esc, nsim, replicas, semilla)
    Y, T = mcA["Y"], mcA["T"]
    mu_sim = [float(Y[T == r].mean()) for r in RUTAS]
    sol_sim = optimizar_rutas(mu_sim, esc["p"], reduccion)
    verif = None
    if sol_sim is not None:
        e2 = dict(esc, p=list(sol_sim["p"]))
        m2 = simular_A(com, e2, nsim, replicas, semilla)
        v = m2["replicas"]["E(Y)"]
        tq = _tq(len(v)); se = float(v.std(ddof=1)) / math.sqrt(len(v))
        verif = {"E": float(v.mean()), "lo": float(v.mean()) - tq * se, "hi": float(v.mean()) + tq * se,
                 "E_teo": teoria_A(com, e2)["EY"]}
    return {"reduccion": reduccion, "mu_teo": mu_teo, "mu_sim": mu_sim, "teo": sol_teo, "sim": sol_sim, "verif": verif,
            "p0": normalizar(esc["p"])}


# ==============================================================================================
# O2.2 · Sensibilidad de la varianza a los parámetros económicos; sensibilidad del estimador
# ==============================================================================================
PARAM_ECON = {"P_fuel": "P_fuel (€/L)", "C_labor": "C_labor (€/h)", "K": "K (€)"}


def sensibilidad_economica(com, esc, variaciones, nsim=20000, semilla=1000):
    """Var(Y) y E(Y) al variar cada parámetro económico manteniendo los demás fijos: teórico (exacto) y simulado con
    números aleatorios comunes (una muestra de nsim entregas)."""
    rng = np.random.default_rng(int(semilla))
    U = rng.random((5, int(nsim)))
    filas = []
    for par in PARAM_ECON:
        for v in variaciones:
            e2 = dict(esc, **{par: esc[par] * (1 + v)})
            E, cov, EY, VY = mezcla([(pp, tipo_pedido(com, e2, r)) for pp, r in zip(normalizar(e2["p"]), RUTAS)])
            Y = generar_pedidos(com, e2, U)["Ysin"]
            filas.append({"Parámetro": PARAM_ECON[par], "Variación": v, "Valor": e2[par],
                          "Teórico · Var(Y)": VY, "Simulado · Var(Y)": float(Y.var(ddof=1)),
                          "Teórico · σ(Y)": math.sqrt(VY), "Teórico · E(Y)": EY, "Simulado · E(Y)": float(Y.mean())})
    df = pd.DataFrame(filas)
    # elasticidad de Var(Y) en el punto base: d ln Var / d ln parámetro (diferencia centrada ±1 %)
    elas = {}
    for par in PARAM_ECON:
        vs = []
        for h in (-0.01, 0.01):
            e2 = dict(esc, **{par: esc[par] * (1 + h)})
            vs.append(mezcla([(pp, tipo_pedido(com, e2, r)) for pp, r in zip(normalizar(e2["p"]), RUTAS)])[3])
        elas[PARAM_ECON[par]] = (math.log(vs[1]) - math.log(vs[0])) / (math.log(1.01) - math.log(0.99)) if min(vs) > 0 else 0.0
    return df, elas


def valores_barrido(vmin, vmax, puntos):
    v = np.geomspace(max(1, vmin), max(vmin + 1, vmax), max(2, int(puntos)))
    return sorted(set(int(round(x)) for x in v))


def sensibilidad_estimador(com, esc, metrica, valores, replicas, semilla, progreso=None):
    """E(Y) por entrega frente a nsim (Bloque A) o coste diario medio frente al nº de días (Bloque B)."""
    t = teoria(com, esc)
    teo = t["A"]["EY"] if metrica == "entrega" else t["B"]["dia_con"]
    sd1 = t["A"]["sdY"] if metrica == "entrega" else t["B"]["sd_dia_con"]
    tq = _tq(int(replicas))
    filas, k, tot = [], 0, len(valores) * int(replicas)
    for n in valores:
        medias = []
        for r in range(int(replicas)):
            if metrica == "entrega":
                rng = np.random.default_rng(int(semilla) + r)
                medias.append(float(generar_pedidos(com, esc, rng.random((5, n)))["Ysin"].mean()))
            else:
                m = simular_B(com, esc, n, 1, int(semilla) + r)
                medias.append(float(m["replicas"]["Coste diario con tráfico"].iloc[0]))
            k += 1
            if progreso:
                progreso(k, tot)
        m = float(np.mean(medias)); sdr = float(np.std(medias, ddof=1)) if len(medias) > 1 else 0.0
        se = sdr / math.sqrt(len(medias))
        filas.append({"n": n, "Media de las réplicas": m, "IC95 inf": m - tq * se, "IC95 sup": m + tq * se,
                      "σ entre réplicas": sdr, "σ teórica del estimador": sd1 / math.sqrt(n), "Teórico": teo,
                      "Error relativo": (m - teo) / teo, "Medias por réplica": medias})
    return pd.DataFrame(filas).set_index("n")


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


def eur(x, d=0):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return num(x, d) + " €"


def fmt_val(k, v):
    """Formato según el nombre del KPI (fila)."""
    if isinstance(v, str) or v is None:
        return "—" if v is None else v
    if isinstance(v, tuple):
        return f"[{fmt_val(k, v[0])}; {fmt_val(k, v[1])}]"
    if "(€)" in k or "(€/día)" in k:
        return eur(v, 0) if any(x in k for x in ("anual", "diario", "P95", "CVaR", "Percentil 95", "día")) else eur(v, 2)
    if k.startswith(("P(", "Teórico · P(", "Simulado · P(")) or "%" in k or "Cuota" in k or "Contribución" in k \
            or "Peso" in k or "Cambio en P" in k:
        return pct(v, 2)
    if k.startswith("CV") or "CV(" in k:
        return num(v, 3)
    if "Varianza" in k or "Pedidos" in k or "pedidos" in k or "Rechazados" in k:
        return num(v, 1)
    if "anual" in k or "diario" in k or "P95" in k or "CVaR" in k or "Percentil 95" in k:
        return eur(v, 0)
    return eur(v, 2) if "€" in k else num(v, 3)


def fmt_kpi(k, c, v):
    return fmt_val(k, v)


def fmt_res(fila, col, v):
    if col == "Teórico en IC95":
        return "—" if v is None else ("Sí" if v else "No")
    if col == "Error relativo":
        return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else pct(v, 2)
    return fmt_val(fila, v)


def fmt_dif(fila, col, v):
    return fmt_val(col, v)


def fmt_obj(fila, col, v):
    if col == "Valor":
        return pct(v, 2) if fila.startswith(("Probabilidad", "P(")) else eur(v, 2 if "entrega" in fila else 0)
    return str(v)


def texto_escenario(nombre, r, com):
    A, B = r["teoria"]["A"], r["teoria"]["B"]
    okA = r["resA"]["Teórico en IC95"].dropna(); okB = r["resB"]["Teórico en IC95"].dropna()
    fuera = list(okA[~okA].index) + list(okB[~okB].index)
    n_kpi = len(okA) + len(okB)
    val = ("la simulación reproduce los valores teóricos (todos dentro del IC95 %)" if not fuera else
           f"la simulación reproduce los valores teóricos en {n_kpi - len(fuera)} de los {n_kpi} KPIs; quedan fuera del "
           f"IC95 % {', '.join(fuera)}, con errores relativos pequeños (con tantos KPIs es esperable que alguno quede "
           "fuera por azar)")
    pr = A["por_ruta"]
    mayor_cola = max(RUTAS, key=lambda x: pr[x]["peso_cola"])
    mayor_cv = max(RUTAS, key=lambda x: pr[x]["CV"])
    dom = int(np.argmax(A["contrib"]))
    txt = (f"{nombre}: una entrega cuesta en media E(Y) = {eur(A['EY'], 2)} (σ = {eur(A['sdY'], 2)}); el 10 % de las "
           f"entregas cuesta más de {eur(A['P90'], 2)} y P(Y > {eur(com['umbral_Y'])}) = {pct(A['Pcola'], 2)}. La ruta "
           f"{NOMBRE_RUTA[mayor_cola]} aporta el {pct(pr[mayor_cola]['peso_cola'], 0)} de las entregas caras y la "
           f"{NOMBRE_RUTA[mayor_cv]} es la de mayor incertidumbre relativa (CV = {num(pr[mayor_cv]['CV'], 2)}); el término "
           f"«{TERMINOS[dom]}» explica el {pct(A['contrib'][dom], 0)} de la varianza. Un día cuesta "
           f"{eur(B['dia_con'])} con tráfico ({eur(B['dia_sin'])} sin él): el tráfico añade {eur(B['dia_delta'])} "
           f"({pct(B['pct_sobre'], 1)}); P95 = {eur(B['P95'])} y P(coste > {eur(com['presupuesto'])}) = "
           f"{pct(B['Ppres'], 2)} (valores teóricos); {val}.")
    n_ok = (r["objetivos"]["Cumple"] == "Sí").sum()
    txt += f" Cumple {n_ok} de los {len(r['objetivos'])} objetivos."
    return txt


def texto_recomendacion(res, base, com):
    filas = [(n, (r["objetivos"]["Cumple"] == "Sí").sum(), r["teoria"], float(r["esc"].get("coste_impl") or 0.0))
             for n, r in res.items()]
    max_ok = max(f[1] for f in filas)
    mejor = sorted([f for f in filas if f[1] == max_ok], key=lambda f: f[2]["B"]["dia_con"] + f[3])[0]
    txt = (f"Se recomienda «{mejor[0]}»: es el escenario que cumple más objetivos ({mejor[1]} de "
           f"{len(res[mejor[0]]['objetivos'])}) y, entre ellos, el de menor coste diario ({eur(mejor[2]['B']['dia_con'])}"
           + (f" más {eur(mejor[3])} de implantación" if mejor[3] else "") + ").")
    if base in res and mejor[0] != base:
        tb = res[base]["teoria"]
        txt += (f" Frente a «{base}», ahorra {eur(tb['B']['dia_con'] - mejor[2]['B']['dia_con'] - mejor[3])} netos al día y "
                f"{eur(tb['A']['EY'] - mejor[2]['A']['EY'], 2)} por entrega "
                f"({eur((tb['A']['EY'] - mejor[2]['A']['EY']) * com['entregas_ano'])} al año con "
                f"{num(com['entregas_ano'], 0)} entregas).")
    barato = min(filas, key=lambda f: f[2]["A"]["EY"])
    if barato[0] != mejor[0]:
        txt += f" El de menor coste por entrega es «{barato[0]}» ({eur(barato[2]['A']['EY'], 2)})."
    return txt


KPIS_CUADRO = [
    ("Coste medio por entrega E(Y) con IC95 %", "nivel de coste unitario; base de la tarificación"),
    ("Percentil 90 por entrega y P(Y > umbral)", "riesgo de entregas caras; dimensiona márgenes y reclamaciones"),
    ("Coste diario medio con y sin tráfico", "presupuesto operativo y peso de la congestión"),
    ("Sobrecoste por tráfico (€/día y % del coste) por franja", "dónde actuar: Punta frente a Llano y Valle"),
    ("Percentil 95 y CVaR 5 % del coste diario", "riesgo financiero de los días malos (colchón de tesorería)"),
    ("P(coste diario > presupuesto)", "probabilidad de desviarse del presupuesto aprobado"),
    ("Pedidos por día y por hora punta; pedidos rechazados", "dimensionamiento de flota y nivel de servicio"),
    ("Mezcla de rutas (p Urbana / Regional / Premium) y coste por ruta", "palanca comercial y de externalización"),
]


def una_entrega(com, esc, uT, uD, uG, uM, franja="Punta"):
    """Una entrega con uniformes dados (mecanismo de composición) y su coste por términos sin y con tráfico."""
    U = np.array([[uT], [uD], [uG], [uM], [1.0]])
    o = generar_pedidos(com, esc, U)
    T = int(o["T"][0])
    fG = com["f_consumo"][franja][T]
    fM = com["f_retraso"][franja][T] * ((1 - esc["reduccion_punta"]) if (franja == "Punta" and esc.get("reduccion_punta")) else 1.0)
    sin = [float(o["fuel"][0]), float(o["labor"][0]), float(o["peaje"][0]), float(o["ext"][0])]
    con = [sin[0] * fG, sin[1] * fM, sin[2], sin[3]]
    if esc.get("externalizar") and T == 3:
        con = sin
    return {"T": T, "D": float(o["D"][0]), "G": float(o["G"][0]), "M": float(o["M"][0]), "sin": sin, "con": con,
            "Ysin": sum(sin), "Ycon": sum(con), "fG": fG, "fM": fM}
