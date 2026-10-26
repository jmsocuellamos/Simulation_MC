# -*- coding: utf-8 -*-
"""EC2 · Farmacoeconomía de la terapia para la superbacteria «Staph-X».

Bloque A · un paciente (método de composición):  fenotipo T ∈ {1, 2, 3} (Rápido, Estándar, Persistente);
    L | T ~ Gamma(α_T, escala), R | T ~ U(r₀, r₁), ε ~ N(1; 0,2);  D = max(0, ln(L + 1)/R + ε);
    Y = C_fijo + C_día·D + C_UCI·I{D > umbral}  (con contrato de pago por resultados: −ρ·C_fijo·I{D > umbral}).
Bloque B · un brote de 90 días:  N_d ~ Poisson(λ_fase) ingresos al día con mezcla de fenotipos por fase; cada día
    solo B camas UCI propias para los pacientes complicados; el resto, a camas de alquiler (si las hay) o a una clínica
    externa. Y_con e Y_sin se calculan con los mismos pacientes (simulación pareada) y Δ = Y_con − Y_sin.

Teoría exacta: la distribución de Z = ln(L + 1)/R se discretiza finamente (Gamma por cuantiles y uniforme) y, sobre
ella, D | Z es una normal truncada en 0: momentos, P(D > umbral), descomposición de la varianza y percentiles de Y. El
número de complicados por día es Poisson (adelgazamiento), de modo que el sobrecoste diario, los días con
desbordamiento, los derivados, los días consecutivos con la UCI llena y la distribución del total de derivados del
brote se calculan exactamente. El coste total del brote usa la aproximación normal solo para su percentil 95.

Material de Javier Morales, equipo IA4LEGOS (CC BY-SA 4.0)."""
import copy
import functools
import math

import numpy as np
import pandas as pd

try:
    from scipy import stats as _stats
    from scipy.special import ndtri as _ndtri, gammaincinv as _gammaincinv
except Exception:          # pragma: no cover
    _stats = None

FENOTIPOS = (1, 2, 3)
NOMBRE_FEN = {1: "Rápido", 2: "Estándar", 3: "Persistente"}
TERMINOS = ("Estancia C_día·D", "Complicación (salto)")
Z95 = 1.959963984540054
N_DIAGRAMA = 40

COMUNES_DEF = {
    "p": [0.50, 0.40, 0.10],
    "alpha": {1: 2.0, 2: 5.0, 3: 10.0}, "escala_L": 1.0,
    "R": {1: (1.5, 2.0), 2: (0.8, 1.2), 3: (0.2, 0.5)},
    "eps_mu": 1.0, "eps_sd": 0.2,
    "fases": [{"nombre": "Inicio", "dias": 20, "lambda": 6.0, "p": [0.65, 0.30, 0.05]},
              {"nombre": "Pico", "dias": 30, "lambda": 25.0, "p": [0.45, 0.35, 0.20]},
              {"nombre": "Remisión", "dias": 40, "lambda": 10.0, "p": [0.58, 0.32, 0.10]}],
    "coste_cama_dia": 250.0,
    "objetivos": {"EY_MAX": 3200.0, "PCOMP_MAX": 0.07, "TOTAL_MAX": 4_550_000.0, "EXT_MAX": 15.0},
}
ESC_BASE = {"C_fijo": 1200.0, "C_dia": 600.0, "C_UCI": 5000.0, "C_ext": 9000.0, "umbral": 7.0, "devolucion": 0.0,
            "coste_extra": 0.0, "factor_L": 1.0, "factor_L3": 1.0, "factor_R3": 1.0, "lambda_pico": None,
            "camas": 3, "camas_alquiler": 0, "coste_alquiler": 6500.0, "tramo_Q": 0, "coste_tramo": 7000.0,
            "coste_impl": 0.0, "descripcion": ""}


def escenario(**cambios):
    e = copy.deepcopy(ESC_BASE)
    e.update(cambios)
    return e


def normalizar(p):
    p = np.clip(np.asarray(p, dtype=float), 0, None)
    return p / p.sum() if p.sum() > 0 else np.full(len(p), 1 / len(p))


def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


_vPhi = np.vectorize(Phi)


def _Phi_arr(x):
    from scipy.special import ndtr
    return ndtr(x)


# ==============================================================================================
# Teoría · un paciente
# ==============================================================================================
@functools.lru_cache(maxsize=64)
def _rejilla_Z(alpha, escala, r0, r1, nL=500, nR=60, nbins=1500):
    """Distribución discretizada de Z = ln(L + 1)/R: L por cuantiles de la Gamma, R por puntos medios de la uniforme;
    se agrupa en `nbins` intervalos finos (valor medio ponderado y probabilidad)."""
    qL = (np.arange(nL) + 0.5) / nL
    L = _gammaincinv(alpha, qL) * escala
    R = r0 + (r1 - r0) * (np.arange(nR) + 0.5) / nR
    Z = (np.log1p(L)[:, None] / R[None, :]).ravel()
    w = np.full(Z.size, 1.0 / Z.size)
    edges = np.linspace(Z.min(), Z.max() * (1 + 1e-12), nbins + 1)
    idx = np.clip(np.searchsorted(edges, Z, side="right") - 1, 0, nbins - 1)
    pw = np.bincount(idx, weights=w, minlength=nbins)
    zm = np.bincount(idx, weights=w * Z, minlength=nbins)
    ok = pw > 0
    return zm[ok] / pw[ok], pw[ok]


def _params_fen(com, esc, f):
    a = float(com["alpha"][f])
    esc_L = float(com["escala_L"]) * float(esc["factor_L"]) * (float(esc["factor_L3"]) if f == 3 else 1.0)
    r0, r1 = com["R"][f]
    k = float(esc["factor_R3"]) if f == 3 else 1.0
    return a, esc_L, float(r0) * k, float(r1) * k


def momentos_D(com, esc, f, u=None):
    """Momentos de D = max(0, Z + ε) para el fenotipo f y cantidades sobre el umbral u: P(D > u), E[D], E[D²],
    E[D·I], E[D²·I], con D | Z normal truncada en 0."""
    u = float(esc["umbral"]) if u is None else float(u)
    z, w = _rejilla_Z(*_params_fen(com, esc, f))
    m = z + float(com["eps_mu"]); s = float(com["eps_sd"])
    a0 = m / s
    P0 = _Phi_arr(a0); f0 = np.exp(-0.5 * a0 ** 2) / math.sqrt(2 * math.pi)
    ED = m * P0 + s * f0
    ED2 = (m * m + s * s) * P0 + m * s * f0
    au = (u - m) / s
    Q = 1 - _Phi_arr(au); fu = np.exp(-0.5 * au ** 2) / math.sqrt(2 * math.pi)
    EDI = m * Q + s * fu
    ED2I = (m * m + s * s) * Q + s * (m + u) * fu
    return {"P": float(w @ Q), "ED": float(w @ ED), "ED2": float(w @ ED2), "EDI": float(w @ EDI),
            "ED2I": float(w @ ED2I), "z": z, "w": w}


def _coefs(esc):
    """Y = a₀ + C_día·D + K·I{D > umbral}: a₀ = C_fijo + coste extra; K = C_UCI − ρ·C_fijo."""
    a0 = float(esc["C_fijo"]) + float(esc["coste_extra"])
    K = float(esc["C_UCI"]) - float(esc["devolucion"]) * float(esc["C_fijo"])
    return a0, float(esc["C_dia"]), K


def momentos_Y_fen(md, esc, coste_comp=None):
    """E(Y), E(Y²), contribuciones de los términos y momentos condicionados a complicar / no complicar."""
    a0, c, K = _coefs(esc)
    if coste_comp is not None:
        K = float(coste_comp) - float(esc["devolucion"]) * float(esc["C_fijo"])
    P, ED, ED2, EDI, ED2I = md["P"], md["ED"], md["ED2"], md["EDI"], md["ED2I"]
    EY = a0 + c * ED + K * P
    EY2 = a0 * a0 + c * c * ED2 + K * K * P + 2 * a0 * c * ED + 2 * a0 * K * P + 2 * c * K * EDI
    # términos X1 = c·D, X2 = K·I
    E = np.array([c * ED, K * P])
    M2 = np.array([[c * c * ED2, c * K * EDI], [c * K * EDI, K * K * P]])
    # condicionados a complicar
    Ec = (a0 * P + c * EDI + K * P) / P if P > 0 else 0.0
    E2c = (a0 * a0 * P + c * c * ED2I + K * K * P + 2 * a0 * c * EDI + 2 * a0 * K * P + 2 * c * K * EDI) / P if P > 0 else 0.0
    Pn = 1 - P
    Enc = (a0 * Pn + c * (ED - EDI)) / Pn if Pn > 0 else 0.0
    E2nc = (a0 * a0 * Pn + c * c * (ED2 - ED2I) + 2 * a0 * c * (ED - EDI)) / Pn if Pn > 0 else 0.0
    return {"EY": EY, "EY2": EY2, "E": E, "M2": M2, "Ec": Ec, "E2c": E2c, "Enc": Enc, "E2nc": E2nc}


def cdf_D(md, com, d):
    d = np.atleast_1d(np.asarray(d, dtype=float))
    m = md["z"] + float(com["eps_mu"]); s = float(com["eps_sd"])
    F = md["w"] @ _Phi_arr((d[None, :] - m[:, None]) / s)
    return np.where(d < 0, 0.0, F)


def cdf_Y(mds, pesos, com, esc, y):
    """P(Y ≤ y) de la mezcla de fenotipos: Y = a₀ + c·D si D ≤ u; a₀ + c·D + K si D > u."""
    a0, c, K = _coefs(esc)
    u = float(esc["umbral"])
    y = np.atleast_1d(np.asarray(y, dtype=float))
    F = np.zeros_like(y)
    for md, p in zip(mds, pesos):
        x1 = np.minimum(u, (y - a0) / c)
        Fd_u = cdf_D(md, com, [u])[0]
        F1 = np.where(x1 >= 0, cdf_D(md, com, np.maximum(x1, 0)), 0.0)
        x2 = (y - a0 - K) / c
        F2 = np.where(x2 > u, cdf_D(md, com, np.maximum(x2, u)) - Fd_u, 0.0)
        F += p * (F1 + F2)
    return F


def cuantil_rejilla(y, F, q):
    i = int(np.searchsorted(F, q))
    if i <= 0:
        return float(y[0])
    if i >= len(y):
        return float(y[-1])
    f0, f1 = F[i - 1], F[i]
    return float(y[i - 1] + (q - f0) / (f1 - f0) * (y[i] - y[i - 1])) if f1 > f0 else float(y[i])


def teoria_A(com, esc, pesos=None, con_cdf=True):
    p = normalizar(pesos if pesos is not None else com["p"])
    mds = [momentos_D(com, esc, f) for f in FENOTIPOS]
    my = [momentos_Y_fen(md, esc) for md in mds]
    EY = float(sum(pi * m["EY"] for pi, m in zip(p, my)))
    EY2 = float(sum(pi * m["EY2"] for pi, m in zip(p, my)))
    VY = EY2 - EY ** 2
    E = sum(pi * m["E"] for pi, m in zip(p, my)); M2 = sum(pi * m["M2"] for pi, m in zip(p, my))
    cov = M2 - np.outer(E, E)
    contrib = cov.sum(axis=1) / VY if VY > 0 else np.zeros(2)
    Pc = float(sum(pi * md["P"] for pi, md in zip(p, mds)))
    ED = float(sum(pi * md["ED"] for pi, md in zip(p, mds)))
    a0, c, K = _coefs(esc)
    if not con_cdf:
        VY_ = max(VY, 0)
        return {"p": p, "EY": EY, "VY": VY, "sdY": math.sqrt(VY_), "CV": math.sqrt(VY_) / EY, "Pcomp": Pc, "ED": ED,
                "contrib": contrib}
    hi = a0 + c * 60 + max(K, 0)
    yg = np.unique(np.concatenate([np.linspace(min(a0, a0 + K) - 1, hi, 2500),
                                   [a0 + c * esc["umbral"], a0 + c * esc["umbral"] + K]]))
    F = cdf_Y(mds, p, com, esc, yg)
    por_fen = {}
    for f, pi, md, m in zip(FENOTIPOS, p, mds, my):
        v = m["EY2"] - m["EY"] ** 2
        por_fen[f] = {"p": float(pi), "EY": m["EY"], "sd": math.sqrt(max(v, 0)),
                      "CV": math.sqrt(max(v, 0)) / m["EY"] if m["EY"] else 0.0, "Pcomp": md["P"], "ED": md["ED"],
                      "peso_comp": pi * md["P"] / Pc if Pc > 0 else 0.0}
    # índice de primer orden del fenotipo: Var(E[Y | T]) / Var(Y)
    S_T = float(sum(pi * (m["EY"] - EY) ** 2 for pi, m in zip(p, my)) / VY) if VY > 0 else 0.0
    return {"p": p, "EY": EY, "VY": VY, "sdY": math.sqrt(max(VY, 0)), "CV": math.sqrt(max(VY, 0)) / EY,
            "Pcomp": Pc, "ED": ED, "P95": cuantil_rejilla(yg, F, 0.95), "P50": cuantil_rejilla(yg, F, 0.50),
            "E_terminos": E, "cov": cov, "contrib": contrib, "por_fen": por_fen, "y": yg, "F": F, "mds": mds, "my": my,
            "S_T": S_T}


# ==============================================================================================
# Teoría · un brote (fases, capacidad UCI, alquiler y clínica externa)
# ==============================================================================================
def _pmf(lam, kmax=None):
    kmax = kmax or int(lam + 12 * math.sqrt(lam + 1) + 25)
    k = np.arange(kmax + 1)
    return k, _stats.poisson.pmf(k, lam)


def fases_esc(com, esc):
    out = []
    for fz in com["fases"]:
        lam = float(fz["lambda"])
        if fz["nombre"] == "Pico" and esc.get("lambda_pico") not in (None, ""):
            lam = float(esc["lambda_pico"])
        out.append({"nombre": fz["nombre"], "dias": int(fz["dias"]), "lambda": lam, "p": normalizar(fz["p"])})
    return out


def _g_dia(k, esc, camas=None, extra=None):
    """Uso de capacidad de un día con k pacientes complicados: (propias, alquiladas, derivadas)."""
    B = int(esc["camas"] if camas is None else camas)
    E = int(esc["camas_alquiler"] if extra is None else extra)
    prop = np.minimum(k, B)
    alq = np.minimum(np.maximum(k - B, 0), E)
    der = np.maximum(k - B - E, 0)
    return prop, alq, der


def teoria_B(com, esc, camas=None, extra=None):
    """Coste del brote con y sin restricción de capacidad, sobrecoste, días con desbordamiento, derivados y riesgo de
    UCI llena en días consecutivos. Exacto salvo el percentil 95 del coste total (aproximación normal)."""
    B = int(esc["camas"] if camas is None else camas)
    E = int(esc["camas_alquiler"] if extra is None else extra)
    mds = [momentos_D(com, esc, f) for f in FENOTIPOS]
    my = [momentos_Y_fen(md, esc) for md in mds]
    Cu, Ca, Cx = float(esc["C_UCI"]), float(esc["coste_alquiler"]), float(esc["C_ext"])
    tot = {"sin": 0.0, "var_sin": 0.0, "delta": 0.0, "var_delta": 0.0, "cov": 0.0, "pacientes": 0.0, "comp": 0.0,
           "desb": 0.0, "der": 0.0, "alq": 0.0, "llena": 0.0}
    por_fase, s_dias, pmfs_der = {}, [], []
    for fz in fases_esc(com, esc):
        p = fz["p"]; lam = fz["lambda"]; nd = fz["dias"]
        q = float(sum(pi * md["P"] for pi, md in zip(p, mds)))
        lc, lnc = lam * q, lam * (1 - q)
        Ec = sum(pi * md["P"] * m["Ec"] for pi, md, m in zip(p, mds, my)) / q if q > 0 else 0.0
        E2c = sum(pi * md["P"] * m["E2c"] for pi, md, m in zip(p, mds, my)) / q if q > 0 else 0.0
        Enc = sum(pi * (1 - md["P"]) * m["Enc"] for pi, md, m in zip(p, mds, my)) / (1 - q) if q < 1 else 0.0
        E2nc = sum(pi * (1 - md["P"]) * m["E2nc"] for pi, md, m in zip(p, mds, my)) / (1 - q) if q < 1 else 0.0
        k, pk = _pmf(lc)
        prop, alq, der = _g_dia(k, esc, B, E)
        g = (Ca - Cu) * alq + (Cx - Cu) * der                 # sobrecoste del día con k complicados
        Eg, Eg2 = float(pk @ g), float(pk @ g ** 2)
        cov_kg = float(pk @ (k * g)) - lc * Eg
        dia_sin = lam * (q * Ec + (1 - q) * Enc)
        var_sin = lc * E2c + lnc * E2nc                         # suma compuesta de Poisson
        dia = {"sin": dia_sin, "var_sin": var_sin, "delta": Eg, "var_delta": Eg2 - Eg * Eg, "cov": Ec * cov_kg,
               "desb": float(pk[k > B + E].sum()) if E else float(pk[k > B].sum()), "der": float(pk @ der),
               "alq": float(pk @ alq), "llena": float(pk[k >= B].sum())}
        for c_ in ("sin", "var_sin", "delta", "var_delta", "cov", "desb", "der", "alq", "llena"):
            tot[c_] += nd * dia[c_]
        tot["pacientes"] += nd * lam; tot["comp"] += nd * lc
        por_fase[fz["nombre"]] = {"dias": nd, "lambda": lam, "q": q, "lambda_comp": lc, "pacientes": nd * lam,
                                  "complicados": nd * lc, "sin": nd * dia_sin, "delta": nd * Eg,
                                  "desb": nd * dia["desb"], "der": nd * dia["der"], "alq": nd * dia["alq"],
                                  "p_persistente": float(p[2]), "delta_por_paciente": Eg / lam if lam else 0.0}
        s_dias += [dia["llena"]] * nd
        pmf_der = np.bincount(der, weights=pk)
        pmfs_der += [pmf_der] * nd
    out = {"camas": B, "extra": E}
    out["total_sin"] = tot["sin"]
    out["delta"] = tot["delta"]
    out["total_con"] = tot["sin"] + tot["delta"]
    out["var_con"] = tot["var_sin"] + tot["var_delta"] + 2 * tot["cov"]
    out["sd_con"] = math.sqrt(max(out["var_con"], 0))
    out["sd_sin"] = math.sqrt(max(tot["var_sin"], 0))
    out["sd_delta"] = math.sqrt(max(tot["var_delta"], 0))
    out["pct_delta"] = tot["delta"] / out["total_con"] if out["total_con"] else 0.0
    out["P95"] = out["total_con"] + 1.6448536269514722 * out["sd_con"]
    out["dias_desb"], out["derivados"], out["alquiladas"] = tot["desb"], tot["der"], tot["alq"]
    out["pacientes"], out["complicados"] = tot["pacientes"], tot["comp"]
    out["var_pareada"] = tot["var_delta"]
    out["var_independiente"] = tot["var_sin"] + out["var_con"]
    # O6.2 · probabilidad de UCI llena (k ≥ B) dos días consecutivos (cadena con días independientes)
    no_pair_0, no_pair_1 = 1.0, 0.0          # prob. sin pareja hasta ahora, terminando en día no lleno / lleno
    for s in s_dias:
        a, b = no_pair_0, no_pair_1
        no_pair_0 = (a + b) * (1 - s)
        no_pair_1 = a * s
    out["P_consecutivos"] = 1 - (no_pair_0 + no_pair_1)
    out["pares_consecutivos"] = float(sum(s_dias[i] * s_dias[i + 1] for i in range(len(s_dias) - 1)))
    out["dias_llena"] = tot["llena"]
    # O5.2 · contrato con tramo: distribución exacta del total de derivados del brote (convolución)
    dist = np.array([1.0])
    for pm in pmfs_der:
        dist = np.convolve(dist, pm)[:2000]
    x = np.arange(len(dist))
    out["dist_derivados"] = dist
    out["sd_derivados"] = math.sqrt(max(float(dist @ x ** 2) - float(dist @ x) ** 2, 0))
    Q = int(esc.get("tramo_Q") or 0)
    if Q > 0:
        pago = Cx * np.minimum(x, Q) + float(esc["coste_tramo"]) * np.maximum(x - Q, 0)
        ahorro = float(dist @ (Cx * x - pago))
        out["delta"] -= ahorro
        out["total_con"] -= ahorro
        out["P95"] -= ahorro
        out["ahorro_tramo"] = ahorro
        out["pct_delta"] = out["delta"] / out["total_con"]
        for fz in por_fase.values():            # el descuento se reparte en proporción a los derivados de cada fase
            fz["delta"] -= ahorro * (fz["der"] / tot["der"] if tot["der"] else 0.0)
    else:
        out["ahorro_tramo"] = 0.0
    for fz in por_fase.values():
        fz["cuota_delta"] = fz["delta"] / out["delta"] if out["delta"] else 0.0
    out["por_fase"] = por_fase
    ndias = sum(f["dias"] for f in por_fase.values())
    out["reserva"] = B * float(com["coste_cama_dia"]) * ndias
    out["total_gestion"] = out["total_con"] + out["reserva"] + float(esc.get("coste_impl") or 0.0)
    out["ndias"] = ndias
    return out


def barrido_camas(com, esc, camas=range(1, 9)):
    """O6.1 · sobrecoste esperado, coste de reservar las camas y coste total para cada número de camas propias."""
    filas, prev = [], None
    for b in camas:
        t = teoria_B(com, esc, camas=b)
        f = {"Camas": b, "Sobrecoste (€)": t["delta"], "Reserva de camas (€)": t["reserva"],
             "Sobrecoste + reserva (€)": t["delta"] + t["reserva"], "Días con desbordamiento": t["dias_desb"],
             "Derivados": t["derivados"], "P(UCI llena dos días seguidos)": t["P_consecutivos"]}
        f["Ahorro marginal de la cama (€)"] = (prev - t["delta"]) if prev is not None else float("nan")
        f["Coste de reservarla (€)"] = float(com["coste_cama_dia"]) * t["ndias"]
        prev = t["delta"]
        filas.append(f)
    df = pd.DataFrame(filas).set_index("Camas")
    return df, int(df["Sobrecoste + reserva (€)"].idxmin())


def cuota_maxima_pico(com, esc, extra_max=4):
    """O8.5 · cuota diaria máxima por cama adicional alquilada solo durante el Pico: ahorro esperado en sobrecoste
    del Pico por cada cama extra, dividido entre los días del Pico."""
    filas = []
    fz = [f for f in fases_esc(com, esc) if f["nombre"] == "Pico"]
    if not fz:
        return pd.DataFrame()
    dias_pico = fz[0]["dias"]
    base = teoria_B(com, esc)["por_fase"]["Pico"]["delta"]
    prev = base
    for k_ in range(1, extra_max + 1):
        d = teoria_B(com, esc, camas=int(esc["camas"]) + k_)["por_fase"]["Pico"]["delta"]
        filas.append({"Camas extra en el Pico": k_, "Sobrecoste del Pico (€)": d, "Ahorro total (€)": base - d,
                      "Ahorro marginal (€)": prev - d, "Cuota máxima por cama y día (€)": (prev - d) / dias_pico})
        prev = d
    return pd.DataFrame(filas).set_index("Camas extra en el Pico")


def teoria(com, esc):
    return {"A": teoria_A(com, esc), "B": teoria_B(com, esc)}


# ==============================================================================================
# Simulación (transformada inversa con números aleatorios comunes)
# ==============================================================================================
def generar_pacientes(com, esc, U, pesos):
    """U: 4 × n uniformes (fenotipo, carga, depuración, ruido); pesos: (n × 3) o vector de probabilidades."""
    n = U.shape[1]
    pesos = np.asarray(pesos, dtype=float)
    if pesos.ndim == 1:
        Fp = np.cumsum(normalizar(pesos))
        T = 1 + (U[0] > Fp[0]).astype(int) + (U[0] > Fp[1]).astype(int)
    else:
        Fp = np.cumsum(pesos / pesos.sum(axis=1, keepdims=True), axis=1)
        T = 1 + (U[0] > Fp[:, 0]).astype(int) + (U[0] > Fp[:, 1]).astype(int)
    L = np.zeros(n); R = np.zeros(n)
    for f in FENOTIPOS:
        mk = T == f
        if not mk.any():
            continue
        a, sc, r0, r1 = _params_fen(com, esc, f)
        L[mk] = _gammaincinv(a, U[1, mk]) * sc
        R[mk] = r0 + (r1 - r0) * U[2, mk]
    eps = float(com["eps_mu"]) + float(com["eps_sd"]) * _ndtri(U[3])
    D = np.maximum(0.0, np.log1p(L) / R + eps)
    comp = D > float(esc["umbral"])
    a0, c, K = _coefs(esc)
    Y = a0 + c * D + K * comp
    return {"T": T, "L": L, "R": R, "eps": eps, "D": D, "comp": comp, "Y": Y, "lin": c * D, "salto": K * comp}


def _tq(n, alpha=0.05):
    if n < 2:
        return float("nan")
    return float(_stats.t.ppf(1 - alpha / 2, n - 1))


def simular_A(com, esc, nsim, replicas, semilla, progreso=None, pesos=None):
    """Bloque A: `replicas` macro-réplicas de `nsim` pacientes (réplica i → semilla + i)."""
    pesos = com["p"] if pesos is None else pesos
    filas, Ys, Ts, traza = [], [], [], None
    for r in range(int(replicas)):
        rng = np.random.default_rng(int(semilla) + r)
        U = rng.random((4, int(nsim)))
        o = generar_pacientes(com, esc, U, pesos)
        Y, T = o["Y"], o["T"]
        cov = np.cov(np.vstack([o["lin"], o["salto"]]))
        vy = float(Y.var(ddof=1))
        fila = {"Réplica": r + 1, "Semilla": int(semilla) + r, "Pacientes": len(Y), "E(Y)": float(Y.mean()),
                "σ(Y)": math.sqrt(vy), "P(complicación)": float(o["comp"].mean()), "E(D)": float(o["D"].mean()),
                "P95": float(np.quantile(Y, 0.95))}
        for k_, nom in enumerate(TERMINOS):
            fila[f"Contrib. {nom}"] = float(cov[k_].sum() / vy) if vy > 0 else 0.0
        for f in FENOTIPOS:
            mk = T == f
            yf = Y[mk]
            nf = NOMBRE_FEN[f]
            fila[f"E(Y | {nf})"] = float(yf.mean()) if mk.any() else float("nan")
            fila[f"CV({nf})"] = float(yf.std(ddof=1) / yf.mean()) if mk.sum() > 1 else float("nan")
            fila[f"P(comp | {nf})"] = float(o["comp"][mk].mean()) if mk.any() else float("nan")
            fila[f"Peso en las complicaciones · {nf}"] = float((mk & o["comp"]).sum() / max(1, o["comp"].sum()))
        filas.append(fila)
        Ys.append(Y); Ts.append(T)
        if r == 0:
            k = N_DIAGRAMA
            traza = pd.DataFrame({"Paciente": np.arange(1, k + 1), "U_T": U[0, :k], "Fenotipo": [NOMBRE_FEN[t] for t in T[:k]],
                                  "L (UFC/ml)": o["L"][:k], "R (log/día)": o["R"][:k], "ε": o["eps"][:k], "D (días)": o["D"][:k],
                                  "Complicación": np.where(o["comp"][:k], "Sí", "No"), "Y (€)": Y[:k]})
        if progreso:
            progreso()
    return {"replicas": pd.DataFrame(filas).set_index("Réplica"), "Y": np.concatenate(Ys), "T": np.concatenate(Ts),
            "traza": traza}


CAMAS_BARRIDO = tuple(range(1, 9))


def _coste_dias(k, esc, camas, extra):
    prop, alq, der = _g_dia(k, esc, camas, extra)
    return (float(esc["coste_alquiler"]) - float(esc["C_UCI"])) * alq + (float(esc["C_ext"]) - float(esc["C_UCI"])) * der, alq, der


def simular_B(com, esc, replicas, semilla, progreso=None, cox_cv=0.0):
    """Bloque B: `replicas` brotes completos (réplica i → semilla + 1000 + i). Dentro de cada día, el orden de llegada
    aleatorio decide qué complicados consiguen cama; para los totales basta el número de complicados del día."""
    fz = fases_esc(com, esc)
    fase_d = np.concatenate([[i] * f["dias"] for i, f in enumerate(fz)])
    lam_d = np.array([fz[i]["lambda"] for i in fase_d])
    P_d = np.array([fz[i]["p"] for i in fase_d])
    nd = len(fase_d)
    B, E = int(esc["camas"]), int(esc["camas_alquiler"])
    Q = int(esc.get("tramo_Q") or 0)
    filas, dias_all = [], []
    for r in range(int(replicas)):
        rng = np.random.default_rng(int(semilla) + 1000 + r)
        if cox_cv and cox_cv > 0:        # O8.4 · proceso de Cox: intensidad diaria λ·G, G ~ Gamma(media 1, CV = cox_cv)
            k_ = 1.0 / cox_cv ** 2
            N = rng.poisson(lam_d * rng.gamma(k_, 1.0 / k_, size=nd))
        else:
            N = rng.poisson(lam_d)
        n_tot = int(N.sum())
        U = rng.random((5, n_tot))
        dia = np.repeat(np.arange(nd), N)
        o = generar_pacientes(com, esc, U[:4], P_d[dia])
        S_sin = np.bincount(dia, weights=o["Y"], minlength=nd)
        Kc = np.bincount(dia, weights=o["comp"].astype(float), minlength=nd).astype(int)
        g, alq, der = _coste_dias(Kc, esc, B, E)
        ahorro_tramo = 0.0
        if Q > 0:
            X = int(der.sum())
            ahorro_tramo = (float(esc["C_ext"]) - float(esc["coste_tramo"])) * max(X - Q, 0)
        S_con = S_sin + g
        total_con = float(S_con.sum()) - ahorro_tramo
        llena = Kc >= B
        fila = {"Réplica": r + 1, "Semilla": int(semilla) + 1000 + r, "Pacientes": n_tot,
                "Complicados": int(Kc.sum()), "Coste total con capacidad": total_con,
                "Coste total sin restricción": float(S_sin.sum()), "Sobrecoste": float(g.sum()) - ahorro_tramo,
                "% sobrecoste": (float(g.sum()) - ahorro_tramo) / total_con,
                "Días con desbordamiento": int((Kc > B + E).sum()), "Pacientes derivados": int(der.sum()),
                "Camas alquiladas usadas": int(alq.sum()),
                "UCI llena dos días seguidos": float(np.any(llena[1:] & llena[:-1])),
                "Pares de días seguidos con UCI llena": int((llena[1:] & llena[:-1]).sum()),
                "Coste total con reserva de camas": total_con + B * float(com["coste_cama_dia"]) * nd + float(esc.get("coste_impl") or 0)}
        for i, f in enumerate(fz):
            mk = fase_d == i
            fila[f"Sobrecoste · {f['nombre']}"] = float(g[mk].sum()) - (ahorro_tramo * der[mk].sum() / der.sum() if der.sum() else 0.0)
            fila[f"Derivados · {f['nombre']}"] = int(der[mk].sum())
            fila[f"Días con desbordamiento · {f['nombre']}"] = int((Kc[mk] > B + E).sum())
        for b in CAMAS_BARRIDO:
            gb, _, _ = _coste_dias(Kc, esc, b, E)
            fila[f"Sobrecoste con {b} camas"] = float(gb.sum())
        filas.append(fila)
        dias_all.append(pd.DataFrame({"Réplica": r + 1, "Día": np.arange(1, nd + 1), "Fase": [fz[i]["nombre"] for i in fase_d],
                                      "Ingresos": N, "Complicados": Kc, "Derivados": der, "Coste sin restricción": S_sin,
                                      "Coste con capacidad": S_con}))
        if progreso:
            progreso()
    return {"replicas": pd.DataFrame(filas).set_index("Réplica"), "dias": pd.concat(dias_all, ignore_index=True)}


def _ic_media(v):
    v = np.asarray(v, dtype=float); v = v[~np.isnan(v)]
    m = float(v.mean())
    if len(v) < 2:
        return m, float("nan"), float("nan")
    se = float(v.std(ddof=1)) / math.sqrt(len(v)); tq = _tq(len(v))
    return m, m - tq * se, m + tq * se


def _ic_ratio(num_, den_):
    """Cociente de medias E(X)/E(W) con IC95 % por el método delta."""
    x = np.asarray(num_, dtype=float); w = np.asarray(den_, dtype=float); n = len(x)
    r = float(x.mean() / w.mean())
    if n < 2:
        return r, float("nan"), float("nan")
    se = float(np.std(x - r * w, ddof=1)) / (math.sqrt(n) * abs(float(w.mean())))
    tq = _tq(n)
    return r, r - tq * se, r + tq * se


def _ic_cuantil(v, q=0.95):
    """Percentil de los totales de las réplicas con IC95 % sin supuestos (estadísticos de orden y binomial)."""
    v = np.sort(np.asarray(v, dtype=float)); n = len(v)
    est = float(np.quantile(v, q))
    lo_i = int(_stats.binom.ppf(0.025, n, q)); hi_i = int(_stats.binom.ppf(0.975, n, q))
    return est, float(v[max(lo_i - 1, 0)]), float(v[min(hi_i, n - 1)])


def resumen(rep, filas):
    out = []
    for nombre, (col, teo, tipo) in filas.items():
        if tipo == "cuantil":
            m, lo, hi = _ic_cuantil(rep[col])
        elif tipo == "ratio":
            m, lo, hi = _ic_ratio(rep[col[0]], rep[col[1]])
        else:
            m, lo, hi = _ic_media(rep[col])
        ok = None
        if teo is not None and lo == lo and hi - lo > 1e-12:     # IC degenerado (suceso raro sin casos): sin veredicto
            tol = 1e-9 * max(1.0, abs(teo))
            ok = bool(lo - tol <= teo <= hi + tol)
        out.append({"KPI": nombre, "Teórico": teo, "Simulado · media": m, "IC95 inf": lo, "IC95 sup": hi,
                    "Error relativo": (m - teo) / abs(teo) if (teo is not None and abs(teo) > 1e-12) else None,
                    "Teórico en IC95": ok})
    return pd.DataFrame(out).set_index("KPI")


def resumen_A(mc, tA):
    d = {"Coste esperado E(Y) (€)": ("E(Y)", tA["EY"], "media"),
         "Desviación típica σ(Y) (€)": ("σ(Y)", tA["sdY"], "media"),
         "Probabilidad de complicación P(D > umbral)": ("P(complicación)", tA["Pcomp"], "media"),
         "Días medios de ingreso E(D)": ("E(D)", tA["ED"], "media"),
         "Percentil 95 de Y (€)": ("P95", tA["P95"], "media")}
    for f in FENOTIPOS:
        nf = NOMBRE_FEN[f]; pf = tA["por_fen"][f]
        d[f"E(Y | {nf}) (€)"] = (f"E(Y | {nf})", pf["EY"], "media")
        d[f"CV(Y | {nf})"] = (f"CV({nf})", pf["CV"], "media")
        d[f"P(complicación | {nf})"] = (f"P(comp | {nf})", pf["Pcomp"], "media")
        d[f"Peso de {nf} en las complicaciones"] = (f"Peso en las complicaciones · {nf}", pf["peso_comp"], "media")
    for k_, nom in enumerate(TERMINOS):
        d[f"Contribución a Var(Y) · {nom}"] = (f"Contrib. {nom}", float(tA["contrib"][k_]), "media")
    return resumen(mc["replicas"], d)


def resumen_B(mc, tB, com):
    d = {"Coste total esperado del brote con capacidad (€)": ("Coste total con capacidad", tB["total_con"], "media"),
         "Coste total esperado sin restricción (€)": ("Coste total sin restricción", tB["total_sin"], "media"),
         "Sobrecoste por falta de capacidad (€)": ("Sobrecoste", tB["delta"], "media"),
         "Sobrecoste en % del coste total": (("Sobrecoste", "Coste total con capacidad"), tB["pct_delta"], "ratio"),
         "Percentil 95 del coste total (€) · aprox. normal": ("Coste total con capacidad", tB["P95"], "cuantil"),
         "Días con desbordamiento": ("Días con desbordamiento", tB["dias_desb"], "media"),
         "Pacientes derivados a la clínica externa": ("Pacientes derivados", tB["derivados"], "media"),
         "Camas de alquiler usadas": ("Camas alquiladas usadas", tB["alquiladas"], "media"),
         "P(UCI llena dos días seguidos)": ("UCI llena dos días seguidos", tB["P_consecutivos"], "media"),
         "Pacientes del brote": ("Pacientes", tB["pacientes"], "media"),
         "Pacientes complicados del brote": ("Complicados", tB["complicados"], "media"),
         "Coste total con reserva de camas e implantación (€)": ("Coste total con reserva de camas", tB["total_gestion"], "media")}
    for nf, f in tB["por_fase"].items():
        d[f"Sobrecoste · {nf} (€)"] = (f"Sobrecoste · {nf}", f["delta"], "media")
    for nf, f in tB["por_fase"].items():
        d[f"Derivados · {nf}"] = (f"Derivados · {nf}", f["der"], "media")
    return resumen(mc["replicas"], d)


# ==============================================================================================
# Objetivos, análisis de escenarios, ahorros e ICER
# ==============================================================================================
OBJ_NOMBRES = {"EY_MAX": "Coste esperado por paciente E(Y)", "PCOMP_MAX": "Probabilidad de complicación",
               "TOTAL_MAX": "Coste total esperado del brote (con reserva)", "EXT_MAX": "Pacientes derivados por brote"}


def evaluar_objetivos(t, obj):
    A, B = t["A"], t["B"]
    filas = []
    def fila(c, valor, txt, cumple):
        filas.append({"Objetivo": OBJ_NOMBRES[c], "Valor": valor, "Umbral": txt, "Cumple": "Sí" if cumple else "No"})
    if obj.get("EY_MAX") is not None:
        fila("EY_MAX", A["EY"], f"≤ {eur(obj['EY_MAX'])}", A["EY"] <= obj["EY_MAX"] + 1e-9)
    if obj.get("PCOMP_MAX") is not None:
        fila("PCOMP_MAX", A["Pcomp"], f"≤ {pct(obj['PCOMP_MAX'], 1)}", A["Pcomp"] <= obj["PCOMP_MAX"] + 1e-12)
    if obj.get("TOTAL_MAX") is not None:
        fila("TOTAL_MAX", B["total_gestion"], f"≤ {eur(obj['TOTAL_MAX'])}", B["total_gestion"] <= obj["TOTAL_MAX"] + 1e-6)
    if obj.get("EXT_MAX") is not None:
        fila("EXT_MAX", B["derivados"], f"≤ {num(obj['EXT_MAX'], 1)}", B["derivados"] <= obj["EXT_MAX"] + 1e-9)
    return pd.DataFrame(filas, columns=["Objetivo", "Valor", "Umbral", "Cumple"]).set_index("Objetivo")


def analizar_escenario(esc, com, aj, progreso=None):
    t = teoria(com, esc)
    mA = simular_A(com, esc, aj["nsim"], aj["replicas"], aj["semilla"], progreso)
    mB = simular_B(com, esc, aj["brotes"], aj["semilla"], progreso)
    return {"esc": dict(esc), "teoria": t, "mcA": mA, "mcB": mB, "resA": resumen_A(mA, t["A"]),
            "resB": resumen_B(mB, t["B"], com), "objetivos": evaluar_objetivos(t, com["objetivos"])}


def _icer(dc, de):
    """ICER = Δcoste / Δefecto (efecto = días o complicaciones evitados por paciente)."""
    if abs(de) < 1e-12:
        return "sin efecto"
    if dc <= 0 and de > 0:
        return "dominante (ahorra y mejora)"
    if dc >= 0 and de < 0:
        return "dominado (cuesta más y empeora)"
    if dc < 0 and de < 0:
        return f"más barato y menos eficaz: ahorra {eur(dc / de)} por unidad perdida"
    return dc / de


def diferencias(res, base):
    if base not in res:
        return None
    rb = res[base]
    filas = []
    for n, r in res.items():
        if n == base:
            continue
        ta, tb = rb["teoria"], r["teoria"]
        dA = r["mcA"]["replicas"]["E(Y)"] - rb["mcA"]["replicas"]["E(Y)"]
        dB = rb["mcB"]["replicas"]["Coste total con reserva de camas"] - r["mcB"]["replicas"]["Coste total con reserva de camas"]
        mA, loA, hiA = _ic_media(dA); mB, loB, hiB = _ic_media(dB)
        dc0 = tb["A"]["EY"] - ta["A"]["EY"]
        impl = (float(r["esc"].get("coste_impl") or 0) / max(tb["B"]["pacientes"], 1e-9)
                - float(rb["esc"].get("coste_impl") or 0) / max(ta["B"]["pacientes"], 1e-9))
        dc = dc0 + impl
        de_dias = ta["A"]["ED"] - tb["A"]["ED"]; de_comp = ta["A"]["Pcomp"] - tb["A"]["Pcomp"]
        filas.append({"Escenario": n,
                      "ΔE(Y) por paciente · teórico (€)": dc0, "ΔE(Y) por paciente · simulado (€)": mA,
                      "IC95 ΔE(Y) (€)": (loA, hiA),
                      "ΔP95 de Y · teórico (€)": tb["A"]["P95"] - ta["A"]["P95"],
                      "Δ coste por paciente con implantación · teórico (€)": dc,
                      "Días de ingreso evitados por paciente": de_dias,
                      "Complicaciones evitadas por paciente": de_comp,
                      "ICER (€ por día evitado)": _icer(dc, de_dias),
                      "ICER (€ por complicación evitada)": _icer(dc, de_comp),
                      "Ahorro por brote · teórico (€)": ta["B"]["total_gestion"] - tb["B"]["total_gestion"],
                      "Ahorro por brote · simulado (€)": mB, "IC95 ahorro por brote (€)": (loB, hiB),
                      "Ahorro en sobrecoste de capacidad · teórico (€)": ta["B"]["delta"] - tb["B"]["delta"],
                      "Coste de implantación (€/brote)": float(r["esc"].get("coste_impl") or 0.0)})
    return pd.DataFrame(filas).set_index("Escenario") if filas else None


def tabla_comparativa(res):
    cols = {}
    for n, r in res.items():
        A, B, e = r["teoria"]["A"], r["teoria"]["B"], r["esc"]
        ra, rb = r["resA"], r["resB"]
        cols[n] = {
            "Parámetro · C_fijo (+ extra) / C_día": f"{num(e['C_fijo'], 0)} (+{num(e['coste_extra'], 0)}) / {num(e['C_dia'], 0)}",
            "Parámetro · C_UCI / C_ext / devolución": f"{num(e['C_UCI'], 0)} / {num(e['C_ext'], 0)} / {pct(e['devolucion'], 0)}",
            "Parámetro · ×L / ×L Persistente / ×R Persistente": f"{num(e['factor_L'], 2)} / {num(e['factor_L3'], 2)} / {num(e['factor_R3'], 2)}",
            "Parámetro · umbral (días) / λ Pico": f"{num(e['umbral'], 0)} / {num(B['por_fase'].get('Pico', {}).get('lambda', 0), 0)}",
            "Parámetro · camas propias / de alquiler": f"{e['camas']} / {e['camas_alquiler']}",
            "Teórico · E(Y) por paciente (€)": A["EY"], "Teórico · P(complicación)": A["Pcomp"],
            "Teórico · P95 de Y (€)": A["P95"], "Teórico · E(D) (días)": A["ED"],
            "Teórico · coste del brote con capacidad (€)": B["total_con"],
            "Teórico · sobrecoste de capacidad (€)": B["delta"], "Teórico · % sobrecoste": B["pct_delta"],
            "Teórico · derivados por brote": B["derivados"], "Teórico · días con desbordamiento": B["dias_desb"],
            "Teórico · coste con reserva de camas (€)": B["total_gestion"],
            "Simulado · E(Y) por paciente (€)": ra.loc["Coste esperado E(Y) (€)", "Simulado · media"],
            "Simulado · P(complicación)": ra.loc["Probabilidad de complicación P(D > umbral)", "Simulado · media"],
            "Simulado · coste del brote con capacidad (€)": rb.loc["Coste total esperado del brote con capacidad (€)", "Simulado · media"],
            "Simulado · sobrecoste de capacidad (€)": rb.loc["Sobrecoste por falta de capacidad (€)", "Simulado · media"],
            "Simulado · derivados por brote": rb.loc["Pacientes derivados a la clínica externa", "Simulado · media"],
            "Objetivos cumplidos": f"{(r['objetivos']['Cumple'] == 'Sí').sum()} de {len(r['objetivos'])}",
        }
    return pd.DataFrame(cols)


def tabla_objetivos(res):
    return pd.DataFrame({n: r["objetivos"]["Cumple"] for n, r in res.items()})


# ==============================================================================================
# Sensibilidad: O2.2 (parámetros e índices de Sobol), O3.4 (umbral) y estimador
# ==============================================================================================
PARAM_SENS = {"alpha": "α de la Gamma (carga L)", "R": "Límites de la uniforme (depuración R)",
              "eps_sd": "σ del ruido ε", "p3": "p del fenotipo Persistente", "C_dia": "C_día", "C_UCI": "C_UCI"}


def _aplicar(com, esc, par, v):
    c2, e2 = copy.deepcopy(com), dict(esc)
    if par == "alpha":
        c2["alpha"] = {f: a * (1 + v) for f, a in com["alpha"].items()}
    elif par == "R":
        c2["R"] = {f: (r[0] * (1 + v), r[1] * (1 + v)) for f, r in com["R"].items()}
    elif par == "eps_sd":
        c2["eps_sd"] = com["eps_sd"] * (1 + v)
    elif par == "p3":
        p = list(normalizar(com["p"])); p3 = min(0.99, p[2] * (1 + v)); s = (1 - p3) / (p[0] + p[1])
        c2["p"] = [p[0] * s, p[1] * s, p3]
    else:
        e2[par] = esc[par] * (1 + v)
    return c2, e2


def sensibilidad_parametros(com, esc, variaciones):
    """Var(Y) y E(Y) teóricos al variar cada parámetro (el resto fijo) y elasticidad de Var(Y)."""
    filas = []
    for par, nom in PARAM_SENS.items():
        for v in variaciones:
            c2, e2 = _aplicar(com, esc, par, v)
            t = teoria_A(c2, e2, con_cdf=False)
            filas.append({"Parámetro": nom, "Variación": v, "Teórico · E(Y) (€)": t["EY"], "Teórico · Var(Y)": t["VY"],
                          "Teórico · σ(Y)": t["sdY"], "Teórico · P(complicación)": t["Pcomp"]})
    elas = {}
    for par, nom in PARAM_SENS.items():
        vs = [teoria_A(*_aplicar(com, esc, par, h), con_cdf=False)["VY"] for h in (-0.01, 0.01)]
        elas[nom] = (math.log(vs[1]) - math.log(vs[0])) / (math.log(1.01) - math.log(0.99)) if min(vs) > 0 else 0.0
    return pd.DataFrame(filas), elas


def indices_sobol(com, esc, n=20000, semilla=1000):
    """Índices de Sobol de primer orden y totales de las entradas aleatorias (fenotipo T, carga L, depuración R y
    ruido ε) sobre Y, con el estimador de Saltelli/Jansen (matrices A, B y A con una columna de B)."""
    rng = np.random.default_rng(int(semilla) + 7)
    UA, UB = rng.random((4, n)), rng.random((4, n))
    yA = generar_pacientes(com, esc, UA, com["p"])["Y"]; yB = generar_pacientes(com, esc, UB, com["p"])["Y"]
    V = float(np.var(np.concatenate([yA, yB]), ddof=1))
    filas = []
    for i, nom in enumerate(("Fenotipo T", "Carga L", "Depuración R", "Ruido ε")):
        UAB = UA.copy(); UAB[i] = UB[i]
        yAB = generar_pacientes(com, esc, UAB, com["p"])["Y"]
        S1 = float(np.mean(yB * (yAB - yA)) / V)
        ST = float(0.5 * np.mean((yA - yAB) ** 2) / V)
        filas.append({"Entrada": nom, "Índice de primer orden S₁": S1, "Índice total S_T": ST})
    df = pd.DataFrame(filas).set_index("Entrada")
    df.loc["Fenotipo T", "Teórico · S₁"] = teoria_A(com, esc)["S_T"]
    return df


def experimento_umbral(com, esc, umbrales=(5.0, 7.0, 10.0), nsim=20000, semilla=1000):
    """O3.4 · distribución de Y y efecto en el brote al recalibrar el umbral de complicación (mismos pacientes)."""
    rng = np.random.default_rng(int(semilla) + 11)
    U = rng.random((4, int(nsim)))
    filas = []
    for u in umbrales:
        e2 = dict(esc, umbral=float(u))
        t = teoria_A(com, e2); b = teoria_B(com, e2)
        o = generar_pacientes(com, e2, U, com["p"])
        filas.append({"Umbral (días)": u, "Teórico · E(Y) (€)": t["EY"], "Simulado · E(Y) (€)": float(o["Y"].mean()),
                      "Teórico · P(complicación)": t["Pcomp"], "Simulado · P(complicación)": float(o["comp"].mean()),
                      "Teórico · P95 de Y (€)": t["P95"], "Simulado · P95 de Y (€)": float(np.quantile(o["Y"], 0.95)),
                      "Teórico · contribución del salto a Var(Y)": float(t["contrib"][1]),
                      "Teórico · sobrecoste del brote (€)": b["delta"], "Teórico · derivados por brote": b["derivados"]})
    return pd.DataFrame(filas).set_index("Umbral (días)")


def valores_barrido(vmin, vmax, puntos):
    v = np.geomspace(max(1, vmin), max(vmin + 1, vmax), max(2, int(puntos)))
    return sorted(set(int(round(x)) for x in v))


def sensibilidad_estimador(com, esc, metrica, valores, replicas, semilla, progreso=None):
    """E(Y) frente al nº de pacientes (Bloque A) o coste medio del brote frente al nº de brotes simulados (Bloque B)."""
    t = teoria(com, esc)
    teo = t["A"]["EY"] if metrica == "paciente" else t["B"]["total_con"]
    sd1 = t["A"]["sdY"] if metrica == "paciente" else t["B"]["sd_con"]
    tq = _tq(int(replicas))
    filas, k, tot = [], 0, len(valores) * int(replicas)
    for n in valores:
        medias = []
        for r in range(int(replicas)):
            if metrica == "paciente":
                rng = np.random.default_rng(int(semilla) + r)
                medias.append(float(generar_pacientes(com, esc, rng.random((4, n)), com["p"])["Y"].mean()))
            else:
                m = simular_B(com, esc, n, int(semilla) + 50_000 * (r + 1))
                medias.append(float(m["replicas"]["Coste total con capacidad"].mean()))
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
# Ampliación optativa (O8)
# ==============================================================================================
def verificacion_pareada(mcB, tB):
    """O8.1 · σ del sobrecoste estimado con el diseño pareado frente a emparejar brotes distintos (independientes)."""
    rep = mcB["replicas"]
    par = rep["Sobrecoste"].to_numpy()
    con = rep["Coste total con capacidad"].to_numpy(); sin = rep["Coste total sin restricción"].to_numpy()
    ind = con - np.roll(sin, 1)                 # Y_con de un brote menos Y_sin de OTRO brote
    n = len(par)
    filas = [{"Diseño": "Pareado (mismos pacientes)", "Media del sobrecoste (€)": float(par.mean()),
              "σ por brote · simulado (€)": float(par.std(ddof=1)), "σ por brote · teórico (€)": tB["sd_delta"],
              "Semiamplitud IC95 (€)": _tq(n) * float(par.std(ddof=1)) / math.sqrt(n)},
             {"Diseño": "Independiente (brotes distintos)", "Media del sobrecoste (€)": float(ind.mean()),
              "σ por brote · simulado (€)": float(ind.std(ddof=1)),
              "σ por brote · teórico (€)": math.sqrt(tB["var_independiente"]),
              "Semiamplitud IC95 (€)": _tq(n) * float(ind.std(ddof=1)) / math.sqrt(n)}]
    df = pd.DataFrame(filas).set_index("Diseño")
    return df, float((ind.std(ddof=1) / par.std(ddof=1)) ** 2) if par.std(ddof=1) > 0 else float("inf")


def simular_ocupacion(com, esc, brotes, semilla, progreso=None):
    """O8.2 · modelo con ocupación de varios días: cada complicado ocupa una cama UCI durante ⌈D − umbral⌉ días (mín. 1);
    si no hay cama libre al llegar, se deriva. Se compara con el modelo «un día, una decisión» con los mismos pacientes."""
    fz = fases_esc(com, esc)
    fase_d = np.concatenate([[i] * f["dias"] for i, f in enumerate(fz)])
    lam_d = np.array([fz[i]["lambda"] for i in fase_d]); P_d = np.array([fz[i]["p"] for i in fase_d])
    nd = len(fase_d); B = int(esc["camas"]); u = float(esc["umbral"])
    sob_d, sob_o, der_d, der_o, ocu = [], [], [], [], []
    for r in range(int(brotes)):
        rng = np.random.default_rng(int(semilla) + 1000 + r)
        N = rng.poisson(lam_d); n_tot = int(N.sum())
        U = rng.random((5, n_tot)); dia = np.repeat(np.arange(nd), N)
        o = generar_pacientes(com, esc, U[:4], P_d[dia])
        comp = o["comp"]; dur = np.maximum(1, np.ceil(o["D"] - u)).astype(int)
        Kc = np.bincount(dia, weights=comp.astype(float), minlength=nd).astype(int)
        der_d.append(float(np.maximum(Kc - B, 0).sum()))
        libres_hasta = np.zeros(B, dtype=int)       # día en que queda libre cada cama
        x, oc = 0, 0
        idx = np.where(comp)[0]
        idx = idx[np.lexsort((U[4, idx], dia[idx]))]   # por día y orden de llegada aleatorio
        for i in idx:
            d = dia[i]
            j = int(np.argmin(libres_hasta))
            if libres_hasta[j] <= d:
                libres_hasta[j] = d + dur[i]; oc += dur[i]
            else:
                x += 1
        der_o.append(float(x)); ocu.append(oc / (B * nd))
        if progreso:
            progreso()
    k = float(esc["C_ext"]) - float(esc["C_UCI"])
    der_d, der_o = np.array(der_d), np.array(der_o)
    filas = []
    for nom, v in (("«Un día, una decisión» (modelo del caso)", der_d), ("Ocupación de ⌈D − umbral⌉ días", der_o)):
        m, lo, hi = _ic_media(v)
        filas.append({"Modelo": nom, "Derivados por brote": m, "IC95 derivados": (lo, hi), "Sobrecoste (€)": k * m})
    df = pd.DataFrame(filas).set_index("Modelo")
    return df, float(np.mean(ocu))


def experimento_cox(com, esc, cvs=(0.0, 0.2, 0.4, 0.6), brotes=200, semilla=1000, progreso=None):
    """O8.4 · intensidad diaria aleatoria (proceso de Cox): λ_d·G_d con G_d ~ Gamma(media 1, CV); mismos uniformes."""
    filas = []
    for cv in cvs:
        m = simular_B(com, esc, brotes, semilla, cox_cv=float(cv))["replicas"]
        fila = {"CV de la intensidad": float(cv)}
        for col, nom in (("Sobrecoste", "Sobrecoste (€)"), ("Pacientes derivados", "Derivados por brote"),
                         ("Días con desbordamiento", "Días con desbordamiento")):
            mm, lo, hi = _ic_media(m[col]); fila[nom] = mm
            if col == "Sobrecoste":
                fila[f"IC95 {nom}"] = (lo, hi)
        fila["σ del coste total (€)"] = float(m["Coste total con capacidad"].std(ddof=1))
        filas.append(fila)
        if progreso:
            progreso()
    return pd.DataFrame(filas).set_index("CV de la intensidad")


def coste_programa(rr, coste_10):
    """Coste por brote del diagnóstico precoz con rendimientos decrecientes: coste_10·(10·r)² (10 % → coste_10)."""
    return float(coste_10) * (10 * float(rr)) ** 2


def optimizacion_conjunta(com, esc, camas=range(2, 8), reducciones=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                          coste_10=12000.0, presupuesto=250000.0):
    """O8.3 · camas UCI reservadas y diagnóstico precoz del Persistente (reduce la carga media L₃ un r %, con coste
    coste_programa(r)). Objetivo: coste total esperado del brote (incluidos la reserva y el programa);
    restricción presupuestaria: reserva + programa ≤ presupuesto."""
    filas = []
    for rr in reducciones:
        e2 = dict(esc, factor_L3=float(esc["factor_L3"]) * (1 - rr))
        for b in camas:
            t = teoria_B(com, e2, camas=b)
            prog = coste_programa(rr, coste_10)
            fijo = t["reserva"] + prog
            filas.append({"Reducción de L₃": rr, "Camas": b, "Programa (€)": prog, "Reserva (€)": t["reserva"],
                          "Gasto fijo (€)": fijo, "Sobrecoste (€)": t["delta"], "Derivados": t["derivados"],
                          "Coste total esperado (€)": t["total_con"] + fijo, "Factible": fijo <= presupuesto + 1e-6})
    df = pd.DataFrame(filas)
    fac = df[df["Factible"]]
    opt = fac.loc[fac["Coste total esperado (€)"].idxmin()] if len(fac) else None
    return df, opt


def ahorro_diagnostico(com, esc, reducciones=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6), coste_10=12000.0):
    """O8.6 · ahorro esperado por brote (sobrecoste de capacidad y coste total) al reducir la carga del Persistente."""
    t0 = teoria_B(com, esc)
    filas = []
    for rr in reducciones:
        t = teoria_B(com, dict(esc, factor_L3=float(esc["factor_L3"]) * (1 - rr)))
        filas.append({"Reducción de L₃": rr, "Ahorro en sobrecoste (€)": t0["delta"] - t["delta"],
                      "Ahorro en coste total (€)": t0["total_con"] - t["total_con"],
                      "Derivados evitados": t0["derivados"] - t["derivados"],
                      "Coste del programa (€)": coste_programa(rr, coste_10),
                      "Beneficio neto (€)": t0["total_con"] - t["total_con"] - coste_programa(rr, coste_10),
                      "Beneficio neto solo con el sobrecoste (€)": t0["delta"] - t["delta"] - coste_programa(rr, coste_10)})
    return pd.DataFrame(filas).set_index("Reducción de L₃")


def un_paciente(com, esc, uT, uL, uR, ue):
    U = np.array([[uT], [uL], [uR], [ue]])
    o = generar_pacientes(com, esc, U, com["p"])
    a0, c, K = _coefs(esc)
    return {"T": int(o["T"][0]), "L": float(o["L"][0]), "R": float(o["R"][0]), "eps": float(o["eps"][0]),
            "D": float(o["D"][0]), "comp": bool(o["comp"][0]), "Y": float(o["Y"][0]),
            "partes": [a0, float(o["lin"][0]), float(o["salto"][0])]}


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
    if isinstance(v, str) or v is None:
        return "—" if v is None else v
    if isinstance(v, tuple):
        return f"[{fmt_val(k, v[0])}; {fmt_val(k, v[1])}]"
    if "ICER" in k:
        return eur(v, 0)
    if "(€" in k or k.endswith("€"):
        return eur(v, 0)
    if "P(" in k or "%" in k or "Contribución" in k or "Peso" in k or "Cuota" in k or "Complicaciones evitadas" in k \
            or "probabilidad" in k.lower() or "S₁" in k or "S_T" in k:
        return pct(v, 2)
    if "CV" in k:
        return num(v, 3)
    if "días" in k.lower() or "Días" in k or "derivados" in k.lower() or "Pacientes" in k or "Camas" in k \
            or "Complicados" in k or "complicados" in k or "E(D)" in k:
        return num(v, 2)
    return num(v, 3)


def fmt_kpi(k, c, v):
    return fmt_val(k, v)


def fmt_res(fila, col, v):
    if col == "Teórico en IC95":
        return "—" if v is None else ("Sí" if v else "No")
    if col == "Error relativo":
        return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else pct(v, 2)
    return fmt_val(fila, v)


def fmt_col(fila, col, v):
    return fmt_val(col, v)


def fmt_obj(fila, col, v):
    if col == "Valor":
        if fila.startswith("Probabilidad"):
            return pct(v, 2)
        if fila.startswith("Pacientes"):
            return num(v, 1)
        return eur(v)
    return str(v)


def texto_escenario(nombre, r):
    A, B = r["teoria"]["A"], r["teoria"]["B"]
    okA = r["resA"]["Teórico en IC95"].dropna().astype(bool); okB = r["resB"]["Teórico en IC95"].dropna().astype(bool)
    fuera = list(okA[~okA].index) + list(okB[~okB].index)
    n_kpi = len(okA) + len(okB)
    val = ("la simulación reproduce los valores teóricos (todos dentro del IC95 %)" if not fuera else
           f"la simulación reproduce los valores teóricos en {n_kpi - len(fuera)} de los {n_kpi} KPIs (fuera del IC95 %: "
           f"{', '.join(fuera)}; con tantos KPIs es esperable que alguno quede fuera por azar)")
    pf = A["por_fen"]
    mayor_cv = max(FENOTIPOS, key=lambda f: pf[f]["CV"])
    dom = int(np.argmax(A["contrib"]))
    fase_max = max(B["por_fase"], key=lambda f: B["por_fase"][f]["delta"])
    txt = (f"{nombre}: un paciente cuesta en media E(Y) = {eur(A['EY'])} (σ = {eur(A['sdY'])}); P(complicación) = "
           f"{pct(A['Pcomp'], 2)} y el 5 % de los pacientes cuesta más de {eur(A['P95'])}. El fenotipo {NOMBRE_FEN[mayor_cv]} "
           f"es el de mayor incertidumbre relativa (CV = {num(pf[mayor_cv]['CV'], 2)}) y el término «{TERMINOS[dom]}» explica el "
           f"{pct(A['contrib'][dom], 0)} de Var(Y). Un brote cuesta {eur(B['total_con'])} con {B['camas']} camas UCI "
           f"(+{eur(B['delta'])} de sobrecoste por capacidad, {pct(B['pct_delta'], 2)}), con {num(B['dias_desb'], 1)} días de "
           f"desbordamiento y {num(B['derivados'], 1)} pacientes derivados, sobre todo en la fase {fase_max} "
           f"({pct(B['por_fase'][fase_max]['cuota_delta'], 0)} del sobrecoste) (valores teóricos); {val}.")
    n_ok = (r["objetivos"]["Cumple"] == "Sí").sum()
    txt += f" Cumple {n_ok} de los {len(r['objetivos'])} objetivos."
    return txt


def recomendaciones(res, base, com, camas_df=None, camas_opt=None):
    """O7.1 · tres recomendaciones con su métrica."""
    filas = [(n, (r["objetivos"]["Cumple"] == "Sí").sum(), r["teoria"]) for n, r in res.items()]
    max_ok = max(f[1] for f in filas)
    mejor = sorted([f for f in filas if f[1] == max_ok], key=lambda f: f[2]["B"]["total_gestion"])[0]
    rec = []
    t = mejor[2]
    txt = (f"Adoptar «{mejor[0]}»: cumple {mejor[1]} de {len(res[mejor[0]]['objetivos'])} objetivos con el menor coste total "
           f"del brote incluida la reserva de camas ({eur(t['B']['total_gestion'])}).")
    if base in res and mejor[0] != base:
        tb = res[base]["teoria"]
        txt += (f" Frente a «{base}» ahorra {eur(tb['B']['total_gestion'] - t['B']['total_gestion'])} por brote y "
                f"{eur(tb['A']['EY'] - t['A']['EY'])} por paciente.")
    rec.append(txt)
    if camas_df is not None and camas_opt is not None:
        rec.append(f"Reservar {camas_opt} camas UCI: minimiza el sobrecoste más el coste de reserva "
                   f"({eur(camas_df.loc[camas_opt, 'Sobrecoste + reserva (€)'])} por brote con "
                   f"{eur(com['coste_cama_dia'])} por cama y día); a partir de ahí el ahorro marginal de otra cama "
                   f"({eur(camas_df['Ahorro marginal de la cama (€)'].get(camas_opt + 1, float('nan')))}) no cubre su reserva "
                   f"({eur(camas_df['Coste de reservarla (€)'].iloc[0])}).")
    if base in res:
        B = res[base]["teoria"]["B"]
        fmax = max(B["por_fase"], key=lambda f: B["por_fase"][f]["delta"])
        pf = B["por_fase"][fmax]
        rec.append(f"Concentrar las medidas en la fase {fmax}: genera el {pct(pf['cuota_delta'], 0)} del sobrecoste por "
                   f"capacidad ({num(pf['lambda'], 0)} ingresos/día y {pct(pf['p_persistente'], 0)} de Persistentes, con "
                   f"{num(pf['lambda_comp'], 2)} complicados al día frente a {B['camas']} camas); medidas como el "
                   "aislamiento preventivo, las camas de alquiler en el Pico o el diagnóstico precoz de Persistentes "
                   "actúan justo ahí.")
    return rec


def texto_recomendacion(res, base, com, camas_df=None, camas_opt=None):
    return " ".join(f"({i + 1}) {r}" for i, r in enumerate(recomendaciones(res, base, com, camas_df, camas_opt)))


KPIS_CUADRO = [
    ("Coste medio por paciente E(Y) y percentil 95", "provisión de fondos por paciente y colchón para los casos caros"),
    ("Probabilidad de complicación y días medios de ingreso", "efectividad clínica del tratamiento"),
    ("Coste total del brote con y sin restricción de capacidad", "presupuesto del brote y peso de la falta de camas"),
    ("Sobrecoste por capacidad por fase (€ y %)", "en qué fase actuar"),
    ("Días con desbordamiento y pacientes derivados", "dimensionamiento de la UCI y nivel de servicio"),
    ("P(UCI llena dos días seguidos)", "riesgo operativo de saturación sostenida"),
    ("ICER de cada alternativa terapéutica", "decisión coste-efectividad de nuevos fármacos"),
]
