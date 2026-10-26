# -*- coding: utf-8 -*-
"""EC3 · Rentabilidad de proyectos de calidad en «Bio-Distri S.A.» (todas las cantidades en miles de €).

Bloque A · un proyecto (método de composición): tipo T ∈ {1, 2, 3, 4} (IoT, IA, Empaque, Auditoría);
    I | T ~ LN(μ_T, σ_T), R | T ~ Beta(α_T, β_T), S | T ~ Gamma(k_T, θ_T), independientes dado T;
    Y = V·R + S − I − Pen(R), Pen(R) = K·I{R < u} + c·(u − R)⁺ (escalón del enunciado: K = 20, c = 0)
    y, con garantía de resultados, + g·I·I{R < u}.
Bloque B · programa plurianual: cada centro mantiene su T durante H años; cada año se generan de nuevo I_t, R⁰_t y
    S⁰_t; aprendizaje R_t = 1 − (1 − R⁰_t)·λ^(t−1) y, sin mantenimiento, S_t = S⁰_t·δ^(t−1) (con mantenimiento,
    S_t = S⁰_t y se paga C_mant). VAN = Σ Y_t/(1 + r)^(t−1). Ambas políticas usan la misma base (pareada).

Teoría exacta: momentos cerrados (Beta truncada, lognormal y Gamma) para E, Var, CV y la descomposición de Var(Y);
la distribución de Y y del VAN se obtiene en una rejilla fina convolucionando las distribuciones exactas de cada término
(FFT), de la que salen P(Y < 0), el VaR (percentil 5) y el CVaR 5 %. La ganancia del mantenimiento,
Δ = VAN_con − VAN_sin = Σ d_t·S⁰_t·(1 − δ^(t−1)) − C_mant·Σ d_t, no depende de R ni de I.

Material de Javier Morales, equipo IA4LEGOS (CC BY-SA 4.0)."""
import copy
import math

import numpy as np
import pandas as pd

try:
    from scipy import stats as _stats
    from scipy.special import ndtri as _ndtri, ndtr as _ndtr, betainc as _betainc, betaincinv as _betaincinv, \
        gammainc as _gammainc, gammaincinv as _gammaincinv
    from scipy.signal import fftconvolve as _fftconvolve
except Exception:          # pragma: no cover
    _stats = None

PROYECTOS = (1, 2, 3, 4)
NOMBRE = {1: "IoT", 2: "IA", 3: "Empaque", 4: "Auditoría"}
NOMBRE_LARGO = {1: "Optimización de cadena de frío (IoT)", 2: "IA para predicción de demanda",
                3: "Empaques activos (bio-polímeros)", 4: "Auditoría de procesos y proveedores"}
TERMINOS = ("Merma evitada V·R", "Ahorro S", "Inversión −I", "Penalización", "Garantía")
N_DIAGRAMA = 40
HMAX = 15
H_A, H_B, H_D = 0.05, 0.5, 0.005          # pasos de la rejilla (miles de €): proyecto, VAN y ganancia Δ
COLA = 1e-10                              # probabilidad despreciada en las colas de la lognormal

COMUNES_DEF = {
    "dist": {1: {"mu_I": 4.0, "sigma_I": 0.5, "a_R": 5.0, "b_R": 2.0, "k_S": 10.0, "theta_S": 1.0},
             2: {"mu_I": 3.5, "sigma_I": 0.8, "a_R": 4.0, "b_R": 4.0, "k_S": 5.0, "theta_S": 2.0},
             3: {"mu_I": 3.0, "sigma_I": 0.3, "a_R": 2.0, "b_R": 5.0, "k_S": 2.0, "theta_S": 1.0},
             4: {"mu_I": 2.5, "sigma_I": 0.2, "a_R": 2.0, "b_R": 2.0, "k_S": 3.0, "theta_S": 1.5}},
    "objetivos": {"EY_MIN": 225.0, "PPERD_MAX": 0.021, "VAN_MIN": 1665.0, "CVAR_MIN": 1175.0},
}
ESC_BASE = {"p": [0.35, 0.25, 0.25, 0.15], "V": 500.0, "K": 20.0, "c_pen": 0.0, "umbral_R": 0.10, "garantia": 0.0,
            "H": 6, "lam": 0.80, "delta": 0.85, "C_mant": 3.0, "r": 0.05, "subvencion": 0.0, "t0": 1,
            "politica": "con", "descripcion": ""}
POLITICAS = {"con": "con mantenimiento", "sin": "sin mantenimiento"}


def escenario(**cambios):
    e = copy.deepcopy(ESC_BASE)
    e.update(cambios)
    return e


def normalizar(p):
    p = np.clip(np.asarray(p, dtype=float), 0, None)
    return p / p.sum() if p.sum() > 0 else np.full(len(p), 1 / len(p))


def _dist(com, f):
    d = com["dist"][f]
    return (float(d["mu_I"]), float(d["sigma_I"]), float(d["a_R"]), float(d["b_R"]), float(d["k_S"]),
            float(d["theta_S"]))


def descuentos(esc):
    H = int(esc["H"]); r = float(esc["r"])
    return np.array([(1 + r) ** -(t - 1) for t in range(1, H + 1)])


def c_efectivo(esc):
    return float(esc["C_mant"]) * (1 - float(esc.get("subvencion") or 0.0))


# ==============================================================================================
# Momentos exactos de un año (Bloque A = año 1 sin experiencia previa)
# ==============================================================================================
def _beta_trunc(a, b, thr):
    """F(thr), E[R·I{R < thr}], E[R²·I{R < thr}] de R ~ Beta(a, b)."""
    if thr <= 0:
        return 0.0, 0.0, 0.0
    thr = min(thr, 1.0)
    F = float(_betainc(a, b, thr))
    m1 = a / (a + b) * float(_betainc(a + 1, b, thr))
    m2 = a * (a + 1) / ((a + b) * (a + b + 1)) * float(_betainc(a + 2, b, thr))
    return F, m1, m2


def momentos_anyo(com, esc, f, e=0, fS=1.0, mant=0):
    """Momentos de los términos de Y_t para el tipo f con exponente de aprendizaje e (L = λ^e), factor de ahorro fS y
    mantenimiento mant ∈ {0, 1}. Términos: V·R_t, S_t, −I_t, −Pen(R_t), garantía g·I_t·I{R_t < u}."""
    mu, sg, a, b, k, th = _dist(com, f)
    V, K, c, u, g = (float(esc["V"]), float(esc["K"]), float(esc["c_pen"]), float(esc["umbral_R"]),
                     float(esc["garantia"]))
    L = float(esc["lam"]) ** e
    ut = 1 - (1 - u) / L if L > 0 else -1.0             # R_t < u ⇔ R⁰ < u_t
    F, m1, m2 = _beta_trunc(a, b, ut)
    mR, E2R = a / (a + b), a * (a + 1) / ((a + b) * (a + b + 1))
    ER = 1 - L + L * mR
    ER2 = (1 - L) ** 2 + 2 * (1 - L) * L * mR + L * L * E2R
    ERJ = (1 - L) * F + L * m1
    EQ = L * (ut * F - m1) if F > 0 else 0.0
    EQ2 = L * L * (ut * ut * F - 2 * ut * m1 + m2) if F > 0 else 0.0
    ERQ = L * ((1 - L) * (ut * F - m1) + L * (ut * m1 - m2)) if F > 0 else 0.0
    EP = K * F + c * EQ
    EP2 = K * K * F + 2 * K * c * EQ + c * c * EQ2
    ERP = K * ERJ + c * ERQ
    EI = math.exp(mu + sg * sg / 2); EI2 = math.exp(2 * mu + 2 * sg * sg)
    ES = fS * k * th; ES2 = fS * fS * (k * k * th * th + k * th * th)
    EG = g * EI * F; EG2 = g * g * EI2 * F
    E = np.array([V * ER, ES, -EI, -EP, EG])
    M = np.zeros((5, 5))
    M[0, 0] = V * V * ER2; M[0, 1] = V * ER * ES; M[0, 2] = -V * ER * EI; M[0, 3] = -V * ERP; M[0, 4] = V * g * EI * ERJ
    M[1, 1] = ES2; M[1, 2] = -ES * EI; M[1, 3] = -ES * EP; M[1, 4] = ES * EG
    M[2, 2] = EI2; M[2, 3] = EI * EP; M[2, 4] = -g * EI2 * F
    M[3, 3] = EP2; M[3, 4] = -g * EI * EP
    M[4, 4] = EG2
    M = np.triu(M) + np.triu(M, 1).T
    cov = M - np.outer(E, E)
    const = -c_efectivo(esc) * mant
    EY = float(E.sum() + const)
    VY = float(cov.sum())
    return {"E": E, "cov": cov, "EY": EY, "VY": VY, "const": const, "F": F, "ut": ut, "L": L, "ER": ER,
            "ES": ES, "EI": EI, "EP": EP}


# ==============================================================================================
# Distribuciones en rejilla (convolución FFT de las distribuciones exactas de cada término)
# ==============================================================================================
class Rejilla:
    """Distribución discreta en la rejilla {(lo + i)·h}."""
    __slots__ = ("lo", "p", "h")

    def __init__(self, lo, p, h):
        self.lo, self.p, self.h = int(lo), np.asarray(p, dtype=float), float(h)

    @property
    def x(self):
        return (self.lo + np.arange(len(self.p))) * self.h

    def recortar(self, eps=1e-15):
        nz = np.nonzero(self.p > eps)[0]
        if len(nz) == 0:
            return self
        i0, i1 = nz[0], nz[-1] + 1
        return Rejilla(self.lo + i0, self.p[i0:i1], self.h)

    def desplazar(self, c):
        return Rejilla(self.lo + int(round(c / self.h)), self.p, self.h)

    def masa(self):
        return float(self.p.sum())

    def cdf(self, y, estricto=True):
        x = self.x
        mk = x < y - 1e-12 if estricto else x <= y + 1e-12
        return float(self.p[mk].sum() + (0.5 * self.p[np.isclose(x, y)].sum() if estricto else 0.0))

    def cuantil(self, q):
        P = np.cumsum(self.p) / self.masa()
        i = int(np.searchsorted(P, q))
        return float(self.x[min(i, len(P) - 1)])

    def cvar(self, alfa=0.05):
        """Media del alfa·100 % de valores más bajos (beneficio medio en los peores casos)."""
        p = self.p / self.masa(); x = self.x
        P = np.cumsum(p)
        i = int(np.searchsorted(P, alfa))
        i = min(i, len(P) - 1)
        prev = P[i - 1] if i > 0 else 0.0
        return float(((p[:i] * x[:i]).sum() + (alfa - prev) * x[i]) / alfa)


def convolucion(A, B):
    p = np.clip(_fftconvolve(A.p, B.p), 0, None)
    return Rejilla(A.lo + B.lo, p, A.h).recortar()


def mezcla(partes):
    """partes: [(peso, Rejilla)] con el mismo paso."""
    partes = [(w, R) for w, R in partes if w > 0 and len(R.p)]
    lo = min(R.lo for _, R in partes); hi = max(R.lo + len(R.p) for _, R in partes)
    p = np.zeros(hi - lo)
    for w, R in partes:
        p[R.lo - lo:R.lo - lo + len(R.p)] += w * R.p
    return Rejilla(lo, p, partes[0][1].h)


def _desde_cdf(cdf, vmin, vmax, h):
    m0 = int(math.floor(vmin / h)) - 1; m1 = int(math.ceil(vmax / h)) + 1
    m = np.arange(m0, m1 + 1)
    bordes = np.concatenate([(m - 0.5) * h, [(m[-1] + 0.5) * h]])
    F = cdf(bordes)
    return Rejilla(m0, np.clip(np.diff(F), 0, None), h).recortar()


def _rej_lognormal(mu, sg, escala, h):
    """Distribución de −escala·I con I ~ LN(μ, σ) (escala > 0)."""
    if escala <= 0:
        return Rejilla(0, [1.0], h)
    hi = escala * math.exp(mu + sg * float(_ndtri(1 - COLA)))
    def cdf(v):          # P(−escala·I ≤ v) = P(I ≥ −v/escala)
        z = np.where(v < 0, (np.log(np.maximum(-v, 1e-300) / escala) - mu) / sg, -np.inf)
        return 1 - _ndtr(z)
    return _desde_cdf(cdf, -hi, 0.0, h)


def _rej_gamma(k, escala, h):
    if escala <= 0:
        return Rejilla(0, [1.0], h)
    hi = escala * float(_gammaincinv(k, 1 - COLA))
    return _desde_cdf(lambda v: _gammainc(k, np.maximum(v, 0) / escala), 0.0, hi, h)


def _rej_R(a, b, V, K, c, u, L, d, parte, h):
    """Distribución de d·(V·R_t − Pen(R_t)) restringida a R_t < u (parte = 1) o R_t ≥ u (parte = 0)."""
    ut = 1 - (1 - u) / L
    F_ut = float(_betainc(a, b, ut)) if ut > 0 else 0.0
    def R0_de(Rt):
        return np.clip(1 - (1 - Rt) / L, 0, 1)
    if parte == 1:
        if F_ut <= 0:
            return None
        lo_v = d * ((V + c) * (1 - L) - c * u - K)
        hi_v = d * (V * u - K)
        def cdf(v):
            Rt = (v / d + c * u + K) / (V + c)
            return np.minimum(_betainc(a, b, R0_de(Rt)), F_ut)
    else:
        lo_v = d * V * max(u, 1 - L); hi_v = d * V
        def cdf(v):
            return np.maximum(_betainc(a, b, R0_de(v / (d * V))) - F_ut, 0.0)
    return _desde_cdf(cdf, lo_v, hi_v, h)


def rejilla_anyo(com, esc, f, e=0, fS=1.0, mant=0, d=1.0, h=H_A):
    """Distribución exacta (en rejilla) de d·Y_t para el tipo f."""
    mu, sg, a, b, k, th = _dist(com, f)
    V, K, c, u, g = (float(esc["V"]), float(esc["K"]), float(esc["c_pen"]), float(esc["umbral_R"]),
                     float(esc["garantia"]))
    L = float(esc["lam"]) ** e
    partes = []
    for parte, esc_I in ((1, d * (1 - g)), (0, d)):
        Rr = _rej_R(a, b, V, K, c, u, L, d, parte, h)
        if Rr is None or Rr.masa() <= 0:
            continue
        partes.append(convolucion(Rr, _rej_lognormal(mu, sg, esc_I, h)))
    X = partes[0] if len(partes) == 1 else mezcla([(1.0, partes[0]), (1.0, partes[1])])
    Y = convolucion(X, _rej_gamma(k, d * fS * th, h))
    return Y.desplazar(d * (-c_efectivo(esc) * mant))


# ==============================================================================================
# Teoría · Bloque A (un proyecto)
# ==============================================================================================
def teoria_A(com, esc, con_rejilla=True):
    p = normalizar(esc["p"])
    ms = {f: momentos_anyo(com, esc, f) for f in PROYECTOS}
    EY = float(sum(pi * ms[f]["EY"] for pi, f in zip(p, PROYECTOS)))
    EY2 = float(sum(pi * (ms[f]["VY"] + ms[f]["EY"] ** 2) for pi, f in zip(p, PROYECTOS)))
    VY = EY2 - EY ** 2
    E = sum(pi * ms[f]["E"] for pi, f in zip(p, PROYECTOS))
    M2 = sum(pi * (ms[f]["cov"] + np.outer(ms[f]["E"], ms[f]["E"])) for pi, f in zip(p, PROYECTOS))
    cov = M2 - np.outer(E, E)
    contrib = cov.sum(axis=1) / VY if VY > 0 else np.zeros(5)
    S_T = float(sum(pi * (ms[f]["EY"] - EY) ** 2 for pi, f in zip(p, PROYECTOS)) / VY) if VY > 0 else 0.0
    out = {"p": p, "EY": EY, "VY": VY, "sdY": math.sqrt(max(VY, 0)), "CV": math.sqrt(max(VY, 0)) / EY if EY else 0.0,
           "contrib": contrib, "E_terminos": E, "S_T": S_T, "mom": ms}
    pf = {}
    for pi, f in zip(p, PROYECTOS):
        m = ms[f]; sd = math.sqrt(max(m["VY"], 0))
        pf[f] = {"p": float(pi), "EY": m["EY"], "sd": sd, "CV": sd / m["EY"] if m["EY"] else float("nan"),
                 "Ppen": m["F"], "EI": m["EI"], "ER": m["ER"], "ES": m["ES"]}
    if con_rejilla:
        rej = {f: rejilla_anyo(com, esc, f) for f in PROYECTOS}
        mix = mezcla([(pi, rej[f]) for pi, f in zip(p, PROYECTOS)])
        out.update({"Pperd": mix.cdf(0.0), "VaR": mix.cuantil(0.05), "CVaR": mix.cvar(0.05),
                    "P50": mix.cuantil(0.5), "rejilla": mix})
        for pi, f in zip(p, PROYECTOS):
            pf[f].update({"Pperd": rej[f].cdf(0.0), "VaR": rej[f].cuantil(0.05), "CVaR": rej[f].cvar(0.05)})
        for pi, f in zip(p, PROYECTOS):
            pf[f]["peso_perd"] = pi * pf[f]["Pperd"] / out["Pperd"] if out["Pperd"] > 0 else 0.0
            pf[f]["peso_cola"] = pi * rej[f].cdf(out["VaR"], estricto=False) / 0.05
    out["por_tipo"] = pf
    return out


# ==============================================================================================
# Teoría · Bloque B (programa plurianual)
# ==============================================================================================
def _exponente(esc, t):
    return t - 1 + int(esc.get("t0") or 1) - 1


def momentos_van(com, esc, f, politica):
    d = descuentos(esc); dl = float(esc["delta"])
    E, V = 0.0, 0.0
    for t, dt in enumerate(d, start=1):
        m = momentos_anyo(com, esc, f, _exponente(esc, t), 1.0 if politica == "con" else dl ** (t - 1),
                          1 if politica == "con" else 0)
        E += dt * m["EY"]; V += dt * dt * m["VY"]
    return E, V


def rejilla_van(com, esc, f, politica, h=H_B):
    d = descuentos(esc); dl = float(esc["delta"])
    R = None
    for t, dt in enumerate(d, start=1):
        Rt = rejilla_anyo(com, esc, f, _exponente(esc, t), 1.0 if politica == "con" else dl ** (t - 1),
                          1 if politica == "con" else 0, d=dt, h=h)
        R = Rt if R is None else convolucion(R, Rt)
    return R


def ganancia_tipo(com, esc, f):
    """Δ = Σ d_t·S⁰_t·(1 − δ^(t−1)) − C_ef·Σ d_t: media, varianza y P(Δ > 0) exactas."""
    mu, sg, a, b, k, th = _dist(com, f)
    d = descuentos(esc); dl = float(esc["delta"])
    w = d * (1 - dl ** np.arange(len(d)))
    C = c_efectivo(esc) * d.sum()
    E = float(k * th * w.sum() - C); V = float(k * th * th * (w ** 2).sum())
    R = None
    for wt in w:
        if wt <= 0:
            continue
        g = _rej_gamma(k, wt * th, H_D)
        R = g if R is None else convolucion(R, g)
    P = 1.0 - R.cdf(C, estricto=False) if R is not None else (1.0 if -C > 0 else 0.0)
    return {"E": E, "V": V, "P": P, "C_ind": float(k * th * w.sum() / d.sum() / (1 - float(esc.get("subvencion") or 0)))
            if (1 - float(esc.get("subvencion") or 0)) > 0 else float("inf"), "ES": k * th}


def teoria_B(com, esc, con_rejilla=True):
    p = normalizar(esc["p"])
    out = {"p": p, "por_tipo": {}}
    for pol in ("con", "sin"):
        Es, Vs = {}, {}
        for f in PROYECTOS:
            Es[f], Vs[f] = momentos_van(com, esc, f, pol)
        E = float(sum(pi * Es[f] for pi, f in zip(p, PROYECTOS)))
        V = float(sum(pi * (Vs[f] + Es[f] ** 2) for pi, f in zip(p, PROYECTOS)) - E * E)
        out[f"E_{pol}"], out[f"V_{pol}"], out[f"sd_{pol}"] = E, V, math.sqrt(max(V, 0))
        for f in PROYECTOS:
            out["por_tipo"].setdefault(f, {})[f"E_{pol}"] = Es[f]
            out["por_tipo"][f][f"sd_{pol}"] = math.sqrt(max(Vs[f], 0))
        if con_rejilla:
            rej = {f: rejilla_van(com, esc, f, pol) for f in PROYECTOS}
            mix = mezcla([(pi, rej[f]) for pi, f in zip(p, PROYECTOS)])
            out[f"VaR_{pol}"], out[f"CVaR_{pol}"] = mix.cuantil(0.05), mix.cvar(0.05)
            out[f"Pperd_{pol}"] = mix.cdf(0.0)
            out[f"rejilla_{pol}"] = mix
    G = {f: ganancia_tipo(com, esc, f) for f in PROYECTOS}
    out["E_gan"] = float(sum(pi * G[f]["E"] for pi, f in zip(p, PROYECTOS)))
    out["V_gan"] = float(sum(pi * (G[f]["V"] + G[f]["E"] ** 2) for pi, f in zip(p, PROYECTOS)) - out["E_gan"] ** 2)
    out["sd_gan"] = math.sqrt(max(out["V_gan"], 0))
    out["P_gan"] = float(sum(pi * G[f]["P"] for pi, f in zip(p, PROYECTOS)))
    for pi, f in zip(p, PROYECTOS):
        out["por_tipo"][f].update({"p": float(pi), "E_gan": G[f]["E"], "sd_gan": math.sqrt(G[f]["V"]), "P_gan": G[f]["P"],
                                   "C_ind": G[f]["C_ind"], "ES": G[f]["ES"],
                                   "cuota_gan": pi * G[f]["E"] / out["E_gan"] if out["E_gan"] else 0.0})
    d = descuentos(esc); dl = float(esc["delta"])
    w = (d * (1 - dl ** np.arange(len(d)))).sum()
    ES = float(sum(pi * G[f]["ES"] for pi, f in zip(p, PROYECTOS)))
    X = float(esc.get("subvencion") or 0)
    out["C_ind"] = ES * w / d.sum() / (1 - X) if X < 1 else float("inf")        # C_mant que iguala ambas políticas
    Cb = float(esc["C_mant"])
    out["X_ind"] = 1 - ES * w / (Cb * d.sum()) if Cb > 0 else 0.0                # subvención que iguala ambas políticas
    out["var_pareada"] = out["V_gan"]
    out["var_independiente"] = out["V_con"] + out["V_sin"]
    pol = esc.get("politica", "con")
    out["E_pol"] = out[f"E_{pol}"]
    if con_rejilla:
        out["CVaR_pol"], out["VaR_pol"] = out[f"CVaR_{pol}"], out[f"VaR_{pol}"]
    out["H"] = int(esc["H"])
    return out


def teoria(com, esc):
    return {"A": teoria_A(com, esc), "B": teoria_B(com, esc)}


# ==============================================================================================
# Simulación (transformada inversa con números aleatorios comunes)
# ==============================================================================================
def tipo_desde_U(u, p):
    F = np.cumsum(normalizar(p))
    return 1 + (u > F[0]).astype(int) + (u > F[1]).astype(int) + (u > F[2]).astype(int)


def variables(com, T, UI, UR, US):
    """I, R⁰ y S⁰ por transformada inversa (arrays de la misma forma que T)."""
    I = np.zeros(T.shape); R = np.zeros(T.shape); S = np.zeros(T.shape)
    for f in PROYECTOS:
        mk = T == f
        if not mk.any():
            continue
        mu, sg, a, b, k, th = _dist(com, f)
        I[mk] = np.exp(mu + sg * _ndtri(UI[mk]))
        R[mk] = _betaincinv(a, b, UR[mk])
        S[mk] = _gammaincinv(k, US[mk]) * th
    return I, R, S


def penalizacion(esc, R):
    u = float(esc["umbral_R"])
    return float(esc["K"]) * (R < u) + float(esc["c_pen"]) * np.maximum(u - R, 0)


def beneficio(esc, R, S, I, mant=0):
    """Términos de Y = V·R + S − I − Pen(R) + g·I·I{R < u} − C·mant."""
    g = float(esc["garantia"]); u = float(esc["umbral_R"])
    t = [float(esc["V"]) * R, S, -I, -penalizacion(esc, R), g * I * (R < u)]
    return t, sum(t) - c_efectivo(esc) * mant


def generar_proyectos(com, esc, U):
    T = tipo_desde_U(U[0], esc["p"])
    I, R, S = variables(com, T, U[1], U[2], U[3])
    terminos, Y = beneficio(esc, R, S, I)
    return {"T": T, "I": I, "R": R, "S": S, "terminos": terminos, "Y": Y}


def _tq(n, alpha=0.05):
    if n < 2:
        return float("nan")
    return float(_stats.t.ppf(1 - alpha / 2, n - 1))


def _cvar_muestra(y, alfa=0.05):
    y = np.sort(np.asarray(y)); n = len(y)
    k = alfa * n; i = int(math.floor(k))
    s = y[:i].sum() + (k - i) * (y[i] if i < n else 0.0)
    return float(s / k)


def simular_A(com, esc, nsim, replicas, semilla, progreso=None):
    """Bloque A: `replicas` macro-réplicas de `nsim` proyectos (réplica i → semilla + i)."""
    filas, Ys, Ts, traza = [], [], [], None
    for r in range(int(replicas)):
        rng = np.random.default_rng(int(semilla) + r)
        U = rng.random((4, int(nsim)))
        o = generar_proyectos(com, esc, U)
        Y, T = o["Y"], o["T"]
        vy = float(Y.var(ddof=1))
        C = np.cov(np.vstack(o["terminos"]))
        fila = {"Réplica": r + 1, "Semilla": int(semilla) + r, "Proyectos": len(Y), "E(Y)": float(Y.mean()),
                "σ(Y)": math.sqrt(vy), "P(Y < 0)": float((Y < 0).mean()), "VaR 5 %": float(np.quantile(Y, 0.05)),
                "CVaR 5 %": _cvar_muestra(Y), "P(R < umbral)": float((o["R"] < float(esc["umbral_R"])).mean())}
        for k_, nom in enumerate(TERMINOS):
            fila[f"Contrib. {nom}"] = float(C[k_].sum() / vy) if vy > 0 else 0.0
        for f in PROYECTOS:
            mk = T == f; yf = Y[mk]; nf = NOMBRE[f]
            fila[f"E(Y | {nf})"] = float(yf.mean()) if mk.any() else float("nan")
            fila[f"CV({nf})"] = float(yf.std(ddof=1) / yf.mean()) if mk.sum() > 1 else float("nan")
            fila[f"P(Y < 0 | {nf})"] = float((yf < 0).mean()) if mk.any() else float("nan")
            fila[f"Peso en las pérdidas · {nf}"] = float((mk & (Y < 0)).sum() / max(1, (Y < 0).sum()))
        filas.append(fila)
        Ys.append(Y); Ts.append(T)
        if r == 0:
            k = N_DIAGRAMA
            traza = pd.DataFrame({"Proyecto": np.arange(1, k + 1), "U_T": U[0, :k], "Tipo": [NOMBRE[t] for t in T[:k]],
                                  "I (k€)": o["I"][:k], "R": o["R"][:k], "S (k€)": o["S"][:k],
                                  "Penalización (k€)": -o["terminos"][3][:k], "Garantía (k€)": o["terminos"][4][:k],
                                  "Y (k€)": Y[:k]})
        if progreso:
            progreso()
    return {"replicas": pd.DataFrame(filas).set_index("Réplica"), "Y": np.concatenate(Ys), "T": np.concatenate(Ts),
            "traza": traza}


# ------------------------------------------------------------------------------------------- Bloque B
def base_programa(com, esc, n, semilla_rep, racha=None):
    """Base común de n centros: T (fijo), U_I, U_R, U_S (n × HMAX) y uniformes extra. `racha` = (tipo, inicio, años):
    todos los centros son del tipo dado y U_R se lleva al 10 % inferior en esos años."""
    rng = np.random.default_rng(int(semilla_rep))
    uT = rng.random(n)
    Hm = max(HMAX, int(esc["H"]))
    U = rng.random((3, n, Hm))
    extra = rng.random((2, n))
    T = tipo_desde_U(uT, esc["p"])
    if racha is not None:
        T = np.full(n, int(racha[0]))
        U = U.copy()
        i0, nr = int(racha[1]) - 1, int(racha[2])
        U[1, :, i0:i0 + nr] *= 0.10
    H = int(esc["H"])
    TT = np.repeat(T[:, None], H, axis=1)
    I, R0, S0 = variables(com, TT, U[0, :, :H], U[1, :, :H], U[2, :, :H])
    return {"T": T, "I": I, "R0": R0, "S0": S0, "uT": uT, "extra": extra}


def aplicar_politica(com, esc, base, politica, umbral_S=None, sustituir=None):
    """Y_t (n × H) de cada centro con la política «con», «sin» o «adaptativa» (mantiene el año t si el ahorro que
    observaría sin mantener, S⁰_t·δ^n, cae por debajo de umbral_S). `sustituir` = año en el que se sustituye el proyecto
    por uno nuevo del mismo tipo (inversión extra y la curva de aprendizaje se reinicia), con mantenimiento."""
    H = int(esc["H"]); lam = float(esc["lam"]); dl = float(esc["delta"])
    n = base["I"].shape[0]
    t = np.arange(1, H + 1)
    e = t - 1 + int(esc.get("t0") or 1) - 1
    if sustituir:
        m = int(sustituir)
        e = np.where(t >= m, t - m, e)
    R = 1 - (1 - base["R0"]) * lam ** e[None, :]
    S0 = base["S0"]
    if politica == "con" or sustituir:
        S = S0.copy(); M = np.ones((n, H))
    elif politica == "sin":
        S = S0 * dl ** (t - 1)[None, :]; M = np.zeros((n, H))
    else:
        S = np.zeros((n, H)); M = np.zeros((n, H)); nd = np.zeros(n)
        for j in range(H):
            obs = S0[:, j] * dl ** nd
            mant = obs < float(umbral_S)
            S[:, j] = np.where(mant, S0[:, j], obs); M[:, j] = mant
            nd = np.where(mant, 1, nd + 1)
    _, Y = beneficio(esc, R, S, base["I"])
    Y = Y - c_efectivo(esc) * M
    if sustituir:
        mu = np.zeros(n); sg = np.zeros(n)
        for f in PROYECTOS:
            mk = base["T"] == f
            mu[mk], sg[mk] = _dist(com, f)[:2]
        Y[:, int(sustituir) - 1] -= np.exp(mu + sg * _ndtri(base["extra"][0]))
    return {"Y": Y, "R": R, "S": S, "M": M}


def van(esc, Y):
    return Y @ descuentos(esc)


def simular_B(com, esc, n_centros, replicas, semilla, progreso=None):
    """Bloque B: `replicas` macro-réplicas de `n_centros` centros (réplica i → semilla + 1000 + i)."""
    filas, ejemplo = [], None
    vc_all, vs_all, T_all = [], [], []
    for r in range(int(replicas)):
        b = base_programa(com, esc, int(n_centros), int(semilla) + 1000 + r)
        pc = aplicar_politica(com, esc, b, "con"); ps = aplicar_politica(com, esc, b, "sin")
        vc, vs = van(esc, pc["Y"]), van(esc, ps["Y"])
        g = vc - vs
        ind = vc - np.roll(vs, 1)
        pol = esc.get("politica", "con")
        vp = vc if pol == "con" else vs
        fila = {"Réplica": r + 1, "Semilla": int(semilla) + 1000 + r, "Centros": int(n_centros),
                "E(VAN) con": float(vc.mean()), "E(VAN) sin": float(vs.mean()), "Ganancia E(Δ)": float(g.mean()),
                "P(Δ > 0)": float((g > 0).mean()), "VaR 5 % con": float(np.quantile(vc, 0.05)),
                "VaR 5 % sin": float(np.quantile(vs, 0.05)), "CVaR 5 % con": _cvar_muestra(vc),
                "CVaR 5 % sin": _cvar_muestra(vs), "P(VAN < 0) con": float((vc < 0).mean()),
                "E(VAN) política": float(vp.mean()), "CVaR 5 % política": _cvar_muestra(vp),
                "σ(Δ) pareado": float(g.std(ddof=1)), "σ(Δ) independiente": float(ind.std(ddof=1))}
        for f in PROYECTOS:
            mk = b["T"] == f; nf = NOMBRE[f]
            fila[f"E(Δ | {nf})"] = float(g[mk].mean()) if mk.any() else float("nan")
            fila[f"E(VAN con | {nf})"] = float(vc[mk].mean()) if mk.any() else float("nan")
            fila[f"E(VAN sin | {nf})"] = float(vs[mk].mean()) if mk.any() else float("nan")
        filas.append(fila)
        vc_all.append(vc); vs_all.append(vs); T_all.append(b["T"])
        if r == 0:
            ejemplo = {"T": b["T"][:N_DIAGRAMA], "Yc": pc["Y"][:N_DIAGRAMA], "Ys": ps["Y"][:N_DIAGRAMA],
                       "Sc": pc["S"][:N_DIAGRAMA], "Ss": ps["S"][:N_DIAGRAMA], "R": pc["R"][:N_DIAGRAMA]}
        if progreso:
            progreso()
    return {"replicas": pd.DataFrame(filas).set_index("Réplica"), "VAN_con": np.concatenate(vc_all),
            "VAN_sin": np.concatenate(vs_all), "T": np.concatenate(T_all), "ejemplo": ejemplo}


# ==============================================================================================
# Resúmenes (teórico frente a simulado)
# ==============================================================================================
def _ic_media(v):
    v = np.asarray(v, dtype=float); v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(v.mean())
    if len(v) < 2:
        return m, float("nan"), float("nan")
    se = float(v.std(ddof=1)) / math.sqrt(len(v)); tq = _tq(len(v))
    return m, m - tq * se, m + tq * se


def resumen(rep, filas):
    out = []
    for nombre, (col, teo) in filas.items():
        m, lo, hi = _ic_media(rep[col])
        ok = None
        if teo is not None and lo == lo and hi - lo > 1e-12:
            tol = 1e-9 * max(1.0, abs(teo))
            ok = bool(lo - tol <= teo <= hi + tol)
        out.append({"KPI": nombre, "Teórico": teo, "Simulado · media": m, "IC95 inf": lo, "IC95 sup": hi,
                    "Error relativo": (m - teo) / abs(teo) if (teo is not None and abs(teo) > 1e-12) else None,
                    "Teórico en IC95": ok})
    return pd.DataFrame(out).set_index("KPI")


def resumen_A(mc, tA):
    d = {"Beneficio esperado E(Y) (k€)": ("E(Y)", tA["EY"]),
         "Desviación típica σ(Y) (k€)": ("σ(Y)", tA["sdY"]),
         "Probabilidad de pérdida P(Y < 0)": ("P(Y < 0)", tA["Pperd"]),
         "VaR 95 % · percentil 5 de Y (k€)": ("VaR 5 %", tA["VaR"]),
         "CVaR 5 % · media del 5 % peor (k€)": ("CVaR 5 %", tA["CVaR"]),
         "Probabilidad de penalización P(R < umbral)": ("P(R < umbral)",
                                                       float(sum(tA["p"][i] * tA["por_tipo"][f]["Ppen"] for i, f in enumerate(PROYECTOS))))}
    for f in PROYECTOS:
        nf = NOMBRE[f]; pf = tA["por_tipo"][f]
        d[f"E(Y | {nf}) (k€)"] = (f"E(Y | {nf})", pf["EY"])
        d[f"CV(Y | {nf})"] = (f"CV({nf})", pf["CV"])
        d[f"P(Y < 0 | {nf})"] = (f"P(Y < 0 | {nf})", pf["Pperd"])
        d[f"Peso de {nf} en las pérdidas"] = (f"Peso en las pérdidas · {nf}", pf["peso_perd"])
    for k_, nom in enumerate(TERMINOS):
        d[f"Contribución a Var(Y) · {nom}"] = (f"Contrib. {nom}", float(tA["contrib"][k_]))
    return resumen(mc["replicas"], d)


def resumen_B(mc, tB):
    d = {"VAN esperado con mantenimiento (k€)": ("E(VAN) con", tB["E_con"]),
         "VAN esperado sin mantenimiento (k€)": ("E(VAN) sin", tB["E_sin"]),
         "Ganancia media pareada E(Δ) (k€)": ("Ganancia E(Δ)", tB["E_gan"]),
         "P(el mantenimiento compensa) P(Δ > 0)": ("P(Δ > 0)", tB["P_gan"]),
         "VaR 95 % del VAN con mantenimiento (k€)": ("VaR 5 % con", tB["VaR_con"]),
         "VaR 95 % del VAN sin mantenimiento (k€)": ("VaR 5 % sin", tB["VaR_sin"]),
         "CVaR 5 % del VAN con mantenimiento (k€)": ("CVaR 5 % con", tB["CVaR_con"]),
         "CVaR 5 % del VAN sin mantenimiento (k€)": ("CVaR 5 % sin", tB["CVaR_sin"]),
         "σ(Δ) por centro · diseño pareado (k€)": ("σ(Δ) pareado", tB["sd_gan"]),
         "σ(Δ) por centro · diseño independiente (k€)": ("σ(Δ) independiente", math.sqrt(tB["var_independiente"]))}
    for f in PROYECTOS:
        nf = NOMBRE[f]
        d[f"E(Δ | {nf}) (k€)"] = (f"E(Δ | {nf})", tB["por_tipo"][f]["E_gan"])
    return resumen(mc["replicas"], d)


# ==============================================================================================
# Objetivos, análisis de escenarios y diferencias
# ==============================================================================================
OBJ_NOMBRES = {"EY_MIN": "Beneficio esperado por proyecto E(Y)", "PPERD_MAX": "Probabilidad de pérdida P(Y < 0)",
               "VAN_MIN": "VAN esperado del programa (política elegida)",
               "CVAR_MIN": "CVaR 5 % del VAN (política elegida)"}


def evaluar_objetivos(t, obj):
    A, B = t["A"], t["B"]
    filas = []
    def fila(c, valor, txt, cumple):
        filas.append({"Objetivo": OBJ_NOMBRES[c], "Valor": valor, "Umbral": txt, "Cumple": "Sí" if cumple else "No"})
    if obj.get("EY_MIN") is not None:
        fila("EY_MIN", A["EY"], f"≥ {keur(obj['EY_MIN'])}", A["EY"] >= obj["EY_MIN"] - 1e-9)
    if obj.get("PPERD_MAX") is not None:
        fila("PPERD_MAX", A["Pperd"], f"≤ {pct(obj['PPERD_MAX'], 1)}", A["Pperd"] <= obj["PPERD_MAX"] + 1e-12)
    if obj.get("VAN_MIN") is not None:
        fila("VAN_MIN", B["E_pol"], f"≥ {keur(obj['VAN_MIN'])}", B["E_pol"] >= obj["VAN_MIN"] - 1e-9)
    if obj.get("CVAR_MIN") is not None:
        fila("CVAR_MIN", B["CVaR_pol"], f"≥ {keur(obj['CVAR_MIN'])}", B["CVaR_pol"] >= obj["CVAR_MIN"] - 1e-9)
    return pd.DataFrame(filas, columns=["Objetivo", "Valor", "Umbral", "Cumple"]).set_index("Objetivo")


def analizar_escenario(esc, com, aj, progreso=None):
    t = teoria(com, esc)
    mA = simular_A(com, esc, aj["nsim"], aj["replicas"], aj["semilla"], progreso)
    mB = simular_B(com, esc, aj["centros"], aj["replicas_B"], aj["semilla"], progreso)
    return {"esc": dict(esc), "teoria": t, "mcA": mA, "mcB": mB, "resA": resumen_A(mA, t["A"]),
            "resB": resumen_B(mB, t["B"]), "objetivos": evaluar_objetivos(t, com["objetivos"])}


def diferencias(res, base):
    """Diferencias frente al escenario base (positivo = mejora): pareadas réplica a réplica."""
    if base not in res:
        return None
    rb = res[base]
    filas = []
    for n, r in res.items():
        if n == base:
            continue
        ta, tb = rb["teoria"], r["teoria"]
        dA = r["mcA"]["replicas"]["E(Y)"] - rb["mcA"]["replicas"]["E(Y)"]
        dB = r["mcB"]["replicas"]["E(VAN) política"] - rb["mcB"]["replicas"]["E(VAN) política"]
        mA, loA, hiA = _ic_media(dA); mB, loB, hiB = _ic_media(dB)
        filas.append({"Escenario": n,
                      "ΔE(Y) por proyecto · teórico (k€)": tb["A"]["EY"] - ta["A"]["EY"],
                      "ΔE(Y) por proyecto · simulado (k€)": mA, "IC95 ΔE(Y) (k€)": (loA, hiA),
                      "ΔP(Y < 0) · teórico": tb["A"]["Pperd"] - ta["A"]["Pperd"],
                      "ΔVaR 95 % · teórico (k€)": tb["A"]["VaR"] - ta["A"]["VaR"],
                      "ΔCVaR 5 % · teórico (k€)": tb["A"]["CVaR"] - ta["A"]["CVaR"],
                      "ΔE(VAN) política · teórico (k€)": tb["B"]["E_pol"] - ta["B"]["E_pol"],
                      "ΔE(VAN) política · simulado (k€)": mB, "IC95 ΔE(VAN) (k€)": (loB, hiB),
                      "ΔCVaR 5 % del VAN · teórico (k€)": tb["B"]["CVaR_pol"] - ta["B"]["CVaR_pol"],
                      "Ganancia del mantenimiento · teórico (k€)": tb["B"]["E_gan"]})
    return pd.DataFrame(filas).set_index("Escenario") if filas else None


def tabla_comparativa(res):
    cols = {}
    for n, r in res.items():
        A, B, e = r["teoria"]["A"], r["teoria"]["B"], r["esc"]
        ra, rb = r["resA"], r["resB"]
        cols[n] = {
            "Teórico · E(Y) por proyecto (k€)": A["EY"], "Teórico · P(Y < 0)": A["Pperd"],
            "Teórico · VaR 95 % de Y (k€)": A["VaR"], "Teórico · CVaR 5 % de Y (k€)": A["CVaR"],
            "Teórico · E(VAN) con mantenimiento (k€)": B["E_con"], "Teórico · E(VAN) sin mantenimiento (k€)": B["E_sin"],
            "Teórico · ganancia E(Δ) (k€)": B["E_gan"], "Teórico · P(Δ > 0)": B["P_gan"],
            "Teórico · CVaR 5 % VAN política (k€)": B["CVaR_pol"],
            "Teórico · C_mant de indiferencia (k€/año)": B["C_ind"],
            "Simulado · E(Y) por proyecto (k€)": ra.loc["Beneficio esperado E(Y) (k€)", "Simulado · media"],
            "Simulado · P(Y < 0)": ra.loc["Probabilidad de pérdida P(Y < 0)", "Simulado · media"],
            "Simulado · ganancia E(Δ) (k€)": rb.loc["Ganancia media pareada E(Δ) (k€)", "Simulado · media"],
            "Simulado · E(VAN) política (k€)": r["mcB"]["replicas"]["E(VAN) política"].mean(),
            "Política elegida": POLITICAS.get(e.get("politica", "con"), "—"),
            "Objetivos cumplidos": f"{(r['objetivos']['Cumple'] == 'Sí').sum()} de {len(r['objetivos'])}",
        }
    return pd.DataFrame(cols)


def tabla_objetivos(res):
    return pd.DataFrame({n: r["objetivos"]["Cumple"] for n, r in res.items()})


# ==============================================================================================
# Mantenimiento: barridos (O5.1, O5.2, O6.1, O6.3, O8.5) y racha (O6.2)
# ==============================================================================================
PARAM_BARRIDO = {"delta": "Tasa de degradación δ", "C_mant": "Coste de mantenimiento C_mant (k€/año)",
                 "H": "Horizonte H (años)", "subvencion": "Subvención del mantenimiento X"}


def barrido(com, esc, par, valores, centros, replicas, semilla, progreso=None):
    """Ganancia del mantenimiento al variar un parámetro: teórica y simulada con los mismos números aleatorios."""
    filas = []
    for v in valores:
        e2 = dict(esc); e2[par] = int(v) if par == "H" else float(v)
        tB = teoria_B(com, e2, con_rejilla=False)
        gs = []
        for r in range(int(replicas)):
            b = base_programa(com, e2, int(centros), int(semilla) + 1000 + r)
            gs.append(float((van(e2, aplicar_politica(com, e2, b, "con")["Y"]) -
                             van(e2, aplicar_politica(com, e2, b, "sin")["Y"])).mean()))
        m, lo, hi = _ic_media(gs)
        fila = {PARAM_BARRIDO[par]: v, "Teórico · E(Δ) (k€)": tB["E_gan"], "Simulado · E(Δ) (k€)": m,
                "IC95 E(Δ) (k€)": (lo, hi), "Teórico · P(Δ > 0)": tB["P_gan"],
                "Teórico · E(VAN) con (k€)": tB["E_con"], "Teórico · E(VAN) sin (k€)": tB["E_sin"],
                "Teórico · C_mant de indiferencia (k€/año)": tB["C_ind"]}
        for f in PROYECTOS:
            fila[f"Teórico · E(Δ | {NOMBRE[f]}) (k€)"] = tB["por_tipo"][f]["E_gan"]
        filas.append(fila)
        if progreso:
            progreso()
    return pd.DataFrame(filas).set_index(PARAM_BARRIDO[par])


def experimento_racha(com, esc, tipo, inicio=2, anyos=3, n=20000, semilla=1000):
    """O6.2 · centros de un tipo con R⁰ en el 10 % inferior durante `anyos` años desde `inicio`, frente a los mismos
    centros sin racha (mismos uniformes en el resto). Evolución año a año de R, S y VAN acumulado por política."""
    H = int(esc["H"]); d = descuentos(esc)
    out = {}
    for nom, rch in (("Sin racha", None), ("Con racha", (tipo, inicio, anyos))):
        b = base_programa(com, esc, int(n), int(semilla) + 777, racha=rch if rch else (tipo, 1, 0))
        for pol in ("con", "sin"):
            p = aplicar_politica(com, esc, b, pol)
            out[(nom, pol)] = {"R": p["R"].mean(axis=0), "S": p["S"].mean(axis=0), "Y": p["Y"].mean(axis=0),
                               "VAN acumulado": np.cumsum(p["Y"].mean(axis=0) * d),
                               "Ppen": (p["R"] < float(esc["umbral_R"])).mean(axis=0)}
    filas = []
    for t in range(H):
        f = {"Año": t + 1}
        for nom in ("Sin racha", "Con racha"):
            f[f"{nom} · R medio"] = out[(nom, "con")]["R"][t]
            f[f"{nom} · P(penalización)"] = out[(nom, "con")]["Ppen"][t]
            f[f"{nom} · S con mant. (k€)"] = out[(nom, "con")]["S"][t]
            f[f"{nom} · S sin mant. (k€)"] = out[(nom, "sin")]["S"][t]
            f[f"{nom} · VAN acum. con (k€)"] = out[(nom, "con")]["VAN acumulado"][t]
            f[f"{nom} · VAN acum. sin (k€)"] = out[(nom, "sin")]["VAN acumulado"][t]
        filas.append(f)
    return pd.DataFrame(filas).set_index("Año"), out


# ==============================================================================================
# Sensibilidad (O2.2, O3.3) y estimador
# ==============================================================================================
PARAM_SENS = {"p_Emp": "p de Empaque (cartera)", "mu_I": "μ de ln I (todos)", "sigma_I": "σ de ln I (todos)",
              "a_R": "α de la Beta de R (todos)", "b_R": "β de la Beta de R (todos)", "k_S": "k de la Gamma de S (todos)",
              "theta_S": "θ de la Gamma de S (todos)", "V": "V_stock"}


def _aplicar(com, esc, par, v):
    c2, e2 = copy.deepcopy(com), dict(esc)
    if par == "p_Emp":
        p = list(normalizar(esc["p"])); p3 = min(0.99, p[2] * (1 + v)); s = (1 - p3) / (1 - p[2])
        e2["p"] = [p[0] * s, p[1] * s, p3, p[3] * s]
    elif par == "V":
        e2["V"] = float(esc["V"]) * (1 + v)
    else:
        for f in PROYECTOS:
            c2["dist"][f][par] = float(com["dist"][f][par]) * (1 + v)
    return c2, e2


def sensibilidad_parametros(com, esc, variaciones):
    filas = []
    for par, nom in PARAM_SENS.items():
        for v in variaciones:
            t = teoria_A(*_aplicar(com, esc, par, v), con_rejilla=False)
            filas.append({"Parámetro": nom, "Variación": v, "Teórico · E(Y) (k€)": t["EY"], "Teórico · Var(Y)": t["VY"],
                          "Teórico · σ(Y) (k€)": t["sdY"]})
    elas = {}
    for par, nom in PARAM_SENS.items():
        vs = [teoria_A(*_aplicar(com, esc, par, h), con_rejilla=False)["VY"] for h in (-0.01, 0.01)]
        elas[nom] = (math.log(vs[1]) - math.log(vs[0])) / (math.log(1.01) - math.log(0.99)) if min(vs) > 0 else 0.0
    return pd.DataFrame(filas), elas


def indices_sobol(com, esc, n=20000, semilla=1000):
    """Índices de Sobol (Saltelli/Jansen) de las entradas aleatorias T, I, R y S sobre Y."""
    rng = np.random.default_rng(int(semilla) + 7)
    UA, UB = rng.random((4, n)), rng.random((4, n))
    yA = generar_proyectos(com, esc, UA)["Y"]; yB = generar_proyectos(com, esc, UB)["Y"]
    V = float(np.var(np.concatenate([yA, yB]), ddof=1))
    filas = []
    for i, nom in enumerate(("Tipo de proyecto T", "Inversión I", "Reducción R", "Ahorro S")):
        UAB = UA.copy(); UAB[i] = UB[i]
        yAB = generar_proyectos(com, esc, UAB)["Y"]
        filas.append({"Entrada": nom, "Índice de primer orden S₁": float(np.mean(yB * (yAB - yA)) / V),
                      "Índice total S_T": float(0.5 * np.mean((yA - yAB) ** 2) / V)})
    df = pd.DataFrame(filas).set_index("Entrada")
    df.loc["Tipo de proyecto T", "Teórico · S₁"] = teoria_A(com, esc, con_rejilla=False)["S_T"]
    return df


def v_equilibrio(com, esc, tipo=3, objetivo=0.0):
    """O3.3 · V_stock con el que el tipo dado alcanza E(Y | tipo) = objetivo (E(Y | T) es lineal en V)."""
    m = momentos_anyo(com, esc, tipo)
    resto = m["EY"] - float(esc["V"]) * m["ER"]
    filas = []
    def fila(nombre, obj):
        Vx = (obj - resto) / m["ER"]
        filas.append({"Meta para el Empaque" if tipo == 3 else "Meta": nombre, "E(Y) objetivo (k€)": obj,
                      "V_stock necesario (k€)": Vx, "Cambio frente al actual": Vx / float(esc["V"]) - 1})
    fila(f"E(Y | {NOMBRE[tipo]}) ≥ {num(objetivo, 0)}", objetivo)
    tA = teoria_A(com, esc, con_rejilla=False)
    fila("Igualar el beneficio medio de la cartera", tA["EY"])
    for f in PROYECTOS:
        if f != tipo:
            fila(f"Igualar a {NOMBRE[f]}", tA["por_tipo"][f]["EY"])
    return pd.DataFrame(filas).set_index(filas[0] and list(filas[0])[0]), m["EY"], m["ER"]


def valores_barrido(vmin, vmax, puntos):
    v = np.geomspace(max(1, vmin), max(vmin + 1, vmax), max(2, int(puntos)))
    return sorted(set(int(round(x)) for x in v))


def sensibilidad_estimador(com, esc, metrica, valores, replicas, semilla, progreso=None):
    """E(Y) frente al nº de proyectos (Bloque A) o E(VAN) con mantenimiento frente al nº de centros (Bloque B)."""
    if metrica == "proyecto":
        t = teoria_A(com, esc, con_rejilla=False); teo, sd1 = t["EY"], t["sdY"]
    else:
        t = teoria_B(com, esc, con_rejilla=False); teo, sd1 = t["E_con"], t["sd_con"]
    tq = _tq(int(replicas))
    filas, k, tot = [], 0, len(valores) * int(replicas)
    for n in valores:
        medias = []
        for r in range(int(replicas)):
            if metrica == "proyecto":
                rng = np.random.default_rng(int(semilla) + r)
                medias.append(float(generar_proyectos(com, esc, rng.random((4, n)))["Y"].mean()))
            else:
                b = base_programa(com, esc, n, int(semilla) + 50_000 * (r + 1))
                medias.append(float(van(esc, aplicar_politica(com, esc, b, "con")["Y"]).mean()))
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
def contraste_ganancia(mcB, tB):
    """O8.1 · contraste H₀: E(Δ) = 0 con los Δ pareados de todos los centros simulados (t de Student y signo)."""
    g = mcB["VAN_con"] - mcB["VAN_sin"]
    T = mcB["T"]
    filas = []
    def fila(nom, x, teo):
        n = len(x); m = float(x.mean()); s = float(x.std(ddof=1)); se = s / math.sqrt(n)
        t = m / se if se > 0 else float("inf")
        pv = float(2 * _stats.t.sf(abs(t), n - 1))
        filas.append({"Grupo": nom, "Centros": n, "Media de Δ (k€)": m, "IC95 de Δ (k€)": (m - _tq(n) * se, m + _tq(n) * se),
                      "Estadístico t": t, "p-valor": pv, "Teórico · E(Δ) (k€)": teo,
                      "Decisión (5 %)": "se rechaza E(Δ) = 0" if pv < 0.05 else "no se rechaza E(Δ) = 0"})
    fila("Toda la red", g, tB["E_gan"])
    for f in PROYECTOS:
        mk = T == f
        if mk.sum() > 2:
            fila(NOMBRE[f], g[mk], tB["por_tipo"][f]["E_gan"])
    return pd.DataFrame(filas).set_index("Grupo")


def comparar_experiencia(com, esc, t0s=(1, 3), centros=300, replicas=20, semilla=1000):
    """O8.2 · VAN esperado si el centro empieza en el punto t0 de la curva de aprendizaje (mismos números aleatorios)."""
    filas = []
    for t0 in t0s:
        e2 = dict(esc, t0=int(t0))
        tB = teoria_B(com, e2, con_rejilla=False)
        vs = []
        for r in range(int(replicas)):
            b = base_programa(com, e2, int(centros), int(semilla) + 1000 + r)
            vs.append(float(van(e2, aplicar_politica(com, e2, b, "con")["Y"]).mean()))
        m, lo, hi = _ic_media(vs)
        filas.append({"Punto de partida de la curva": f"t = {t0}", "Teórico · E(VAN) con (k€)": tB["E_con"],
                      "Simulado · E(VAN) con (k€)": m, "IC95 (k€)": (lo, hi), "Teórico · E(VAN) sin (k€)": tB["E_sin"],
                      "Teórico · ganancia E(Δ) (k€)": tB["E_gan"]})
    df = pd.DataFrame(filas).set_index("Punto de partida de la curva")
    return df


def politica_adaptativa(com, esc, umbrales, centros=300, replicas=20, semilla=1000):
    """O8.3 · mantener el año t solo si el ahorro que se observaría sin mantener cae por debajo del umbral."""
    filas = []
    for u in list(umbrales):
        vs, mf = [], []
        for r in range(int(replicas)):
            b = base_programa(com, esc, int(centros), int(semilla) + 1000 + r)
            p = aplicar_politica(com, esc, b, "adaptativa", umbral_S=u)
            vs.append(float(van(esc, p["Y"]).mean())); mf.append(float(p["M"].mean()))
        m, lo, hi = _ic_media(vs)
        filas.append({"Umbral de ahorro (k€)": u, "Simulado · E(VAN) (k€)": m, "IC95 (k€)": (lo, hi),
                      "Años con mantenimiento": float(np.mean(mf))})
    df = pd.DataFrame(filas).set_index("Umbral de ahorro (k€)")
    tB = teoria_B(com, esc, con_rejilla=False)
    return df, tB["E_con"], tB["E_sin"]


def actualizacion_bayesiana(com, esc, lam_min=0.6, lam_max=0.95, n=4000, semilla=1000, rejilla=200):
    """O8.4 · cada centro tiene su propia λ ~ U(lam_min, lam_max); tras observar R₁ y R₂ se actualiza la creencia
    sobre λ con la verosimilitud exacta de R₂ = 1 − (1 − R⁰₂)·λ (R₁ no informa sobre λ). Se mide cuánto se reduce la
    incertidumbre y se decide continuar si el VAN esperado de los años restantes es positivo."""
    rng = np.random.default_rng(int(semilla) + 31)
    T = tipo_desde_U(rng.random(n), esc["p"])
    lam_true = lam_min + (lam_max - lam_min) * rng.random(n)
    U = rng.random((3, n))
    TT = T
    I, R0, S0 = variables(com, TT, U[0], U[1], U[2])
    R2 = 1 - (1 - R0) * lam_true
    g = np.linspace(lam_min, lam_max, rejilla)
    post_m = np.zeros(n); post_sd = np.zeros(n)
    for f in PROYECTOS:
        mk = T == f
        if not mk.any():
            continue
        a, b = _dist(com, f)[2:4]
        r0 = 1 - (1 - R2[mk])[:, None] / g[None, :]
        ok = (r0 > 0) & (r0 < 1)
        lik = np.where(ok, _stats.beta.pdf(np.clip(r0, 1e-12, 1 - 1e-12), a, b) / g[None, :], 0.0)
        w = lik / np.maximum(lik.sum(axis=1, keepdims=True), 1e-300)
        post_m[mk] = w @ g
        post_sd[mk] = np.sqrt(np.maximum(w @ g ** 2 - post_m[mk] ** 2, 0))
    sd_prior = (lam_max - lam_min) / math.sqrt(12)
    filas = []
    for f in PROYECTOS:
        mk = T == f
        if mk.sum() < 3:
            continue
        filas.append({"Tipo": NOMBRE[f], "Centros": int(mk.sum()), "σ a priori de λ": sd_prior,
                      "σ a posteriori media": float(post_sd[mk].mean()),
                      "Reducción de la incertidumbre": 1 - float(post_sd[mk].mean()) / sd_prior,
                      "Correlación (λ real, media a posteriori)": float(np.corrcoef(lam_true[mk], post_m[mk])[0, 1])})
    df = pd.DataFrame(filas).set_index("Tipo")
    # decisión: VAN esperado de los años 3..H con λ = media a posteriori (con mantenimiento)
    H = int(esc["H"])
    van_rest = []
    for lam_ in (lam_min, (lam_min + lam_max) / 2, lam_max):
        e2 = dict(esc, lam=lam_)
        d = descuentos(e2)
        p = normalizar(esc["p"])
        v = sum(pi * sum(d[t - 1] * momentos_anyo(com, e2, f, _exponente(e2, t), 1.0, 1)["EY"] for t in range(3, H + 1))
                for pi, f in zip(p, PROYECTOS))
        van_rest.append({"λ supuesta": lam_, "VAN esperado de los años 3 a H (k€)": v,
                         "Decisión": "continuar" if v > 0 else "abandonar"})
    return df, pd.DataFrame(van_rest).set_index("λ supuesta")


def sustitucion(com, esc, centros=300, replicas=20, semilla=1000):
    """O8.6 · mantener el proyecto con mantenimiento frente a sustituirlo a mitad del horizonte por uno nuevo del mismo
    tipo (se paga otra inversión I y la curva de aprendizaje vuelve a t = 1)."""
    H = int(esc["H"]); m = H // 2 + 1
    res = {"Con mantenimiento": [], "Sin mantenimiento": [], f"Sustituir en el año {m} (con mantenimiento)": []}
    cv = {k: [] for k in res}
    for r in range(int(replicas)):
        b = base_programa(com, esc, int(centros), int(semilla) + 1000 + r)
        for k, (pol, sus) in zip(res, (("con", None), ("sin", None), ("con", m))):
            v = van(esc, aplicar_politica(com, esc, b, pol, sustituir=sus)["Y"])
            res[k].append(float(v.mean())); cv[k].append(_cvar_muestra(v))
    filas = []
    for k in res:
        mm, lo, hi = _ic_media(res[k])
        filas.append({"Alternativa": k, "Simulado · E(VAN) (k€)": mm, "IC95 (k€)": (lo, hi),
                      "Simulado · CVaR 5 % (k€)": float(np.mean(cv[k]))})
    return pd.DataFrame(filas).set_index("Alternativa")


# ==============================================================================================
# Mecanismo
# ==============================================================================================
def un_proyecto(com, esc, uT, uI, uR, uS):
    U = np.array([[uT], [uI], [uR], [uS]])
    o = generar_proyectos(com, esc, U)
    t = [float(x[0]) for x in o["terminos"]]
    return {"T": int(o["T"][0]), "I": float(o["I"][0]), "R": float(o["R"][0]), "S": float(o["S"][0]),
            "Y": float(o["Y"][0]), "terminos": t}


def trayectoria(com, esc, uT, uI, uR, uS):
    """Un centro con los mismos U todos los años: R_t, S_t y Y_t con y sin mantenimiento."""
    H = int(esc["H"])
    T = tipo_desde_U(np.array([uT]), esc["p"])
    b = {"T": T, "I": None}
    TT = np.repeat(T[:, None], H, axis=1)
    I, R0, S0 = variables(com, TT, np.full((1, H), uI), np.full((1, H), uR), np.full((1, H), uS))
    b.update({"I": I, "R0": R0, "S0": S0, "extra": np.full((2, 1), 0.5)})
    pc, ps = aplicar_politica(com, esc, b, "con"), aplicar_politica(com, esc, b, "sin")
    return {"T": int(T[0]), "R": pc["R"][0], "Sc": pc["S"][0], "Ss": ps["S"][0], "Yc": pc["Y"][0], "Ys": ps["Y"][0],
            "VANc": float(van(esc, pc["Y"])[0]), "VANs": float(van(esc, ps["Y"])[0])}


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


def keur(x, d=1):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return num(x, d) + " k€"


def fmt_val(k, v):
    if isinstance(v, str) or v is None:
        return "—" if v is None else v
    if isinstance(v, tuple):
        return f"[{fmt_val(k, v[0])}; {fmt_val(k, v[1])}]"
    if "(k€" in k or k.endswith("k€"):
        return keur(v, 2 if abs(v) < 10 else 1)
    if "P(" in k or "%" in k or "Contribución" in k or "Peso" in k or "Cuota" in k or "Reducción de la" in k \
            or "probabilidad" in k.lower() or "S₁" in k or "S_T" in k or "Cambio" in k or "Años con" in k \
            or "subvención" in k.lower():
        return pct(v, 2)
    if "CV" in k or "σ" in k or "λ" in k or "Correlación" in k or "δ" in k:
        return num(v, 3)
    if "p-valor" in k:
        return num(v, 4) if v >= 1e-4 else "< 0,0001"
    if "Centros" in k or "Proyectos" in k:
        return num(v, 0)
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
        return pct(v, 2) if fila.startswith("Probabilidad") else keur(v)
    return str(v)


def texto_escenario(nombre, r):
    A, B = r["teoria"]["A"], r["teoria"]["B"]
    okA = r["resA"]["Teórico en IC95"].dropna().astype(bool); okB = r["resB"]["Teórico en IC95"].dropna().astype(bool)
    fuera = list(okA[~okA].index) + list(okB[~okB].index)
    n_kpi = len(okA) + len(okB)
    val = ("la simulación reproduce los valores teóricos (todos dentro del IC95 %)" if not fuera else
           f"la simulación reproduce los valores teóricos en {n_kpi - len(fuera)} de los {n_kpi} KPIs (fuera del IC95 %: "
           f"{', '.join(fuera)}; con tantos KPIs es esperable que alguno quede fuera por azar)")
    pf = A["por_tipo"]
    mayor_cv = max(PROYECTOS, key=lambda f: pf[f]["CV"] if pf[f]["p"] > 0 else -1)
    dom = int(np.argmax(np.abs(A["contrib"])))
    pol = POLITICAS.get(r["esc"].get("politica", "con"))
    gan = B["E_gan"]
    txt = (f"{nombre}: un proyecto rinde en media E(Y) = {keur(A['EY'])} (σ = {keur(A['sdY'])}); P(Y < 0) = "
           f"{pct(A['Pperd'], 2)}, VaR 95 % = {keur(A['VaR'])} y CVaR 5 % = {keur(A['CVaR'])}. El proyecto "
           f"{NOMBRE[mayor_cv]} tiene la mayor incertidumbre relativa (CV = {num(pf[mayor_cv]['CV'], 3)}) y el término "
           f"«{TERMINOS[dom]}» explica el {pct(A['contrib'][dom], 0)} de Var(Y). En el programa de {B['H']} años, "
           f"E(VAN) = {keur(B['E_con'])} con mantenimiento y {keur(B['E_sin'])} sin él: el mantenimiento "
           f"{'gana' if gan > 0 else 'pierde'} {keur(abs(gan))} por centro (P(Δ > 0) = {pct(B['P_gan'], 1)}; compensa si "
           f"C_mant < {keur(B['C_ind'], 2)}/año). Política elegida: {pol} (valores teóricos); {val}.")
    n_ok = (r["objetivos"]["Cumple"] == "Sí").sum()
    txt += f" Cumple {n_ok} de los {len(r['objetivos'])} objetivos."
    return txt


def recomendaciones(res, base, com):
    """O7.1 · recomendación sobre la política de mantenimiento con sus condiciones."""
    rec = []
    if base in res:
        r = res[base]; B = r["teoria"]["B"]; esc = r["esc"]
        ic = r["resB"].loc["Ganancia media pareada E(Δ) (k€)", ["IC95 inf", "IC95 sup"]]
        gan = B["E_gan"]
        verbo = "No mantener" if gan < 0 else "Mantener"
        rec.append(f"{verbo} con carácter general en «{base}»: la ganancia media del mantenimiento es {keur(gan, 2)} por "
                   f"centro (IC95 % simulado [{keur(ic.iloc[0], 2)}; {keur(ic.iloc[1], 2)}]) y solo compensa en el "
                   f"{pct(B['P_gan'], 0)} de los centros; el coste de indiferencia es {keur(B['C_ind'], 2)}/año frente a "
                   f"{keur(c_efectivo(esc), 2)}/año.")
        pt = B["por_tipo"]
        activos = [f for f in PROYECTOS if pt[f]["p"] > 0] or list(PROYECTOS)
        mx = max(pt[f]["E_gan"] for f in activos)
        mejores = [f for f in activos if abs(pt[f]["E_gan"] - mx) < 1e-9]
        peor = min(activos, key=lambda f: pt[f]["E_gan"])
        nm = " e ".join(NOMBRE[f] for f in mejores)
        rec.append(f"Decidir por tipo de proyecto: el mantenimiento es más valioso en {nm} (E(Δ) = "
                   f"{keur(pt[mejores[0]]['E_gan'], 2)}, compensa si C_mant < {keur(pt[mejores[0]]['C_ind'], 2)}/año; ahorro "
                   f"S medio {keur(pt[mejores[0]]['ES'], 1)}) y el menos valioso en {NOMBRE[peor]} "
                   f"({keur(pt[peor]['E_gan'], 2)}; S medio {keur(pt[peor]['ES'], 1)}): su valor es proporcional al "
                   "ahorro que protege, así que solo podría justificarse en los proyectos de ahorro alto"
                   + (" y con un coste algo menor." if pt[mejores[0]]["E_gan"] < 0 else "."))
        largo = [n for n, x in res.items() if int(x["esc"]["H"]) > int(esc["H"])]
        caro = [n for n, x in res.items() if float(x["esc"]["C_mant"]) > float(esc["C_mant"])]
        severo = [n for n, x in res.items() if float(x["esc"]["delta"]) < float(esc["delta"])]
        barato = [n for n, x in res.items() if float(x["esc"]["C_mant"]) < float(esc["C_mant"])]
        txt = "Revisar la decisión si cambian las condiciones: "
        partes = []
        if largo:
            x = res[largo[0]]["teoria"]["B"]
            partes.append(f"con H = {res[largo[0]]['esc']['H']} años la ganancia pasa a {keur(x['E_gan'], 2)} (indiferencia "
                          f"{keur(x['C_ind'], 2)}/año)")
        if severo:
            x = res[severo[0]]["teoria"]["B"]
            partes.append(f"con δ = {num(res[severo[0]]['esc']['delta'], 2)} la ganancia es {keur(x['E_gan'], 2)}")
        if barato:
            x = res[barato[0]]["teoria"]["B"]
            partes.append(f"con C_mant = {keur(res[barato[0]]['esc']['C_mant'], 1)}/año, {keur(x['E_gan'], 2)}")
        if B["X_ind"] > 0:
            partes.append(f"una subvención del {pct(B['X_ind'], 0)} del coste lo haría rentable en promedio")
        if caro:
            x = res[caro[0]]["teoria"]["B"]
            partes.append(f"con C_mant = {keur(res[caro[0]]['esc']['C_mant'], 1)}/año la pérdida sube a {keur(x['E_gan'], 2)}")
        rec.append(txt + "; ".join(partes) + "." if partes else txt + "horizontes más largos y degradaciones más severas "
                   "aumentan el valor del mantenimiento.")
    filas = [(n, (r["objetivos"]["Cumple"] == "Sí").sum(), r["teoria"]) for n, r in res.items()]
    max_ok = max(f[1] for f in filas)
    mejor = sorted([f for f in filas if f[1] == max_ok], key=lambda f: -f[2]["B"]["E_pol"])[0]
    rec.append(f"Entre los escenarios analizados, «{mejor[0]}» cumple más objetivos ({mejor[1]} de "
               f"{len(res[mejor[0]]['objetivos'])}): E(Y) = {keur(mejor[2]['A']['EY'])} por proyecto, P(Y < 0) = "
               f"{pct(mejor[2]['A']['Pperd'], 2)} y E(VAN) = {keur(mejor[2]['B']['E_pol'])} por centro con la política elegida.")
    return rec


def texto_recomendacion(res, base, com):
    return " ".join(f"({i + 1}) {r}" for i, r in enumerate(recomendaciones(res, base, com)))


KPIS_CUADRO = [
    ("Beneficio esperado por proyecto E(Y) y por tipo", "rentabilidad media de la cartera de proyectos"),
    ("Probabilidad de pérdida P(Y < 0)", "frecuencia de proyectos que destruyen valor"),
    ("VaR 95 % y CVaR 5 % del beneficio", "tamaño del peor caso para provisionar el riesgo"),
    ("VAN esperado del programa con y sin mantenimiento", "valor del programa plurianual por centro"),
    ("Ganancia del mantenimiento E(Δ) y P(Δ > 0)", "si compensa reinvertir y en cuántos centros"),
    ("C_mant de indiferencia por tipo de proyecto", "precio máximo a negociar con el proveedor de mantenimiento"),
    ("CVaR 5 % del VAN por política", "protección frente a los centros con peor resultado"),
]
