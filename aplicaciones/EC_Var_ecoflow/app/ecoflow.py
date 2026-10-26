# -*- coding: utf-8 -*-
"""EC4 · Gestión de riesgos en la micro-red de «Eco-Flow Renewables» (energía en MWh, dinero en €).

Bloque A · un día (método de composición): escenario climático T ∈ {1, 2, 3, 4} (Estival, Frontal, Monzónico,
    Anticiclónico); dado T, independientes: solar G_s ~ Tri(a, m, b), eólica G_w ~ Weibull(k, λ), hidráulica
    G_h ~ LN(μ, σ) y demanda D ~ N(μ, σ). B = G − D; Y = P_exp·B⁺ − P_pen·B⁻ − C_fijo − c·G.
    Palancas: precio spot aleatorio (lognormal de media P_pen), más capacidad solar, contrato a plazo y respuesta de
    demanda (D se reduce un r % si B < −umbral, con un descuento por MWh no consumido).
Bloque B · batería de capacidad Cap y eficiencia η, recorrida día a día durante 365 días: carga con el superávit,
    cubre el déficit con E·η y el resto se compra en el mercado. Se compara con la operación sin batería sobre la misma
    base climática (diseño pareado).

Teoría: la distribución de G se obtiene en una rejilla fina convolucionando por FFT las distribuciones exactas de las
tres fuentes; D se integra por cuantiles. De ahí salen E(Y), Var(Y), su descomposición, P(B < 0), el VaR y el CVaR.
La batería es una cadena de Markov en el estado de carga (discretizado): su ley día a día da la ganancia esperada, el
estado de carga medio, la probabilidad de acudir al mercado y el LOLE.

Material de Javier Morales, equipo IA4LEGOS (CC BY-SA 4.0)."""
import copy
import math

import numpy as np
import pandas as pd

try:
    from scipy import stats as _stats
    from scipy.special import ndtri as _ndtri, ndtr as _ndtr
    from scipy.signal import fftconvolve as _fftconvolve
except Exception:          # pragma: no cover
    _stats = None

ESCENARIOS = (1, 2, 3, 4)
NOMBRE = {1: "Estival", 2: "Frontal", 3: "Monzónico", 4: "Anticiclónico"}
TERMINOS = ("Mercado P·B", "Coste variable −c·G", "Descuento por respuesta de demanda")
N_DIAS = 365
N_DIAGRAMA = 40
H_G, H_B, H_Y = 0.25, 0.1, 2.0     # pasos de las rejillas: generación y balance (MWh) y beneficio (€)
N_D = 100                          # nodos (cuantiles) para integrar la demanda
N_P = 20                           # nodos para el precio spot aleatorio
N_E = 200                          # estados del estado de carga de la batería
COLA = 1e-7

COMUNES_DEF = {
    "dist": {1: {"tri": [80.0, 120.0, 150.0], "weib": [1.5, 5.0], "ln": [2.0, 0.2], "norm": [135.0, 10.0]},
             2: {"tri": [0.0, 15.0, 35.0], "weib": [3.0, 80.0], "ln": [2.5, 0.3], "norm": [115.0, 15.0]},
             3: {"tri": [0.0, 10.0, 15.0], "weib": [2.0, 40.0], "ln": [4.0, 0.5], "norm": [125.0, 15.0]},
             4: {"tri": [5.0, 15.0, 20.0], "weib": [1.2, 10.0], "ln": [2.0, 0.1], "norm": [145.0, 10.0]}},
    "objetivos": {"EY_MIN": -3000.0, "PDEF_MAX": 0.65, "CVAR_MIN": -11000.0, "NETO_MIN": 0.0},
}
ESC_BASE = {"p": [0.40, 0.30, 0.20, 0.10], "P_exp": 40.0, "P_pen": 80.0, "cv_pen": 0.0, "C_fijo": 1000.0, "c_var": 2.0,
            "solar_factor": 1.0, "solar_extra": 0.0, "contrato": False, "P_contrato": 60.0, "prima": 150.0,
            "dr_red": 0.0, "dr_umbral": 20.0, "dr_desc": 20.0,
            "cap": 20.0, "eta": 0.98, "E1": 0.0, "amort": 1800.0, "tarifa": 0.0, "descripcion": ""}


def escenario(**cambios):
    e = copy.deepcopy(ESC_BASE)
    e.update(cambios)
    return e


def normalizar(p):
    p = np.clip(np.asarray(p, dtype=float), 0, None)
    return p / p.sum() if p.sum() > 0 else np.full(len(p), 1 / len(p))


def precio_deficit(esc):
    """Precio medio de la energía comprada y su segundo momento (spot aleatorio o contrato a plazo)."""
    if esc.get("contrato"):
        P = float(esc["P_contrato"]); return P, P * P
    P = float(esc["P_pen"]); cv = float(esc.get("cv_pen") or 0.0)
    return P, P * P * (1 + cv * cv)


def nodos_precio(esc):
    if esc.get("contrato") or not float(esc.get("cv_pen") or 0) > 0:
        return np.array([precio_deficit(esc)[0]])
    P = float(esc["P_pen"]); cv = float(esc["cv_pen"]); s2 = math.log(1 + cv * cv)
    z = _ndtri((np.arange(N_P) + 0.5) / N_P)
    return np.exp(math.log(P) - s2 / 2 + math.sqrt(s2) * z)


def coste_fijo_dia(esc):
    return float(esc["C_fijo"]) + (float(esc["prima"]) if esc.get("contrato") else 0.0)

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



# ==============================================================================================
# Distribuciones de las fuentes (CDF e inversa)
# ==============================================================================================
def cdf_tri(x, a, m, b):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        izq = np.where(m > a, (x - a) ** 2 / ((b - a) * (m - a)), 0.0)
        der = np.where(b > m, 1 - (b - x) ** 2 / ((b - a) * (b - m)), 1.0)
    return np.where(x <= a, 0.0, np.where(x >= b, 1.0, np.where(x <= m, izq, der)))


def inv_tri(u, a, m, b):
    Fc = (m - a) / (b - a)
    return np.where(u < Fc, a + np.sqrt(u * (b - a) * (m - a)), b - np.sqrt((1 - u) * (b - a) * (b - m)))


def _par(com, f):
    d = com["dist"][f]
    return [float(x) for x in d["tri"]], [float(x) for x in d["weib"]], [float(x) for x in d["ln"]], \
        [float(x) for x in d["norm"]]


def momentos_fuentes(com, esc, f):
    """Medias y desviaciones típicas exactas de G_s (con la ampliación solar), G_w, G_h y D."""
    (a, m, b), (k, lam), (mu, sg), (md, sd) = _par(com, f)
    fs, es = float(esc["solar_factor"]), float(esc["solar_extra"])
    Es = (a + m + b) / 3; Vs = (a * a + m * m + b * b - a * m - a * b - m * b) / 18
    Ew = lam * math.gamma(1 + 1 / k); Vw = lam * lam * (math.gamma(1 + 2 / k) - math.gamma(1 + 1 / k) ** 2)
    Eh = math.exp(mu + sg * sg / 2); Vh = (math.exp(sg * sg) - 1) * math.exp(2 * mu + sg * sg)
    return {"Gs": (fs * Es + es, fs * math.sqrt(Vs)), "Gw": (Ew, math.sqrt(Vw)), "Gh": (Eh, math.sqrt(Vh)), "D": (md, sd)}


def rejilla_G(com, esc, f, h=H_G):
    """Distribución exacta (en rejilla) de G = G_s + G_w + G_h para el escenario f."""
    (a, m, b), (k, lam), (mu, sg), _ = _par(com, f)
    fs, es = float(esc["solar_factor"]), float(esc["solar_extra"])
    Rs = _desde_cdf(lambda x: cdf_tri((x - es) / fs, a, m, b), fs * a + es, fs * b + es, h) if fs > 0 else \
        Rejilla(int(round(es / h)), [1.0], h)
    hi_w = lam * (-math.log(COLA)) ** (1 / k)
    Rw = _desde_cdf(lambda x: 1 - np.exp(-(np.maximum(x, 0) / lam) ** k), 0.0, hi_w, h)
    hi_h = math.exp(mu + sg * float(_ndtri(1 - COLA)))
    Rh = _desde_cdf(lambda x: _ndtr((np.log(np.maximum(x, 1e-300)) - mu) / sg) * (x > 0), 0.0, hi_h, h)
    return convolucion(convolucion(Rs, Rw), Rh)


# ==============================================================================================
# Teoría · Bloque A (un día)
# ==============================================================================================
def _nodos_D(md, sd):
    return md + sd * _ndtri((np.arange(N_D) + 0.5) / N_D)


def _dia_tipo(com, esc, f, con_dist=True):
    """Valores del día sobre la rejilla G × nodos de D: momentos, probabilidades y distribuciones de Y y del balance."""
    RG = rejilla_G(com, esc, f)
    g = RG.x; pg = RG.p / RG.masa()
    _, _, _, (md, sd) = _par(com, f)
    d = _nodos_D(md, sd)
    W = np.broadcast_to(pg[:, None] / N_D, (len(g), N_D))
    B = g[:, None] - d[None, :]
    r, u, cd = float(esc["dr_red"]), float(esc["dr_umbral"]), float(esc["dr_desc"])
    if r > 0:
        sev = B < -u
        Bp = np.where(sev, g[:, None] - (1 - r) * d[None, :], B)
        desc = np.where(sev, cd * r * d[None, :], 0.0)
    else:
        Bp, desc = B, np.zeros_like(B)
    Bpos = np.maximum(Bp, 0); Bneg = np.maximum(-Bp, 0)
    Pe = float(esc["P_exp"]); EP, EP2 = precio_deficit(esc)
    c = float(esc["c_var"]); C0 = coste_fijo_dia(esc)
    M = Pe * Bpos - EP * Bneg
    X2 = -c * np.broadcast_to(g[:, None], B.shape)
    X3 = -desc
    s = lambda A: float((W * A).sum())
    E = np.array([s(M), s(X2), s(X3)])
    M2 = np.zeros((3, 3))
    M2[0, 0] = s(Pe * Pe * Bpos ** 2 + EP2 * Bneg ** 2)
    M2[1, 1] = s(X2 * X2); M2[2, 2] = s(X3 * X3)
    M2[0, 1] = M2[1, 0] = s(M * X2); M2[0, 2] = M2[2, 0] = s(M * X3); M2[1, 2] = M2[2, 1] = s(X2 * X3)
    cov = M2 - np.outer(E, E)
    out = {"E": E, "cov": cov, "EY": float(E.sum() - C0), "VY": float(cov.sum()), "Pdef": s(B < 0),
           "Pspot": s(Bp < 0), "Edef": s(Bneg), "Esup": s(Bpos), "EG": float((pg * g).sum()),
           "Pdr": s(desc > 0) if r > 0 else 0.0}
    if con_dist:
        Ys = []
        for P in nodos_precio(esc):
            Ys.append(Pe * Bpos - P * Bneg + X2 + X3 - C0)
        idx = [np.rint(Y / H_Y).astype(np.int64) for Y in Ys]
        lo = min(int(i.min()) for i in idx); hi = max(int(i.max()) for i in idx)
        pY = np.zeros(hi - lo + 1)
        for i in idx:
            pY += np.bincount((i - lo).ravel(), weights=W.ravel() / len(idx), minlength=hi - lo + 1)
        out["rejY"] = Rejilla(lo, pY, H_Y).recortar()
        ib = np.rint(Bp / H_B).astype(np.int64); lb = int(ib.min())
        out["rejB"] = Rejilla(lb, np.bincount((ib - lb).ravel(), weights=W.ravel()), H_B).recortar()
    return out


def teoria_A(com, esc, con_dist=True):
    p = normalizar(esc["p"])
    tt = {f: _dia_tipo(com, esc, f, con_dist) for f in ESCENARIOS}
    EY = float(sum(pi * tt[f]["EY"] for pi, f in zip(p, ESCENARIOS)))
    VY = float(sum(pi * (tt[f]["VY"] + tt[f]["EY"] ** 2) for pi, f in zip(p, ESCENARIOS)) - EY * EY)
    E = sum(pi * tt[f]["E"] for pi, f in zip(p, ESCENARIOS))
    M2 = sum(pi * (tt[f]["cov"] + np.outer(tt[f]["E"], tt[f]["E"])) for pi, f in zip(p, ESCENARIOS))
    cov = M2 - np.outer(E, E)
    contrib = cov.sum(axis=1) / VY if VY > 0 else np.zeros(3)
    S_T = float(sum(pi * (tt[f]["EY"] - EY) ** 2 for pi, f in zip(p, ESCENARIOS)) / VY) if VY > 0 else 0.0
    sm = lambda k: float(sum(pi * tt[f][k] for pi, f in zip(p, ESCENARIOS)))
    out = {"p": p, "EY": EY, "VY": VY, "sdY": math.sqrt(max(VY, 0)), "contrib": contrib, "S_T": S_T,
           "Pdef": sm("Pdef"), "Pspot": sm("Pspot"), "Edef": sm("Edef"), "Esup": sm("Esup"), "EG": sm("EG"),
           "Pdr": sm("Pdr")}
    pf = {}
    for pi, f in zip(p, ESCENARIOS):
        t = tt[f]; sd = math.sqrt(max(t["VY"], 0))
        pf[f] = {"p": float(pi), "EY": t["EY"], "sd": sd, "CV": sd / abs(t["EY"]) if t["EY"] else float("nan"),
                 "Pdef": t["Pdef"], "Edef": t["Edef"], "Esup": t["Esup"], "EG": t["EG"]}
    if con_dist:
        mix = mezcla([(pi, tt[f]["rejY"]) for pi, f in zip(p, ESCENARIOS)])
        out.update({"VaR": mix.cuantil(0.05), "CVaR": mix.cvar(0.05), "P50": mix.cuantil(0.5), "rejilla": mix,
                    "Pperd": mix.cdf(0.0)})
        out["rejB"] = mezcla([(pi, tt[f]["rejB"]) for pi, f in zip(p, ESCENARIOS)])
        out["rejB_tipo"] = {f: tt[f]["rejB"] for f in ESCENARIOS}
        for pi, f in zip(p, ESCENARIOS):
            R = tt[f]["rejY"]
            pf[f].update({"VaR": R.cuantil(0.05), "CVaR": R.cvar(0.05),
                          "peso_cola": pi * R.cdf(out["VaR"], estricto=False) / 0.05,
                          "peso_def": pi * pf[f]["Pdef"] / out["Pdef"] if out["Pdef"] > 0 else 0.0})
    out["por_tipo"] = pf
    return out


# ==============================================================================================
# Teoría · Bloque B (batería como cadena de Markov en el estado de carga)
# ==============================================================================================
def _cdf_rej(R):
    """Función de distribución continua (interpolada) de una rejilla."""
    x = R.x; cum = np.cumsum(R.p) / R.masa()
    xp = np.concatenate([[x[0] - R.h / 2], x + R.h / 2]); fp = np.concatenate([[0.0], cum])
    return lambda v: np.interp(v, xp, fp, left=0.0, right=1.0)


def _estados(esc):
    cap = float(esc["cap"])
    return np.linspace(0, cap, N_E + 1) if cap > 0 else np.zeros(1)


def matriz_transicion(esc, RB):
    """P[k, k'] = P(E_{t+1} ≈ e_k' | E_t = e_k) con la regla reactiva (cargar con el superávit, descargar con el déficit)."""
    e = _estados(esc); cap = float(esc["cap"]); eta = float(esc["eta"])
    n = len(e)
    if n == 1:
        return np.ones((1, 1))
    F = _cdf_rej(RB)
    de = cap / N_E
    xs = np.concatenate([[-np.inf], (np.arange(n - 1) + 0.5) * de, [np.inf]])
    X = xs[None, :]; E = e[:, None]
    with np.errstate(invalid="ignore"):
        bx = np.where(X < E, eta * (X - E), X - E)
    bx = np.where(np.isinf(X), X, bx)
    C = F(bx)
    return np.clip(np.diff(C, axis=1), 0, None)


def ganancia_estado(esc, RB):
    """E[Δ | E_t = e], E[Δ² | e], P(compra en el mercado | e) y P(déficit cubierto del todo | e)."""
    e = _estados(esc); cap = float(esc["cap"]); eta = float(esc["eta"])
    b = RB.x; q = RB.p / RB.masa()
    Pe = float(esc["P_exp"]); EP, EP2 = precio_deficit(esc)
    pos = b >= 0
    carga = np.where(pos[None, :], np.minimum(b[None, :], cap - e[:, None]), 0.0)
    cob = np.where(~pos[None, :], np.minimum(-b[None, :], eta * e[:, None]), 0.0)
    D1 = (-Pe * carga + EP * cob) @ q
    D2 = (Pe * Pe * carga ** 2 + EP2 * cob ** 2) @ q
    spot = ((~pos[None, :]) & (-b[None, :] > eta * e[:, None] + 1e-12)) @ q
    cubierto = ((~pos[None, :]) & (-b[None, :] <= eta * e[:, None] + 1e-12)) @ q
    return D1, D2, spot, cubierto


def teoria_bateria(com, esc, tA=None, racha=None):
    """Ley día a día del estado de carga (E₁ dado) y ganancia esperada de la batería. `racha` = (día inicial, días):
    esos días son Anticiclónicos con seguridad."""
    tA = tA or teoria_A(com, esc)
    RB = tA["rejB"]
    e = _estados(esc)
    P = matriz_transicion(esc, RB)
    g1, g2, sp, cub = ganancia_estado(esc, RB)
    if racha is not None:
        R4 = tA["rejB_tipo"][4]
        P4 = matriz_transicion(esc, R4); g4, g24, sp4, cub4 = ganancia_estado(esc, R4)
        d0, nr = int(racha[0]), int(racha[1])
    pi = np.zeros(len(e)); pi[int(round(float(esc["E1"]) * (len(e) - 1)))] = 1.0
    acum = np.zeros(len(e)); Esoc, Egan, Pspot = [], [], []
    for t in range(1, N_DIAS + 1):
        en_racha = racha is not None and d0 <= t < d0 + nr
        G1, SP, PP = (g4, sp4, P4) if en_racha else (g1, sp, P)
        acum += pi
        Esoc.append(float(pi @ e)); Egan.append(float(pi @ G1)); Pspot.append(float(pi @ SP))
        pi = pi @ PP
    pibar = acum / N_DIAS
    out = {"estados": e, "pibar": pibar, "E_soc": np.array(Esoc), "E_gan_dia": np.array(Egan), "Pspot_dia": np.array(Pspot)}
    out["E_gan"] = float(np.mean(Egan))
    out["E2_gan"] = float(pibar @ g2)
    out["sd_gan"] = math.sqrt(max(out["E2_gan"] - out["E_gan"] ** 2, 0))
    out["Pspot_con"] = float(np.mean(Pspot))
    out["Pspot_sin"] = tA["Pspot"]
    out["cubiertos"] = float(pibar @ cub) / tA["Pspot"] if tA["Pspot"] > 0 else 0.0
    out["LOLE_con"] = N_DIAS * out["Pspot_con"]; out["LOLE_sin"] = N_DIAS * out["Pspot_sin"]
    out["EY_sin"] = tA["EY"]; out["EY_con"] = tA["EY"] + out["E_gan"]
    out["SOC_medio"] = float(np.mean(Esoc))
    out["gan_anual"] = N_DIAS * out["E_gan"]
    cap = float(esc["cap"])
    out["amortizacion"] = float(esc["amort"]) * cap
    out["ingreso_red"] = N_DIAS * float(esc.get("tarifa") or 0) * cap
    out["neto_anual"] = out["gan_anual"] + out["ingreso_red"] - out["amortizacion"]
    out["tarifa_equilibrio"] = (out["amortizacion"] - out["gan_anual"]) / (N_DIAS * cap) if cap > 0 else float("nan")
    out["tarifa_sola"] = float(esc["amort"]) / N_DIAS
    por = {}
    for f in ESCENARIOS:
        G1f = ganancia_estado(esc, tA["rejB_tipo"][f])
        por[f] = {"E_gan": float(pibar @ G1f[0]), "Pspot_con": float(pibar @ G1f[2]),
                  "Pdef": tA["por_tipo"][f]["Pdef"], "Edef": tA["por_tipo"][f]["Edef"],
                  "Esup": tA["por_tipo"][f]["Esup"], "p": tA["por_tipo"][f]["p"]}
    for f in ESCENARIOS:
        por[f]["cuota_gan"] = por[f]["p"] * por[f]["E_gan"] / out["E_gan"] if out["E_gan"] else 0.0
    out["por_tipo"] = por
    return out


def teoria(com, esc):
    A = teoria_A(com, esc)
    return {"A": A, "B": teoria_bateria(com, esc, A)}


# ==============================================================================================
# Simulación (transformada inversa con números aleatorios comunes)
# ==============================================================================================
def tipo_desde_U(u, p):
    F = np.cumsum(normalizar(p))
    return 1 + (u > F[0]).astype(int) + (u > F[1]).astype(int) + (u > F[2]).astype(int)


def generar_dias(com, esc, U, T=None):
    """U: 6 × … uniformes (escenario, solar, eólica, hidráulica, demanda, precio spot)."""
    if T is None:
        T = tipo_desde_U(U[0], esc["p"])
    Gs = np.zeros(T.shape); Gw = np.zeros(T.shape); Gh = np.zeros(T.shape); D = np.zeros(T.shape)
    for f in ESCENARIOS:
        mk = T == f
        if not mk.any():
            continue
        (a, m, b), (k, lam), (mu, sg), (md, sd) = _par(com, f)
        Gs[mk] = inv_tri(U[1][mk], a, m, b)
        Gw[mk] = lam * (-np.log1p(-U[2][mk])) ** (1 / k)
        Gh[mk] = np.exp(mu + sg * _ndtri(U[3][mk]))
        D[mk] = md + sd * _ndtri(U[4][mk])
    Gs = float(esc["solar_factor"]) * Gs + float(esc["solar_extra"])
    G = Gs + Gw + Gh
    B = G - D
    r, u, cd = float(esc["dr_red"]), float(esc["dr_umbral"]), float(esc["dr_desc"])
    if r > 0:
        sev = B < -u
        Bp = np.where(sev, G - (1 - r) * D, B); desc = np.where(sev, cd * r * D, 0.0)
    else:
        Bp, desc = B, np.zeros(B.shape)
    if esc.get("contrato"):
        Pd = np.full(B.shape, float(esc["P_contrato"]))
    elif float(esc.get("cv_pen") or 0) > 0:
        cv = float(esc["cv_pen"]); s2 = math.log(1 + cv * cv)
        Pd = np.exp(math.log(float(esc["P_pen"])) - s2 / 2 + math.sqrt(s2) * _ndtri(U[5]))
    else:
        Pd = np.full(B.shape, float(esc["P_pen"]))
    Pe = float(esc["P_exp"]); c = float(esc["c_var"])
    M = np.where(Bp >= 0, Pe * Bp, Pd * Bp)
    Y = M - c * G - desc - coste_fijo_dia(esc)
    return {"T": T, "Gs": Gs, "Gw": Gw, "Gh": Gh, "G": G, "D": D, "B": B, "Bp": Bp, "desc": desc, "Pd": Pd,
            "M": M, "Y": Y, "terminos": [M, -c * G, -desc]}


def _tq(n, alpha=0.05):
    if n < 2:
        return float("nan")
    return float(_stats.t.ppf(1 - alpha / 2, n - 1))


def _cvar_muestra(y, alfa=0.05):
    y = np.sort(np.asarray(y).ravel()); n = len(y)
    k = alfa * n; i = int(math.floor(k))
    s = y[:i].sum() + (k - i) * (y[i] if i < n else 0.0)
    return float(s / k)


def simular_A(com, esc, nsim, replicas, semilla, progreso=None):
    """Bloque A: `replicas` macro-réplicas de `nsim` días independientes (réplica i → semilla + i)."""
    filas, Ys, Ts, traza = [], [], [], None
    for r in range(int(replicas)):
        rng = np.random.default_rng(int(semilla) + r)
        o = generar_dias(com, esc, rng.random((6, int(nsim))))
        Y, T = o["Y"], o["T"]
        vy = float(Y.var(ddof=1)); C = np.cov(np.vstack(o["terminos"]))
        fila = {"Réplica": r + 1, "Semilla": int(semilla) + r, "Días": len(Y), "E(Y)": float(Y.mean()),
                "σ(Y)": math.sqrt(vy), "P(B < 0)": float((o["B"] < 0).mean()), "P(compra en el mercado)": float((o["Bp"] < 0).mean()),
                "VaR 5 %": float(np.quantile(Y, 0.05)), "CVaR 5 %": _cvar_muestra(Y), "E(déficit)": float(np.maximum(-o["Bp"], 0).mean()),
                "E(G)": float(o["G"].mean())}
        for k_, nom in enumerate(TERMINOS):
            fila[f"Contrib. {nom}"] = float(C[k_].sum() / vy) if vy > 0 else 0.0
        for f in ESCENARIOS:
            mk = T == f; yf = Y[mk]; nf = NOMBRE[f]
            fila[f"E(Y | {nf})"] = float(yf.mean()) if mk.any() else float("nan")
            fila[f"CV({nf})"] = float(yf.std(ddof=1) / abs(yf.mean())) if mk.sum() > 1 else float("nan")
            fila[f"P(B < 0 | {nf})"] = float((o["B"][mk] < 0).mean()) if mk.any() else float("nan")
        filas.append(fila)
        Ys.append(Y); Ts.append(T)
        if r == 0:
            k = N_DIAGRAMA
            traza = pd.DataFrame({"Día": np.arange(1, k + 1), "Escenario": [NOMBRE[t] for t in T[:k]],
                                  "Solar (MWh)": o["Gs"][:k], "Eólica (MWh)": o["Gw"][:k], "Hidráulica (MWh)": o["Gh"][:k],
                                  "Demanda (MWh)": o["D"][:k], "Balance (MWh)": o["Bp"][:k], "Y (€)": Y[:k]})
        if progreso:
            progreso()
    return {"replicas": pd.DataFrame(filas).set_index("Réplica"), "Y": np.concatenate(Ys), "T": np.concatenate(Ts),
            "traza": traza}


# ------------------------------------------------------------------------------------------- Bloque B
def base_anual(com, esc, anyos, semilla, racha=None):
    """Base común: un generador por año (año i → semilla + 1000 + i) con 6 × 365 uniformes."""
    U = np.stack([np.random.default_rng(int(semilla) + 1000 + r).random((6, N_DIAS)) for r in range(int(anyos))], axis=1)
    T = tipo_desde_U(U[0], esc["p"])
    if racha is not None:
        d0, nr = int(racha[0]), int(racha[1])
        T = T.copy(); T[:, d0 - 1:d0 - 1 + nr] = 4
    return generar_dias(com, esc, U, T)


def simular_bateria(esc, base, E1=None, activa=None):
    """Recorre los 365 días en orden (vectorizado sobre los años). `activa` = (precios de exportación (años × días),
    umbral): si el precio del día supera el umbral se vende también lo almacenado (O8.3)."""
    Bp, G = base["Bp"], base["G"]
    ny, nd = Bp.shape
    cap = float(esc["cap"]); eta = float(esc["eta"])
    E = np.full(ny, (float(esc["E1"]) if E1 is None else float(E1)) * cap)
    Pe = float(esc["P_exp"]); c = float(esc["c_var"])
    out = {k: np.empty((ny, nd)) for k in ("E_ini", "carga", "cob", "vend", "res", "Y", "E_fin")}
    for t in range(nd):
        b = Bp[:, t]; s = b >= 0
        out["E_ini"][:, t] = E
        carga = np.where(s, np.minimum(b, cap - E), 0.0)
        cob = np.where(~s, np.minimum(-b, E * eta), 0.0)
        vend = np.where(s, b - carga, 0.0); res = np.where(~s, -b - cob, 0.0)
        Pexp_t = Pe
        extra = 0.0
        E = np.clip(E + carga - cob / eta, 0.0, cap)
        if activa is not None:
            pr = activa[0][:, t]
            vender = pr > activa[1]
            extra = np.where(vender, E * eta * pr, 0.0)
            E = np.where(vender, 0.0, E)
            Pexp_t = pr
        out["Y"][:, t] = vend * Pexp_t - res * base["Pd"][:, t] - c * G[:, t] - base["desc"][:, t] - coste_fijo_dia(esc) + extra
        out["carga"][:, t] = carga; out["cob"][:, t] = cob; out["vend"][:, t] = vend; out["res"][:, t] = res
        out["E_fin"][:, t] = E
    return out


def simular_B(com, esc, anyos, semilla, progreso=None):
    """Bloque B: `anyos` años de 365 días; ambas políticas sobre la misma base. Cada año es una macro-réplica (las
    estimaciones son medias por día; los años solo sirven para el intervalo de confianza)."""
    base = base_anual(com, esc, anyos, semilla)
    bat = simular_bateria(esc, base)
    Ys, Yc = base["Y"], bat["Y"]
    G = Yc - Ys
    T = base["T"]
    filas = []
    for r in range(int(anyos)):
        def_dias = base["Bp"][r] < 0
        fila = {"Año": r + 1, "Semilla": int(semilla) + 1000 + r, "E(Y) sin batería": float(Ys[r].mean()),
                "E(Y) con batería": float(Yc[r].mean()), "Ganancia E(Δ)": float(G[r].mean()),
                "P(compra) sin batería": float(def_dias.mean()), "P(compra) con batería": float((bat["res"][r] > 1e-12).mean()),
                "Déficits cubiertos por completo": float(((bat["res"][r] <= 1e-12) & def_dias).sum() / max(def_dias.sum(), 1)),
                "VaR 5 % sin batería": float(np.quantile(Ys[r], 0.05)), "VaR 5 % con batería": float(np.quantile(Yc[r], 0.05)),
                "CVaR 5 % sin batería": _cvar_muestra(Ys[r]), "CVaR 5 % con batería": _cvar_muestra(Yc[r]),
                "Estado de carga medio": float(bat["E_ini"][r].mean()), "LOLE sin batería": float(def_dias.sum()),
                "LOLE con batería": float((bat["res"][r] > 1e-12).sum()), "σ(Δ) diario": float(G[r].std(ddof=1)),
                "Ganancia anual": float(G[r].sum())}
        for f in ESCENARIOS:
            mk = T[r] == f
            fila[f"E(Δ | {NOMBRE[f]})"] = float(G[r][mk].mean()) if mk.any() else float("nan")
        filas.append(fila)
        if progreso:
            progreso()
    ind = Yc - np.roll(Ys, 1, axis=0)
    return {"replicas": pd.DataFrame(filas).set_index("Año"), "Ys": Ys, "Yc": Yc, "T": T, "base": base, "bat": bat,
            "sd_par": float(G.std(ddof=1)), "sd_ind": float(ind.std(ddof=1))}


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


def resumen(rep, filas, agregados=None):
    """filas: {KPI: (columna de réplicas, teórico)}; agregados: {KPI: valor con todos los días} (para VaR y CVaR,
    cuyo IC se toma de la variabilidad entre réplicas)."""
    out = []
    agregados = agregados or {}
    for nombre, (col, teo) in filas.items():
        m, lo, hi = _ic_media(rep[col])
        if nombre in agregados and lo == lo:
            c = agregados[nombre]; lo, hi = c - (m - lo), c + (hi - m); m = c
        ok = None
        if teo is not None and lo == lo and hi - lo > 1e-12:
            tol = 1e-9 * max(1.0, abs(teo))
            ok = bool(lo - tol <= teo <= hi + tol)
        out.append({"KPI": nombre, "Teórico": teo, "Simulado · media": m, "IC95 inf": lo, "IC95 sup": hi,
                    "Error relativo": (m - teo) / abs(teo) if (teo is not None and abs(teo) > 1e-12) else None,
                    "Teórico en IC95": ok})
    return pd.DataFrame(out).set_index("KPI")


def resumen_A(mc, tA):
    d = {"Beneficio diario esperado E(Y) (€)": ("E(Y)", tA["EY"]),
         "Desviación típica σ(Y) (€)": ("σ(Y)", tA["sdY"]),
         "Probabilidad de déficit P(B < 0)": ("P(B < 0)", tA["Pdef"]),
         "Probabilidad de comprar en el mercado": ("P(compra en el mercado)", tA["Pspot"]),
         "VaR 95 % · percentil 5 de Y (€)": ("VaR 5 %", tA["VaR"]),
         "CVaR 5 % · media del 5 % peor (€)": ("CVaR 5 %", tA["CVaR"]),
         "Déficit medio comprado (MWh)": ("E(déficit)", tA["Edef"]),
         "Generación media E(G) (MWh)": ("E(G)", tA["EG"])}
    for f in ESCENARIOS:
        nf = NOMBRE[f]; pf = tA["por_tipo"][f]
        d[f"E(Y | {nf}) (€)"] = (f"E(Y | {nf})", pf["EY"])
        d[f"CV(Y | {nf})"] = (f"CV({nf})", pf["CV"])
        d[f"P(B < 0 | {nf})"] = (f"P(B < 0 | {nf})", pf["Pdef"])
    for k_, nom in enumerate(TERMINOS):
        d[f"Contribución a Var(Y) · {nom}"] = (f"Contrib. {nom}", float(tA["contrib"][k_]))
    Y = mc["Y"]
    return resumen(mc["replicas"], d, {"VaR 95 % · percentil 5 de Y (€)": float(np.quantile(Y, 0.05)),
                                        "CVaR 5 % · media del 5 % peor (€)": _cvar_muestra(Y)})


def resumen_B(mc, tB, tA):
    d = {"Beneficio diario esperado sin batería (€)": ("E(Y) sin batería", tB["EY_sin"]),
         "Beneficio diario esperado con batería (€)": ("E(Y) con batería", tB["EY_con"]),
         "Ganancia media diaria pareada E(Δ) (€)": ("Ganancia E(Δ)", tB["E_gan"]),
         "P(comprar en el mercado) sin batería": ("P(compra) sin batería", tB["Pspot_sin"]),
         "P(comprar en el mercado) con batería": ("P(compra) con batería", tB["Pspot_con"]),
         "Déficits cubiertos por completo por la batería": ("Déficits cubiertos por completo", tB["cubiertos"]),
         "VaR 95 % diario sin batería (€)": ("VaR 5 % sin batería", tA.get("VaR")),
         "VaR 95 % diario con batería (€)": ("VaR 5 % con batería", None),
         "CVaR 5 % diario sin batería (€)": ("CVaR 5 % sin batería", tA.get("CVaR")),
         "CVaR 5 % diario con batería (€)": ("CVaR 5 % con batería", None),
         "Estado de carga medio (MWh)": ("Estado de carga medio", tB["SOC_medio"]),
         "LOLE sin batería (días/año)": ("LOLE sin batería", tB["LOLE_sin"]),
         "LOLE con batería (días/año)": ("LOLE con batería", tB["LOLE_con"]),
         "σ(Δ) diario (€)": ("σ(Δ) diario", tB["sd_gan"])}
    for f in ESCENARIOS:
        nf = NOMBRE[f]
        d[f"E(Δ | {nf}) (€)"] = (f"E(Δ | {nf})", tB["por_tipo"][f]["E_gan"])
    Ys, Yc = mc["Ys"], mc["Yc"]
    agr = {"VaR 95 % diario sin batería (€)": float(np.quantile(Ys, 0.05)), "VaR 95 % diario con batería (€)": float(np.quantile(Yc, 0.05)),
           "CVaR 5 % diario sin batería (€)": _cvar_muestra(Ys), "CVaR 5 % diario con batería (€)": _cvar_muestra(Yc),
           "σ(Δ) diario (€)": float((Yc - Ys).std(ddof=1))}
    return resumen(mc["replicas"], d, agr)


# ==============================================================================================
# Objetivos, análisis de escenarios y diferencias
# ==============================================================================================
OBJ_NOMBRES = {"EY_MIN": "Beneficio diario esperado E(Y) (con batería)", "PDEF_MAX": "Probabilidad de comprar en el mercado (con batería)",
               "CVAR_MIN": "CVaR 5 % diario (sin batería)", "NETO_MIN": "Ganancia neta anual de la batería"}


def evaluar_objetivos(t, obj):
    A, B = t["A"], t["B"]
    filas = []
    def fila(c, valor, txt, cumple):
        filas.append({"Objetivo": OBJ_NOMBRES[c], "Valor": valor, "Umbral": txt, "Cumple": "Sí" if cumple else "No"})
    if obj.get("EY_MIN") is not None:
        fila("EY_MIN", B["EY_con"], f"≥ {eur(obj['EY_MIN'])}", B["EY_con"] >= obj["EY_MIN"] - 1e-9)
    if obj.get("PDEF_MAX") is not None:
        fila("PDEF_MAX", B["Pspot_con"], f"≤ {pct(obj['PDEF_MAX'], 1)}", B["Pspot_con"] <= obj["PDEF_MAX"] + 1e-12)
    if obj.get("CVAR_MIN") is not None:
        fila("CVAR_MIN", A["CVaR"], f"≥ {eur(obj['CVAR_MIN'])}", A["CVaR"] >= obj["CVAR_MIN"] - 1e-9)
    if obj.get("NETO_MIN") is not None:
        fila("NETO_MIN", B["neto_anual"], f"≥ {eur(obj['NETO_MIN'])}", B["neto_anual"] >= obj["NETO_MIN"] - 1e-9)
    return pd.DataFrame(filas, columns=["Objetivo", "Valor", "Umbral", "Cumple"]).set_index("Objetivo")


def analizar_escenario(esc, com, aj, progreso=None):
    t = teoria(com, esc)
    mA = simular_A(com, esc, aj["nsim"], aj["replicas"], aj["semilla"], progreso)
    mB = simular_B(com, esc, aj["anyos"], aj["semilla"], progreso)
    return {"esc": dict(esc), "teoria": t, "mcA": mA, "mcB": mB, "resA": resumen_A(mA, t["A"]),
            "resB": resumen_B(mB, t["B"], t["A"]), "objetivos": evaluar_objetivos(t, com["objetivos"])}


def diferencias(res, base):
    """Diferencias frente al escenario base (positivo = mejora), pareadas réplica a réplica."""
    if base not in res:
        return None
    rb = res[base]
    filas = []
    for n, r in res.items():
        if n == base:
            continue
        ta, tb = rb["teoria"], r["teoria"]
        dA = r["mcA"]["replicas"]["E(Y)"] - rb["mcA"]["replicas"]["E(Y)"]
        dB = r["mcB"]["replicas"]["E(Y) con batería"] - rb["mcB"]["replicas"]["E(Y) con batería"]
        mA, loA, hiA = _ic_media(dA); mB, loB, hiB = _ic_media(dB)
        filas.append({"Escenario": n,
                      "ΔE(Y) diario sin batería · teórico (€)": tb["A"]["EY"] - ta["A"]["EY"],
                      "ΔE(Y) diario sin batería · simulado (€)": mA, "IC95 ΔE(Y) (€)": (loA, hiA),
                      "ΔP(B < 0) · teórico": tb["A"]["Pdef"] - ta["A"]["Pdef"],
                      "ΔVaR 95 % · teórico (€)": tb["A"]["VaR"] - ta["A"]["VaR"],
                      "ΔCVaR 5 % · teórico (€)": tb["A"]["CVaR"] - ta["A"]["CVaR"],
                      "ΔE(Y) diario con batería · teórico (€)": tb["B"]["EY_con"] - ta["B"]["EY_con"],
                      "ΔE(Y) diario con batería · simulado (€)": mB, "IC95 ΔE(Y) con batería (€)": (loB, hiB),
                      "Ganancia de la batería · teórico (€/día)": tb["B"]["E_gan"],
                      "Δ ganancia neta anual de la batería · teórico (€)": tb["B"]["neto_anual"] - ta["B"]["neto_anual"]})
    return pd.DataFrame(filas).set_index("Escenario") if filas else None


def tabla_comparativa(res):
    cols = {}
    for n, r in res.items():
        A, B, e = r["teoria"]["A"], r["teoria"]["B"], r["esc"]
        ra, rb = r["resA"], r["resB"]
        cols[n] = {
            "Teórico · E(Y) diario sin batería (€)": A["EY"], "Teórico · P(B < 0)": A["Pdef"],
            "Teórico · VaR 95 % (€)": A["VaR"], "Teórico · CVaR 5 % (€)": A["CVaR"],
            "Teórico · E(Y) diario con batería (€)": B["EY_con"], "Teórico · ganancia de la batería (€/día)": B["E_gan"],
            "Teórico · P(compra) con batería": B["Pspot_con"], "Teórico · LOLE con batería (días/año)": B["LOLE_con"],
            "Teórico · ganancia neta anual de la batería (€)": B["neto_anual"],
            "Simulado · E(Y) diario sin batería (€)": ra.loc["Beneficio diario esperado E(Y) (€)", "Simulado · media"],
            "Simulado · ganancia de la batería (€/día)": rb.loc["Ganancia media diaria pareada E(Δ) (€)", "Simulado · media"],
            "Simulado · CVaR 5 % con batería (€)": rb.loc["CVaR 5 % diario con batería (€)", "Simulado · media"],
            "Objetivos cumplidos": f"{(r['objetivos']['Cumple'] == 'Sí').sum()} de {len(r['objetivos'])}",
        }
    return pd.DataFrame(cols)


def tabla_objetivos(res):
    return pd.DataFrame({n: r["objetivos"]["Cumple"] for n, r in res.items()})


# ==============================================================================================
# Batería: barridos (O5.1, O5.2, O6.1, O8.5), racha (O6.2), estado inicial (O8.2)
# ==============================================================================================
PARAM_BARRIDO = {"cap": "Capacidad de la batería (MWh)", "eta": "Eficiencia η"}


def barrido(com, esc, par, valores, anyos, semilla, tA=None, progreso=None):
    """Ganancia de la batería al variar su capacidad o su eficiencia: teórica (cadena) y simulada (misma base)."""
    tA = tA or teoria_A(com, esc)
    base = base_anual(com, esc, anyos, semilla)
    filas = []
    for v in valores:
        e2 = dict(esc); e2[par] = float(v)
        tB = teoria_bateria(com, e2, tA)
        bat = simular_bateria(e2, base)
        gr_ = (bat["Y"] - base["Y"]).mean(axis=1)
        m, lo, hi = _ic_media(gr_)
        fila = {PARAM_BARRIDO[par]: float(v), "Teórico · E(Δ) (€/día)": tB["E_gan"], "Simulado · E(Δ) (€/día)": m,
                "IC95 E(Δ) (€/día)": (lo, hi), "Teórico · P(compra) con batería": tB["Pspot_con"],
                "Teórico · LOLE con batería (días/año)": tB["LOLE_con"], "Teórico · ganancia anual (€)": tB["gan_anual"],
                "Amortización anual (€)": tB["amortizacion"], "Teórico · ganancia neta anual (€)": tB["neto_anual"]}
        if par == "cap":
            fila["Ganancia por MWh de capacidad (€/año)"] = tB["gan_anual"] / v if v > 0 else float("nan")
        filas.append(fila)
        if progreso:
            progreso()
    df = pd.DataFrame(filas).set_index(PARAM_BARRIDO[par])
    if par == "cap":
        prev = df["Teórico · ganancia anual (€)"].shift(1); cprev = df.index.to_series().shift(1)
        df["Ganancia marginal por MWh (€/año)"] = (df["Teórico · ganancia anual (€)"] - prev) / (df.index.to_series() - cprev)
    return df


def capacidad_optima(df):
    return float(df["Teórico · ganancia neta anual (€)"].idxmax())


def experimento_racha(com, esc, inicio=100, dias=20, anyos=200, semilla=1000, tA=None):
    """O6.2 · 20 días seguidos Anticiclónicos desde `inicio`: estado de carga medio día a día (teórico y simulado),
    frente al mismo año sin racha (misma base)."""
    tA = tA or teoria_A(com, esc)
    t0 = teoria_bateria(com, esc, tA); t1 = teoria_bateria(com, esc, tA, racha=(inicio, dias))
    b0 = base_anual(com, esc, anyos, semilla + 5000); b1 = base_anual(com, esc, anyos, semilla + 5000, racha=(inicio, dias))
    s0 = simular_bateria(esc, b0); s1 = simular_bateria(esc, b1)
    df = pd.DataFrame({"Día": np.arange(1, N_DIAS + 1),
                       "Teórico · carga media sin racha (MWh)": t0["E_soc"], "Teórico · carga media con racha (MWh)": t1["E_soc"],
                       "Simulado · carga media sin racha (MWh)": s0["E_ini"].mean(axis=0),
                       "Simulado · carga media con racha (MWh)": s1["E_ini"].mean(axis=0),
                       "Teórico · ganancia con racha (€)": t1["E_gan_dia"],
                       "Simulado · Y con batería, racha (€)": s1["Y"].mean(axis=0),
                       "Simulado · Y sin batería, racha (€)": b1["Y"].mean(axis=0)}).set_index("Día")
    fin = inicio + dias - 1
    sin_, con_ = t0["E_soc"], t1["E_soc"]
    rec = None
    for t in range(fin, N_DIAS):
        if con_[t] >= 0.95 * sin_[t]:
            rec = t + 1 - fin
            break
    resumen_ = {"Carga media al empezar la racha (MWh)": float(con_[inicio - 1]),
                "Carga media al acabar la racha (MWh)": float(con_[fin]),
                "Días de racha hasta vaciar el 90 % de la carga": next((t - inicio + 1 for t in range(inicio - 1, fin + 1)
                                                                        if con_[t] <= 0.1 * con_[inicio - 1]), float("nan")),
                "Días hasta recuperar el 95 % de la carga habitual": rec if rec is not None else float("nan"),
                "Pérdida media durante la racha sin batería (€/día)": float(b1["Y"][:, inicio - 1:fin].mean()),
                "Pérdida media durante la racha con batería (€/día)": float(s1["Y"][:, inicio - 1:fin].mean()),
                "Pérdida del año por la racha, con batería (€)": float((s0["Y"] - s1["Y"]).sum(axis=1).mean())}
    return df, resumen_


def estado_inicial(com, esc, valores=(0.0, 0.5, 1.0), anyos=200, semilla=1000, tA=None):
    """O8.2 · beneficio anual esperado con batería según el estado de carga inicial (misma base)."""
    tA = tA or teoria_A(com, esc)
    base = base_anual(com, esc, anyos, semilla)
    filas = []
    for v in valores:
        tB = teoria_bateria(com, dict(esc, E1=float(v)), tA)
        s = simular_bateria(esc, base, E1=float(v))
        m, lo, hi = _ic_media(s["Y"].sum(axis=1))
        filas.append({"Carga inicial": float(v), "Teórico · beneficio anual con batería (€)": N_DIAS * tB["EY_con"],
                      "Simulado · beneficio anual con batería (€)": m, "IC95 (€)": (lo, hi),
                      "Teórico · ganancia anual de la batería (€)": tB["gan_anual"]})
    return pd.DataFrame(filas).set_index("Carga inicial")


def contraste_ganancia(mcB, tB):
    """O8.1 · H₀: E(Δ) = 0. Los días de un año están correlacionados (la batería tiene memoria), así que el contraste
    válido usa las medias anuales (macro-réplicas); se muestra también el ingenuo que trata los días como independientes."""
    G = mcB["Yc"] - mcB["Ys"]
    filas = []
    for nom, x, n_ef in (("Medias por año (macro-réplicas)", G.mean(axis=1), None), ("Días tratados como independientes", G.ravel(), None)):
        n = len(x); m = float(x.mean()); se = float(x.std(ddof=1)) / math.sqrt(n)
        t = m / se if se > 0 else float("inf")
        pv = float(2 * _stats.t.sf(abs(t), n - 1))
        filas.append({"Procedimiento": nom, "n": n, "Media de Δ (€/día)": m, "Error estándar (€/día)": se,
                      "IC95 (€/día)": (m - _tq(n) * se, m + _tq(n) * se), "Estadístico t": t, "p-valor": pv,
                      "Decisión (5 %)": "se rechaza H₀" if pv < 0.05 else "no se rechaza H₀"})
    df = pd.DataFrame(filas).set_index("Procedimiento")
    x = G[:, :-1].ravel(); y = G[:, 1:].ravel()
    rho = float(np.corrcoef(x, y)[0, 1])
    return df, rho, tB["E_gan"]


def gestion_activa(com, esc, cv_exp=0.3, umbrales=(60.0, 70.0, 80.0, 90.0, 100.0), anyos=100, semilla=1000):
    """O8.3 · precio de exportación diario aleatorio (lognormal de media P_exp y CV dado); si supera el umbral, además
    de vender el excedente se vende lo almacenado. Comparación con la regla reactiva bajo los mismos precios."""
    base = base_anual(com, esc, anyos, semilla)
    rng = np.random.default_rng(int(semilla) + 999)
    s2 = math.log(1 + cv_exp ** 2)
    pr = np.exp(math.log(float(esc["P_exp"])) - s2 / 2 + math.sqrt(s2) * rng.standard_normal(base["Bp"].shape))
    filas = []
    react = simular_bateria(esc, base, activa=(pr, np.inf))
    m0 = react["Y"].mean(axis=1)
    for u in [None] + list(umbrales):
        s = react if u is None else simular_bateria(esc, base, activa=(pr, float(u)))
        m, lo, hi = _ic_media(s["Y"].mean(axis=1) - m0)
        filas.append({"Regla": "reactiva (sin vender lo almacenado)" if u is None else f"vender lo almacenado si P_exp > {num(u, 0)} €/MWh",
                      "Simulado · E(Y) con batería (€/día)": float(s["Y"].mean()),
                      "Diferencia frente a la reactiva (€/día)": m, "IC95 diferencia (€/día)": (lo, hi),
                      "P(compra en el mercado)": float((s["res"] > 1e-12).mean())})
    return pd.DataFrame(filas).set_index("Regla")


def bateria_vs_solar(com, esc, coste_solar=9000.0, tA=None):
    """O8.6 · con el mismo coste anual que la batería, ampliar la solar en Δ MWh/día de media (escalando G_s)."""
    tA = tA or teoria_A(com, esc)
    tB = teoria_bateria(com, esc, tA)
    presupuesto = tB["amortizacion"]
    ms = sum(pi * momentos_fuentes(com, esc, f)["Gs"][0] for pi, f in zip(normalizar(esc["p"]), ESCENARIOS))
    extra = presupuesto / coste_solar if coste_solar > 0 else 0.0
    e2 = dict(esc, solar_factor=float(esc["solar_factor"]) * (1 + extra / ms) if ms > 0 else 1.0)
    tA2 = teoria_A(com, e2)
    filas = {"Batería de " + num(float(esc["cap"]), 0) + " MWh": {
                 "Coste anual (€)": presupuesto, "Ganancia anual (€)": tB["gan_anual"], "Ganancia neta anual (€)": tB["neto_anual"],
                 "CVaR 5 % diario (€)": float("nan"), "P(compra en el mercado)": tB["Pspot_con"]},
             f"Solar +{num(extra, 1)} MWh/día de media": {
                 "Coste anual (€)": presupuesto, "Ganancia anual (€)": N_DIAS * (tA2["EY"] - tA["EY"]),
                 "Ganancia neta anual (€)": N_DIAS * (tA2["EY"] - tA["EY"]) - presupuesto, "CVaR 5 % diario (€)": tA2["CVaR"],
                 "P(compra en el mercado)": tA2["Pspot"]}}
    return pd.DataFrame(filas).T, extra


# ==============================================================================================
# Sensibilidad (O2.2, O3.3) y estimador
# ==============================================================================================
PARAM_SENS = {"p4": "p del escenario Anticiclónico", "tri": "Solar G_s (a, m, b)", "weib_l": "Eólica · escala λ",
              "weib_k": "Eólica · forma k", "ln_mu": "Hidro · μ de ln G_h", "ln_sg": "Hidro · σ de ln G_h",
              "norm_mu": "Demanda · media", "norm_sd": "Demanda · desv. típica", "P_exp": "P_exp", "P_pen": "P_pen"}


def _aplicar(com, esc, par, v):
    c2, e2 = copy.deepcopy(com), dict(esc)
    if par == "p4":
        p = list(normalizar(esc["p"])); p4 = min(0.99, p[3] * (1 + v)); s = (1 - p4) / (1 - p[3])
        e2["p"] = [p[0] * s, p[1] * s, p[2] * s, p4]
    elif par in ("P_exp", "P_pen"):
        e2[par] = float(esc[par]) * (1 + v)
    else:
        for f in ESCENARIOS:
            d = c2["dist"][f]
            if par == "tri":
                d["tri"] = [x * (1 + v) for x in d["tri"]]
            elif par == "weib_l":
                d["weib"] = [d["weib"][0], d["weib"][1] * (1 + v)]
            elif par == "weib_k":
                d["weib"] = [d["weib"][0] * (1 + v), d["weib"][1]]
            elif par == "ln_mu":
                d["ln"] = [d["ln"][0] * (1 + v), d["ln"][1]]
            elif par == "ln_sg":
                d["ln"] = [d["ln"][0], d["ln"][1] * (1 + v)]
            elif par == "norm_mu":
                d["norm"] = [d["norm"][0] * (1 + v), d["norm"][1]]
            elif par == "norm_sd":
                d["norm"] = [d["norm"][0], d["norm"][1] * (1 + v)]
    return c2, e2


def sensibilidad_parametros(com, esc, variaciones):
    filas = []
    for par, nom in PARAM_SENS.items():
        for v in variaciones:
            t = teoria_A(*_aplicar(com, esc, par, v), con_dist=False)
            filas.append({"Parámetro": nom, "Variación": v, "Teórico · E(Y) (€)": t["EY"], "Teórico · Var(Y)": t["VY"],
                          "Teórico · σ(Y) (€)": t["sdY"]})
    elas = {}
    for par, nom in PARAM_SENS.items():
        vs = [teoria_A(*_aplicar(com, esc, par, h), con_dist=False)["VY"] for h in (-0.01, 0.01)]
        elas[nom] = (math.log(vs[1]) - math.log(vs[0])) / (math.log(1.01) - math.log(0.99)) if min(vs) > 0 else 0.0
    return pd.DataFrame(filas), elas


def indices_sobol(com, esc, n=20000, semilla=1000):
    """Índices de Sobol (Saltelli/Jansen) de las entradas aleatorias T, G_s, G_w, G_h y D sobre Y."""
    rng = np.random.default_rng(int(semilla) + 7)
    UA, UB = rng.random((6, n)), rng.random((6, n))
    yA = generar_dias(com, esc, UA)["Y"]; yB = generar_dias(com, esc, UB)["Y"]
    V = float(np.var(np.concatenate([yA, yB]), ddof=1))
    filas = []
    for i, nom in enumerate(("Escenario T", "Solar G_s", "Eólica G_w", "Hidráulica G_h", "Demanda D")):
        UAB = UA.copy(); UAB[i] = UB[i]
        yAB = generar_dias(com, esc, UAB)["Y"]
        filas.append({"Entrada": nom, "Índice de primer orden S₁": float(np.mean(yB * (yAB - yA)) / V),
                      "Índice total S_T": float(0.5 * np.mean((yA - yAB) ** 2) / V)})
    df = pd.DataFrame(filas).set_index("Entrada")
    df.loc["Escenario T", "Teórico · S₁"] = teoria_A(com, esc, con_dist=False)["S_T"]
    return df


def solar_necesaria(com, esc, objetivo=0.0):
    """O3.3 · ampliación solar con la que E(Y) = objetivo: (a) desplazamiento fijo de Δ MWh todos los días;
    (b) más capacidad (G_s escalada), expresada en MWh/día de media. Bisección sobre la teoría exacta."""
    ms = sum(pi * momentos_fuentes(com, esc, f)["Gs"][0] for pi, f in zip(normalizar(esc["p"]), ESCENARIOS))
    def EY(kind, x):
        e2 = dict(esc, solar_extra=float(esc["solar_extra"]) + x) if kind == "despl" else \
            dict(esc, solar_factor=float(esc["solar_factor"]) * (1 + x / ms))
        return teoria_A(com, e2, con_dist=False)["EY"]
    filas = []
    for kind, nom in (("despl", "Desplazamiento fijo de G_s (mismos MWh todos los días)"),
                      ("escala", "Más capacidad solar (G_s escalada, MWh/día de media)")):
        lo, hi = 0.0, 50.0
        while EY(kind, hi) < objetivo and hi < 5000:
            hi *= 2
        for _ in range(40):
            mid = (lo + hi) / 2
            if EY(kind, mid) < objetivo:
                lo = mid
            else:
                hi = mid
        filas.append({"Forma de la ampliación": nom, "MWh/día de media adicionales": hi,
                      "Aumento de la solar media": hi / ms, "E(Y) resultante (€)": EY(kind, hi),
                      "Ganancia por MWh/día añadido (€/día)": (EY(kind, 1.0) - EY(kind, 0.0))})
    return pd.DataFrame(filas).set_index("Forma de la ampliación"), ms


def valores_barrido(vmin, vmax, puntos):
    v = np.geomspace(max(1, vmin), max(vmin + 1, vmax), max(2, int(puntos)))
    return sorted(set(int(round(x)) for x in v))


def sensibilidad_estimador(com, esc, metrica, valores, replicas, semilla, progreso=None):
    """E(Y) frente al nº de días (Bloque A) o ganancia de la batería frente al nº de años simulados (Bloque B)."""
    tA = teoria_A(com, esc, con_dist=metrica != "dia")
    if metrica == "dia":
        teo, sd1 = tA["EY"], tA["sdY"]
    else:
        tB = teoria_bateria(com, esc, tA); teo, sd1 = tB["E_gan"], float("nan")
    tq = _tq(int(replicas))
    filas, k, tot = [], 0, len(valores) * int(replicas)
    for n in valores:
        medias = []
        for r in range(int(replicas)):
            if metrica == "dia":
                rng = np.random.default_rng(int(semilla) + r)
                medias.append(float(generar_dias(com, esc, rng.random((6, n)))["Y"].mean()))
            else:
                b = base_anual(com, esc, n, int(semilla) + 50_000 * (r + 1))
                medias.append(float((simular_bateria(esc, b)["Y"] - b["Y"]).mean()))
            k += 1
            if progreso:
                progreso(k, tot)
        m = float(np.mean(medias)); sdr = float(np.std(medias, ddof=1)) if len(medias) > 1 else 0.0
        se = sdr / math.sqrt(len(medias))
        filas.append({"n": n, "Media de las réplicas": m, "IC95 inf": m - tq * se, "IC95 sup": m + tq * se,
                      "σ entre réplicas": sdr, "σ teórica del estimador": sd1 / math.sqrt(n) if sd1 == sd1 else float("nan"),
                      "Teórico": teo, "Error relativo": (m - teo) / abs(teo), "Medias por réplica": medias})
    return pd.DataFrame(filas).set_index("n")


# ==============================================================================================
# Mecanismo
# ==============================================================================================
def un_dia(com, esc, uT, us, uw, uh, ud):
    U = np.array([[uT], [us], [uw], [uh], [ud], [0.5]])
    o = generar_dias(com, esc, U)
    return {k: (float(v[0]) if hasattr(v, "__len__") else v) for k, v in o.items() if k != "terminos"} | \
        {"terminos": [float(x[0]) for x in o["terminos"]]}


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
    if isinstance(v, (bool, np.bool_)):
        return "Sí" if v else "No"
    if isinstance(v, tuple):
        return f"[{fmt_val(k, v[0])}; {fmt_val(k, v[1])}]"
    if "(€" in k or k.endswith("€"):
        return eur(v, 1 if abs(v) < 100 else 0)
    if "(MWh" in k or "MWh/día" in k or "(días" in k or "Días" in k:
        return num(v, 2)
    if "P(" in k or "%" in k or "Contribución" in k or "Peso" in k or "Cuota" in k or "Aumento" in k \
            or "probabilidad" in k.lower() or "S₁" in k or "S_T" in k or "cubiertos" in k or "Carga inicial" in k:
        return pct(v, 2)
    if "p-valor" in k:
        return num(v, 4) if v >= 1e-4 else "< 0,0001"
    if "CV" in k or "σ" in k or "η" in k or "Eficiencia" in k or "Estadístico" in k:
        return num(v, 3)
    if k == "n":
        return num(v, 0)
    return num(v, 3)


def fmt_kpi(k, c, v):
    return fmt_val(k, v)


def fmt_res(fila, col, v):
    if col == "Teórico en IC95":
        return "—" if v is None else ("Sí" if v else "No")
    if col == "Error relativo":
        return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else pct(v, 2)
    if col == "Teórico" and v is not None and isinstance(v, float) and math.isnan(v):
        return "solo simulación"
    if col == "Teórico" and v is None:
        return "solo simulación"
    return fmt_val(fila, v)


def fmt_col(fila, col, v):
    return fmt_val(col, v)


def fmt_obj(fila, col, v):
    if col == "Valor":
        return pct(v, 2) if fila.startswith("Probabilidad") else eur(v)
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
    act = [f for f in ESCENARIOS if pf[f]["p"] > 0]
    mayor_cv = max(act, key=lambda f: pf[f]["CV"])
    dom = int(np.argmax(np.abs(A["contrib"])))
    cvar_con = r["resB"].loc["CVaR 5 % diario con batería (€)", "Simulado · media"]
    txt = (f"{nombre}: un día deja en media E(Y) = {eur(A['EY'])} (σ = {eur(A['sdY'])}); P(B < 0) = {pct(A['Pdef'], 1)}, "
           f"VaR 95 % = {eur(A['VaR'])} y CVaR 5 % = {eur(A['CVaR'])}. El escenario {NOMBRE[mayor_cv]} es el de mayor "
           f"incertidumbre relativa (CV = {num(pf[mayor_cv]['CV'], 3)}) y el término «{TERMINOS[dom]}» explica el "
           f"{pct(A['contrib'][dom], 0)} de Var(Y). La batería de {num(float(r['esc']['cap']), 0)} MWh gana {eur(B['E_gan'], 1)} "
           f"al día ({eur(B['gan_anual'])} al año, {eur(B['neto_anual'])} netos de amortización), reduce los días de compra "
           f"en el mercado de {num(B['LOLE_sin'], 0)} a {num(B['LOLE_con'], 0)} al año y lleva el CVaR 5 % diario a "
           f"{eur(cvar_con)} (simulado) (valores teóricos salvo indicación); {val}.")
    n_ok = (r["objetivos"]["Cumple"] == "Sí").sum()
    txt += f" Cumple {n_ok} de los {len(r['objetivos'])} objetivos."
    return txt


def recomendaciones(res, base, com, barr_cap=None):
    """O7.1 · recomendación de inversión en la batería y condiciones que la cambiarían."""
    rec = []
    if base in res:
        r = res[base]; B = r["teoria"]["B"]; e = r["esc"]
        ic = r["resB"].loc["Ganancia media diaria pareada E(Δ) (€)", ["IC95 inf", "IC95 sup"]]
        verbo = "Instalar" if B["neto_anual"] > 0 else "No instalar"
        rec.append(f"{verbo} la batería de {num(float(e['cap']), 0)} MWh: gana {eur(B['E_gan'], 1)} al día (IC95 % simulado "
                   f"[{eur(ic.iloc[0], 1)}; {eur(ic.iloc[1], 1)}]), es decir {eur(B['gan_anual'])} al año frente a una "
                   f"amortización de {eur(B['amortizacion'])} ({eur(float(e['amort']))} por MWh y año): neto "
                   f"{eur(B['neto_anual'])} al año. Compensa mientras la amortización no supere "
                   f"{eur(B['gan_anual'] / float(e['cap']) if float(e['cap']) > 0 else 0)} por MWh y año.")
        if barr_cap is not None and len(barr_cap):
            cst = capacidad_optima(barr_cap)
            rec.append(f"Dimensionar la batería en {num(cst, 0)} MWh: maximiza la ganancia neta anual "
                       f"({eur(barr_cap.loc[cst, 'Teórico · ganancia neta anual (€)'])}); la ganancia crece con rendimientos "
                       "decrecientes, porque una batería grande rara vez se llena con superávits tan escasos.")
        rec.append(f"La batería no resuelve el problema de fondo: incluso con ella E(Y) = {eur(B['EY_con'])} al día y se compra "
                   f"en el mercado el {pct(B['Pspot_con'], 0)} de los días. La palanca principal es la generación (o la demanda): "
                   "véanse la solar necesaria para E(Y) ≥ 0 y la comparación batería frente a solar.")
    filas = [(n, (r["objetivos"]["Cumple"] == "Sí").sum(), r["teoria"]) for n, r in res.items()]
    max_ok = max(f[1] for f in filas)
    mejor = sorted([f for f in filas if f[1] == max_ok], key=lambda f: -f[2]["B"]["EY_con"])[0]
    rec.append(f"Entre los escenarios analizados, «{mejor[0]}» cumple más objetivos ({mejor[1]} de "
               f"{len(res[mejor[0]]['objetivos'])}), con E(Y) = {eur(mejor[2]['B']['EY_con'])} al día con batería.")
    return rec


def texto_recomendacion(res, base, com, barr_cap=None):
    return " ".join(f"({i + 1}) {r}" for i, r in enumerate(recomendaciones(res, base, com, barr_cap)))


KPIS_CUADRO = [
    ("Beneficio diario esperado E(Y), con y sin batería", "rentabilidad media de la micro-red"),
    ("Probabilidad de déficit y de compra en el mercado", "frecuencia con que la generación no cubre la demanda"),
    ("VaR 95 % y CVaR 5 % diarios", "tamaño de las pérdidas en los días malos para provisionar liquidez"),
    ("LOLE (días al año con compra en el mercado)", "nivel de servicio y dependencia del mercado spot"),
    ("Ganancia de la batería y ganancia neta de amortización", "rentabilidad de la inversión en almacenamiento"),
    ("Estado de carga medio y recuperación tras rachas", "robustez frente a rachas de mal tiempo"),
    ("Coste por escenario climático", "qué clima concentra las pérdidas (Anticiclónico)"),
]
