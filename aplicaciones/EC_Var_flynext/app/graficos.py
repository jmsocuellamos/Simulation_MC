# -*- coding: utf-8 -*-
"""Gráficos (matplotlib) de la aplicación Fly-Next. Cada función devuelve una Figure."""
import io
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import flynext as fx

# Paleta categórica validada (orden fijo; el color sigue a la entidad, nunca a su posición)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COLOR_RUTA = {1: SERIES[0], 2: SERIES[3], 3: SERIES[6]}
COLOR_FRANJA = {"Valle": SERIES[2], "Llano": SERIES[3], "Punta": SERIES[7]}
COLOR_TERM = [SERIES[1], SERIES[0], SERIES[4], SERIES[5]]          # combustible, mano de obra, peaje, externalización
SIN, CON = "#8a8984", SERIES[7]
BUENO, CRITICO = "#0b7a0b", "#b23232"
TEXTO, TEXTO2, REJILLA = "#0b0b0b", "#52514e", "#e1e0d9"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.edgecolor": "#b9b8b2", "axes.labelcolor": TEXTO2, "xtick.color": TEXTO2, "ytick.color": TEXTO2,
    "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
_EUR0 = mticker.FuncFormatter(lambda y, _: fx.num(y, 0))
_PCT = mticker.PercentFormatter(1.0, decimals=0)


def color_escenario(i):
    return SERIES[i % len(SERIES)]


def fig_a_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _etiquetas(ax, nombres):
    ax.set_xticks(range(len(nombres)))
    ax.set_xticklabels(nombres, rotation=30 if len(nombres) > 2 else 0, ha="right" if len(nombres) > 2 else "center",
                       fontsize=8)


# ---------------------------------------------------------------------------------------------- mecanismo
def fig_mecanismo(p, uT, partes_sin, partes_con, titulo=None):
    """Izquierda: composición, la ruta se elige con U_T sobre las probabilidades acumuladas. Derecha: coste de la
    entrega generada, por términos, sin y con tráfico."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.2, 1]})
    F = np.cumsum(fx.normalizar(p)); esc = np.concatenate([[0], F])
    for i, r in enumerate(fx.RUTAS):
        a1.add_patch(plt.Rectangle((-0.6, esc[i]), 3.2, esc[i + 1] - esc[i], color=COLOR_RUTA[r], alpha=0.12, lw=0))
        a1.plot([i - 0.5, i + 0.5], [F[i], F[i]], color=COLOR_RUTA[r], lw=3)
        a1.text(2.62, (esc[i] + esc[i + 1]) / 2, f"{fx.NOMBRE_RUTA[r]}\n({fx.num(esc[i], 2)}, {fx.num(F[i], 2)}]",
                va="center", fontsize=8, color=COLOR_RUTA[r])
    k = int(np.searchsorted(F, uT))
    k = min(k, 2)
    a1.annotate("", xy=(k, uT), xytext=(-0.55, uT), arrowprops=dict(arrowstyle="->", color=TEXTO, lw=1.6))
    a1.annotate("", xy=(k, -0.02), xytext=(k, uT), arrowprops=dict(arrowstyle="->", color=TEXTO, lw=1.6))
    a1.text(-0.55, min(uT + 0.03, 1.0), f"U_T = {fx.num(uT, 2)}", fontsize=9)
    a1.set_xlim(-0.6, 3.3); a1.set_ylim(-0.03, 1.05)
    a1.set_xticks(range(3)); a1.set_xticklabels([fx.NOMBRE_RUTA[r] for r in fx.RUTAS])
    a1.set_ylabel("F(T)"); a1.set_title("Paso 1 · ruta por transformada inversa", fontsize=9.5)
    nombres = ["sin tráfico", "con tráfico"]
    base = np.zeros(2)
    for j, nom in enumerate(fx.TERMINOS):
        v = np.array([partes_sin[j], partes_con[j]])
        if v.max() <= 0:
            continue
        a2.bar(nombres, v, bottom=base, color=COLOR_TERM[j], width=0.55, label=nom.split(" ")[0])
        base += v
    for i, b in enumerate(base):
        a2.text(i, b, fx.eur(b, 2), ha="center", va="bottom", fontsize=8.5)
    a2.set_ylim(0, max(base.max(), 1) * 1.2)
    a2.set_ylabel("€"); a2.set_title("Paso 2 · D, G, M y coste de la entrega", fontsize=9.5)
    a2.legend(loc="upper left", fontsize=7.5)
    if titulo:
        fig.suptitle(titulo, fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------- bloque A
def fig_hist_Y(tA, Y, umbral, titulo=None):
    Y = np.asarray(Y)
    hi = float(np.quantile(Y, 0.995))
    bins = np.linspace(0, hi, 70)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.hist(Y[Y <= hi], bins=bins, density=True, color=SERIES[0], alpha=0.5, edgecolor="white", label="simulación")
    Fb = np.interp(bins, tA["y"], tA["F"])
    dens = np.diff(Fb) / np.diff(bins)
    ax.step(bins[:-1], dens, where="post", color=TEXTO, lw=1.6, label="teórica (integración numérica)")
    ax.axvline(tA["EY"], color=SERIES[0], ls="--", lw=1.4, label=f"E(Y) = {fx.eur(tA['EY'], 2)}")
    ax.axvline(tA["P90"], color=SERIES[3], ls="-.", lw=1.4, label=f"P90 = {fx.eur(tA['P90'], 2)}")
    ax.axvline(umbral, color=CRITICO, ls=":", lw=1.5, label=f"umbral {fx.eur(umbral)}")
    ax.set_yscale("log")
    ax.set_xlabel("coste de una entrega Y (€) · hasta el percentil 99,5"); ax.set_ylabel("densidad (escala log)")
    ax.set_title(titulo or "Distribución del coste de una entrega")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def fig_por_ruta(tA, resA, umbral, titulo=None):
    rutas = [fx.NOMBRE_RUTA[r] for r in fx.RUTAS]
    pr = tA["por_ruta"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    u = fx.num(umbral, 0)
    datos = [("E(Y | ruta) (€)", [pr[r]["EY"] for r in fx.RUTAS], [resA.loc[f"E(Y | {fx.NOMBRE_RUTA[r]}) (€)", "Simulado · media"] for r in fx.RUTAS], 2),
             (f"P(Y > {u} € | ruta)", [pr[r]["Pcola"] for r in fx.RUTAS], [resA.loc[f"P(Y > {u} € | {fx.NOMBRE_RUTA[r]})", "Simulado · media"] for r in fx.RUTAS], "%"),
             ("Coeficiente de variación", [pr[r]["CV"] for r in fx.RUTAS], [resA.loc[f"CV(Y | {fx.NOMBRE_RUTA[r]})", "Simulado · media"] for r in fx.RUTAS], 3)]
    for ax, (tit, v, s, d) in zip(axs, datos):
        bars = ax.bar(range(3), v, color=[COLOR_RUTA[r] for r in fx.RUTAS], width=0.6)
        ax.plot(range(3), s, "D", color=TEXTO, ms=6, label="simulado")
        for b_, x in zip(bars, v):
            ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height(), fx.pct(x, 1) if d == "%" else fx.num(x, d),
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(3)); ax.set_xticklabels(rutas)
        ax.set_title(tit, fontsize=9.5)
        ax.set_ylim(0, max(max(v), max(s)) * 1.25)
        if d == "%":
            ax.yaxis.set_major_formatter(_PCT)
    axs[0].legend(loc="upper left", fontsize=8)
    if titulo:
        fig.suptitle(titulo, fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    return fig


def fig_descomposicion(tA, resA, titulo=None):
    """Contribución de cada término a Var(Y): Cov(Xᵢ, Y)/Var(Y), teórica (barras) y simulada (rombos)."""
    v = list(tA["contrib"])
    s = [resA.loc[f"Contribución a Var(Y) · {n}", "Simulado · media"] for n in fx.TERMINOS]
    keep = [i for i in range(4) if abs(v[i]) > 1e-9 or abs(s[i]) > 1e-9]
    fig, ax = plt.subplots(figsize=(9, 3.0))
    x = np.arange(len(keep))
    ax.barh(x, [v[i] for i in keep], color=[COLOR_TERM[i] for i in keep], height=0.55)
    ax.plot([s[i] for i in keep], x, "D", color=TEXTO, ms=6, label="simulado")
    for j, i in enumerate(keep):
        ax.text(v[i], j, "  " + fx.pct(v[i], 1), va="center", fontsize=8.5)
    ax.set_yticks(x); ax.set_yticklabels([fx.TERMINOS[i] for i in keep])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(_PCT)
    ax.set_xlim(min(0, min(v)) * 1.2, max(v) * 1.25)
    ax.set_title(titulo or "Descomposición de Var(Y): Cov(término, Y) / Var(Y)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def fig_traza_A(traza, esc, titulo=None):
    """Primeras entregas simuladas: coste por términos (combustible, mano de obra, peaje) y ruta asignada."""
    tr = traza
    fuel = tr["D (km)"] * tr["G (L/km)"] * esc["P_fuel"]
    lab = tr["M (h)"] * esc["C_labor"]
    otros = tr["Y (€)"] - fuel - lab
    if esc.get("externalizar"):
        prem = tr["Ruta"] == "Premium"
        fuel = np.where(prem, 0, fuel); lab = np.where(prem, 0, lab); otros = tr["Y (€)"] - fuel - lab
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = tr["Entrega"]
    ax.bar(x, fuel, color=COLOR_TERM[0], width=0.8, label="combustible")
    ax.bar(x, lab, bottom=fuel, color=COLOR_TERM[1], width=0.8, label="mano de obra")
    ax.bar(x, otros, bottom=fuel + lab, color=COLOR_TERM[2], width=0.8, label="peaje / externalización")
    for r in fx.RUTAS:
        mk = tr["Ruta"] == fx.NOMBRE_RUTA[r]
        ax.plot(x[mk], np.full(mk.sum(), -4), "s", color=COLOR_RUTA[r], ms=5, label=f"ruta {fx.NOMBRE_RUTA[r]}")
    ax.set_xlabel("entrega simulada"); ax.set_ylabel("€")
    ax.set_title(titulo or f"Funcionamiento simulado · primeras {len(tr)} entregas")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=6, fontsize=7.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------- bloque B
def fig_hist_dias(tB, dias, presupuesto, titulo=None):
    d = np.asarray(dias)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.hist(d, bins=35, density=True, color=CON, alpha=0.45, edgecolor="white", label="días simulados")
    xx = np.linspace(min(d.min(), tB["dia_con"] - 4 * tB["sd_dia_con"]), max(d.max(), presupuesto * 1.02), 300)
    ax.plot(xx, np.exp(-0.5 * ((xx - tB["dia_con"]) / tB["sd_dia_con"]) ** 2) / (tB["sd_dia_con"] * math.sqrt(2 * math.pi)),
            color=TEXTO, lw=2, label="teórica (aprox. normal)")
    ax.axvline(tB["dia_con"], color=CON, ls="--", lw=1.4, label=f"E = {fx.eur(tB['dia_con'])}")
    ax.axvline(tB["P95"], color=SERIES[3], ls="-.", lw=1.4, label=f"P95 = {fx.eur(tB['P95'])}")
    ax.axvline(tB["CVaR"], color=SERIES[6], ls=":", lw=1.6, label=f"CVaR 5 % = {fx.eur(tB['CVaR'])}")
    ax.axvline(presupuesto, color=CRITICO, lw=1.6, label=f"presupuesto {fx.eur(presupuesto)}")
    ax.xaxis.set_major_formatter(_EUR0)
    ax.set_xlabel("coste diario con tráfico (€)"); ax.set_ylabel("densidad")
    ax.set_title(titulo or f"Coste diario · {fx.num(len(d), 0)} días simulados")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    return fig


def fig_franjas(tB, resB, titulo=None):
    fr = tB["franjas"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(3)
    a1.bar(x - 0.2, [fr[f]["Ysin"] for f in fx.FRANJAS], width=0.4, color=SIN, label="sin tráfico")
    a1.bar(x + 0.2, [fr[f]["Ycon"] for f in fx.FRANJAS], width=0.4, color=CON, label="con tráfico")
    s = [resB.loc[f"E(Y_con) por pedido · {f} (€)", "Simulado · media"] for f in fx.FRANJAS]
    a1.plot(x + 0.2, s, "D", color=TEXTO, ms=5, label="simulado (con tráfico)")
    for i, f in enumerate(fx.FRANJAS):
        a1.text(i + 0.2, fr[f]["Ycon"], f"+{fx.num(fr[f]['delta'], 2)}", ha="center", va="bottom", fontsize=8, color=CON)
    a1.set_xticks(x); a1.set_xticklabels(fx.FRANJAS); a1.set_ylabel("€ por pedido")
    a1.set_title("Coste medio por pedido y sobrecoste (+) por franja", fontsize=9.5)
    a1.set_ylim(0, max(fr[f]["Ycon"] for f in fx.FRANJAS) * 1.25)
    a1.legend(loc="upper left", fontsize=7.5)
    v = [fr[f]["cuota_sobre"] for f in fx.FRANJAS]
    a2.bar(x, v, color=[COLOR_FRANJA[f] for f in fx.FRANJAS], width=0.6)
    for i, y in enumerate(v):
        a2.text(i, y, fx.pct(y, 1), ha="center", va="bottom", fontsize=8.5)
    a2.set_xticks(x); a2.set_xticklabels(fx.FRANJAS); a2.yaxis.set_major_formatter(_PCT)
    a2.set_ylim(0, max(max(v), 0.01) * 1.2)
    a2.set_title("Cuota del sobrecoste diario", fontsize=9.5)
    if titulo:
        fig.suptitle(titulo, fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    return fig


def fig_horas(com, tB, titulo=None):
    """Perfil horario teórico: pedidos esperados por hora y coste esperado por hora con y sin tráfico."""
    h = np.arange(24)
    fran = [fx.HORAS_FRANJA[i] for i in h]
    lam = np.array([float(com["lambda"][f]) for f in fran])
    ycon = np.array([tB["franjas"][f]["Ycon"] for f in fran]); ysin = np.array([tB["franjas"][f]["Ysin"] for f in fran])
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(h, lam * ysin, color=SIN, width=0.8, label="coste sin tráfico")
    ax.bar(h, lam * (ycon - ysin), bottom=lam * ysin, color=CON, width=0.8, label="sobrecoste por tráfico")
    for i in h:
        ax.text(i, -60, fran[i][0], ha="center", va="top", fontsize=7, color=COLOR_FRANJA[fran[i]])
    ax.yaxis.set_major_formatter(_EUR0)
    ax.set_xticks(h); ax.set_xlabel("hora del día (V = Valle, L = Llano, P = Punta)"); ax.set_ylabel("€ por hora")
    ax.set_title(titulo or "Coste esperado por hora del día (λ_h · coste medio por pedido)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(-200, (lam * ycon).max() * 1.15)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------- comparativas
def fig_comparativa(res, com):
    nombres = list(res)
    cols = [color_escenario(i) for i in range(len(nombres))]
    fig, axs = plt.subplots(1, 4, figsize=(12.5, 4.0))
    x = np.arange(len(nombres))
    A = [res[n]["teoria"]["A"] for n in nombres]; B = [res[n]["teoria"]["B"] for n in nombres]
    vals = [a["EY"] for a in A]
    axs[0].bar(x, vals, color=cols, width=0.65)
    for i, v in enumerate(vals):
        axs[0].text(i, v, fx.num(v, 2), ha="center", va="bottom", fontsize=7.5)
    lo, hi = min(vals), max(vals); m = (hi - lo) or 1
    axs[0].set_ylim(lo - 1.2 * m, hi + 0.5 * m)
    axs[0].set_title("E(Y) por entrega (€)", fontsize=9)
    vals = [a["Pcola"] for a in A]
    axs[1].bar(x, vals, color=cols, width=0.65)
    for i, v in enumerate(vals):
        axs[1].text(i, v, fx.num(100 * v, 2), ha="center", va="bottom", fontsize=7.5)
    axs[1].yaxis.set_major_formatter(_PCT); axs[1].set_ylim(0, max(vals) * 1.25)
    axs[1].set_title(f"P(Y > {fx.num(com['umbral_Y'], 0)} €)", fontsize=9)
    s_ = [b["dia_sin"] for b in B]; d_ = [b["dia_delta"] for b in B]
    axs[2].bar(x, s_, color=SIN, width=0.65, label="sin tráfico")
    axs[2].bar(x, d_, bottom=s_, color=CON, width=0.65, label="sobrecoste")
    for i in x:
        axs[2].text(i, s_[i] + d_[i], fx.num((s_[i] + d_[i]) / 1000, 1) + "k", ha="center", va="bottom", fontsize=7.5)
    axs[2].yaxis.set_major_formatter(_EUR0); axs[2].set_ylim(0, max(a + b for a, b in zip(s_, d_)) * 1.18)
    axs[2].set_title("Coste diario esperado (€)", fontsize=9); axs[2].legend(fontsize=7, loc="lower right")
    vals = [b["Ppres"] for b in B]
    axs[3].bar(x, vals, color=cols, width=0.65)
    for i, v in enumerate(vals):
        axs[3].text(i, v, fx.num(100 * v, 2), ha="center", va="bottom", fontsize=7.5)
    axs[3].yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=1))
    axs[3].set_ylim(0, max(max(vals), 0.001) * 1.25)
    axs[3].set_title(f"P(coste diario > {fx.num(com['presupuesto'] / 1000, 0)}k €)", fontsize=9)
    for ax in axs:
        _etiquetas(ax, nombres)
    fig.text(0.5, 0.005, "El eje de E(Y) no empieza en 0 para que se aprecien las diferencias.", ha="center", fontsize=7.5,
             color=TEXTO2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def fig_ahorros(dif, titulo=None):
    """Ahorro diario frente al escenario base (teórico, simulado con IC95 % pareado) y ahorro neto de implantación."""
    nombres = list(dif.index)
    x = np.arange(len(nombres))
    teo = dif["Ahorro diario · teórico (€)"].to_numpy(float)
    sim = dif["Ahorro diario · simulado (€)"].to_numpy(float)
    ic = np.array([list(v) for v in dif["IC95 ahorro diario (€)"]])
    neto = dif["Ahorro diario neto · teórico (€)"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(x - 0.18, teo, width=0.36, color=[BUENO if v >= 0 else CRITICO for v in teo], alpha=0.8, label="teórico")
    ax.bar(x + 0.18, neto, width=0.36, color=SERIES[0], alpha=0.6, label="neto de implantación (teórico)")
    ax.errorbar(x - 0.18, sim, yerr=[sim - ic[:, 0], ic[:, 1] - sim], fmt="D", color=TEXTO, ms=5, capsize=3,
                label="simulado ± IC95 % (pareado)")
    for i, v in enumerate(teo):
        ax.text(i - 0.18, v, fx.num(v, 0), ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)
    ax.axhline(0, color=TEXTO2, lw=1)
    ax.yaxis.set_major_formatter(_EUR0)
    _etiquetas(ax, nombres)
    ax.set_ylabel("€ por día")
    ax.set_title(titulo or "Ahorro diario frente al escenario base")
    ax.legend(loc="best", fontsize=7.5)
    ax.margins(y=0.15)
    fig.tight_layout()
    return fig


def fig_validacion(res, com):
    nombres = list(res)
    u = fx.num(com["umbral_Y"], 0)
    kA = [("Coste esperado E(Y) (€)", "E(Y)"), (f"P(Y > {u} €)", "P(Y>umbral)"), ("Percentil 90 de Y (€)", "P90")]
    kB = [("Coste diario esperado con tráfico (€)", "coste diario"), ("Sobrecoste diario por tráfico (€)", "sobrecoste"),
          ("Percentil 95 del coste diario (€) · aprox. normal", "P95 diario")]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ks = kA + kB
    ancho = 0.8 / len(nombres)
    for i, n in enumerate(nombres):
        v = []
        for k, _ in ks:
            tab = res[n]["resA"] if (k, _) in kA else res[n]["resB"]
            e = tab.loc[k, "Error relativo"]
            v.append(0.0 if e is None or e != e else 100 * e)
        ax.bar(np.arange(len(ks)) + i * ancho - 0.4 + ancho / 2, v, width=ancho * 0.9, color=color_escenario(i), label=n)
    ax.axhline(0, c=TEXTO2, lw=1)
    ax.set_xticks(range(len(ks))); ax.set_xticklabels([e for _, e in ks], fontsize=8.5)
    ax.set_ylabel("error relativo (%)")
    ax.set_title("Validación: error relativo de la simulación frente a la teoría")
    ax.legend(fontsize=7, ncol=min(3, len(nombres)))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------- O2.2 · O3.1
def fig_sens_econ(df, elas, titulo=None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={"width_ratios": [2, 1]})
    for i, (par, d) in enumerate(df.groupby("Parámetro", sort=False)):
        base = d.loc[d["Variación"].abs().idxmin(), "Teórico · Var(Y)"]
        a1.plot(100 * d["Variación"], d["Teórico · Var(Y)"] / base, "-", color=SERIES[i], lw=2, label=f"{par} (teórico)")
        a1.plot(100 * d["Variación"], d["Simulado · Var(Y)"] / base, "o", color=SERIES[i], ms=5)
    a1.axhline(1, color=TEXTO2, lw=0.8)
    a1.set_xlabel("variación del parámetro (%)"); a1.set_ylabel("Var(Y) relativa al escenario")
    a1.set_title("Var(Y) al variar cada parámetro (puntos: simulado)", fontsize=9.5)
    a1.legend(fontsize=7.5)
    nombres = list(elas); v = [elas[k] for k in nombres]
    a2.barh(nombres, v, color=SERIES[:len(v)], height=0.55)
    for i, y in enumerate(v):
        a2.text(y, i, "  " + fx.num(y, 2), va="center", fontsize=8.5)
    a2.invert_yaxis(); a2.set_xlim(0, max(v) * 1.35 if max(v) > 0 else 1)
    a2.set_title("Elasticidad de Var(Y)", fontsize=9.5)
    if titulo:
        fig.suptitle(titulo, fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    return fig


def fig_optimizacion(opt, titulo=None):
    """Símplex de probabilidades (p_Urbana, p_Regional): E(Y) por niveles, región factible y solución."""
    mu = np.array(opt["mu_teo"])
    g = np.linspace(0, 1, 201)
    p1, p2 = np.meshgrid(g, g, indexing="ij")
    p3 = 1 - p1 - p2
    E = np.where(p3 >= 0, p1 * mu[0] + p2 * mu[1] + np.clip(p3, 0, 1) * mu[2], np.nan)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    cs = ax.contourf(p1, p2, E, levels=20, cmap="Blues_r", alpha=0.8)
    cb = fig.colorbar(cs, ax=ax); cb.set_label("E(Y) teórico (€)")
    if opt["teo"] is not None:
        ax.contour(p1, p2, E, levels=[opt["teo"]["objetivo"]], colors=[CRITICO], linewidths=1.8)
        ax.plot([], [], color=CRITICO, label=f"E(Y) = {fx.eur(opt['teo']['objetivo'], 2)} (−{fx.pct(opt['reduccion'], 0)})")
    ax.plot(opt["p0"][0], opt["p0"][1], "o", color=TEXTO, ms=8, label="actual p₀")
    if opt["teo"] is not None:
        ax.plot(opt["teo"]["p"][0], opt["teo"]["p"][1], "*", color=SERIES[3], ms=15, mec=TEXTO, label="óptimo (teórico)")
    if opt["sim"] is not None:
        ax.plot(opt["sim"]["p"][0], opt["sim"]["p"][1], "x", color=CRITICO, ms=10, mew=2, label="óptimo (simulado)")
    ax.set_xlabel("p Urbana"); ax.set_ylabel("p Regional  (p Premium = 1 − p Urbana − p Regional)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(titulo or "O3.1 · probabilidades de ruta más cercanas a las actuales con un 10 % menos de coste")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------- estimador
def fig_sens_convergencia(df, metrica, titulo=None):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(df))
    for i, (_, r) in enumerate(df.iterrows()):
        ax.plot([i] * len(r["Medias por réplica"]), r["Medias por réplica"], "o", color=SERIES[0], alpha=0.3, ms=5)
    m = df["Media de las réplicas"]
    ax.errorbar(x, m, yerr=[m - df["IC95 inf"], df["IC95 sup"] - m], fmt="o-", color=SERIES[0], ms=7, capsize=4,
                label="media de las réplicas ± IC95 %")
    ax.plot([], [], "o", color=SERIES[0], alpha=0.3, label="cada réplica")
    ax.axhline(df["Teórico"].iloc[0], color=TEXTO, ls="--", lw=1.4, label=f"teórico {fx.eur(df['Teórico'].iloc[0], 2)}")
    ax.set_xticks(x); ax.set_xticklabels([fx.num(v, 0) for v in df.index])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: fx.num(y, 1 if metrica == "entrega" else 0)))
    ax.set_xlabel("entregas simuladas" if metrica == "entrega" else "días simulados"); ax.set_ylabel("€")
    ax.set_title(titulo or ("Convergencia · E(Y) por entrega" if metrica == "entrega" else "Convergencia · coste diario"))
    ax.legend(loc="best", fontsize=8)
    ax.margins(y=0.15)
    fig.tight_layout()
    return fig


def fig_sens_variabilidad(df, metrica, titulo=None):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    x = np.arange(len(df))
    v = df["σ entre réplicas"]
    ax.bar(x, v, color=SERIES[0], width=0.6, label="σ entre réplicas (simulada)")
    ax.plot(x, df["σ teórica del estimador"], color=TEXTO, marker="o", lw=1.5, label="σ teórica del estimador")
    for i, y in enumerate(v):
        ax.text(i, y, fx.num(y, 2 if metrica == "entrega" else 0), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([fx.num(t, 0) for t in df.index])
    ax.set_xlabel("entregas simuladas" if metrica == "entrega" else "días simulados"); ax.set_ylabel("€")
    ax.set_title(titulo or "Variabilidad de la estimación entre réplicas")
    ax.legend(loc="upper right", fontsize=8)
    ax.margins(y=0.15)
    fig.tight_layout()
    return fig
