# -*- coding: utf-8 -*-
"""Gráficos (matplotlib) de la aplicación Eco-Flow. Cada función devuelve una Figure."""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import ecoflow as ef

# Paleta categórica validada (orden fijo; el color sigue a la entidad, nunca a su posición)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COLOR_TIPO = {1: SERIES[3], 2: SERIES[0], 3: SERIES[2], 4: "#8a8984"}
COLOR_TERM = [SERIES[0], SERIES[1], SERIES[6]]          # mercado, coste variable, descuento
COLOR_FUENTE = {"Solar": SERIES[3], "Eólica": SERIES[0], "Hidráulica": SERIES[2]}
CON, SIN = SERIES[6], SERIES[1]
BUENO, CRITICO = "#0b7a0b", "#b23232"
TEXTO, TEXTO2, REJILLA = "#0b0b0b", "#52514e", "#e1e0d9"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.edgecolor": "#b9b8b2", "axes.labelcolor": TEXTO2, "xtick.color": TEXTO2, "ytick.color": TEXTO2,
    "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
_K0 = mticker.FuncFormatter(lambda y, _: ef.num(y, 0))
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
                       fontsize=7.5)


def _densidad(rej, bins):
    """Densidad de una distribución en rejilla promediada en los intervalos `bins`."""
    F = np.concatenate([[0.0], np.cumsum(rej.p)]) / rej.masa()
    xe = np.concatenate([[rej.x[0] - rej.h / 2], rej.x + rej.h / 2])
    Fb = np.interp(bins, xe, F)
    return np.diff(Fb) / np.diff(bins)


# ---------------------------------------------------------------------------------------------- mecanismo
def fig_sens_param(df, elas, titulo=None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.7), gridspec_kw={"width_ratios": [1.6, 1]})
    for i, (par, d) in enumerate(df.groupby("Parámetro", sort=False)):
        a1.plot(100 * d["Variación"], d["Teórico · Var(Y)"], "o-", color=SERIES[i % 8], lw=1.8, ms=4, label=par)
    a1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: ef.num(y / 1e6, 1) + "M"))
    a1.set_xlabel("variación del parámetro (%)"); a1.set_ylabel("Var(Y) teórica (€²)")
    a1.set_title("Var(Y) al variar cada parámetro (resto fijo)", fontsize=9.5)
    a1.legend(fontsize=6.5, ncol=2)
    nombres = list(elas); v = [elas[k] for k in nombres]
    a2.barh(nombres, [abs(y) for y in v], color=[SERIES[i % 8] for i in range(len(v))], height=0.55)
    for i, y in enumerate(v):
        a2.text(abs(y), i, "  " + ef.num(y, 2), va="center", fontsize=8)
    a2.invert_yaxis(); a2.set_xlim(0, max(abs(y) for y in v) * 1.35)
    a2.tick_params(axis="y", labelsize=7.5)
    a2.set_title("|Elasticidad| de Var(Y) (signo en la etiqueta)", fontsize=9.5)
    if titulo:
        fig.suptitle(titulo, fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    return fig


def fig_sobol(df, titulo=None):
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["Índice de primer orden S₁"], width=0.4, color=SERIES[0], label="S₁ (primer orden)")
    ax.bar(x + 0.2, df["Índice total S_T"], width=0.4, color=SERIES[1], label="S_T (total, con interacciones)")
    if "Teórico · S₁" in df:
        t = df["Teórico · S₁"].to_numpy(float); ok = ~np.isnan(t)
        ax.plot(x[ok] - 0.2, t[ok], "D", color=TEXTO, ms=6, label="S₁ teórico del escenario climático")
    for i, (a, b) in enumerate(zip(df["Índice de primer orden S₁"], df["Índice total S_T"])):
        ax.text(i - 0.2, max(a, 0), ef.pct(a, 0), ha="center", va="bottom", fontsize=7.5)
        ax.text(i + 0.2, b, ef.pct(b, 0), ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(df.index)
    ax.yaxis.set_major_formatter(_PCT); ax.set_ylim(min(0, df["Índice de primer orden S₁"].min()) - 0.02, 1.15)
    ax.set_title(titulo or "O2.2 · índices de Sobol de las entradas aleatorias sobre Y")
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    return fig


def fig_sens_convergencia(df, metrica, titulo=None):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(df))
    for i, (_, r) in enumerate(df.iterrows()):
        ax.plot([i] * len(r["Medias por réplica"]), r["Medias por réplica"], "o", color=SERIES[0], alpha=0.3, ms=5)
    m = df["Media de las réplicas"]
    ax.errorbar(x, m, yerr=[m - df["IC95 inf"], df["IC95 sup"] - m], fmt="o-", color=SERIES[0], ms=7, capsize=4,
                label="media de las réplicas ± IC95 %")
    ax.plot([], [], "o", color=SERIES[0], alpha=0.3, label="cada réplica")
    ax.axhline(df["Teórico"].iloc[0], color=TEXTO, ls="--", lw=1.4, label=f"teórico {ef.eur(df['Teórico'].iloc[0])}")
    ax.set_xticks(x); ax.set_xticklabels([ef.num(v, 0) for v in df.index])
    ax.set_xlabel("días simulados" if metrica == "dia" else "años simulados"); ax.set_ylabel("€")
    ax.set_title(titulo or ("Convergencia · E(Y) diario" if metrica == "dia" else "Convergencia · ganancia diaria de la batería"))
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
        ax.text(i, y, ef.num(y, 2), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([ef.num(t, 0) for t in df.index])
    ax.set_xlabel("días simulados" if metrica == "dia" else "años simulados"); ax.set_ylabel("€")
    ax.set_title(titulo or "Variabilidad de la estimación entre réplicas")
    ax.legend(loc="upper right", fontsize=8)
    ax.margins(y=0.15)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------- mecanismo
def fig_mecanismo(p, uT, dia, titulo=None):
    """Izquierda: el escenario climático se elige con U_T. Centro: generación por fuente frente a la demanda.
    Derecha: beneficio del día por términos."""
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11.5, 3.6), gridspec_kw={"width_ratios": [1.4, 0.9, 1]})
    F = np.cumsum(ef.normalizar(p)); esc = np.concatenate([[0], F])
    for i, f in enumerate(ef.ESCENARIOS):
        if esc[i + 1] - esc[i] <= 0:
            continue
        a1.add_patch(plt.Rectangle((-0.6, esc[i]), 4.2, esc[i + 1] - esc[i], color=COLOR_TIPO[f], alpha=0.15, lw=0))
        a1.plot([i - 0.5, i + 0.5], [F[i], F[i]], color=COLOR_TIPO[f], lw=3)
        a1.text(3.65, (esc[i] + esc[i + 1]) / 2, f"{ef.NOMBRE[f]} ({ef.num(esc[i], 2)}, {ef.num(F[i], 2)}]",
                va="center", fontsize=7.5, color=TEXTO2)
    k = min(int(np.searchsorted(F, uT)), 3)
    a1.annotate("", xy=(k, uT), xytext=(-0.55, uT), arrowprops=dict(arrowstyle="->", color=TEXTO, lw=1.6))
    a1.annotate("", xy=(k, -0.02), xytext=(k, uT), arrowprops=dict(arrowstyle="->", color=TEXTO, lw=1.6))
    a1.text(-0.55, min(uT + 0.03, 1.0), f"U_T = {ef.num(uT, 2)}", fontsize=9)
    a1.set_xlim(-0.6, 6.2); a1.set_ylim(-0.03, 1.05)
    a1.set_xticks(range(4)); a1.set_xticklabels([ef.NOMBRE[f][:5] + "." for f in ef.ESCENARIOS], fontsize=8)
    a1.set_ylabel("F(T)"); a1.set_title("Paso 1 · escenario climático", fontsize=9.5)
    base = 0.0
    for nom, v in (("Solar", dia["Gs"]), ("Eólica", dia["Gw"]), ("Hidráulica", dia["Gh"])):
        a2.bar([0], [v], bottom=[base], color=COLOR_FUENTE[nom], width=0.5, label=nom); base += v
    a2.bar([1], [dia["D"]], color=TEXTO2, width=0.5, label="demanda")
    a2.text(0, base, ef.num(base, 1), ha="center", va="bottom", fontsize=8)
    a2.text(1, dia["D"], ef.num(dia["D"], 1), ha="center", va="bottom", fontsize=8)
    a2.set_xticks([0, 1]); a2.set_xticklabels(["G", "D"]); a2.set_ylabel("MWh")
    a2.set_title(f"Paso 2 · balance B = {ef.num(dia['Bp'], 1)} MWh", fontsize=9.5)
    a2.legend(fontsize=7, loc="upper right"); a2.margins(y=0.25)
    t = dia["terminos"]; noms = ["mercado", "coste var.", "descuento"]
    x = 0; acum = 0.0; usados = []
    for j in range(3):
        if j == 2 and abs(t[j]) < 1e-9:
            continue
        a3.bar([x], [t[j]], bottom=[acum], color=COLOR_TERM[j], width=0.6)
        a3.text(x, acum + t[j], ef.num(t[j], 0), ha="center", va="bottom" if t[j] >= 0 else "top", fontsize=7.5)
        acum += t[j]; x += 1; usados.append(noms[j])
    cf = dia["Y"] - acum
    a3.bar([x], [cf], bottom=[acum], color="#b9b8b2", width=0.6); usados.append("fijo"); acum += cf; x += 1
    a3.bar([x], [dia["Y"]], color=TEXTO, width=0.6); usados.append("Y")
    a3.text(x, dia["Y"], ef.num(dia["Y"], 0), ha="center", va="bottom" if dia["Y"] >= 0 else "top", fontsize=8)
    a3.axhline(0, color=TEXTO2, lw=0.8)
    a3.set_xticks(range(len(usados))); a3.set_xticklabels(usados, fontsize=7.5)
    a3.yaxis.set_major_formatter(_K0); a3.set_ylabel("€"); a3.margins(y=0.2)
    a3.set_title("Paso 3 · beneficio del día", fontsize=9.5)
    if titulo:
        fig.suptitle(titulo, fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------- bloque A
def fig_hist_Y(tA, Y, titulo=None):
    Y = np.asarray(Y)
    lo, hi = float(np.quantile(Y, 0.001)), float(np.quantile(Y, 0.999))
    bins = np.linspace(lo, hi, 90)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.hist(Y[(Y >= lo) & (Y <= hi)], bins=bins, density=True, color=SERIES[0], alpha=0.45, edgecolor="white",
            label="simulación")
    ax.step(bins[:-1], _densidad(tA["rejilla"], bins), where="post", color=TEXTO, lw=1.4, label="teórica")
    ax.axvline(tA["CVaR"], color=SERIES[6], ls=":", lw=1.6, label=f"CVaR 5 % = {ef.eur(tA['CVaR'])}")
    ax.axvline(tA["VaR"], color=SERIES[3], ls="-.", lw=1.4, label=f"VaR 95 % = {ef.eur(tA['VaR'])}")
    ax.axvline(tA["EY"], color=SERIES[0], ls="--", lw=1.4, label=f"E(Y) = {ef.eur(tA['EY'])}")
    ax.axvline(0, color=CRITICO, lw=1.2, label="beneficio 0")
    ax.xaxis.set_major_formatter(_K0)
    ax.set_xlabel("beneficio de un día Y (€) · del percentil 0,1 al 99,9"); ax.set_ylabel("densidad")
    ax.set_title(titulo or "Distribución del beneficio diario")
    ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    return fig


def fig_por_tipo(tA, resA, titulo=None):
    tipos = [f for f in ef.ESCENARIOS if tA["por_tipo"][f]["p"] > 0]
    noms = [ef.NOMBRE[f] for f in tipos]
    pf = tA["por_tipo"]
    g = lambda k: resA.loc[k, "Simulado · media"]
    datos = [("E(Y | escenario) (€)", [pf[f]["EY"] for f in tipos], [g(f"E(Y | {n}) (€)") for n in noms], 0),
             ("P(B < 0 | escenario)", [pf[f]["Pdef"] for f in tipos], [g(f"P(B < 0 | {n})") for n in noms], "%"),
             ("Coeficiente de variación", [pf[f]["CV"] for f in tipos], [g(f"CV(Y | {n})") for n in noms], 3)]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    x = np.arange(len(tipos))
    for ax, (tit, v, s, d) in zip(axs, datos):
        bars = ax.bar(x, v, color=[COLOR_TIPO[f] for f in tipos], width=0.6)
        ax.plot(x, s, "D", color=TEXTO, ms=6, label="simulado")
        for b_, y in zip(bars, v):
            ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height(), ef.pct(y, 1) if d == "%" else ef.num(y, d),
                    ha="center", va="bottom" if y >= 0 else "top", fontsize=7.5)
        ax.set_xticks(x); ax.set_xticklabels([n[:6] + "." if len(n) > 7 else n for n in noms], fontsize=8)
        ax.set_title(tit, fontsize=9.5)
        if d == "%":
            ax.yaxis.set_major_formatter(_PCT); ax.set_ylim(0, 1.15)
        elif d == 0:
            ax.yaxis.set_major_formatter(_K0); ax.margins(y=0.15)
        else:
            ax.set_ylim(0, max(max(v), np.nanmax(s)) * 1.25)
    axs[0].axhline(0, color=TEXTO2, lw=0.8)
    axs[0].legend(loc="lower left", fontsize=7.5)
    if titulo:
        fig.suptitle(titulo, fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    return fig


def fig_descomposicion(tA, resA, titulo=None):
    v = list(tA["contrib"])
    s = [resA.loc[f"Contribución a Var(Y) · {n}", "Simulado · media"] for n in ef.TERMINOS]
    keep = [i for i in range(3) if abs(v[i]) > 1e-9 or abs(s[i]) > 1e-9]
    fig, ax = plt.subplots(figsize=(9, 0.7 + 0.6 * len(keep)))
    x = np.arange(len(keep))
    ax.barh(x, [v[i] for i in keep], color=[COLOR_TERM[i] for i in keep], height=0.55)
    ax.plot([s[i] for i in keep], x, "D", color=TEXTO, ms=6, label="simulado")
    for j, i in enumerate(keep):
        ax.text(max(v[i], 0), j, "  " + ef.pct(v[i], 1), va="center", fontsize=8.5)
    ax.set_yticks(x); ax.set_yticklabels([ef.TERMINOS[i] for i in keep])
    ax.invert_yaxis(); ax.xaxis.set_major_formatter(_PCT)
    ax.set_xlim(min(0, min(v)) * 1.3 - 0.05, max(v) * 1.2)
    ax.axvline(0, color=TEXTO2, lw=0.8)
    ax.set_title(titulo or "Descomposición de Var(Y): Cov(término, Y) / Var(Y)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def fig_traza_A(traza, titulo=None):
    tr = traza; x = tr["Día"].to_numpy()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    base = np.zeros(len(x))
    for nom, col in (("Solar", "Solar (MWh)"), ("Eólica", "Eólica (MWh)"), ("Hidráulica", "Hidráulica (MWh)")):
        a1.bar(x, tr[col], bottom=base, color=COLOR_FUENTE[nom], width=0.8, label=nom); base += tr[col].to_numpy()
    a1.plot(x, tr["Demanda (MWh)"], "o-", color=TEXTO, ms=3, lw=1.2, label="demanda")
    a1.set_ylabel("MWh"); a1.legend(fontsize=7, ncol=4, loc="upper right")
    a1.set_title(titulo or f"Funcionamiento simulado · primeros {len(tr)} días")
    Y = tr["Y (€)"].to_numpy()
    tipos = [[k for k, v in ef.NOMBRE.items() if v == t][0] for t in tr["Escenario"]]
    a2.bar(x, Y, color=[COLOR_TIPO[t] for t in tipos], width=0.8)
    a2.axhline(0, color=TEXTO2, lw=0.8); a2.set_ylabel("Y (€)"); a2.yaxis.set_major_formatter(_K0)
    from matplotlib.patches import Patch
    a2.legend([Patch(color=COLOR_TIPO[f]) for f in ef.ESCENARIOS], [ef.NOMBRE[f] for f in ef.ESCENARIOS],
              fontsize=7, ncol=4, loc="lower right")
    a2.set_xlabel("día simulado")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------- bloque B
def fig_soc(tB, mcB, esc, titulo=None):
    t = np.arange(1, ef.N_DIAS + 1)
    E = mcB["bat"]["E_ini"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})
    a1.plot(t, E[0], color=SERIES[2], lw=0.7, alpha=0.6, label="un año simulado")
    a1.plot(t, E.mean(axis=0), color=SERIES[0], lw=1.5, label="media simulada")
    a1.plot(t, tB["E_soc"], "--", color=TEXTO, lw=1.5, label="teórica (cadena de Markov)")
    a1.set_xlim(1, ef.N_DIAS); a1.set_ylim(0, max(float(esc["cap"]), 1) * 1.05)
    a1.set_xlabel("día del año"); a1.set_ylabel("estado de carga (MWh)")
    a1.set_title(f"Estado de carga de la batería ({ef.num(float(esc['cap']), 0)} MWh)", fontsize=9.5)
    a1.legend(fontsize=7, ncol=2, loc="upper right")
    e = tB["estados"]; pb = tB["pibar"]
    if len(e) > 1:
        a2.bar(e, pb / (e[1] - e[0]), width=(e[1] - e[0]), color=SERIES[6], alpha=0.8)
        a2.set_yscale("log")
    a2.set_xlabel("estado de carga (MWh)"); a2.set_ylabel("densidad media del año (log)")
    a2.set_title(f"Distribución teórica · vacía el {ef.pct(pb[0], 0)} de los días", fontsize=9.5)
    if titulo:
        fig.suptitle(titulo, fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    return fig


def fig_hist_bat(tA, mcB, titulo=None):
    ys, yc = mcB["Ys"].ravel(), mcB["Yc"].ravel()
    lo, hi = float(np.quantile(ys, 0.001)), float(np.quantile(yc, 0.999))
    bins = np.linspace(lo, hi, 80)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.hist(ys, bins=bins, density=True, color=SIN, alpha=0.35, label="sin batería (simulado)")
    ax.hist(yc, bins=bins, density=True, color=CON, alpha=0.35, label="con batería (simulado)")
    ax.step(bins[:-1], _densidad(tA["rejilla"], bins), where="post", color=SIN, lw=1.4, label="teórica sin batería")
    cs, cc = ef._cvar_muestra(ys), ef._cvar_muestra(yc)
    ax.axvline(cs, color=SIN, ls=":", lw=1.6, label=f"CVaR 5 % sin = {ef.eur(cs)}")
    ax.axvline(cc, color=CON, ls=":", lw=1.6, label=f"CVaR 5 % con = {ef.eur(cc)}")
    ax.xaxis.set_major_formatter(_K0)
    ax.set_xlabel("beneficio diario (€)"); ax.set_ylabel("densidad")
    ax.set_title(titulo or "Beneficio diario con y sin batería (misma base climática)")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    return fig


def fig_ganancia_tipo(tB, mcB, titulo=None):
    tipos = [f for f in ef.ESCENARIOS if tB["por_tipo"][f]["p"] > 0]
    x = np.arange(len(tipos))
    v = [tB["por_tipo"][f]["E_gan"] for f in tipos]
    s = [mcB["replicas"][f"E(Δ | {ef.NOMBRE[f]})"].mean() for f in tipos]
    pdf = [tB["por_tipo"][f]["Pdef"] for f in tipos]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.3))
    a1.bar(x, v, color=[COLOR_TIPO[f] for f in tipos], width=0.6)
    a1.plot(x, s, "D", color=TEXTO, ms=6, label="simulado")
    for i, f in enumerate(tipos):
        a1.text(i, v[i], f"{ef.num(v[i], 1)} €\n({ef.pct(tB['por_tipo'][f]['cuota_gan'], 0)} del total)", ha="center",
                va="bottom", fontsize=7.5)
    a1.set_xticks(x); a1.set_xticklabels([ef.NOMBRE[f] for f in tipos], fontsize=8)
    a1.set_title("Ganancia de la batería por escenario (€/día) (O5.3)", fontsize=9.5)
    a1.legend(fontsize=7.5, loc="upper left"); a1.margins(y=0.3)
    w = 0.38
    a2.bar(x - w / 2, [tB["por_tipo"][f]["Edef"] for f in tipos], width=w, color=CRITICO, alpha=0.8, label="déficit medio (MWh)")
    a2.bar(x + w / 2, [tB["por_tipo"][f]["Esup"] for f in tipos], width=w, color=BUENO, alpha=0.8, label="superávit medio (MWh)")
    for i, p_ in enumerate(pdf):
        a2.text(i, max(tB["por_tipo"][tipos[i]]["Edef"], tB["por_tipo"][tipos[i]]["Esup"]), f"P(B<0) = {ef.pct(p_, 0)}",
                ha="center", va="bottom", fontsize=7)
    a2.set_xticks(x); a2.set_xticklabels([ef.NOMBRE[f] for f in tipos], fontsize=8)
    a2.set_title("Déficit y superávit medios por escenario", fontsize=9.5); a2.legend(fontsize=7.5); a2.margins(y=0.2)
    if titulo:
        fig.suptitle(titulo, fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------- comparativas
def fig_comparativa(res):
    nombres = list(res)
    cols = [color_escenario(i) for i in range(len(nombres))]
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.6))
    axs = axs.ravel()
    x = np.arange(len(nombres))
    A = [res[n]["teoria"]["A"] for n in nombres]; B = [res[n]["teoria"]["B"] for n in nombres]
    s_ = np.array([a["EY"] for a in A]); c_ = np.array([b["EY_con"] for b in B])
    axs[0].bar(x - 0.2, s_, width=0.4, color=SIN, label="sin batería")
    axs[0].bar(x + 0.2, c_, width=0.4, color=CON, label="con batería")
    axs[0].axhline(0, color=TEXTO2, lw=0.8); axs[0].yaxis.set_major_formatter(_K0)
    axs[0].set_title("E(Y) diario (€)", fontsize=9); axs[0].legend(fontsize=7, loc="lower right")
    s_ = [a["Pdef"] for a in A]; c_ = [b["Pspot_con"] for b in B]
    axs[1].bar(x - 0.2, s_, width=0.4, color=SIN, label="P(B < 0)")
    axs[1].bar(x + 0.2, c_, width=0.4, color=CON, label="P(compra) con batería")
    axs[1].yaxis.set_major_formatter(_PCT); axs[1].set_ylim(0, 1); axs[1].legend(fontsize=7, loc="lower right")
    axs[1].set_title("Días de déficit y de compra en el mercado", fontsize=9)
    vals = [a["CVaR"] for a in A]
    axs[2].bar(x, vals, color=cols, width=0.65)
    for i, v in enumerate(vals):
        axs[2].text(i, v, ef.num(v, 0), ha="center", va="top", fontsize=6.5)
    axs[2].yaxis.set_major_formatter(_K0); axs[2].margins(y=0.15)
    axs[2].set_title("CVaR 5 % diario sin batería (€)", fontsize=9)
    vals = [b["neto_anual"] for b in B]
    axs[3].bar(x, vals, color=[BUENO if v > 0 else CRITICO for v in vals], width=0.65)
    for i, v in enumerate(vals):
        axs[3].text(i, v, ef.num(v / 1000, 1) + "k", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    axs[3].axhline(0, color=TEXTO2, lw=0.8); axs[3].margins(y=0.2); axs[3].yaxis.set_major_formatter(_K0)
    axs[3].set_title("Ganancia neta anual de la batería (€)", fontsize=9)
    for ax in axs:
        _etiquetas(ax, nombres)
    fig.tight_layout()
    return fig


def fig_diferencias(dif, titulo=None):
    nombres = list(dif.index)
    x = np.arange(len(nombres))
    teo = dif["ΔE(Y) diario con batería · teórico (€)"].to_numpy(float)
    sim = dif["ΔE(Y) diario con batería · simulado (€)"].to_numpy(float)
    ic = np.array([list(v) for v in dif["IC95 ΔE(Y) con batería (€)"]])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(x, teo, width=0.55, color=[BUENO if v >= 0 else CRITICO for v in teo], alpha=0.8, label="teórico")
    ax.errorbar(x, sim, yerr=[sim - ic[:, 0], ic[:, 1] - sim], fmt="D", color=TEXTO, ms=5, capsize=3,
                label="simulado ± IC95 % (pareado)")
    for i, v in enumerate(teo):
        ax.text(i + 0.3, v, ef.num(v, 0), ha="left", va="center", fontsize=7.5)
    ax.axhline(0, color=TEXTO2, lw=1)
    ax.yaxis.set_major_formatter(_K0)
    _etiquetas(ax, nombres)
    ax.set_ylabel("€ por día")
    ax.set_title(titulo or "Diferencia de E(Y) diario con batería frente al escenario base")
    ax.legend(loc="best", fontsize=7.5); ax.margins(y=0.15)
    fig.tight_layout()
    return fig


def fig_validacion(res):
    nombres = list(res)
    kA = [("Beneficio diario esperado E(Y) (€)", "E(Y)"), ("Probabilidad de déficit P(B < 0)", "P(B<0)"),
          ("VaR 95 % · percentil 5 de Y (€)", "VaR"), ("CVaR 5 % · media del 5 % peor (€)", "CVaR")]
    kB = [("Ganancia media diaria pareada E(Δ) (€)", "E(Δ)"), ("P(comprar en el mercado) con batería", "P(compra) con"),
          ("Estado de carga medio (MWh)", "carga media")]
    fig, ax = plt.subplots(figsize=(9, 3.3))
    ks = kA + kB
    ancho = 0.8 / len(nombres)
    for i, n in enumerate(nombres):
        v = []
        for k, e_ in ks:
            tab = res[n]["resA"] if (k, e_) in kA else res[n]["resB"]
            lo, hi, m, teo = tab.loc[k, "IC95 inf"], tab.loc[k, "IC95 sup"], tab.loc[k, "Simulado · media"], tab.loc[k, "Teórico"]
            se = (hi - lo) / (2 * 1.96)
            v.append(0.0 if tab.loc[k, "Teórico en IC95"] is None or not se > 0 else (m - teo) / se)
        ax.bar(np.arange(len(ks)) + i * ancho - 0.4 + ancho / 2, v, width=ancho * 0.9, color=color_escenario(i), label=n)
    ax.axhline(0, c=TEXTO2, lw=1)
    for y in (-2, 2):
        ax.axhline(y, c=CRITICO, lw=1, ls="--")
    ax.set_xticks(range(len(ks))); ax.set_xticklabels([e for _, e in ks], fontsize=8.5)
    ax.set_ylabel("(simulado − teórico) / error estándar")
    ax.set_ylim(min(-3, ax.get_ylim()[0]), max(3, ax.get_ylim()[1]))
    ax.set_title("Validación: error estandarizado de la simulación (±2 ≈ IC95 %)")
    ax.legend(fontsize=6.5, ncol=min(3, len(nombres)))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------- batería
def fig_barrido(df, par, opt=None, titulo=None):
    x = df.index.to_numpy(float)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.4))
    m = df["Simulado · E(Δ) (€/día)"].to_numpy(float); ic = np.array([list(v) for v in df["IC95 E(Δ) (€/día)"]])
    a1.plot(x, df["Teórico · E(Δ) (€/día)"], "-", color=TEXTO, lw=2, label="teórico (cadena de Markov)")
    a1.errorbar(x, m, yerr=[m - ic[:, 0], ic[:, 1] - m], fmt="D", color=SERIES[0], ms=5, capsize=3, label="simulado ± IC95 %")
    if par == "cap" and len(x) > 1:
        a1.plot(x, x * (df["Teórico · E(Δ) (€/día)"].iloc[1] / x[1]) if x[1] > 0 else x * 0, ":", color=TEXTO2,
                label="proporcional a la capacidad")
    a1.set_xlabel(df.index.name); a1.set_ylabel("€/día"); a1.set_title("Ganancia diaria de la batería", fontsize=9.5)
    a1.legend(fontsize=7.5)
    if par == "cap":
        a2.plot(x, df["Teórico · ganancia anual (€)"], "o-", color=CON, lw=2, label="ganancia anual")
        a2.plot(x, df["Amortización anual (€)"], "s-", color=SIN, lw=2, label="amortización anual")
        a2.plot(x, df["Teórico · ganancia neta anual (€)"], "D-", color=SERIES[0], lw=2.4, label="ganancia neta")
        if opt is not None:
            a2.plot([opt], [df.loc[opt, "Teórico · ganancia neta anual (€)"]], "*", color=SERIES[3], ms=16, mec=TEXTO,
                    label=f"óptimo: {ef.num(opt, 0)} MWh")
        a2.axhline(0, color=TEXTO2, lw=0.8)
        a2.set_title("Ganancia, amortización y ganancia neta anual (O6.1)", fontsize=9.5)
        a2.yaxis.set_major_formatter(_K0)
    else:
        a2.plot(x, df["Teórico · LOLE con batería (días/año)"], "o-", color=SERIES[6], lw=2)
        a2.set_title("LOLE con batería (días/año)", fontsize=9.5)
    a2.set_xlabel(df.index.name); a2.legend(fontsize=7.5) if par == "cap" else None
    if titulo:
        fig.suptitle(titulo, fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    return fig


def fig_racha(df, inicio, dias, titulo=None):
    t = df.index.to_numpy()
    lo, hi = max(1, inicio - 15), min(ef.N_DIAS, inicio + dias + 40)
    mk = (t >= lo) & (t <= hi)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4))
    a1.axvspan(inicio, inicio + dias - 1, color="#8a8984", alpha=0.18, label="racha Anticiclónica")
    a1.plot(t[mk], df["Teórico · carga media sin racha (MWh)"][mk], "--", color=TEXTO2, lw=1.4, label="sin racha · teórico")
    a1.plot(t[mk], df["Teórico · carga media con racha (MWh)"][mk], "-", color=CON, lw=2, label="con racha · teórico")
    a1.plot(t[mk], df["Simulado · carga media con racha (MWh)"][mk], "o", color=CON, ms=3, alpha=0.6, label="con racha · simulado")
    a1.set_xlabel("día del año"); a1.set_ylabel("estado de carga medio (MWh)")
    a1.set_title("Vaciado y recuperación de la batería (O6.2)", fontsize=9.5); a1.legend(fontsize=7)
    a2.axvspan(inicio, inicio + dias - 1, color="#8a8984", alpha=0.18)
    a2.plot(t[mk], df["Simulado · Y sin batería, racha (€)"][mk], "-", color=SIN, lw=1.5, label="sin batería")
    a2.plot(t[mk], df["Simulado · Y con batería, racha (€)"][mk], "-", color=CON, lw=1.5, label="con batería")
    a2.yaxis.set_major_formatter(_K0)
    a2.set_xlabel("día del año"); a2.set_ylabel("€"); a2.set_title("Beneficio medio diario (simulado)", fontsize=9.5)
    a2.legend(fontsize=7.5)
    if titulo:
        fig.suptitle(titulo, fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    return fig
