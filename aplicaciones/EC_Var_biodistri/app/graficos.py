# -*- coding: utf-8 -*-
"""Gráficos (matplotlib) de la aplicación Bio-Distri. Cada función devuelve una Figure."""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import biodistri as bd

# Paleta categórica validada (orden fijo; el color sigue a la entidad, nunca a su posición)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COLOR_TIPO = {1: SERIES[0], 2: SERIES[6], 3: SERIES[3], 4: SERIES[2]}
COLOR_TERM = [SERIES[2], SERIES[0], SERIES[7], SERIES[1], SERIES[6]]      # V·R, S, −I, −Pen, garantía
CON, SIN = SERIES[0], SERIES[1]
BUENO, CRITICO = "#0b7a0b", "#b23232"
TEXTO, TEXTO2, REJILLA = "#0b0b0b", "#52514e", "#e1e0d9"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.edgecolor": "#b9b8b2", "axes.labelcolor": TEXTO2, "xtick.color": TEXTO2, "ytick.color": TEXTO2,
    "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
_K0 = mticker.FuncFormatter(lambda y, _: bd.num(y, 0))
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
def fig_mecanismo(p, uT, pr, titulo=None):
    """Izquierda: el tipo de proyecto se elige con U_T. Derecha: beneficio del proyecto por términos (cascada)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.2, 1]})
    F = np.cumsum(bd.normalizar(p)); esc = np.concatenate([[0], F])
    for i, f in enumerate(bd.PROYECTOS):
        if esc[i + 1] - esc[i] <= 0:
            continue
        a1.add_patch(plt.Rectangle((-0.6, esc[i]), 4.2, esc[i + 1] - esc[i], color=COLOR_TIPO[f], alpha=0.12, lw=0))
        a1.plot([i - 0.5, i + 0.5], [F[i], F[i]], color=COLOR_TIPO[f], lw=3)
        a1.text(3.65, (esc[i] + esc[i + 1]) / 2, f"{bd.NOMBRE[f]} ({bd.num(esc[i], 2)}, {bd.num(F[i], 2)}]",
                va="center", fontsize=7.5, color=COLOR_TIPO[f])
    k = min(int(np.searchsorted(F, uT)), 3)
    a1.annotate("", xy=(k, uT), xytext=(-0.55, uT), arrowprops=dict(arrowstyle="->", color=TEXTO, lw=1.6))
    a1.annotate("", xy=(k, -0.02), xytext=(k, uT), arrowprops=dict(arrowstyle="->", color=TEXTO, lw=1.6))
    a1.text(-0.55, min(uT + 0.03, 1.0), f"U_T = {bd.num(uT, 2)}", fontsize=9)
    a1.set_xlim(-0.6, 5.6); a1.set_ylim(-0.03, 1.05)
    a1.set_xticks(range(4)); a1.set_xticklabels([bd.NOMBRE[f] for f in bd.PROYECTOS], fontsize=8)
    a1.set_ylabel("F(T)"); a1.set_title("Paso 1 · tipo de proyecto por transformada inversa", fontsize=9.5)
    t = pr["terminos"]; noms = ["V·R", "S", "−I", "−Pen", "garantía"]
    acum = 0.0; x = 0
    for j in range(5):
        if abs(t[j]) < 1e-9 and j >= 3:
            continue
        a2.bar([x], [t[j]], bottom=[acum], color=COLOR_TERM[j], width=0.6)
        a2.text(x, acum + t[j] if t[j] >= 0 else acum + t[j], bd.num(t[j], 1), ha="center",
                va="bottom" if t[j] >= 0 else "top", fontsize=8)
        acum += t[j]; x += 1
    a2.bar([x], [pr["Y"]], color=TEXTO2, width=0.6)
    a2.text(x, max(pr["Y"], 0), f"Y = {bd.num(pr['Y'], 1)}", ha="center", va="bottom", fontsize=8.5)
    usados = [noms[j] for j in range(5) if not (abs(t[j]) < 1e-9 and j >= 3)]
    a2.set_xticks(range(x + 1)); a2.set_xticklabels(usados + ["Y"])
    a2.axhline(0, color=TEXTO2, lw=0.8)
    a2.set_ylabel("k€"); a2.set_title("Paso 2 · beneficio del proyecto por términos", fontsize=9.5)
    a2.margins(y=0.15)
    if titulo:
        fig.suptitle(titulo, fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    return fig


def fig_trayectoria(tr, esc, titulo=None):
    """Un centro con los mismos U todos los años: aprendizaje de R, ahorro S y beneficio anual por política."""
    H = len(tr["R"]); t = np.arange(1, H + 1)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11, 3.2))
    a1.plot(t, tr["R"], "o-", color=SERIES[2], lw=2)
    a1.axhline(float(esc["umbral_R"]), color=CRITICO, ls="--", lw=1.2, label="umbral de penalización")
    a1.set_ylim(0, 1.05); a1.yaxis.set_major_formatter(_PCT)
    a1.set_title("R_t · curva de aprendizaje", fontsize=9.5); a1.legend(fontsize=7.5, loc="lower right")
    a2.plot(t, tr["Sc"], "o-", color=CON, lw=2, label="con mantenimiento")
    a2.plot(t, tr["Ss"], "s--", color=SIN, lw=2, label="sin mantenimiento")
    a2.set_ylim(0, max(tr["Sc"].max(), 1e-9) * 1.25)
    a2.set_title("S_t · ahorro operativo (k€)", fontsize=9.5); a2.legend(fontsize=7.5)
    a3.bar(t - 0.2, tr["Yc"], width=0.4, color=CON, label=f"con (VAN {bd.num(tr['VANc'], 1)})")
    a3.bar(t + 0.2, tr["Ys"], width=0.4, color=SIN, label=f"sin (VAN {bd.num(tr['VANs'], 1)})")
    a3.set_title("Y_t · beneficio anual (k€)", fontsize=9.5); a3.legend(fontsize=7.5, loc="lower right")
    a3.yaxis.set_major_formatter(_K0)
    for ax in (a1, a2, a3):
        ax.set_xticks(t); ax.set_xlabel("año")
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
    ax.axvline(0, color=CRITICO, lw=1.4, label=f"pérdida (P(Y < 0) = {bd.pct(tA['Pperd'], 2)})")
    ax.axvline(tA["CVaR"], color=SERIES[6], ls=":", lw=1.6, label=f"CVaR 5 % = {bd.keur(tA['CVaR'])}")
    ax.axvline(tA["VaR"], color=SERIES[3], ls="-.", lw=1.4, label=f"VaR 95 % = {bd.keur(tA['VaR'])}")
    ax.axvline(tA["EY"], color=SERIES[0], ls="--", lw=1.4, label=f"E(Y) = {bd.keur(tA['EY'])}")
    ax.xaxis.set_major_formatter(_K0)
    ax.set_xlabel("beneficio neto de un proyecto Y (k€) · del percentil 0,1 al 99,9"); ax.set_ylabel("densidad")
    ax.set_title(titulo or "Distribución del beneficio de un proyecto")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    return fig


def fig_por_tipo(tA, resA, titulo=None):
    tipos = [f for f in bd.PROYECTOS if tA["por_tipo"][f]["p"] > 0]
    noms = [bd.NOMBRE[f] for f in tipos]
    pf = tA["por_tipo"]
    g = lambda k: resA.loc[k, "Simulado · media"]
    datos = [("E(Y | tipo) (k€)", [pf[f]["EY"] for f in tipos], [g(f"E(Y | {n}) (k€)") for n in noms], 1),
             ("P(Y < 0 | tipo)", [pf[f]["Pperd"] for f in tipos], [g(f"P(Y < 0 | {n})") for n in noms], "%"),
             ("Coeficiente de variación", [pf[f]["CV"] for f in tipos], [g(f"CV(Y | {n})") for n in noms], 3)]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    x = np.arange(len(tipos))
    for ax, (tit, v, s, d) in zip(axs, datos):
        bars = ax.bar(x, v, color=[COLOR_TIPO[f] for f in tipos], width=0.6)
        ax.plot(x, s, "D", color=TEXTO, ms=6, label="simulado")
        for b_, y in zip(bars, v):
            ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height(), bd.pct(y, 1) if d == "%" else bd.num(y, d),
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(noms)
        ax.set_title(tit, fontsize=9.5)
        ax.set_ylim(0, max(max(v), np.nanmax(s), 1e-9) * 1.25)
        if d == "%":
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=1))
    axs[0].legend(loc="upper right", fontsize=8)
    if titulo:
        fig.suptitle(titulo, fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    return fig


def fig_descomposicion(tA, resA, titulo=None):
    v = list(tA["contrib"])
    s = [resA.loc[f"Contribución a Var(Y) · {n}", "Simulado · media"] for n in bd.TERMINOS]
    keep = [i for i in range(5) if abs(v[i]) > 1e-9 or abs(s[i]) > 1e-9]
    fig, ax = plt.subplots(figsize=(9, 0.6 + 0.55 * len(keep)))
    x = np.arange(len(keep))
    ax.barh(x, [v[i] for i in keep], color=[COLOR_TERM[i] for i in keep], height=0.55)
    ax.plot([s[i] for i in keep], x, "D", color=TEXTO, ms=6, label="simulado")
    for j, i in enumerate(keep):
        ax.text(max(v[i], 0), j, "  " + bd.pct(v[i], 1), va="center", fontsize=8.5)
    ax.set_yticks(x); ax.set_yticklabels([bd.TERMINOS[i] for i in keep])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(_PCT)
    ax.set_xlim(min(0, min(v)) * 1.3 - 0.02, max(v) * 1.2)
    ax.axvline(0, color=TEXTO2, lw=0.8)
    ax.set_title(titulo or "Descomposición de Var(Y): Cov(término, Y) / Var(Y)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def fig_traza_A(traza, titulo=None):
    tr = traza
    x = tr["Proyecto"].to_numpy()
    Y = tr["Y (k€)"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(x, Y, color=[COLOR_TIPO[[k for k, v in bd.NOMBRE.items() if v == t][0]] for t in tr["Tipo"]], width=0.8)
    pen = tr["Penalización (k€)"].to_numpy() > 0
    ax.plot(x[pen], Y[pen], "v", color=CRITICO, ms=7, label="penalización (R < umbral)")
    for f in bd.PROYECTOS:
        ax.bar([], [], color=COLOR_TIPO[f], label=bd.NOMBRE[f])
    ax.axhline(0, color=TEXTO2, lw=0.8)
    ax.set_xlabel("proyecto simulado"); ax.set_ylabel("Y (k€)")
    ax.set_title(titulo or f"Funcionamiento simulado · primeros {len(tr)} proyectos")
    from matplotlib.patches import Patch
    h = [Patch(color=COLOR_TIPO[f]) for f in bd.PROYECTOS]; l = [bd.NOMBRE[f] for f in bd.PROYECTOS]
    hh, ll = ax.get_legend_handles_labels()
    ax.legend(h + hh[:1], l + ll[:1], loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=5, fontsize=7.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------- bloque B
def fig_hist_van(tB, mcB, titulo=None):
    vc, vs = mcB["VAN_con"], mcB["VAN_sin"]
    lo = float(min(np.quantile(vc, 0.001), np.quantile(vs, 0.001))); hi = float(max(np.quantile(vc, 0.999), np.quantile(vs, 0.999)))
    bins = np.linspace(lo, hi, 70)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.hist(vc, bins=bins, density=True, color=CON, alpha=0.35, edgecolor="white", label="con mantenimiento (simulado)")
    ax.hist(vs, bins=bins, density=True, color=SIN, alpha=0.35, edgecolor="white", label="sin mantenimiento (simulado)")
    ax.step(bins[:-1], _densidad(tB["rejilla_con"], bins), where="post", color=CON, lw=1.5, label="teórica · con")
    ax.step(bins[:-1], _densidad(tB["rejilla_sin"], bins), where="post", color=SIN, lw=1.5, ls="--", label="teórica · sin")
    ax.axvline(tB["CVaR_con"], color=CON, ls=":", lw=1.4, label=f"CVaR 5 % con = {bd.keur(tB['CVaR_con'], 0)}")
    ax.axvline(tB["CVaR_sin"], color=SIN, ls=":", lw=1.4, label=f"CVaR 5 % sin = {bd.keur(tB['CVaR_sin'], 0)}")
    ax.xaxis.set_major_formatter(_K0)
    ax.set_xlabel("VAN del programa de un centro (k€)"); ax.set_ylabel("densidad")
    ax.set_title(titulo or "Distribución del VAN por centro")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    return fig


def fig_ganancia(tB, mcB, titulo=None):
    g = mcB["VAN_con"] - mcB["VAN_sin"]; T = mcB["T"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.4), gridspec_kw={"width_ratios": [1.3, 1]})
    for f in bd.PROYECTOS:
        mk = T == f
        if mk.sum():
            a1.hist(g[mk], bins=40, color=COLOR_TIPO[f], alpha=0.55, label=bd.NOMBRE[f], edgecolor="white")
    a1.axvline(0, color=CRITICO, lw=1.4)
    a1.axvline(tB["E_gan"], color=TEXTO, ls="--", lw=1.4, label=f"E(Δ) = {bd.keur(tB['E_gan'], 2)}")
    a1.set_xlabel("Δ = VAN_con − VAN_sin por centro (k€)"); a1.set_ylabel("centros simulados")
    a1.set_title(f"Ganancia del mantenimiento · P(Δ > 0) = {bd.pct(tB['P_gan'], 1)}", fontsize=9.5)
    a1.legend(fontsize=7.5)
    tipos = [f for f in bd.PROYECTOS if tB["por_tipo"][f]["p"] > 0]
    x = np.arange(len(tipos))
    v = [tB["por_tipo"][f]["E_gan"] for f in tipos]
    s = [mcB["replicas"][f"E(Δ | {bd.NOMBRE[f]})"].mean() for f in tipos]
    a2.bar(x, v, color=[BUENO if y > 0 else CRITICO for y in v], width=0.6, alpha=0.85)
    a2.plot(x, s, "D", color=TEXTO, ms=6, label="simulado")
    for i, f in enumerate(tipos):
        a2.text(i, v[i], f"{bd.num(v[i], 2)}\nS̄ = {bd.num(tB['por_tipo'][f]['ES'], 1)}", ha="center",
                va="bottom" if v[i] >= 0 else "top", fontsize=7.5)
    a2.axhline(0, color=TEXTO2, lw=0.8)
    a2.set_xticks(x); a2.set_xticklabels([bd.NOMBRE[f] for f in tipos])
    a2.set_title("E(Δ | tipo) (k€) y ahorro medio S̄ (O5.3)", fontsize=9.5)
    a2.legend(fontsize=7.5, loc="lower left")
    a2.margins(y=0.3)
    if titulo:
        fig.suptitle(titulo, fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    return fig


def fig_ejemplo_B(ej, esc, titulo=None):
    """Primeros centros simulados: ahorro S_t con y sin mantenimiento y beneficio medio por año."""
    H = ej["Sc"].shape[1]; t = np.arange(1, H + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.3))
    for i in range(min(12, len(ej["T"]))):
        c = COLOR_TIPO[int(ej["T"][i])]
        a1.plot(t, ej["Sc"][i], "-", color=c, lw=1, alpha=0.7)
        a1.plot(t, ej["Ss"][i], ":", color=c, lw=1.3, alpha=0.9)
    a1.plot([], [], "-", color=TEXTO2, label="con mantenimiento"); a1.plot([], [], ":", color=TEXTO2, label="sin mantenimiento")
    a1.set_xticks(t); a1.set_xlabel("año"); a1.set_ylabel("S_t (k€)")
    a1.set_title("Ahorro de 12 centros simulados (color = tipo)", fontsize=9.5); a1.legend(fontsize=7.5)
    a2.plot(t, ej["Yc"].mean(axis=0), "o-", color=CON, lw=2, label="con mantenimiento")
    a2.plot(t, ej["Ys"].mean(axis=0), "s--", color=SIN, lw=2, label="sin mantenimiento")
    a2.set_xticks(t); a2.set_xlabel("año"); a2.set_ylabel("k€")
    a2.set_title(f"Beneficio medio por año ({len(ej['T'])} centros)", fontsize=9.5); a2.legend(fontsize=7.5)
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
    vals = [a["EY"] for a in A]
    axs[0].bar(x, vals, color=cols, width=0.65)
    for i, v in enumerate(vals):
        axs[0].text(i, v, bd.num(v, 1), ha="center", va="bottom", fontsize=7)
    lo, hi = min(vals), max(vals); m = (hi - lo) or 1
    axs[0].set_ylim(max(0, lo - 1.2 * m), hi + 0.4 * m)
    axs[0].set_title("E(Y) por proyecto (k€)", fontsize=9)
    vals = [a["Pperd"] for a in A]
    axs[1].bar(x, vals, color=cols, width=0.65)
    for i, v in enumerate(vals):
        axs[1].text(i, v, bd.num(100 * v, 2), ha="center", va="bottom", fontsize=7)
    axs[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=1)); axs[1].set_ylim(0, max(vals) * 1.25)
    axs[1].set_title("P(Y < 0)", fontsize=9)
    c_ = np.array([b["E_con"] for b in B]); s_ = np.array([b["E_sin"] for b in B])
    axs[2].bar(x - 0.2, c_, width=0.4, color=CON, label="con mantenimiento")
    axs[2].bar(x + 0.2, s_, width=0.4, color=SIN, label="sin mantenimiento")
    lo, hi = min(c_.min(), s_.min()), max(c_.max(), s_.max()); m = (hi - lo) or 1
    axs[2].set_ylim(max(0, lo - 0.3 * m), hi + 0.25 * m); axs[2].yaxis.set_major_formatter(_K0)
    axs[2].set_title("E(VAN) por centro (k€)", fontsize=9); axs[2].legend(fontsize=7, loc="upper left")
    vals = [b["E_gan"] for b in B]
    axs[3].bar(x, vals, color=[BUENO if v > 0 else CRITICO for v in vals], width=0.65)
    for i, v in enumerate(vals):
        axs[3].text(i, v, bd.num(v, 2), ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    axs[3].axhline(0, color=TEXTO2, lw=0.8); axs[3].margins(y=0.2)
    axs[3].set_title("Ganancia del mantenimiento E(Δ) (k€)", fontsize=9)
    for ax in axs:
        _etiquetas(ax, nombres)
    fig.text(0.5, 0.005, "Los ejes de E(Y) y E(VAN) no empiezan en 0 para que se aprecien las diferencias.",
             ha="center", fontsize=7.5, color=TEXTO2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def fig_diferencias(dif, titulo=None):
    nombres = list(dif.index)
    x = np.arange(len(nombres))
    teo = dif["ΔE(VAN) política · teórico (k€)"].to_numpy(float)
    sim = dif["ΔE(VAN) política · simulado (k€)"].to_numpy(float)
    ic = np.array([list(v) for v in dif["IC95 ΔE(VAN) (k€)"]])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(x, teo, width=0.55, color=[BUENO if v >= 0 else CRITICO for v in teo], alpha=0.8, label="teórico")
    ax.errorbar(x, sim, yerr=[sim - ic[:, 0], ic[:, 1] - sim], fmt="D", color=TEXTO, ms=5, capsize=3,
                label="simulado ± IC95 % (pareado)")
    for i, v in enumerate(teo):
        ax.text(i + 0.3, v, bd.num(v, 1), ha="left", va="center", fontsize=7.5)
    ax.axhline(0, color=TEXTO2, lw=1)
    ax.yaxis.set_major_formatter(_K0)
    _etiquetas(ax, nombres)
    ax.set_ylabel("k€ por centro")
    ax.set_title(titulo or "Diferencia de E(VAN) (política elegida) frente al escenario base")
    ax.legend(loc="best", fontsize=7.5)
    ax.margins(y=0.15)
    fig.tight_layout()
    return fig


def fig_validacion(res):
    nombres = list(res)
    kA = [("Beneficio esperado E(Y) (k€)", "E(Y)"), ("Probabilidad de pérdida P(Y < 0)", "P(Y < 0)"),
          ("VaR 95 % · percentil 5 de Y (k€)", "VaR Y")]
    kB = [("VAN esperado con mantenimiento (k€)", "E(VAN) con"), ("Ganancia media pareada E(Δ) (k€)", "E(Δ)"),
          ("P(el mantenimiento compensa) P(Δ > 0)", "P(Δ > 0)"), ("CVaR 5 % del VAN con mantenimiento (k€)", "CVaR VAN")]
    fig, ax = plt.subplots(figsize=(9, 3.3))
    ks = kA + kB
    ancho = 0.8 / len(nombres)
    for i, n in enumerate(nombres):
        v = []
        for k, e_ in ks:
            tab = res[n]["resA"] if (k, e_) in kA else res[n]["resB"]
            lo, hi, m, teo = (tab.loc[k, "IC95 inf"], tab.loc[k, "IC95 sup"], tab.loc[k, "Simulado · media"],
                              tab.loc[k, "Teórico"])
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


# ---------------------------------------------------------------------------------------------- mantenimiento
def fig_barrido(df, titulo=None, marca=None):
    par = df.index.name
    x = df.index.to_numpy(float)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.4))
    a1.plot(x, df["Teórico · E(Δ) (k€)"], "-", color=TEXTO, lw=2, label="teórico (red)")
    m = df["Simulado · E(Δ) (k€)"].to_numpy(float); ic = np.array([list(v) for v in df["IC95 E(Δ) (k€)"]])
    a1.errorbar(x, m, yerr=[m - ic[:, 0], ic[:, 1] - m], fmt="D", color=SERIES[0], ms=5, capsize=3, label="simulado ± IC95 %")
    for f in bd.PROYECTOS:
        a1.plot(x, df[f"Teórico · E(Δ | {bd.NOMBRE[f]}) (k€)"], "--", color=COLOR_TIPO[f], lw=1.2, label=bd.NOMBRE[f])
    a1.axhline(0, color=CRITICO, lw=1)
    if marca is not None:
        a1.axvline(marca, color=TEXTO2, ls=":", lw=1)
    a1.set_xlabel(par); a1.set_ylabel("E(Δ) (k€ por centro)")
    a1.set_title("Ganancia del mantenimiento", fontsize=9.5); a1.legend(fontsize=7, ncol=2)
    a2.plot(x, df["Teórico · P(Δ > 0)"], "o-", color=SERIES[6], lw=2)
    a2.yaxis.set_major_formatter(_PCT); a2.set_ylim(0, 1.05)
    a2.set_xlabel(par); a2.set_title("P(Δ > 0): centros en los que compensa", fontsize=9.5)
    if marca is not None:
        a2.axvline(marca, color=TEXTO2, ls=":", lw=1)
    if titulo:
        fig.suptitle(titulo, fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    return fig


def fig_racha(df, titulo=None):
    t = df.index.to_numpy()
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11.5, 3.3))
    for nom, ls in (("Sin racha", "-"), ("Con racha", "--")):
        a1.plot(t, df[f"{nom} · R medio"], ls, marker="o", color=SERIES[2] if nom == "Sin racha" else CRITICO, label=nom)
        a2.plot(t, df[f"{nom} · S con mant. (k€)"], ls, marker="o", color=CON, label=f"{nom} · con")
        a2.plot(t, df[f"{nom} · S sin mant. (k€)"], ls, marker="s", color=SIN, label=f"{nom} · sin")
        a3.plot(t, df[f"{nom} · VAN acum. con (k€)"], ls, marker="o", color=CON, label=f"{nom} · con")
        a3.plot(t, df[f"{nom} · VAN acum. sin (k€)"], ls, marker="s", color=SIN, label=f"{nom} · sin")
    a1.yaxis.set_major_formatter(_PCT); a1.set_ylim(0, 1.05)
    a1.set_title("R medio", fontsize=9.5); a1.legend(fontsize=7)
    a2.set_title("Ahorro S medio (k€)", fontsize=9.5); a2.legend(fontsize=6.5)
    a3.set_title("VAN acumulado medio (k€)", fontsize=9.5); a3.legend(fontsize=6.5)
    a3.yaxis.set_major_formatter(_K0)
    for ax in (a1, a2, a3):
        ax.set_xticks(t); ax.set_xlabel("año")
    if titulo:
        fig.suptitle(titulo, fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------- sensibilidad
def fig_sens_param(df, elas, titulo=None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.7), gridspec_kw={"width_ratios": [1.6, 1]})
    for i, (par, d) in enumerate(df.groupby("Parámetro", sort=False)):
        a1.plot(100 * d["Variación"], d["Teórico · Var(Y)"], "o-", color=SERIES[i % 8], lw=1.8, ms=4, label=par)
    a1.yaxis.set_major_formatter(_K0)
    a1.set_xlabel("variación del parámetro (%)"); a1.set_ylabel("Var(Y) teórica (k€²)")
    a1.set_title("Var(Y) al variar cada parámetro (resto fijo)", fontsize=9.5)
    a1.legend(fontsize=6.5, ncol=2)
    nombres = list(elas); v = [elas[k] for k in nombres]
    a2.barh(nombres, [abs(y) for y in v], color=[SERIES[i % 8] for i in range(len(v))], height=0.55)
    for i, y in enumerate(v):
        a2.text(abs(y), i, "  " + bd.num(y, 2), va="center", fontsize=8)
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
        ax.plot(x[ok] - 0.2, t[ok], "D", color=TEXTO, ms=6, label="S₁ teórico del tipo de proyecto")
    for i, (a, b) in enumerate(zip(df["Índice de primer orden S₁"], df["Índice total S_T"])):
        ax.text(i - 0.2, max(a, 0), bd.pct(a, 0), ha="center", va="bottom", fontsize=7.5)
        ax.text(i + 0.2, b, bd.pct(b, 0), ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(df.index)
    ax.yaxis.set_major_formatter(_PCT); ax.set_ylim(min(0, df["Índice de primer orden S₁"].min()) - 0.02, 1.15)
    ax.set_title(titulo or "O2.2 · índices de Sobol de las entradas aleatorias sobre Y")
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    return fig


def fig_vequilibrio(df, Vact, titulo=None):
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    y = np.arange(len(df))
    v = df["V_stock necesario (k€)"].to_numpy(float)
    ax.barh(y, v, color=SERIES[3], height=0.55)
    ax.axvline(Vact, color=TEXTO, ls="--", lw=1.4, label=f"V_stock actual = {bd.keur(Vact, 0)}")
    for i, x_ in enumerate(v):
        ax.text(x_, i, "  " + bd.keur(x_, 1), va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(df.index, fontsize=8); ax.invert_yaxis()
    ax.xaxis.set_major_formatter(_K0); ax.set_xlim(0, max(v.max(), Vact) * 1.25)
    ax.set_title(titulo or "O3.3 · V_stock que necesita el Empaque para cada meta")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    return fig


def fig_adaptativa(df, Ec, Es, titulo=None):
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    x = df.index.to_numpy(float)
    m = df["Simulado · E(VAN) (k€)"].to_numpy(float); ic = np.array([list(v) for v in df["IC95 (k€)"]])
    ax.errorbar(x, m, yerr=[m - ic[:, 0], ic[:, 1] - m], fmt="D-", color=SERIES[6], ms=5, capsize=3,
                label="política adaptativa (simulada)")
    ax.axhline(Ec, color=CON, ls="--", lw=1.4, label=f"siempre mantener · {bd.keur(Ec, 1)}")
    ax.axhline(Es, color=SIN, ls=":", lw=1.6, label=f"nunca mantener · {bd.keur(Es, 1)}")
    ax.set_xlabel("umbral de ahorro observado para mantener (k€)"); ax.set_ylabel("E(VAN) (k€)")
    ax.set_title(titulo or "O8.3 · política adaptativa de mantenimiento")
    ax.legend(fontsize=7.5); ax.margins(y=0.25)
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
    ax.axhline(df["Teórico"].iloc[0], color=TEXTO, ls="--", lw=1.4, label=f"teórico {bd.keur(df['Teórico'].iloc[0])}")
    ax.set_xticks(x); ax.set_xticklabels([bd.num(v, 0) for v in df.index])
    ax.set_xlabel("proyectos simulados" if metrica == "proyecto" else "centros simulados"); ax.set_ylabel("k€")
    ax.set_title(titulo or ("Convergencia · E(Y) por proyecto" if metrica == "proyecto" else "Convergencia · E(VAN) con mantenimiento"))
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
        ax.text(i, y, bd.num(y, 2), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([bd.num(t, 0) for t in df.index])
    ax.set_xlabel("proyectos simulados" if metrica == "proyecto" else "centros simulados"); ax.set_ylabel("k€")
    ax.set_title(titulo or "Variabilidad de la estimación entre réplicas")
    ax.legend(loc="upper right", fontsize=8)
    ax.margins(y=0.15)
    fig.tight_layout()
    return fig
