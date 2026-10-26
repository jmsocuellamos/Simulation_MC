# -*- coding: utf-8 -*-
"""Gráficos (matplotlib) de la aplicación de riesgo y retorno. Cada función devuelve una Figure."""
import io
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import inversion as iv

# Paleta categórica validada (orden fijo; el color sigue a la entidad, nunca a su posición)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GANANCIA, PERDIDA = "#1baf7a", "#e34948"            # tramos con R ≥ I₀ y con R < I₀
BUENO, CRITICO = "#0b7a0b", "#b23232"
TEXTO, TEXTO2, REJILLA = "#0b0b0b", "#52514e", "#e1e0d9"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.edgecolor": "#b9b8b2", "axes.labelcolor": TEXTO2, "xtick.color": TEXTO2, "ytick.color": TEXTO2,
    "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def _eur(x, d=0):
    return iv.eur(x, d)


def _fmt_eur(ax, eje="y"):
    f = mticker.FuncFormatter(lambda v, _: iv.num(v, 0))
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(f)


def color_escenario(i):
    return SERIES[i % len(SERIES)]


def _color_tramo(v, I0):
    return PERDIDA if v < I0 else GANANCIA


def fig_a_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _vacia(texto, figsize=(9, 3.2)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.text(0.5, 0.5, texto, ha="center", va="center", color=TEXTO2, fontsize=11, transform=ax.transAxes)
    return fig


def fig_escalera(sis, u=None, titulo=None):
    """Transformada inversa: probabilidades acumuladas F(i) de los tramos y, si se indica, la altura U y el tramo
    donde cae."""
    F = np.array(sis["F"]); v = sis["v"]; n = len(F)
    esc = np.concatenate([[0.0], F])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i in range(n):
        c = _color_tramo(v[i], sis["I0"])
        ax.add_patch(plt.Rectangle((-0.6, esc[i]), n + 0.2, esc[i + 1] - esc[i], color=c, alpha=0.08 + 0.06 * (i % 2), lw=0))
        ax.plot([i - 0.5, i + 0.5], [F[i], F[i]], color=c, lw=3, solid_capstyle="butt")
        ax.hlines(F[i], i - 0.5, n - 0.4, color=c, lw=0.8, ls=":")
        ax.text(n - 0.35, (esc[i] + esc[i + 1]) / 2, f"tramo {i + 1}: {_eur(v[i])}\n({iv.num(esc[i], 2)}, {iv.num(F[i], 2)}]",
                va="center", fontsize=8, color=c)
    if u is not None:
        k = int(iv.transformada_inversa(np.array([u]), F)[0])
        ax.annotate("", xy=(k, u), xytext=(-0.55, u), arrowprops=dict(arrowstyle="->", color=TEXTO, lw=1.6))
        ax.annotate("", xy=(k, -0.02), xytext=(k, u), arrowprops=dict(arrowstyle="->", color=TEXTO, lw=1.6))
        ax.plot([k], [u], "o", color=TEXTO, ms=6)
        ax.text(-0.55, min(u + 0.025, 1.0), f"U = {iv.num(u, 2)}", fontsize=9, color=TEXTO)
    ax.set_xlim(-0.6, n + 0.75); ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(range(n)); ax.set_xticklabels([f"tramo {i + 1}\n{_eur(v[i])}" for i in range(n)])
    ax.set_ylabel("F(i) = p₁ + … + pᵢ")
    ax.set_title(titulo or "Transformada inversa: tramo = menor i tal que U ≤ F(i)")
    fig.tight_layout()
    return fig


def fig_pmf(sis, mc=None, titulo=None):
    """Distribución del retorno: probabilidad teórica de cada valor (barras; rojo si hay pérdida, R < I₀) y
    frecuencia simulada (rombos), con la inversión inicial I₀."""
    v, p, I0 = np.array(sis["v"]), np.array(sis["p"]), sis["I0"]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    x = np.arange(len(v))
    ax.bar(x, p, width=0.6, color=[_color_tramo(a, I0) for a in v])
    for i in x:
        ax.text(i, p[i], iv.pct(p[i], 1), ha="center", va="bottom", fontsize=8.5, color=TEXTO)
    if mc is not None:
        fr = np.bincount(mc["idx"], minlength=len(v)) / len(mc["idx"])
        ax.plot(x, fr, "D", color=TEXTO, ms=6, label="simulada")
    ax.set_xticks(x)
    hueco = max(0, 3 - len(v)) * 0.5               # con 1 o 2 tramos, barras de anchura normal
    ax.set_xlim(-0.6 - hueco, len(v) - 0.4 + hueco)
    ax.set_xticklabels([f"{_eur(a)}\n(neto {_eur(a - I0)})" for a in v])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 1.15)
    from matplotlib.patches import Patch
    h = [Patch(color=GANANCIA, label="teórica · R ≥ I₀"), Patch(color=PERDIDA, label="teórica · R < I₀ (pérdida)")]
    if mc is not None:
        from matplotlib.lines import Line2D
        h.append(Line2D([], [], ls="none", marker="D", color=TEXTO, label="simulada"))
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=8)
    ax.set_title(titulo or f"Distribución del retorno R · I₀ = {_eur(I0)}")
    fig.tight_layout()
    return fig


def fig_cdf(sis, R=None, titulo=None):
    """Función de distribución del retorno: teórica (escalones) y empírica de la simulación, con I₀."""
    v, p, I0 = np.array(sis["v"]), np.array(sis["p"]), sis["I0"]
    o = np.argsort(v); vs, F = v[o], np.cumsum(p[o])
    lo = min(vs.min(), I0); hi = max(vs.max(), I0); m = (hi - lo) * 0.08 or max(1.0, abs(hi) * 0.02)
    xs = np.concatenate([[lo - m], vs, [hi + m]]); ys = np.concatenate([[0], F, [1]])
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.step(xs, ys, where="post", color=TEXTO, lw=2, label="teórica")
    if R is not None:
        rs = np.sort(np.asarray(R)); ye = np.arange(1, len(rs) + 1) / len(rs)
        ax.step(np.concatenate([[lo - m], rs, [hi + m]]), np.concatenate([[0], ye, [1]]), where="post",
                color=SERIES[0], lw=1.2, ls="--", label="empírica (simulación)")
    ax.axvline(I0, color=CRITICO, ls=":", lw=1.5, label=f"I₀ = {_eur(I0)}")
    _fmt_eur(ax, "x")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("retorno R (€)"); ax.set_ylabel("P(R ≤ x)")
    ax.set_title(titulo or "Función de distribución del retorno")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def fig_traza(traza, sis, titulo=None):
    """Diagrama de funcionamiento: U generado en cada una de las primeras simulaciones y tramo obtenido."""
    if traza is None or len(traza) == 0:
        return _vacia("Sin simulaciones que mostrar")
    F = np.array(sis["F"]); esc = np.concatenate([[0.0], F]); v = sis["v"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i in range(len(F)):
        c = _color_tramo(v[i], sis["I0"])
        ax.axhspan(esc[i], esc[i + 1], color=c, alpha=0.08 + 0.06 * (i % 2), lw=0)
        ax.axhline(F[i], color=c, lw=0.8, ls=":")
        d = traza[traza["Tramo"] == i + 1]
        if len(d):
            ax.vlines(d["Simulación"], 0, d["U"], color=c, lw=1, alpha=0.5)
            ax.plot(d["Simulación"], d["U"], "o" if i % 2 == 0 else "s", color=c, ms=5,
                    label=f"tramo {i + 1}: {_eur(v[i])} ({len(d)})")
    ax.set_xlim(0, len(traza) + 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("simulación"); ax.set_ylabel("U ~ Unif(0, 1)")
    ax.set_title(titulo or f"Funcionamiento de la transformada inversa · primeras {len(traza)} simulaciones")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=min(4, len(F)), fontsize=8)
    fig.tight_layout()
    return fig


def fig_replicas(mc, sis, titulo=None):
    rep = mc["replicas"]
    se = rep["σ(R)"] / np.sqrt(rep["Simulaciones"])
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.errorbar(rep.index, rep["Media R"], yerr=iv.Z95 * se, fmt="o", color=SERIES[0], ms=6, capsize=4,
                label="media de la réplica ± IC95 %")
    ax.axhline(sis["E"], color=TEXTO, ls="--", lw=1.4, label=f"teórico {_eur(sis['E'])}")
    _fmt_eur(ax)
    if sis["sd"] == 0:
        ax.set_ylim(sis["E"] - 100, sis["E"] + 100)
    ax.set_xticks(rep.index); ax.set_xlabel("réplica (semilla)"); ax.set_ylabel("€")
    ax.set_title(titulo or "Retorno medio en cada réplica")
    ax.legend(loc="best", fontsize=8)
    ax.margins(y=0.3)
    fig.tight_layout()
    return fig


def fig_riesgo_retorno(res, titulo=None):
    """Riesgo σ(R) frente a retorno medio de cada plan: teórico (círculo) y simulado (cruz con IC95 %)."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for i, (n, r) in enumerate(res.items()):
        t, rm = r["teoria"], r["resumen_mc"]
        c = color_escenario(i)
        ax.scatter([t["sd"]], [t["E"]], s=260, color=c, alpha=0.25, lw=0)
        ax.scatter([t["sd"]], [t["E"]], s=60, color=c, edgecolor="white", lw=1.5, zorder=3)
        ms, ss = rm.loc["Retorno medio E(R)"], rm.loc["Riesgo σ(R)"]
        ax.errorbar([ss["Simulado · media"]], [ms["Simulado · media"]],
                    xerr=[[ss["Simulado · media"] - ss["IC95 inf"]], [ss["IC95 sup"] - ss["Simulado · media"]]],
                    yerr=[[ms["Simulado · media"] - ms["IC95 inf"]], [ms["IC95 sup"] - ms["Simulado · media"]]],
                    fmt="x", color=TEXTO, ms=7, capsize=3, zorder=4)
        ax.annotate(f"{n}\nP(pérdida) {iv.pct(t['Ploss'], 0)}", (t["sd"], t["E"]), xytext=(12, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=c, fontweight="bold")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", color=TEXTO2, label="teórico"),
                       Line2D([], [], ls="none", marker="x", color=TEXTO, label="simulado ± IC95 %")],
              loc="lower right", fontsize=8)
    sds = [r["teoria"]["sd"] for r in res.values()]; es = [r["teoria"]["E"] for r in res.values()]
    ax.set_xlim(-0.05 * max(max(sds), 1), max(max(sds), 1) * 1.35)
    m = (max(es) - min(es)) or max(1.0, abs(max(es)) * 0.02)
    ax.set_ylim(min(es) - 0.3 * m, max(es) + 0.3 * m)
    _fmt_eur(ax); _fmt_eur(ax, "x")
    ax.set_xlabel("riesgo σ(R) (€) → menos seguro"); ax.set_ylabel("retorno medio (€) → más rentable")
    ax.set_title(titulo or "Riesgo frente a retorno de los planes")
    fig.tight_layout()
    return fig


def fig_comparativa(res):
    """Comparación de planes: retorno esperado, riesgo, probabilidad de pérdida y rentabilidad esperada."""
    nombres = list(res)
    cols = [color_escenario(i) for i in range(len(nombres))]
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.9))
    med = [("Retorno esperado E(R) (€)", [res[n]["teoria"]["E"] for n in nombres], "€z"),
           ("Riesgo σ(R) (€)", [res[n]["teoria"]["sd"] for n in nombres], "€"),
           ("Probabilidad de pérdida (%)", [100 * res[n]["teoria"]["Ploss"] for n in nombres], "%"),
           ("Rentabilidad esperada (%)", [100 * res[n]["teoria"]["rent"] for n in nombres], "%")]
    for ax, (tit, v, u) in zip(axs, med):
        barras = ax.bar(range(len(nombres)), v, color=cols, width=0.65)
        for bb, x in zip(barras, v):
            ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height(), iv.num(x, 1 if u == "%" else 0), ha="center",
                    va="bottom", fontsize=7.5, color=TEXTO)
        if u == "€z":                       # eje sin empezar en 0 para que se aprecien las diferencias
            lo, hi = min(v), max(v); m = (hi - lo) or abs(hi) * 0.05 or 1
            ax.set_ylim(lo - 1.2 * m, hi + 0.5 * m)
        else:
            ax.set_ylim(0, max(v) * 1.2 if max(v) > 0 else 1)
        if u.startswith("€"):
            _fmt_eur(ax)
        ax.set_title(tit, fontsize=9)
        ax.set_xticks(range(len(nombres)))
        ax.set_xticklabels(nombres, rotation=30 if len(nombres) > 2 else 0,
                           ha="right" if len(nombres) > 2 else "center", fontsize=8)
    fig.text(0.5, 0.005, "El eje del retorno esperado no empieza en 0 para que se aprecien las diferencias entre planes.",
             ha="center", fontsize=7.5, color=TEXTO2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def fig_validacion(res):
    """Error relativo de la simulación frente a la teoría en E(R), σ(R) y P(pérdida) de cada plan."""
    nombres = list(res)
    kpis = [("Retorno medio E(R)", "E(R)"), ("Riesgo σ(R)", "σ(R)"), ("Probabilidad de pérdida P(R < I₀)", "P(pérdida)"),
            ("Rentabilidad esperada", "rentabilidad")]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ancho = 0.8 / len(nombres)
    for i, n in enumerate(nombres):
        rm = res[n]["resumen_mc"]
        v = []
        for k, _ in kpis:
            e = rm.loc[k, "Error relativo"]
            v.append(0.0 if e is None or e != e else 100 * e)
        ax.bar(np.arange(len(kpis)) + i * ancho - 0.4 + ancho / 2, v, width=ancho * 0.9, color=color_escenario(i), label=n)
    ax.axhline(0, c=TEXTO2, lw=1)
    ax.set_xticks(range(len(kpis))); ax.set_xticklabels([e for _, e in kpis])
    ax.set_ylabel("error relativo (%)")
    ax.set_title("Validación: error relativo de la simulación frente a la teoría (0 si el teórico es 0)")
    ax.legend(fontsize=8, ncol=min(4, len(nombres)))
    fig.tight_layout()
    return fig


def fig_sens_convergencia(df, titulo=None):
    """Medias por réplica (puntos claros), media combinada con IC95 % (punto sólido) y retorno teórico."""
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(df))
    for i, (_, f) in enumerate(df.iterrows()):
        ax.plot([i] * len(f["Medias por réplica"]), f["Medias por réplica"], "o", color=SERIES[0], alpha=0.3, ms=5)
    ax.errorbar(x, df["Media combinada"], yerr=[df["Media combinada"] - df["IC95 inf"], df["IC95 sup"] - df["Media combinada"]],
                fmt="o-", color=SERIES[0], ms=7, capsize=4, label="media combinada ± IC95 %")
    ax.plot([], [], "o", color=SERIES[0], alpha=0.3, label="media de cada réplica")
    ax.axhline(df["Teórico"].iloc[0], color=TEXTO, ls="--", lw=1.4, label=f"teórico {_eur(df['Teórico'].iloc[0])}")
    ax.set_xticks(x); ax.set_xticklabels([iv.num(v, 0) for v in df.index])
    _fmt_eur(ax)
    ax.set_xlabel("nsim (simulaciones por réplica)"); ax.set_ylabel("€")
    ax.set_title(titulo or "Convergencia de la estimación hacia el retorno teórico")
    ax.legend(loc="best", fontsize=8)
    ax.margins(y=0.15)
    fig.tight_layout()
    return fig


def fig_sens_variabilidad(df, titulo=None):
    """σ entre réplicas de la media estimada (barras) y la predicción teórica σ(R)/√nsim (línea)."""
    fig, ax = plt.subplots(figsize=(9, 3.2))
    x = np.arange(len(df))
    v = df["σ entre réplicas"]
    ax.bar(x, v, color=SERIES[0], width=0.6, label="σ entre réplicas (simulada)")
    ax.plot(x, df["σ teórica de la media"], color=TEXTO, marker="o", lw=1.5, label="σ(R)/√nsim (teórica)")
    for i, y in enumerate(v):
        ax.text(i, y, iv.num(y, 1), ha="center", va="bottom", fontsize=8, color=TEXTO)
    ax.set_xticks(x); ax.set_xticklabels([iv.num(t, 0) for t in df.index])
    ax.set_xlabel("nsim (simulaciones por réplica)"); ax.set_ylabel("€")
    _fmt_eur(ax)
    ax.set_title(titulo or "Variabilidad de la media estimada entre réplicas")
    ax.legend(loc="upper right", fontsize=8)
    ax.margins(y=0.15)
    fig.tight_layout()
    return fig
