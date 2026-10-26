# -*- coding: utf-8 -*-
"""Gráficos (matplotlib) de la aplicación de beneficio por mixtura. Cada función devuelve una Figure."""
import io
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import mixtura as mx

# Paleta categórica validada (orden fijo; el color sigue a la entidad, nunca a su posición)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NO_COMPRA_COLOR = "#8a8984"                 # «no compra» (0 €): no es un riesgo, solo la otra rama de la mixtura
COLORES_MODELO = [c for c in SERIES]         # un color por modelo, en orden
BUENO, CRITICO = "#0b7a0b", "#b23232"
TEXTO, TEXTO2, REJILLA = "#0b0b0b", "#52514e", "#e1e0d9"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.edgecolor": "#b9b8b2", "axes.labelcolor": TEXTO2, "xtick.color": TEXTO2, "ytick.color": TEXTO2,
    "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def _eur(x, d=2):
    return mx.eur(x, d)


def color_modelo(i):
    return COLORES_MODELO[i % len(COLORES_MODELO)]


def color_escenario(i):
    return SERIES[i % len(SERIES)]


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


def fig_mecanismo(sis, u1=None, u2=None, titulo=None):
    """Mixtura en dos pasos. Izquierda: U₁ frente a Pr(compra) (¿compra?). Derecha: U₂ y la escalera F de los
    modelos (qué modelo, solo si compra)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 2.2]})
    pc = sis["pc"]
    a1.bar([0], [pc], width=0.6, color=BUENO, alpha=0.35)
    a1.bar([0], [1 - pc], bottom=[pc], width=0.6, color=NO_COMPRA_COLOR, alpha=0.35)
    a1.text(0, pc / 2, f"compra\nU₁ < {mx.num(pc, 2)}", ha="center", va="center", fontsize=8.5, color=BUENO)
    a1.text(0, pc + (1 - pc) / 2, "no compra\nB = 0 €", ha="center", va="center", fontsize=8.5, color=TEXTO2)
    a1.axhline(pc, color=TEXTO2, lw=1, ls=":")
    if u1 is not None:
        a1.annotate("", xy=(0.3, u1), xytext=(-0.55, u1), arrowprops=dict(arrowstyle="->", color=TEXTO, lw=1.6))
        a1.text(-0.55, min(u1 + 0.03, 1.0), f"U₁ = {mx.num(u1, 2)}", fontsize=9, color=TEXTO)
    a1.set_xlim(-0.6, 0.6); a1.set_ylim(0, 1.04); a1.set_xticks([])
    a1.set_ylabel("U₁ ~ Unif(0, 1)")
    a1.set_title("Paso 1 · ¿compra?")
    F = np.array(sis["F"]); esc = np.concatenate([[0.0], F]); n = len(F)
    compra = u1 is not None and u1 < pc
    for i in range(n):
        c = color_modelo(i)
        a2.add_patch(plt.Rectangle((-0.6, esc[i]), n + 0.2, esc[i + 1] - esc[i], color=c, alpha=0.10, lw=0))
        a2.plot([i - 0.5, i + 0.5], [F[i], F[i]], color=c, lw=3, solid_capstyle="butt")
        a2.text(n - 0.35, (esc[i] + esc[i + 1]) / 2, f"{sis['modelos'][i]}: {_eur(sis['b'][i], 0)}\n"
                f"({mx.num(esc[i], 2)}, {mx.num(F[i], 2)}]", va="center", fontsize=8, color=c)
    if u2 is not None:
        k = int(mx.transformada_inversa(np.array([u2]), F)[0])
        col = TEXTO if compra else "#b9b8b2"
        a2.annotate("", xy=(k, u2), xytext=(-0.55, u2), arrowprops=dict(arrowstyle="->", color=col, lw=1.6))
        a2.annotate("", xy=(k, -0.02), xytext=(k, u2), arrowprops=dict(arrowstyle="->", color=col, lw=1.6))
        a2.text(-0.55, min(u2 + 0.03, 1.0), f"U₂ = {mx.num(u2, 2)}" + ("" if compra else " (no se usa)"), fontsize=9,
                color=col)
    a2.set_xlim(-0.6, n + 0.9); a2.set_ylim(-0.03, 1.05)
    a2.set_xticks(range(n)); a2.set_xticklabels(sis["modelos"])
    a2.set_ylabel("F(i) = p₁ + … + pᵢ")
    a2.set_title("Paso 2 · qué modelo (solo si compra)")
    if titulo:
        fig.suptitle(titulo, fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    return fig


def fig_desenlaces(sis, mc=None, titulo=None):
    """Distribución del beneficio por cliente B: probabilidad de cada desenlace (barras teóricas; rombos simulados)."""
    nombres = [mx.NO_COMPRA] + list(sis["modelos"])
    p = [1 - sis["pc"]] + [sis["pc"] * q for q in sis["p"]]
    cols = [NO_COMPRA_COLOR] + [color_modelo(i) for i in range(len(sis["modelos"]))]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    x = np.arange(len(nombres))
    ax.bar(x, p, width=0.6, color=cols)
    for i in x:
        ax.text(i, p[i], mx.pct(p[i], 1), ha="center", va="bottom", fontsize=8.5, color=TEXTO)
    if mc is not None:
        n = len(mc["B"])
        sim = [float((~mc["T"]).sum()) / n] + [float(((mc["idx"] == i) & mc["T"]).sum()) / n for i in range(len(sis["b"]))]
        ax.plot(x, sim, "D", color=TEXTO, ms=6, label="simulada")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{nm}\nB = {_eur(b, 0)}" for nm, b in zip(nombres, [0.0] + list(sis["b"]))])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, max(p) * 1.25)
    ax.set_title(titulo or "Distribución del beneficio por cliente B (barras: teórica)")
    fig.tight_layout()
    return fig


def fig_descomposicion(sis, titulo=None):
    """Contribución de cada desenlace a E(B) = Σ Pr(compra)·pᵢ·bᵢ, con E(B) y E(B | compra) de referencia."""
    nombres = [mx.NO_COMPRA] + list(sis["modelos"])
    v = [0.0] + list(sis["contrib"])
    cols = [NO_COMPRA_COLOR] + [color_modelo(i) for i in range(len(sis["modelos"]))]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    x = np.arange(len(nombres))
    ax.bar(x, v, width=0.6, color=cols)
    for i, y in enumerate(v):
        ax.text(i, y, _eur(y), ha="center", va="bottom", fontsize=8.5, color=TEXTO)
    ax.axhline(sis["E"], color=TEXTO, ls="--", lw=1.3, label=f"E(B) = {_eur(sis['E'])} (suma de las barras)")
    ax.set_xticks(x); ax.set_xticklabels(nombres)
    ax.set_ylabel("€ por cliente")
    ax.set_ylim(0, max(max(v), sis["E"]) * 1.25)
    ax.set_title(titulo or "Descomposición del beneficio esperado: Pr(desenlace) × beneficio")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def fig_traza(traza, sis, titulo=None):
    """Funcionamiento simulado: U₁ de los primeros clientes (compran si U₁ < Pr(compra)); color = modelo elegido."""
    if traza is None or len(traza) == 0:
        return _vacia("Sin clientes que mostrar")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    pc = sis["pc"]
    ax.axhspan(0, pc, color=BUENO, alpha=0.07, lw=0)
    ax.axhline(pc, color=TEXTO2, lw=1, ls=":", label=f"Pr(compra) = {mx.num(pc, 2)}")
    nc = traza[~traza["Compra"]]
    ax.vlines(nc["Cliente"], 0, nc["U1"], color=NO_COMPRA_COLOR, lw=1, alpha=0.5)
    ax.plot(nc["Cliente"], nc["U1"], "x", color=NO_COMPRA_COLOR, ms=6, mew=1.8, label=f"no compra ({len(nc)})")
    for i, m in enumerate(sis["modelos"]):
        d = traza[traza["Modelo"] == m]
        if len(d):
            ax.vlines(d["Cliente"], 0, d["U1"], color=color_modelo(i), lw=1, alpha=0.5)
            ax.plot(d["Cliente"], d["U1"], "o", color=color_modelo(i), ms=5, label=f"compra · {m} ({len(d)})")
    ax.set_xlim(0, len(traza) + 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("cliente"); ax.set_ylabel("U₁ ~ Unif(0, 1)")
    ax.set_title(titulo or f"Funcionamiento de la mixtura · primeros {len(traza)} clientes")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=min(5, len(sis["modelos"]) + 2), fontsize=8)
    fig.tight_layout()
    return fig


def fig_replicas(mc, sis, titulo=None):
    """E(B) y E(B | compra) de cada réplica con su IC95 % y los valores teóricos."""
    rep = mc["replicas"]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    se1 = rep["σ(B)"] / np.sqrt(rep["Clientes"])
    ax.errorbar(rep.index - 0.08, rep["E(B)"], yerr=mx.Z95 * se1, fmt="o", color=SERIES[0], ms=6, capsize=4,
                label="E(B) · todos ± IC95 %")
    ax.axhline(sis["E"], color=SERIES[0], ls="--", lw=1.3, label=f"teórico {_eur(sis['E'])}")
    se2 = sis["sd_compra"] / np.sqrt(rep["Compradores"].clip(lower=1))
    ax.errorbar(rep.index + 0.08, rep["E(B | compra)"], yerr=mx.Z95 * se2, fmt="s", color=BUENO, ms=6, capsize=4,
                label="E(B | compra) ± IC95 %")
    ax.axhline(sis["E_compra"], color=BUENO, ls="--", lw=1.3, label=f"teórico {_eur(sis['E_compra'])}")
    ax.set_xticks(rep.index); ax.set_xlabel("réplica (semilla)"); ax.set_ylabel("€ por cliente")
    ax.set_title(titulo or "Beneficio medio en cada réplica: todos los clientes y solo compradores")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    ax.margins(y=0.15)
    fig.tight_layout()
    return fig


def fig_comparativa(res):
    """Comparación de estrategias: P(compra), E(B | compra), beneficio neto por cliente y beneficio neto diario."""
    nombres = list(res)
    cols = [color_escenario(i) for i in range(len(nombres))]
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.9))
    med = [("Proporción que compra (%)", [100 * res[n]["teoria"]["pc"] for n in nombres], "%"),
           ("E(B | compra) (€)", [res[n]["teoria"]["E_compra"] for n in nombres], "€z"),
           ("Beneficio neto por cliente (€)", [res[n]["teoria"]["E_neto"] for n in nombres], "€z"),
           ("Beneficio neto diario (€)", [res[n]["teoria"]["dia_neto"] for n in nombres], "€z0")]
    for ax, (tit, v, u) in zip(axs, med):
        barras = ax.bar(range(len(nombres)), v, color=cols, width=0.65)
        for bb, x in zip(barras, v):
            ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height(), mx.num(x, 0 if u == "€z0" else (1 if u == "%" else 2)),
                    ha="center", va="bottom", fontsize=7.5, color=TEXTO)
        if u.startswith("€z"):
            lo, hi = min(v), max(v); m = (hi - lo) or abs(hi) * 0.05 or 1
            ax.set_ylim(lo - 1.2 * m, hi + 0.5 * m)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: mx.num(y, 0 if u == "€z0" else 1)))
        else:
            ax.set_ylim(0, max(v) * 1.2 if max(v) > 0 else 1)
        ax.set_title(tit, fontsize=9)
        ax.set_xticks(range(len(nombres)))
        ax.set_xticklabels(nombres, rotation=30 if len(nombres) > 2 else 0,
                           ha="right" if len(nombres) > 2 else "center", fontsize=8)
    fig.text(0.5, 0.005, "Los ejes en € no empiezan en 0 para que se aprecien las diferencias entre estrategias.",
             ha="center", fontsize=7.5, color=TEXTO2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def fig_validacion(res):
    nombres = list(res)
    kpis = [("Beneficio medio por cliente E(B)", "E(B)"), ("Beneficio medio por comprador E(B | compra)", "E(B | compra)"),
            ("Proporción que compra P(compra)", "P(compra)"), ("Desviación típica σ(B)", "σ(B)")]
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
    ax.set_title("Validación: error relativo de la simulación frente a la teoría")
    ax.legend(fontsize=8, ncol=min(4, len(nombres)))
    fig.tight_layout()
    return fig


def fig_sens_convergencia(df, metrica, titulo=None):
    f = mx.fmt_metrica(metrica)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(df))
    for i, (_, r) in enumerate(df.iterrows()):
        ax.plot([i] * len(r["Medias por réplica"]), r["Medias por réplica"], "o", color=SERIES[0], alpha=0.3, ms=5)
    ax.errorbar(x, df["Media de las réplicas"],
                yerr=[df["Media de las réplicas"] - df["IC95 inf"], df["IC95 sup"] - df["Media de las réplicas"]],
                fmt="o-", color=SERIES[0], ms=7, capsize=4, label="media de las réplicas ± IC95 %")
    ax.plot([], [], "o", color=SERIES[0], alpha=0.3, label="cada réplica")
    ax.axhline(df["Teórico"].iloc[0], color=TEXTO, ls="--", lw=1.4, label=f"teórico {f(df['Teórico'].iloc[0])}")
    ax.set_xticks(x); ax.set_xticklabels([mx.num(v, 0) for v in df.index])
    if metrica == "proporcion":
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=1))
    ax.set_xlabel("nsim (clientes simulados por réplica)"); ax.set_ylabel("%" if metrica == "proporcion" else "€")
    ax.set_title(titulo or f"Convergencia · {mx.METRICAS[metrica]}")
    ax.legend(loc="best", fontsize=8)
    ax.margins(y=0.15)
    fig.tight_layout()
    return fig


def fig_sens_variabilidad(df, metrica, titulo=None):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    x = np.arange(len(df))
    k = 100 if metrica == "proporcion" else 1
    v = df["σ entre réplicas"] * k
    ax.bar(x, v, color=SERIES[0], width=0.6, label="σ entre réplicas (simulada)")
    ax.plot(x, df["σ teórica del estimador"] * k, color=TEXTO, marker="o", lw=1.5, label="σ teórica del estimador")
    for i, y in enumerate(v):
        ax.text(i, y, mx.num(y, 3 if metrica == "proporcion" else 2), ha="center", va="bottom", fontsize=8, color=TEXTO)
    ax.set_xticks(x); ax.set_xticklabels([mx.num(t, 0) for t in df.index])
    ax.set_xlabel("nsim (clientes simulados por réplica)")
    ax.set_ylabel("puntos porcentuales" if metrica == "proporcion" else "€")
    ax.set_title(titulo or "Variabilidad de la estimación entre réplicas")
    ax.legend(loc="upper right", fontsize=8)
    ax.margins(y=0.15)
    fig.tight_layout()
    return fig
