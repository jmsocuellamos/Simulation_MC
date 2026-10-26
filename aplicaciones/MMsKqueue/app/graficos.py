# -*- coding: utf-8 -*-
"""Gráficos (matplotlib) de la aplicación M/M/s/K. Cada función devuelve una Figure."""
import io
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Paleta categórica validada (orden fijo; el color sigue a la entidad, nunca a su posición)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SERVIDORES, COLA, LLENO = SERIES[0], SERIES[1], SERIES[7]   # n < s, s ≤ n < K y n = K (bloqueo)
# colores de servidor individuales en el diagrama de funcionamiento (sin el naranja ni el rojo, reservados)
COLORES_SERVIDOR = [c for c in SERIES if c not in (COLA, LLENO)]
BUENO, CRITICO = "#0b7a0b", "#b23232"
TEXTO, TEXTO2, REJILLA = "#0b0b0b", "#52514e", "#e1e0d9"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.edgecolor": "#b9b8b2", "axes.labelcolor": TEXTO2, "xtick.color": TEXTO2, "ytick.color": TEXTO2,
    "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


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


def _color_n(k, s, K):
    return LLENO if k == K else (SERVIDORES if k < s else COLA)


def _leyenda_regiones(ax, extra=()):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SERVIDORES, label="n < s (servidores)"), Patch(color=COLA, label="s ≤ n < K (cola)"),
                       Patch(color=LLENO, label="n = K (lleno · bloqueo)"), *extra], loc="upper center",
              bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=8)


def fig_pn(sis, titulo=None):
    """Distribución límite p_n, n = 0 … K: azul los servidores, naranja la cola y rojo el estado lleno (bloqueo)."""
    pn, s, K = sis["pn"], sis["s"], sis["K"]
    n = np.arange(len(pn))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(n, pn, width=0.8, color=[_color_n(k, s, K) for k in n], edgecolor="white", linewidth=1)
    ax.axvline(s - 0.5, ls=":", c=TEXTO2, lw=1)
    ax.set_xlabel("n (clientes en el sistema)"); ax.set_ylabel("p_n")
    if K <= 40:
        ax.set_xticks(n)
    ax.set_title(titulo or f"Distribución límite p_n · M/M/{s}/{K}")
    _leyenda_regiones(ax)
    fig.tight_layout()
    return fig


def fig_pn_comparada(pn_mc, sis, titulo=None):
    """p_n simulada (barras, ponderada por tiempo) frente a la teórica (línea con puntos)."""
    s, K = sis["s"], sis["K"]
    n = np.arange(K + 1)
    a = np.zeros(K + 1); m = min(K + 1, len(pn_mc)); a[:m] = pn_mc[:m]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(n, a, width=0.8, color=[_color_n(x, s, K) for x in n], alpha=0.55, edgecolor="white", linewidth=1)
    ax.plot(n, sis["pn"], color=TEXTO, lw=2, marker="o", ms=4)
    ax.axvline(s - 0.5, ls=":", c=TEXTO2, lw=1)
    ax.set_xlabel("n (clientes en el sistema)"); ax.set_ylabel("p_n")
    if K <= 40:
        ax.set_xticks(n)
    ax.set_title(titulo or "Distribución p_n: simulación (barras) frente a teoría (línea)")
    from matplotlib.lines import Line2D
    _leyenda_regiones(ax, [Line2D([], [], color=TEXTO, lw=2, marker="o", ms=4, label="teórica")])
    fig.tight_layout()
    return fig


def fig_traza(traza, s, unidad, titulo=None, K=None):
    """Trayectoria N(t) de una réplica de ejemplo (función escalonada), con las líneas n = s y n = K."""
    if not traza:
        return _vacia("Sin trayectoria")
    t = [p[0] for p in traza]; n = [p[1] for p in traza]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.step(t, n, where="post", color=SERVIDORES, lw=1.6)
    ax.axhline(s, ls=":", c=COLA, lw=1.5, label=f"n = s = {s}")
    if K is not None:
        ax.axhline(K, ls="--", c=LLENO, lw=1.5, label=f"n = K = {K} (lleno)")
        ax.set_ylim(-0.3, K + 0.8)
    ax.set_xlabel(f"tiempo ({unidad})"); ax.set_ylabel("N(t)")
    ax.set_title(titulo or "Trayectoria N(t) · clientes en el sistema (réplica de ejemplo)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def fig_diagrama(clientes, s, horizonte, unidad, titulo=None, max_clientes=60):
    """Diagrama de funcionamiento: para cada llegada, su espera en cola (naranja) y su servicio en el servidor que le
    atiende; las llegadas rechazadas (sistema lleno) se marcan con ✕ rojo. Debajo, el número de clientes en cola."""
    if clientes is None or len(clientes) == 0:
        return _vacia("No llega ningún cliente en el horizonte simulado.")
    c = clientes.head(max_clientes)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    rech = c[c["Rechazado"]] if "Rechazado" in c else c.iloc[0:0]
    for _, r in c.iterrows():
        y = r["Cliente"]
        if "Rechazado" in c and r["Rechazado"]:
            continue
        if r["Espera"] > 1e-12:
            ax.barh(y, r["Espera"], left=r["Llegada"], height=0.7, color=COLA)
        ax.barh(y, r["Servicio"], left=r["Inicio"], height=0.7,
                color=COLORES_SERVIDOR[int(r["Servidor"]) - 1] if s <= len(COLORES_SERVIDOR) else SERVIDORES)
    if len(rech):
        ax.plot(rech["Llegada"], rech["Cliente"], ls="none", marker="x", ms=7, mew=2, color=LLENO)
    ax.invert_yaxis()
    ax.set_ylabel("cliente (llegada)")
    ax.set_title(titulo or f"Diagrama de funcionamiento · primeras {len(c)} llegadas (color de servicio = servidor)")
    from matplotlib.patches import Patch
    hand = [Patch(color=COLA, label="espera en cola")]
    if s <= len(COLORES_SERVIDOR):
        hand += [Patch(color=COLORES_SERVIDOR[k], label=f"servicio · servidor {k + 1}") for k in range(s)]
    else:
        hand += [Patch(color=SERVIDORES, label="servicio")]
    if len(rech):
        from matplotlib.lines import Line2D
        hand.append(Line2D([], [], ls="none", marker="x", ms=7, mew=2, color=LLENO, label="rechazado (sistema lleno)"))
    ax.legend(handles=hand, loc="lower left", ncol=min(4, len(hand)), fontsize=8)
    # número en cola a lo largo del tiempo
    esperan = clientes[clientes["Espera"] > 1e-12]       # solo los clientes que realmente pasan por la cola
    ev = sorted([(t, +1) for t in esperan["Llegada"]] + [(t, -1) for t in esperan["Inicio"]], key=lambda x: (x[0], -x[1]))
    tt, q, cur = [0.0], [0], 0
    for t, d in ev:
        cur += d; tt.append(t); q.append(cur)
    ax2.step(tt, q, where="post", color=COLA, lw=1.6)
    ax2.set_ylabel("en cola"); ax2.set_xlabel(f"tiempo ({unidad})")
    t_fin = float(np.nanmax([c["Fin"].max(), c["Llegada"].max()])) if horizonte is None else min(horizonte, float(c["Fin"].max()))
    ax2.set_xlim(0, max(t_fin, 1e-9))
    fig.tight_layout()
    return fig


def fig_costes(cst, unidad_s, titulo=None):
    """Desglose del coste total por unidad de tiempo."""
    nombres = ["CTS\nservidores ocupados", "CTSn\nservidores libres", "CTQ\nespera en cola"]
    v = [cst["CTS"], cst["CTSn"], cst["CTQ"]]
    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.barh(nombres, v, color=[SERIES[0], SERIES[2], SERIES[1]], height=0.6)
    for i, x in enumerate(v):
        ax.text(x, i, f"  {x:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."), va="center", color=TEXTO)
    ax.invert_yaxis()
    ax.set_xlabel(f"€ / {unidad_s}")
    ax.set_title(titulo or f"Desglose de costes · CT = {cst['CT']:,.2f} €/{unidad_s}".replace(",", "X").replace(".", ",").replace("X", "."))
    ax.set_xlim(0, max(v) * 1.25 if max(v) > 0 else 1)
    fig.tight_layout()
    return fig


def fig_barrido(df, s_actual, s_opt, unidad_s, titulo=None):
    """Coste total frente al número de servidores (s = 1 … min(s máx, K)) y probabilidad de bloqueo."""
    if df.empty:
        return _vacia("No hay valores de s que explorar.")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [3, 2]})
    ax.plot(df.index, df["CT"], color=SERIES[0], lw=2, marker="o", ms=6, label="CT (coste total)")
    ax.plot(df.index, df["CTQ"], color=SERIES[1], lw=1.4, ls="--", label="CTQ (espera)")
    ax.plot(df.index, df["CTS"] + df["CTSn"], color=SERIES[2], lw=1.4, ls="--", label="CTS + CTSn (servidores)")
    if s_opt is not None:
        ax.plot([s_opt], [df.loc[s_opt, "CT"]], marker="o", ms=12, color=BUENO, ls="none", label=f"mínimo: s = {s_opt}")
    if s_actual in df.index:
        ax.plot([s_actual], [df.loc[s_actual, "CT"]], marker="o", ms=15, mfc="none", mec=TEXTO, mew=1.5, ls="none",
                label=f"actual: s = {s_actual}")
    ax.set_xticks(df.index)
    ax.set_xlabel("s (número de servidores)"); ax.set_ylabel(f"€ / {unidad_s}")
    ax.set_title(titulo or "Coste total frente al número de servidores (K fijo)")
    ax.legend(loc="best", fontsize=7.5)
    ax2.bar(df.index, 100 * df["P(bloqueo)"], color=[LLENO if k == s_actual else "#f0a3a3" for k in df.index], width=0.6)
    ax2.set_xticks(df.index)
    ax2.set_xlabel("s (número de servidores)"); ax2.set_ylabel("%")
    ax2.set_title("P(bloqueo) según s")
    fig.tight_layout()
    return fig


def fig_comparativa(res, unidad):
    """Comparación de escenarios: ρ̄, P(bloqueo), Wq y coste total (un panel por medida, sin doble eje)."""
    nombres = list(res)
    cols = [color_escenario(i) for i in range(len(nombres))]
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.8))
    medidas = [("ρ̄ (ocupación efectiva, %)", [res[n]["teoria"]["rho_eff"] * 100 for n in nombres], "%"),
               ("P(bloqueo) (%)", [res[n]["teoria"]["Pbloqueo"] * 100 for n in nombres], "%2"),
               (f"Wq (tiempo en cola, {unidad})", [res[n]["teoria"]["Wq"] for n in nombres], ""),
               ("CT (coste total, €)", [res[n]["costes"]["CT"] for n in nombres], "€")]
    for ax, (tit, v, u) in zip(axs, medidas):
        barras = ax.bar(range(len(nombres)), v, color=cols, width=0.65)
        for b, x in zip(barras, v):
            lab = f"{x:.1f}" if u == "%" else (f"{x:.2f}" if u == "%2" else (f"{x:,.2f}" if u == "€" else f"{x:,.3f}"))
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), lab.replace(",", "X").replace(".", ",").replace("X", "."),
                    ha="center", va="bottom", fontsize=8, color=TEXTO)
        ax.set_title(tit, fontsize=9.5)
        ax.set_xticks(range(len(nombres)))
        ax.set_xticklabels(nombres, rotation=30 if len(nombres) > 2 else 0, ha="right" if len(nombres) > 2 else "center",
                           fontsize=8)
        ax.margins(y=0.15)
    fig.tight_layout()
    return fig


def fig_validacion(res):
    """Error relativo de la simulación frente a la teoría (L, Lq, W, Wq, P(bloqueo)) en cada escenario."""
    nombres = list(res)
    if not nombres:
        return _vacia("No hay escenarios que validar.")
    kpis = ["L", "Lq", "W", "Wq", "P(bloqueo)"]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ancho = 0.8 / len(nombres)
    for i, n in enumerate(nombres):
        rm = res[n]["resumen_mc"]
        v = [100 * (rm.loc[k, "Error relativo"] or 0) for k in kpis]
        ax.bar(np.arange(len(kpis)) + i * ancho - 0.4 + ancho / 2, v, width=ancho * 0.9, color=color_escenario(list(res).index(n)),
               label=n)
    ax.axhline(0, c=TEXTO2, lw=1)
    ax.set_xticks(range(len(kpis))); ax.set_xticklabels(kpis)
    ax.set_ylabel("error relativo (%)")
    ax.set_title("Validación: error relativo de la simulación frente a la teoría")
    ax.legend(fontsize=8, ncol=min(4, len(nombres)))
    fig.tight_layout()
    return fig
