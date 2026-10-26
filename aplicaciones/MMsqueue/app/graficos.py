# -*- coding: utf-8 -*-
"""Gráficos (matplotlib) de la aplicación M/M/s. Cada función devuelve una Figure."""
import io
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Paleta categórica validada (orden fijo; el color sigue a la entidad, nunca a su posición)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SERVIDORES, COLA = SERIES[0], SERIES[1]          # región de servidores (n < s) y de cola (n ≥ s)
# colores de servidor individuales en el diagrama de funcionamiento (sin el naranja, reservado a la cola)
COLORES_SERVIDOR = [c for c in SERIES if c != COLA]
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


def fig_pn(sis, titulo=None):
    """Distribución límite p_n: azul la región de servidores (n < s), naranja la cola (n ≥ s)."""
    if not sis["estable"]:
        return _vacia("Sistema inestable (ρ ≥ 100 %): la distribución límite no existe.")
    pn, s = sis["pn"], sis["s"]
    n = np.arange(len(pn))
    corte = max(s + 1, int(np.searchsorted(np.cumsum(pn), 0.999)) + 1)
    n, pn = n[:corte + 1], pn[:corte + 1]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    col = [SERVIDORES if k < s else COLA for k in n]
    ax.bar(n, pn, width=0.8, color=col, edgecolor="white", linewidth=1)
    ax.axvline(s - 0.5, ls=":", c=TEXTO2, lw=1)
    ax.set_xlabel("n (clientes en el sistema)"); ax.set_ylabel("p_n")
    ax.set_title(titulo or f"Distribución límite p_n · M/M/{s}")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SERVIDORES, label="n < s (servidores)"), Patch(color=COLA, label="n ≥ s (cola)")],
              loc="upper right")
    fig.tight_layout()
    return fig


def fig_pn_comparada(pn_mc, sis, titulo=None):
    """p_n simulada (barras, ponderada por tiempo) frente a la teórica (línea con puntos)."""
    if not sis["estable"]:
        return _vacia("Sistema inestable (ρ ≥ 100 %): no hay distribución límite con la que comparar.")
    s = sis["s"]
    k = max(len(pn_mc), len(sis["pn"]))
    a = np.zeros(k); a[:len(pn_mc)] = pn_mc
    b = np.zeros(k); b[:len(sis["pn"])] = sis["pn"]
    corte = max(s + 1, int(np.searchsorted(np.cumsum(b), 0.999)) + 1)
    n = np.arange(min(k, corte + 1))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(n, a[:len(n)], width=0.8, color=[SERVIDORES if x < s else COLA for x in n], alpha=0.55,
           edgecolor="white", linewidth=1, label="Simulación")
    ax.plot(n, b[:len(n)], color=TEXTO, lw=2, marker="o", ms=4, label="Teórica")
    ax.axvline(s - 0.5, ls=":", c=TEXTO2, lw=1)
    ax.set_xlabel("n (clientes en el sistema)"); ax.set_ylabel("p_n")
    ax.set_title(titulo or "Distribución p_n: simulación frente a teoría")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_traza(traza, s, unidad, titulo=None):
    """Trayectoria N(t) de una réplica de ejemplo (función escalonada)."""
    if not traza:
        return _vacia("Sin trayectoria")
    t = [p[0] for p in traza]; n = [p[1] for p in traza]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.step(t, n, where="post", color=SERVIDORES, lw=1.6)
    ax.axhline(s, ls=":", c=COLA, lw=1.5, label=f"n = s = {s}")
    ax.set_xlabel(f"tiempo ({unidad})"); ax.set_ylabel("N(t)")
    ax.set_title(titulo or "Trayectoria N(t) · clientes en el sistema (réplica de ejemplo)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_diagrama(clientes, s, horizonte, unidad, titulo=None, max_clientes=60):
    """Diagrama de funcionamiento: para cada cliente, su espera en cola (naranja) y su servicio (azul) en el servidor
    que le atiende; debajo, el número de clientes en cola a lo largo del horizonte."""
    if clientes is None or len(clientes) == 0:
        return _vacia("No llega ningún cliente en el horizonte simulado.")
    c = clientes.head(max_clientes)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    for _, r in c.iterrows():
        y = r["Cliente"]
        if r["Espera"] > 1e-12:
            ax.barh(y, r["Espera"], left=r["Llegada"], height=0.7, color=COLA)
        ax.barh(y, r["Servicio"], left=r["Inicio"], height=0.7,
                color=COLORES_SERVIDOR[int(r["Servidor"]) - 1] if s <= len(COLORES_SERVIDOR) else SERVIDORES)
    ax.invert_yaxis()
    ax.set_ylabel("cliente")
    ax.set_title(titulo or f"Diagrama de funcionamiento · primeros {len(c)} clientes (color de servicio = servidor)")
    from matplotlib.patches import Patch
    hand = [Patch(color=COLA, label="espera en cola")]
    if s <= len(COLORES_SERVIDOR):
        hand += [Patch(color=COLORES_SERVIDOR[k], label=f"servicio · servidor {k + 1}") for k in range(s)]
    else:
        hand += [Patch(color=SERVIDORES, label="servicio")]
    ax.legend(handles=hand, loc="lower left", ncol=min(4, len(hand)), fontsize=8)
    # número en cola a lo largo del tiempo
    esperan = clientes[clientes["Espera"] > 1e-12]       # solo los clientes que realmente pasan por la cola
    ev = sorted([(t, +1) for t in esperan["Llegada"]] + [(t, -1) for t in esperan["Inicio"]], key=lambda x: (x[0], -x[1]))
    tt, q, cur = [0.0], [0], 0
    for t, d in ev:
        cur += d; tt.append(t); q.append(cur)
    ax2.step(tt, q, where="post", color=COLA, lw=1.6)
    ax2.set_ylabel("en cola"); ax2.set_xlabel(f"tiempo ({unidad})")
    t_fin = float(c["Fin"].max()) if horizonte is None else min(horizonte, float(c["Fin"].max()))
    ax2.set_xlim(0, max(t_fin, 1e-9))
    fig.tight_layout()
    return fig


def fig_costes(cst, unidad_s, titulo=None):
    """Desglose del coste total por unidad de tiempo."""
    if math.isinf(cst["CT"]):
        return _vacia("Sistema inestable: el coste de espera (CTQ) y el coste total son infinitos.")
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
    """Coste total frente al número de servidores (solo configuraciones estables)."""
    est = df[df["Estable"]]
    if est.empty:
        return _vacia("Ningún valor de s explorado es estable.")
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(est.index, est["CT"], color=SERIES[0], lw=2, marker="o", ms=6, label="CT (coste total)")
    ax.plot(est.index, est["CTQ"], color=SERIES[1], lw=1.4, ls="--", label="CTQ (espera)")
    ax.plot(est.index, est["CTS"] + est["CTSn"], color=SERIES[2], lw=1.4, ls="--", label="CTS + CTSn (servidores)")
    if s_opt is not None:
        ax.plot([s_opt], [est.loc[s_opt, "CT"]], marker="o", ms=12, color=BUENO, ls="none", label=f"mínimo: s = {s_opt}")
    if s_actual in est.index:
        ax.plot([s_actual], [est.loc[s_actual, "CT"]], marker="o", ms=15, mfc="none", mec=TEXTO, mew=1.5, ls="none",
                label=f"actual: s = {s_actual}")
    inest = df.index[~df["Estable"]]
    if len(inest):
        ax.axvspan(0.5, inest.max() + 0.5, color=CRITICO, alpha=0.08, lw=0)
        ax.text(inest.max() / 2 + 0.5, ax.get_ylim()[1] * 0.95, "inestable", color=CRITICO, ha="center", va="top", fontsize=8)
    ax.set_xticks(df.index)
    ax.set_xlabel("s (número de servidores)"); ax.set_ylabel(f"€ / {unidad_s}")
    ax.set_title(titulo or "Coste total frente al número de servidores")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def fig_comparativa(res, unidad):
    """Comparación de escenarios: ρ, Wq y coste total (un panel por medida, sin doble eje)."""
    nombres = list(res)
    cols = [color_escenario(i) for i in range(len(nombres))]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    medidas = [("ρ (carga por servidor)", [res[n]["teoria"]["rho"] * 100 for n in nombres], "%"),
               (f"Wq (tiempo en cola, {unidad})", [res[n]["teoria"]["Wq"] for n in nombres], ""),
               ("CT (coste total, €)", [res[n]["costes"]["CT"] for n in nombres], "€")]
    for ax, (tit, v, u) in zip(axs, medidas):
        vv = [x if math.isfinite(x) else 0 for x in v]
        barras = ax.bar(range(len(nombres)), vv, color=cols, width=0.65)
        for b, x in zip(barras, v):
            lab = "∞" if not math.isfinite(x) else (f"{x:.1f}" if u == "%" else f"{x:,.2f}")
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), lab.replace(",", "X").replace(".", ",").replace("X", "."),
                    ha="center", va="bottom", fontsize=8, color=TEXTO)
        ax.set_title(tit, fontsize=9.5)
        ax.set_xticks(range(len(nombres)))
        ax.set_xticklabels(nombres, rotation=30 if len(nombres) > 2 else 0, ha="right" if len(nombres) > 2 else "center",
                           fontsize=8)
        if u == "%":
            ax.axhline(100, ls=":", c=CRITICO, lw=1)
    fig.tight_layout()
    return fig


def fig_validacion(res):
    """Error relativo de la simulación frente a la teoría (L, Lq, W, Wq) en cada escenario estable."""
    nombres = [n for n in res if res[n]["teoria"]["estable"]]
    if not nombres:
        return _vacia("Ningún escenario estable que validar.")
    kpis = ["L", "Lq", "W", "Wq"]
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
