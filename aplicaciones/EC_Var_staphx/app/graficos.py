# -*- coding: utf-8 -*-
"""Gráficos (matplotlib) de la aplicación Staph-X. Cada función devuelve una Figure."""
import io
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import staphx as sx

# Paleta categórica validada (orden fijo; el color sigue a la entidad, nunca a su posición)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COLOR_FEN = {1: SERIES[2], 2: SERIES[0], 3: SERIES[7]}
COLOR_FASE = {"Inicio": SERIES[2], "Pico": SERIES[7], "Remisión": SERIES[3]}
COLOR_TERM = ["#8a8984", SERIES[0], SERIES[1]]          # fijo, estancia, complicación
SIN, CON = "#8a8984", SERIES[7]
BUENO, CRITICO = "#0b7a0b", "#b23232"
TEXTO, TEXTO2, REJILLA = "#0b0b0b", "#52514e", "#e1e0d9"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.edgecolor": "#b9b8b2", "axes.labelcolor": TEXTO2, "xtick.color": TEXTO2, "ytick.color": TEXTO2,
    "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
_EUR0 = mticker.FuncFormatter(lambda y, _: sx.num(y, 0))
_MILES = mticker.FuncFormatter(lambda y, _: sx.num(y / 1000, 0) + "k")
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


# ---------------------------------------------------------------------------------------------- mecanismo
def fig_mecanismo(p, uT, pac, umbral, titulo=None):
    """Izquierda: el fenotipo se elige con U_T sobre las probabilidades acumuladas. Derecha: días de ingreso D frente
    al umbral y coste del paciente por términos."""
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11.5, 3.6), gridspec_kw={"width_ratios": [1.5, 0.7, 1]})
    F = np.cumsum(sx.normalizar(p)); esc = np.concatenate([[0], F])
    for i, f in enumerate(sx.FENOTIPOS):
        a1.add_patch(plt.Rectangle((-0.6, esc[i]), 3.2, esc[i + 1] - esc[i], color=COLOR_FEN[f], alpha=0.12, lw=0))
        a1.plot([i - 0.5, i + 0.5], [F[i], F[i]], color=COLOR_FEN[f], lw=3)
        a1.text(2.62, (esc[i] + esc[i + 1]) / 2, f"{sx.NOMBRE_FEN[f]}\n({sx.num(esc[i], 2)}, {sx.num(F[i], 2)}]",
                va="center", fontsize=7.5, color=COLOR_FEN[f])
    k = min(int(np.searchsorted(F, uT)), 2)
    a1.annotate("", xy=(k, uT), xytext=(-0.55, uT), arrowprops=dict(arrowstyle="->", color=TEXTO, lw=1.6))
    a1.annotate("", xy=(k, -0.02), xytext=(k, uT), arrowprops=dict(arrowstyle="->", color=TEXTO, lw=1.6))
    a1.text(-0.55, min(uT + 0.03, 1.0), f"U_T = {sx.num(uT, 2)}", fontsize=9)
    a1.set_xlim(-0.6, 3.9); a1.set_ylim(-0.03, 1.05)
    a1.set_xticks(range(3)); a1.set_xticklabels([sx.NOMBRE_FEN[f] for f in sx.FENOTIPOS], fontsize=8)
    a1.set_ylabel("F(T)"); a1.set_title("Paso 1 · fenotipo por transformada inversa", fontsize=9.5)
    D = pac["D"]
    a2.bar([0], [D], color=COLOR_FEN[pac["T"]], width=0.5)
    a2.axhline(umbral, color=CRITICO, ls="--", lw=1.5)
    a2.text(0.42, umbral, f"umbral {sx.num(umbral, 0)} d", color=CRITICO, fontsize=8, va="bottom", ha="right")
    a2.text(0, D, f"{sx.num(D, 2)} d", ha="center", va="bottom", fontsize=9)
    a2.set_xticks([0]); a2.set_xticklabels(["D"]); a2.set_xlim(-0.6, 0.6)
    a2.set_ylim(0, max(D, umbral) * 1.25); a2.set_ylabel("días")
    a2.set_title("Paso 2 · D = máx(0, ln(L+1)/R + ε)", fontsize=9.5)
    partes = pac["partes"]; noms = ["fijo (+ extra)", "estancia C_día·D", "complicación"]
    base = 0.0
    for j in range(3):
        if abs(partes[j]) < 1e-9:
            continue
        a3.bar([0], [partes[j]], bottom=[base], color=COLOR_TERM[j], width=0.5, label=noms[j])
        base += partes[j]
    a3.text(0, base, sx.eur(base), ha="center", va="bottom", fontsize=9)
    a3.set_xticks([0]); a3.set_xticklabels(["Y"]); a3.set_xlim(-0.6, 1.4)
    a3.set_ylim(0, max(base, 1) * 1.2); a3.yaxis.set_major_formatter(_EUR0)
    a3.set_title("Paso 3 · coste del paciente", fontsize=9.5)
    a3.legend(loc="upper right", fontsize=7.5)
    if titulo:
        fig.suptitle(titulo, fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------- bloque A
def fig_hist_Y(tA, Y, titulo=None):
    Y = np.asarray(Y)
    hi = float(np.quantile(Y, 0.999))
    bins = np.linspace(min(Y.min(), tA["y"][0]), hi, 90)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.hist(Y[Y <= hi], bins=bins, density=True, color=SERIES[0], alpha=0.5, edgecolor="white", label="simulación")
    Fb = np.interp(bins, tA["y"], tA["F"])
    dens = np.diff(Fb) / np.diff(bins)
    ax.step(bins[:-1], dens, where="post", color=TEXTO, lw=1.4, label="teórica")
    ax.axvline(tA["EY"], color=SERIES[0], ls="--", lw=1.4, label=f"E(Y) = {sx.eur(tA['EY'])}")
    ax.axvline(tA["P95"], color=SERIES[3], ls="-.", lw=1.4, label=f"P95 = {sx.eur(tA['P95'])}")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(_EUR0)
    ax.set_xlabel("coste de un paciente Y (€) · hasta el percentil 99,9"); ax.set_ylabel("densidad (escala log)")
    ax.set_title(titulo or "Distribución del coste de un paciente (el salto separa a los complicados)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def fig_por_fen(tA, resA, titulo=None):
    noms = [sx.NOMBRE_FEN[f] for f in sx.FENOTIPOS]
    pf = tA["por_fen"]
    g = lambda k: resA.loc[k, "Simulado · media"]
    datos = [("E(Y | fenotipo) (€)", [pf[f]["EY"] for f in sx.FENOTIPOS], [g(f"E(Y | {n}) (€)") for n in noms], 0),
             ("P(complicación | fenotipo)", [pf[f]["Pcomp"] for f in sx.FENOTIPOS], [g(f"P(complicación | {n})") for n in noms], "%"),
             ("Coeficiente de variación", [pf[f]["CV"] for f in sx.FENOTIPOS], [g(f"CV(Y | {n})") for n in noms], 3)]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (tit, v, s, d) in zip(axs, datos):
        bars = ax.bar(range(3), v, color=[COLOR_FEN[f] for f in sx.FENOTIPOS], width=0.6)
        ax.plot(range(3), s, "D", color=TEXTO, ms=6, label="simulado")
        for b_, x in zip(bars, v):
            ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height(), sx.pct(x, 1) if d == "%" else sx.num(x, d),
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(3)); ax.set_xticklabels(noms)
        ax.set_title(tit, fontsize=9.5)
        ax.set_ylim(0, max(max(v), max(s), 1e-9) * 1.25)
        if d == "%":
            ax.yaxis.set_major_formatter(_PCT)
        elif d == 0:
            ax.yaxis.set_major_formatter(_EUR0)
    axs[0].legend(loc="upper left", fontsize=8)
    if titulo:
        fig.suptitle(titulo, fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    return fig


def fig_descomposicion(tA, resA, titulo=None):
    """Contribución de cada término a Var(Y): Cov(Xᵢ, Y)/Var(Y), teórica (barras) y simulada (rombos)."""
    v = list(tA["contrib"])
    s = [resA.loc[f"Contribución a Var(Y) · {n}", "Simulado · media"] for n in sx.TERMINOS]
    fig, ax = plt.subplots(figsize=(9, 2.6))
    x = np.arange(2)
    ax.barh(x, v, color=COLOR_TERM[1:], height=0.55)
    ax.plot(s, x, "D", color=TEXTO, ms=6, label="simulado")
    for j in range(2):
        ax.text(v[j], j, "  " + sx.pct(v[j], 1), va="center", fontsize=8.5)
    ax.set_yticks(x); ax.set_yticklabels(sx.TERMINOS)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(_PCT)
    ax.set_xlim(min(0, min(v)) * 1.2, max(v) * 1.25)
    ax.set_title(titulo or "Descomposición de Var(Y): Cov(término, Y) / Var(Y)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def fig_traza_A(traza, esc, titulo=None):
    """Primeros pacientes simulados: coste por términos y fenotipo asignado."""
    tr = traza
    a0, c, K = sx._coefs(esc)
    lin = c * tr["D (días)"]
    salto = np.where(tr["Complicación"] == "Sí", K, 0.0)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = tr["Paciente"]
    ax.bar(x, np.full(len(x), a0), color=COLOR_TERM[0], width=0.8, label="fijo")
    ax.bar(x, lin, bottom=a0, color=COLOR_TERM[1], width=0.8, label="estancia")
    ax.bar(x, salto, bottom=a0 + lin, color=COLOR_TERM[2], width=0.8, label="complicación")
    ymax = float((a0 + lin + salto).max())
    for f in sx.FENOTIPOS:
        mk = tr["Fenotipo"] == sx.NOMBRE_FEN[f]
        ax.plot(x[mk], np.full(mk.sum(), -0.04 * ymax), "s", color=COLOR_FEN[f], ms=5, label=f"{sx.NOMBRE_FEN[f]}")
    ax.yaxis.set_major_formatter(_EUR0)
    ax.set_xlabel("paciente simulado"); ax.set_ylabel("€")
    ax.set_title(titulo or f"Funcionamiento simulado · primeros {len(tr)} pacientes")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=6, fontsize=7.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------- bloque B
def fig_hist_brote(tB, totales, titulo=None):
    d = np.asarray(totales)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.hist(d, bins=30, density=True, color=CON, alpha=0.45, edgecolor="white", label="brotes simulados")
    m, s = tB["total_con"], tB["sd_con"]
    xx = np.linspace(min(d.min(), m - 4 * s), max(d.max(), m + 4 * s), 300)
    ax.plot(xx, np.exp(-0.5 * ((xx - m) / s) ** 2) / (s * math.sqrt(2 * math.pi)), color=TEXTO, lw=2,
            label="teórica (aprox. normal)")
    ax.axvline(m, color=CON, ls="--", lw=1.4, label=f"E = {sx.eur(m)}")
    ax.axvline(tB["P95"], color=SERIES[3], ls="-.", lw=1.4, label=f"P95 = {sx.eur(tB['P95'])}")
    ax.xaxis.set_major_formatter(_MILES)
    ax.set_xlabel("coste total del brote con capacidad limitada (€)"); ax.set_ylabel("densidad")
    ax.set_title(titulo or f"Coste total del brote · {len(d)} brotes simulados")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    return fig


def fig_fases(tB, resB, titulo=None):
    pf = tB["por_fase"]; fases = list(pf)
    x = np.arange(len(fases))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    v = [pf[f]["delta"] for f in fases]
    s = [resB.loc[f"Sobrecoste · {f} (€)", "Simulado · media"] for f in fases]
    a1.bar(x, v, color=[COLOR_FASE.get(f, SERIES[0]) for f in fases], width=0.6)
    a1.plot(x, s, "D", color=TEXTO, ms=6, label="simulado")
    for i, y in enumerate(v):
        a1.text(i, y, f"{sx.eur(y)}\n({sx.pct(pf[fases[i]]['cuota_delta'], 0)})", ha="center", va="bottom", fontsize=7.5)
    a1.set_xticks(x); a1.set_xticklabels(fases); a1.yaxis.set_major_formatter(_EUR0)
    a1.set_ylim(0, max(max(v), 1) * 1.35)
    a1.set_title("Sobrecoste por falta de capacidad por fase (O5.3)", fontsize=9.5)
    a1.legend(loc="upper right", fontsize=7.5)
    lc = [pf[f]["lambda_comp"] for f in fases]
    a2.bar(x, lc, color=[COLOR_FASE.get(f, SERIES[0]) for f in fases], width=0.6, alpha=0.85)
    a2.axhline(tB["camas"], color=CRITICO, ls="--", lw=1.5, label=f"{tB['camas']} camas UCI propias")
    if tB["extra"]:
        a2.axhline(tB["camas"] + tB["extra"], color=SERIES[6], ls=":", lw=1.5, label=f"+{tB['extra']} de alquiler")
    for i, f in enumerate(fases):
        a2.text(i, lc[i], f"{sx.num(lc[i], 2)}/día\nλ = {sx.num(pf[f]['lambda'], 0)} · P = {sx.pct(pf[f]['p_persistente'], 0)}",
                ha="center", va="bottom", fontsize=7.5)
    a2.set_xticks(x); a2.set_xticklabels(fases)
    a2.set_ylim(0, max(max(lc), tB["camas"] + tB["extra"]) * 1.4)
    a2.set_title("Complicados esperados por día frente a las camas", fontsize=9.5)
    a2.legend(loc="upper right", fontsize=7.5)
    if titulo:
        fig.suptitle(titulo, fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    return fig


def fig_dias(dias, camas, extra=0, titulo=None):
    """Un brote simulado: complicados por día frente a las camas; los derivados, en rojo."""
    d = dias[dias["Réplica"] == dias["Réplica"].min()]
    fig, ax = plt.subplots(figsize=(10, 3.3))
    cols = [COLOR_FASE.get(f, SERIES[0]) for f in d["Fase"]]
    atend = d["Complicados"] - d["Derivados"]
    ax.bar(d["Día"], atend, color=cols, width=0.85, alpha=0.75)
    ax.bar(d["Día"], d["Derivados"], bottom=atend, color=CRITICO, width=0.85, label="derivados a la clínica externa")
    ax.axhline(camas, color=TEXTO, ls="--", lw=1.3, label=f"{camas} camas propias")
    if extra:
        ax.axhline(camas + extra, color=SERIES[6], ls=":", lw=1.3, label=f"+{extra} de alquiler")
    from matplotlib.patches import Patch
    h, l = ax.get_legend_handles_labels()
    for f, c in COLOR_FASE.items():
        h.append(Patch(facecolor=c, alpha=0.75)); l.append(f"complicados atendidos · {f}")
    ax.set_xlabel("día del brote"); ax.set_ylabel("pacientes complicados")
    ax.set_title(titulo or "Un brote simulado: complicados por día y desbordamientos")
    ax.legend(h, l, loc="upper right", fontsize=7.2, ncol=2)
    ax.set_ylim(0, max(d["Complicados"].max(), camas + extra) * 1.45)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------- capacidad
def fig_camas(df, b_opt, titulo=None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.5))
    b = df.index.to_numpy()
    a1.plot(b, df["Sobrecoste (€)"], "o-", color=CON, lw=2, label="sobrecoste esperado")
    a1.plot(b, df["Reserva de camas (€)"], "s-", color=SIN, lw=2, label="reserva de camas")
    a1.plot(b, df["Sobrecoste + reserva (€)"], "D-", color=SERIES[0], lw=2.4, label="sobrecoste + reserva")
    a1.plot([b_opt], [df.loc[b_opt, "Sobrecoste + reserva (€)"]], "*", color=SERIES[3], ms=16, mec=TEXTO, label=f"óptimo: {b_opt} camas")
    a1.yaxis.set_major_formatter(_MILES)
    a1.set_xlabel("camas UCI reservadas"); a1.set_ylabel("€ por brote")
    a1.set_title("O6.1 · sobrecoste y coste de reserva", fontsize=9.5)
    a1.legend(fontsize=7.5)
    m = df["Ahorro marginal de la cama (€)"].to_numpy(float)
    a2.bar(b[1:], m[1:], color=[BUENO if v > df["Coste de reservarla (€)"].iloc[0] else CRITICO for v in m[1:]], width=0.6)
    a2.axhline(df["Coste de reservarla (€)"].iloc[0], color=TEXTO, ls="--", lw=1.4,
               label=f"coste de reservar una cama ({sx.eur(df['Coste de reservarla (€)'].iloc[0])})")
    for x_, y in zip(b[1:], m[1:]):
        a2.text(x_, y, sx.num(y / 1000, 1) + "k", ha="center", va="bottom", fontsize=7.5)
    a2.yaxis.set_major_formatter(_MILES)
    a2.set_xlabel("cama añadida (de b − 1 a b)"); a2.set_ylabel("€ por brote")
    a2.set_title("Ahorro marginal de añadir una cama", fontsize=9.5)
    a2.legend(fontsize=7.5)
    if titulo:
        fig.suptitle(titulo, fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    return fig


def fig_cuota(df, titulo=None):
    fig, ax = plt.subplots(figsize=(8, 3.0))
    x = df.index.to_numpy()
    v = df["Cuota máxima por cama y día (€)"].to_numpy(float)
    ax.bar(x, v, color=SERIES[6], width=0.6)
    for a, y in zip(x, v):
        ax.text(a, y, sx.eur(y), ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("cama adicional alquilada en el Pico (1.ª, 2.ª, …)"); ax.set_ylabel("€ por cama y día")
    ax.yaxis.set_major_formatter(_EUR0)
    ax.set_title(titulo or "O8.5 · cuota diaria máxima por cama de alquiler durante el Pico")
    ax.margins(y=0.15)
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
        axs[0].text(i, v, sx.num(v, 0), ha="center", va="bottom", fontsize=7)
    lo, hi = min(vals), max(vals); m = (hi - lo) or 1
    axs[0].set_ylim(max(0, lo - 1.2 * m), hi + 0.5 * m); axs[0].yaxis.set_major_formatter(_EUR0)
    axs[0].set_title("E(Y) por paciente (€)", fontsize=9)
    vals = [a["Pcomp"] for a in A]
    axs[1].bar(x, vals, color=cols, width=0.65)
    for i, v in enumerate(vals):
        axs[1].text(i, v, sx.num(100 * v, 1), ha="center", va="bottom", fontsize=7)
    axs[1].yaxis.set_major_formatter(_PCT); axs[1].set_ylim(0, max(vals) * 1.25)
    axs[1].set_title("P(complicación)", fontsize=9)
    s_ = np.array([b["total_sin"] for b in B]); d_ = np.array([b["delta"] for b in B])
    r_ = np.array([b["reserva"] + float(res[n]["esc"].get("coste_impl") or 0) for n, b in zip(nombres, B)])
    axs[2].bar(x, s_, color=SIN, width=0.65, label="sin restricción")
    axs[2].bar(x, d_, bottom=s_, color=CON, width=0.65, label="sobrecoste")
    axs[2].bar(x, r_, bottom=s_ + d_, color=SERIES[6], width=0.65, label="reserva + implantación")
    tot = s_ + d_ + r_
    for i in x:
        axs[2].text(i, tot[i], sx.num(tot[i] / 1e6, 2) + "M", ha="center", va="bottom", fontsize=7)
    lo, hi = tot.min(), tot.max(); m = (hi - lo) or 1
    axs[2].set_ylim(max(0, lo - 2 * m), hi + 0.4 * m)
    axs[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: sx.num(y / 1e6, 1) + "M"))
    axs[2].set_title("Coste total del brote (€)", fontsize=9); axs[2].legend(fontsize=7, loc="upper right", ncol=3)
    axs[2].set_ylim(max(0, lo - 2 * m), hi + 0.8 * m)
    vals = [b["derivados"] for b in B]
    axs[3].bar(x, vals, color=cols, width=0.65)
    for i, v in enumerate(vals):
        axs[3].text(i, v, sx.num(v, 1), ha="center", va="bottom", fontsize=7)
    axs[3].set_ylim(0, max(max(vals), 1) * 1.25)
    axs[3].set_title("Pacientes derivados por brote", fontsize=9)
    for ax in axs:
        _etiquetas(ax, nombres)
    fig.text(0.5, 0.005, "Los ejes de E(Y) y del coste total no empiezan en 0 para que se aprecien las diferencias.",
             ha="center", fontsize=7.5, color=TEXTO2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def fig_ahorros(dif, titulo=None):
    """Ahorro por brote frente al escenario base (teórico y simulado con IC95 % pareado)."""
    nombres = list(dif.index)
    x = np.arange(len(nombres))
    teo = dif["Ahorro por brote · teórico (€)"].to_numpy(float)
    sim = dif["Ahorro por brote · simulado (€)"].to_numpy(float)
    ic = np.array([list(v) for v in dif["IC95 ahorro por brote (€)"]])
    sob = dif["Ahorro en sobrecoste de capacidad · teórico (€)"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(x - 0.18, teo, width=0.36, color=[BUENO if v >= 0 else CRITICO for v in teo], alpha=0.8, label="ahorro total (teórico)")
    ax.bar(x + 0.18, sob, width=0.36, color=SERIES[0], alpha=0.6, label="ahorro en sobrecoste de capacidad (teórico)")
    ax.errorbar(x - 0.18, sim, yerr=[sim - ic[:, 0], ic[:, 1] - sim], fmt="D", color=TEXTO, ms=5, capsize=3,
                label="simulado ± IC95 % (pareado)")
    for i, v in enumerate(teo):
        ax.text(i - 0.18, v, sx.num(v / 1000, 0) + "k", ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)
    ax.axhline(0, color=TEXTO2, lw=1)
    ax.yaxis.set_major_formatter(_MILES)
    _etiquetas(ax, nombres)
    ax.set_ylabel("€ por brote")
    ax.set_title(titulo or "Ahorro por brote frente al escenario base (incluida reserva e implantación)")
    ax.legend(loc="best", fontsize=7.5)
    ax.margins(y=0.15)
    fig.tight_layout()
    return fig


def fig_validacion(res):
    nombres = list(res)
    kA = [("Coste esperado E(Y) (€)", "E(Y)"), ("Probabilidad de complicación P(D > umbral)", "P(comp.)"),
          ("Percentil 95 de Y (€)", "P95 de Y")]
    kB = [("Coste total esperado del brote con capacidad (€)", "coste brote"),
          ("Sobrecoste por falta de capacidad (€)", "sobrecoste"),
          ("Pacientes derivados a la clínica externa", "derivados")]
    fig, ax = plt.subplots(figsize=(9, 3.3))
    ks = kA + kB
    ancho = 0.8 / len(nombres)
    for i, n in enumerate(nombres):
        v = []
        for k, _ in ks:
            tab = res[n]["resA"] if (k, _) in kA else res[n]["resB"]
            e = tab.loc[k, "Error relativo"]
            sin_veredicto = tab.loc[k, "Teórico en IC95"] is None      # suceso raro sin casos simulados
            v.append(0.0 if e is None or e != e or sin_veredicto else 100 * e)
        ax.bar(np.arange(len(ks)) + i * ancho - 0.4 + ancho / 2, v, width=ancho * 0.9, color=color_escenario(i), label=n)
    ax.axhline(0, c=TEXTO2, lw=1)
    ax.set_xticks(range(len(ks))); ax.set_xticklabels([e for _, e in ks], fontsize=8.5)
    ax.set_ylabel("error relativo (%)")
    ax.set_title("Validación: error relativo de la simulación frente a la teoría")
    ax.legend(fontsize=6.5, ncol=min(3, len(nombres)))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------- sensibilidad
def fig_sens_param(df, elas, titulo=None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.7), gridspec_kw={"width_ratios": [1.6, 1]})
    for i, (par, d) in enumerate(df.groupby("Parámetro", sort=False)):
        a1.plot(100 * d["Variación"], d["Teórico · Var(Y)"], "o-", color=SERIES[i], lw=1.8, ms=4, label=par)
    a1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: sx.num(y / 1e6, 1) + "M"))
    a1.set_xlabel("variación del parámetro (%)"); a1.set_ylabel("Var(Y) teórica (€²)")
    a1.set_title("Var(Y) al variar cada parámetro (resto fijo)", fontsize=9.5)
    a1.legend(fontsize=6.8)
    nombres = list(elas); v = [elas[k] for k in nombres]
    a2.barh(nombres, [abs(y) for y in v], color=SERIES[:len(v)], height=0.55)
    for i, y in enumerate(v):
        a2.text(abs(y), i, "  " + sx.num(y, 2), va="center", fontsize=8.5)
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
        t = df["Teórico · S₁"].to_numpy(float)
        ok = ~np.isnan(t)
        ax.plot(x[ok] - 0.2, t[ok], "D", color=TEXTO, ms=6, label="S₁ teórico del fenotipo")
    for i, (a, b) in enumerate(zip(df["Índice de primer orden S₁"], df["Índice total S_T"])):
        ax.text(i - 0.2, a, sx.pct(a, 0), ha="center", va="bottom", fontsize=7.5)
        ax.text(i + 0.2, b, sx.pct(b, 0), ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(df.index)
    ax.yaxis.set_major_formatter(_PCT); ax.set_ylim(0, 1.15)
    ax.set_title(titulo or "O2.2 · índices de Sobol de las entradas aleatorias sobre Y")
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    return fig


def fig_umbral(df, titulo=None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.3))
    x = np.arange(len(df)); et = [f"{sx.num(u, 0)} días" for u in df.index]
    a1.bar(x - 0.2, df["Teórico · E(Y) (€)"], width=0.4, color=SERIES[0], label="E(Y)")
    a1.bar(x + 0.2, df["Teórico · P95 de Y (€)"], width=0.4, color=SERIES[3], label="P95 de Y")
    a1.plot(x - 0.2, df["Simulado · E(Y) (€)"], "D", color=TEXTO, ms=5, label="simulado")
    a1.plot(x + 0.2, df["Simulado · P95 de Y (€)"], "D", color=TEXTO, ms=5)
    a1.set_xticks(x); a1.set_xticklabels(et); a1.yaxis.set_major_formatter(_EUR0)
    a1.set_title("Coste por paciente según el umbral", fontsize=9.5); a1.legend(fontsize=7.5)
    a2.bar(x, df["Teórico · P(complicación)"], color=CON, width=0.55)
    for i, (p_, s_) in enumerate(zip(df["Teórico · P(complicación)"], df["Teórico · sobrecoste del brote (€)"])):
        a2.text(i, p_, f"{sx.pct(p_, 1)}\nsobrecoste {sx.num(s_ / 1000, 0)}k", ha="center", va="bottom", fontsize=7.5)
    a2.set_xticks(x); a2.set_xticklabels(et); a2.yaxis.set_major_formatter(_PCT)
    a2.set_ylim(0, df["Teórico · P(complicación)"].max() * 1.45)
    a2.set_title("P(complicación) y sobrecoste del brote", fontsize=9.5)
    if titulo:
        fig.suptitle(titulo, fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    return fig


def fig_cox(df, titulo=None):
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    x = np.arange(len(df))
    v = df["Sobrecoste (€)"].to_numpy(float)
    ic = np.array([list(t) for t in df["IC95 Sobrecoste (€)"]])
    ax.bar(x, v, color=SERIES[6], width=0.55)
    ax.errorbar(x, v, yerr=[v - ic[:, 0], ic[:, 1] - v], fmt="none", color=TEXTO, capsize=4)
    for i, y in enumerate(v):
        ax.text(i, y, sx.num(y / 1000, 1) + "k", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"CV = {sx.num(c, 1)}" for c in df.index])
    ax.yaxis.set_major_formatter(_MILES); ax.set_ylabel("€ por brote")
    ax.set_title(titulo or "O8.4 · sobrecoste con intensidad diaria aleatoria (proceso de Cox)")
    ax.margins(y=0.18)
    fig.tight_layout()
    return fig


def fig_optim(df, opt, titulo=None):
    piv = df.pivot(index="Reducción de L₃", columns="Camas", values="Coste total esperado (€)")
    fac = df.pivot(index="Reducción de L₃", columns="Camas", values="Factible")
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    im = ax.imshow(piv.to_numpy() / 1e6, cmap="Blues_r", aspect="auto")
    cb = fig.colorbar(im, ax=ax); cb.set_label("coste total esperado (M€)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ok = bool(fac.iloc[i, j])
            ax.text(j, i, sx.num(piv.iloc[i, j] / 1e6, 3) if ok else "fuera", ha="center", va="center",
                    fontsize=7 if ok else 6.5, color=TEXTO if ok else "white", style="normal" if ok else "italic")
    if opt is not None:
        i = list(piv.index).index(opt["Reducción de L₃"]); j = list(piv.columns).index(opt["Camas"])
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=SERIES[3], lw=3))
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels([sx.pct(r, 0) for r in piv.index])
    ax.set_xlabel("camas UCI reservadas"); ax.set_ylabel("reducción de la carga L del Persistente")
    ax.grid(False)
    ax.set_title(titulo or "O8.3 · coste total esperado en M€ («fuera» = supera el presupuesto; recuadro = óptimo)")
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
    ax.axhline(df["Teórico"].iloc[0], color=TEXTO, ls="--", lw=1.4, label=f"teórico {sx.eur(df['Teórico'].iloc[0])}")
    ax.set_xticks(x); ax.set_xticklabels([sx.num(v, 0) for v in df.index])
    ax.yaxis.set_major_formatter(_EUR0 if metrica == "paciente" else _MILES)
    ax.set_xlabel("pacientes simulados" if metrica == "paciente" else "brotes simulados"); ax.set_ylabel("€")
    ax.set_title(titulo or ("Convergencia · E(Y) por paciente" if metrica == "paciente" else "Convergencia · coste del brote"))
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
        ax.text(i, y, sx.num(y, 1 if metrica == "paciente" else 0), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([sx.num(t, 0) for t in df.index])
    ax.set_xlabel("pacientes simulados" if metrica == "paciente" else "brotes simulados"); ax.set_ylabel("€")
    ax.yaxis.set_major_formatter(_EUR0)
    ax.set_title(titulo or "Variabilidad de la estimación entre réplicas")
    ax.legend(loc="upper right", fontsize=8)
    ax.margins(y=0.15)
    fig.tight_layout()
    return fig
